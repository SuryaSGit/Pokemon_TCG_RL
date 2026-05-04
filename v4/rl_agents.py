"""
RL Agents for Pokemon TCG v4 Environment
─────────────────────────────────────────
Two agents trained via self-play:
  • Agent 0 (Lucario deck)  — trained with PPO
  • Agent 1 (Starmie deck)  — trained with DQN

Both use NumPy neural networks (no PyTorch/TF dependency).
Self-play: each agent plays as its fixed deck, alternating sides.
"""

from __future__ import annotations
import numpy as np
import copy
import time
import os
import sys
from collections import deque
from typing import List, Tuple, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptcg_env import (
    PokemonTCGEnv, StateEncoder, ActionMapper,
    ActionType, HeuristicAgent, RandomAgent,
    build_lucario_deck, build_starmie_deck,
)

OBS_SIZE = StateEncoder.STATE_SIZE      # 173
ACT_SIZE = ActionMapper.TOTAL_ACTIONS   # 217

# ══════════════════════════════════════════════════════════════════════════════
# NUMPY NEURAL NETWORK PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class LinearLayer:
    """Dense layer with He initialisation."""

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.standard_normal((in_dim, out_dim)).astype(np.float32) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t  = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dW = self._input.T @ dout if self._input.ndim > 1 else np.outer(self._input, dout)
        self.db = dout.sum(axis=0) if dout.ndim > 1 else dout
        return dout @ self.W.T

    def adam_update(self, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.vW = beta2 * self.vW + (1 - beta2) * self.dW ** 2
        mW_hat  = self.mW / (1 - beta1 ** self.t)
        vW_hat  = self.vW / (1 - beta2 ** self.t)
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

        self.mb = beta1 * self.mb + (1 - beta1) * self.db
        self.vb = beta2 * self.vb + (1 - beta2) * self.db ** 2
        mb_hat  = self.mb / (1 - beta1 ** self.t)
        vb_hat  = self.vb / (1 - beta2 ** self.t)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def get_params(self) -> dict:
        return {"W": self.W.copy(), "b": self.b.copy()}

    def set_params(self, p: dict):
        self.W[:] = p["W"]
        self.b[:] = p["b"]


class MLP:
    """Multi-layer perceptron: [in → hidden → hidden → out] with ReLU."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 rng: np.random.Generator):
        self.layers = [
            LinearLayer(in_dim,  hidden, rng),
            LinearLayer(hidden,  hidden, rng),
            LinearLayer(hidden,  out_dim, rng),
        ]
        self._pre_acts: List[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._pre_acts = []
        h = x.astype(np.float32)
        for i, layer in enumerate(self.layers):
            h = layer.forward(h)
            if i < len(self.layers) - 1:
                self._pre_acts.append(h.copy())
                h = relu(h)
        return h

    def backward(self, dout: np.ndarray) -> np.ndarray:
        d = dout.astype(np.float32)
        for i in reversed(range(len(self.layers))):
            if i < len(self.layers) - 1:
                d = d * relu_grad(self._pre_acts[i])
            d = self.layers[i].backward(d)
        return d

    def update(self, lr: float):
        for layer in self.layers:
            layer.adam_update(lr)

    def get_params(self) -> List[dict]:
        return [l.get_params() for l in self.layers]

    def set_params(self, params: List[dict]):
        for l, p in zip(self.layers, params):
            l.set_params(p)

    def copy_params_from(self, other: 'MLP'):
        self.set_params(other.get_params())


# ══════════════════════════════════════════════════════════════════════════════
# ACTOR-CRITIC NETWORK (for PPO)
# ══════════════════════════════════════════════════════════════════════════════

class ActorCritic:
    """
    Shared 2-layer trunk → actor head (policy logits) + critic head (value).
    """

    def __init__(self, obs_size: int, act_size: int, hidden: int,
                 rng: np.random.Generator):
        self.t0     = LinearLayer(obs_size, hidden, rng)
        self.t1     = LinearLayer(hidden,   hidden, rng)
        self.actor  = LinearLayer(hidden, act_size, rng)
        self.critic = LinearLayer(hidden, 1, rng)

    def forward(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        h0   = relu(self.t0.forward(obs))
        feat = relu(self.t1.forward(h0))
        logits = self.actor.forward(feat)
        value  = float(self.critic.forward(feat)[0])
        return logits, value

    def masked_policy(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(obs)
        logits = logits + (mask - 1) * 1e9
        return softmax(logits)

    def get_params(self) -> dict:
        return {
            "t0":     self.t0.get_params(),
            "t1":     self.t1.get_params(),
            "actor":  self.actor.get_params(),
            "critic": self.critic.get_params(),
        }

    def set_params(self, p: dict):
        self.t0.set_params(p["t0"])
        self.t1.set_params(p["t1"])
        self.actor.set_params(p["actor"])
        self.critic.set_params(p["critic"])

    def copy_params_from(self, other: 'ActorCritic'):
        self.set_params(other.get_params())

    def update_actor_critic(self, obs: np.ndarray, advantages: np.ndarray,
                             returns: np.ndarray, old_log_probs: np.ndarray,
                             actions: np.ndarray, masks: np.ndarray,
                             lr: float, clip_eps: float = 0.2,
                             vf_coef: float = 0.5, ent_coef: float = 0.01):
        B = obs.shape[0]
        if B == 0:
            return {"pg_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0}

        obs = obs.astype(np.float32)

        pre0 = obs @ self.t0.W + self.t0.b
        h0   = relu(pre0)
        pre1 = h0  @ self.t1.W + self.t1.b
        feat = relu(pre1)

        logits  = feat @ self.actor.W  + self.actor.b
        values  = (feat @ self.critic.W + self.critic.b).squeeze(-1)

        logits_m = logits + (masks.astype(np.float32) - 1.0) * 1e9
        logits_m -= logits_m.max(axis=1, keepdims=True)
        exp_l = np.exp(logits_m)
        probs = exp_l / (exp_l.sum(axis=1, keepdims=True) + 1e-10)

        log_pas = np.log(probs[np.arange(B), actions] + 1e-8)
        ratios  = np.exp(log_pas - old_log_probs.astype(np.float32))

        pg1     = ratios * advantages
        pg2     = np.clip(ratios, 1-clip_eps, 1+clip_eps) * advantages
        pg_loss = float(-np.mean(np.minimum(pg1, pg2)))
        vf_loss = float(0.5 * np.mean((values - returns) ** 2))
        entropy = float(-np.mean(np.sum(probs * np.log(probs + 1e-8), axis=1)))

        d_val    = (vf_coef * (values - returns) / B).astype(np.float32)
        dW_crit  = (feat.T @ d_val[:, None]).astype(np.float32)
        db_crit  = d_val.sum(keepdims=True).reshape(1,).astype(np.float32)
        d_feat_c = (d_val[:, None] * self.critic.W.T).astype(np.float32)

        clipped_high   = (advantages > 0) & (ratios > 1.0 + clip_eps)
        clipped_low    = (advantages < 0) & (ratios < 1.0 - clip_eps)
        gradient_flows = ~(clipped_high | clipped_low)
        coeff    = advantages * ratios * gradient_flows / B
        d_log_pa = -coeff

        d_p          = np.zeros_like(probs)
        p_chosen     = probs[np.arange(B), actions]
        d_p[np.arange(B), actions] = d_log_pa / (p_chosen + 1e-8)
        d_ent        = -ent_coef * (np.log(probs + 1e-8) + 1.0) / B
        d_p         += d_ent
        dot          = (d_p * probs).sum(axis=1, keepdims=True)
        d_logits_in  = probs * (d_p - dot)

        dW_act   = (feat.T @ d_logits_in).astype(np.float32)
        db_act   = d_logits_in.sum(axis=0).astype(np.float32)
        d_feat_a = (d_logits_in @ self.actor.W.T).astype(np.float32)

        d_feat   = (d_feat_a + d_feat_c).astype(np.float32)
        d_pre1   = d_feat * relu_grad(pre1)
        dW_t1    = (h0.T @ d_pre1).astype(np.float32)
        db_t1    = d_pre1.sum(axis=0).astype(np.float32)
        d_h0     = (d_pre1 @ self.t1.W.T) * relu_grad(pre0)
        dW_t0    = (obs.T @ d_h0).astype(np.float32)
        db_t0    = d_h0.sum(axis=0).astype(np.float32)

        self.critic.dW = dW_crit; self.critic.db = db_crit
        self.actor.dW  = dW_act;  self.actor.db  = db_act
        self.t1.dW = dW_t1; self.t1.db = db_t1
        self.t0.dW = dW_t0; self.t0.db = db_t0

        self.critic.adam_update(lr)
        self.actor.adam_update(lr)
        self.t1.adam_update(lr)
        self.t0.adam_update(lr)

        return {"pg_loss": pg_loss, "vf_loss": vf_loss, "entropy": entropy}


# ══════════════════════════════════════════════════════════════════════════════
# Q-NETWORK (for DQN)
# ══════════════════════════════════════════════════════════════════════════════

class QNetwork:
    def __init__(self, obs_size: int, act_size: int, hidden: int,
                 rng: np.random.Generator):
        self.net = MLP(obs_size, hidden, act_size, rng)

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        return self.net.forward(obs.astype(np.float32))

    def masked_greedy(self, obs: np.ndarray, mask: np.ndarray) -> int:
        q = self.q_values(obs)
        q_masked = np.where(mask > 0, q, -np.inf)
        return int(np.argmax(q_masked))

    def update(self, obs: np.ndarray, actions: np.ndarray, targets: np.ndarray,
               masks: np.ndarray, lr: float):
        B = obs.shape[0]
        h0 = obs @ self.net.layers[0].W + self.net.layers[0].b
        a0 = relu(h0)
        h1 = a0 @ self.net.layers[1].W + self.net.layers[1].b
        a1 = relu(h1)
        q  = a1 @ self.net.layers[2].W + self.net.layers[2].b

        q_a = q[np.arange(B), actions]
        loss = 0.5 * np.mean((q_a - targets) ** 2)

        d_q = np.zeros_like(q)
        d_q[np.arange(B), actions] = (q_a - targets) / B

        d_a1 = d_q @ self.net.layers[2].W.T
        self.net.layers[2].dW = a1.T @ d_q
        self.net.layers[2].db = d_q.sum(axis=0)

        d_h1 = d_a1 * relu_grad(h1)
        d_a0 = d_h1 @ self.net.layers[1].W.T
        self.net.layers[1].dW = a0.T @ d_h1
        self.net.layers[1].db = d_h1.sum(axis=0)

        d_h0 = d_a0 * relu_grad(h0)
        self.net.layers[0].dW = obs.T @ d_h0
        self.net.layers[0].db = d_h0.sum(axis=0)

        for layer in self.net.layers:
            layer.adam_update(lr)

        return float(loss)

    def get_params(self) -> List[dict]:
        return self.net.get_params()

    def set_params(self, p: List[dict]):
        self.net.set_params(p)

    def copy_params_from(self, other: 'QNetwork'):
        self.set_params(other.get_params())


# ══════════════════════════════════════════════════════════════════════════════
# REPLAY BUFFER (for DQN)
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity: int, obs_size: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng      = rng
        self.ptr      = 0
        self.size     = 0
        self.obs      = np.zeros((capacity, obs_size), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.actions  = np.zeros(capacity, dtype=np.int32)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)
        self.masks    = np.zeros((capacity, ACT_SIZE), dtype=np.float32)
        self.next_masks = np.zeros((capacity, ACT_SIZE), dtype=np.float32)

    def push(self, obs, action, reward, next_obs, done, mask, next_mask):
        i = self.ptr
        self.obs[i]        = obs
        self.next_obs[i]   = next_obs
        self.actions[i]    = action
        self.rewards[i]    = reward
        self.dones[i]      = float(done)
        self.masks[i]      = mask
        self.next_masks[i] = next_mask
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = self.rng.integers(0, self.size, size=batch_size)
        return (self.obs[idxs], self.actions[idxs], self.rewards[idxs],
                self.next_obs[idxs], self.dones[idxs],
                self.masks[idxs], self.next_masks[idxs])

    def __len__(self):
        return self.size


# ══════════════════════════════════════════════════════════════════════════════
# SELF-PLAY ENVIRONMENT WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class SelfPlayEnv:
    """
    Wraps PokemonTCGEnv for self-play training.

    Agent 0 plays as Lucario (player 0) — trained with PPO.
    Agent 1 plays as Starmie (player 1) — trained with DQN.

    Reward shaping is applied on top of the base environment rewards.
    """

    def __init__(self, seed: int = 42):
        self.env   = PokemonTCGEnv(seed=seed)
        self._seed = seed

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        s = seed if seed is not None else self._seed
        self.env.reset(seed=s)
        return self._get_obs(0), self._get_obs(1)

    def _get_obs(self, player_idx: int) -> np.ndarray:
        gs = self.env.gs
        orig = gs.current_player
        gs.current_player = player_idx
        obs = StateEncoder.encode(gs)
        gs.current_player = orig
        return obs

    def _get_mask(self, player_idx: int) -> np.ndarray:
        from ptcg_env import compute_legal_mask
        gs = self.env.gs
        orig = gs.current_player
        gs.current_player = player_idx
        mask = compute_legal_mask(gs)
        gs.current_player = orig
        return mask

    @property
    def current_player(self) -> int:
        return self.env.gs.current_player

    @property
    def done(self) -> bool:
        return self.env.gs.game_over

    @property
    def winner(self) -> int:
        return self.env.gs.winner

    def obs_and_mask(self, player_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._get_obs(player_idx), self._get_mask(player_idx)

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
        """
        Execute action for current player.
        Returns (obs_p0, obs_p1, reward_p0, reward_p1, done).

        ── Reward Shaping ──────────────────────────────────────────────────
        ENERGY ATTACHMENT
          +0.20  Attach energy whose type matches any non-colorless cost on target
          +0.05  Attach colorless/generic energy

        ATTACK
          +0.30  Base reward for attacking
          +dmg/100  Proportional damage bonus
          +0.10  Extra if chosen attack is highest-damage legal move
          -0.15  Penalty for suboptimal attack (≥20 damage left on table)

        EVOLVE
          +0.25  Evolve into a higher stage (more so for Mega ex)

        PLAY POKEMON
          +0.05  Bench a pokemon that is an evolution base
          +0.02  Bench any basic

        USE_ABILITY
          +0.15  Use an active ability (meaningful game action)

        ATTACH_TOOL
          +0.10  Attach a tool to a Pokémon (one-time benefit)

        USE_STADIUM
          +0.05  Play a stadium card (changing board state)

        RETREAT
          +0.08  Retreat to a higher-HP benched Pokémon (tactical switch)
          -0.05  Retreat to a lower-HP Pokémon (possibly suboptimal)

        PROMOTE
          +0.05  Promote scaled by HP fraction of new active

        END_TURN
          -0.20  End turn while an attack was legal (strong passivity penalty)
          -0.05  End turn when energy could have been attached but wasn't

        USE_SUPPORTER / USE_ITEM
          +0.03  Small positive for using card effects
        ─────────────────────────────────────────────────────────────────────
        """
        from ptcg_env import (
            compute_legal_mask, ActionMapper as AM, ActionType as AT,
            EnergyType, EnergyCard, PokemonCard, TrainerCard,
            can_pay_cost, Stage,
        )

        cp = self.current_player
        gs = self.env.gs
        me = gs.players[cp]

        mask_before = compute_legal_mask(gs)
        attack_was_legal = any(
            mask_before[AM.ATTACK_START + i] > 0 for i in range(2)
        )
        energy_attach_legal = any(
            mask_before[AM.ATTACH_START + i] > 0
            for i in range(AM.ATTACH_COUNT)
        )
        energy_used_before = me.energy_used
        had_energy_in_hand = any(isinstance(c, EnergyCard) for c in me.hand)

        atype, params = AM.decode(action)

        # Pre-step info for attack shaping
        attack_damage_chosen  = 0
        attack_damage_best    = 0
        chosen_is_best_attack = False
        suboptimal_attack     = False

        if atype == AT.ATTACK and me.active:
            ai = params.get("atk_idx", 0)
            if ai < len(me.active.attacks):
                attack_damage_chosen = me.active.attacks[ai].damage
            for atk_i, atk in enumerate(me.active.attacks):
                if atk_i < AM.ATTACK_COUNT and mask_before[AM.ATTACK_START + atk_i]:
                    attack_damage_best = max(attack_damage_best, atk.damage)
            chosen_is_best_attack = (attack_damage_chosen >= attack_damage_best)
            suboptimal_attack = (
                not chosen_is_best_attack and
                attack_damage_best - attack_damage_chosen >= 20
            )

        # Pre-step info for energy attachment shaping
        energy_type_attached  = None
        energy_target_pokemon = None
        energy_matches_attack = False

        if atype == AT.ATTACH_ENERGY:
            slot = params.get("slot", 0)
            for c in me.hand:
                if isinstance(c, EnergyCard):
                    energy_type_attached = c.energy_type
                    break
            if slot == 0:
                energy_target_pokemon = me.active
            else:
                bi = slot - 1
                if bi < len(me.bench):
                    energy_target_pokemon = me.bench[bi]
            if energy_target_pokemon and energy_type_attached is not None:
                for atk in energy_target_pokemon.attacks:
                    for etype, count in atk.energy_cost.items():
                        if (etype != EnergyType.COLORLESS and
                                etype == energy_type_attached and count > 0):
                            energy_matches_attack = True
                            break

        # Pre-step info for evolve shaping
        evolve_is_mega = False
        evolve_stage   = None
        if atype == AT.EVOLVE:
            hi = params.get("hand_idx", 0)
            if hi < len(me.hand) and isinstance(me.hand[hi], PokemonCard):
                evolve_stage   = me.hand[hi].stage
                evolve_is_mega = me.hand[hi].is_mega_ex

        # Pre-step info for bench play shaping
        benched_is_evo_base = False
        if atype == AT.PLAY_POKEMON:
            hi = params.get("hand_idx", 0)
            if hi < len(me.hand) and isinstance(me.hand[hi], PokemonCard):
                played = me.hand[hi]
                all_cards = me.hand + me.deck + me.bench
                for c in all_cards:
                    if isinstance(c, PokemonCard) and c.evolves_from == played.name:
                        benched_is_evo_base = True
                        break

        # Pre-step info for retreat shaping
        active_hp_before = me.active.current_hp if me.active else 0
        retreat_target_hp = 0
        if atype == AT.RETREAT:
            bench_slot = params.get("bench_slot", 0)
            if bench_slot < len(me.bench):
                retreat_target_hp = me.bench[bench_slot].current_hp

        # ── Execute action ───────────────────────────────────────────────
        _, base_reward, done, info = self.env.step(action)

        # ── Compute shaping reward ───────────────────────────────────────
        shape = 0.0

        if atype == AT.ATTACK:
            shape += 0.30
            shape += attack_damage_chosen / 100.0
            if chosen_is_best_attack:
                shape += 0.10
            if suboptimal_attack:
                shape -= 0.15

        elif atype == AT.ATTACH_ENERGY:
            if energy_matches_attack:
                shape += 0.20
            else:
                shape += 0.05

        elif atype == AT.EVOLVE:
            if evolve_stage is not None:
                shape += 0.35 if evolve_is_mega else 0.25

        elif atype == AT.PLAY_POKEMON:
            shape += 0.02
            if benched_is_evo_base:
                shape += 0.05

        elif atype == AT.USE_ABILITY:
            shape += 0.15

        elif atype == AT.ATTACH_TOOL:
            shape += 0.10

        elif atype == AT.USE_STADIUM:
            shape += 0.05

        elif atype == AT.RETREAT:
            if retreat_target_hp > active_hp_before:
                shape += 0.08
            else:
                shape -= 0.05

        elif atype == AT.PROMOTE:
            if me.active is not None:
                hp_frac = me.active.current_hp / max(me.active.hp, 1)
                shape += 0.05 * hp_frac
            else:
                gs_after = self.env.gs
                promoted = gs_after.players[cp].active
                if promoted:
                    hp_frac = promoted.current_hp / max(promoted.hp, 1)
                    shape += 0.05 * hp_frac

        elif atype == AT.END_TURN:
            if attack_was_legal:
                shape -= 0.20
            elif had_energy_in_hand and not energy_used_before and energy_attach_legal:
                shape -= 0.05

        elif atype in (AT.USE_SUPPORTER, AT.USE_ITEM):
            shape += 0.03

        total = base_reward + shape
        r0 = total if cp == 0 else -total
        r1 = total if cp == 1 else -total
        return self._get_obs(0), self._get_obs(1), r0, r1, done


# ══════════════════════════════════════════════════════════════════════════════
# PPO AGENT (Lucario — Player 0)
# ══════════════════════════════════════════════════════════════════════════════

class PPOAgent:
    """On-policy PPO agent with GAE and clipped surrogate loss."""

    def __init__(self,
                 player_idx: int,
                 obs_size:   int   = OBS_SIZE,
                 act_size:   int   = ACT_SIZE,
                 hidden:     int   = 256,
                 lr:         float = 3e-4,
                 gamma:      float = 0.99,
                 lam:        float = 0.95,
                 clip_eps:   float = 0.2,
                 vf_coef:    float = 0.5,
                 ent_coef:   float = 0.02,
                 n_epochs:   int   = 4,
                 batch_size: int   = 64,
                 seed:       int   = 0):
        self.player_idx = player_idx
        self.lr         = lr
        self.gamma      = gamma
        self.lam        = lam
        self.clip_eps   = clip_eps
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef
        self.n_epochs   = n_epochs
        self.batch_size = batch_size

        self.rng = np.random.default_rng(seed)
        self.net = ActorCritic(obs_size, act_size, hidden, self.rng)

        self.obs_buf:  List[np.ndarray] = []
        self.act_buf:  List[int]        = []
        self.rew_buf:  List[float]      = []
        self.val_buf:  List[float]      = []
        self.logp_buf: List[float]      = []
        self.done_buf: List[bool]       = []
        self.mask_buf: List[np.ndarray] = []

        self.train_steps   = 0
        self.total_rewards: List[float] = []

    def act(self, obs: np.ndarray, mask: np.ndarray,
            deterministic: bool = False) -> Tuple[int, float, float]:
        logits, value = self.net.forward(obs)
        masked_logits = logits + (mask - 1) * 1e9
        probs = softmax(masked_logits)

        if deterministic:
            action = int(np.argmax(probs))
        else:
            legal = np.where(mask > 0)[0]
            legal_probs = probs[legal]
            uniform = np.ones(len(legal)) / len(legal)
            legal_probs = 0.95 * legal_probs + 0.05 * uniform
            legal_probs = legal_probs / legal_probs.sum()
            action = int(self.rng.choice(legal, p=legal_probs))

        log_prob = float(np.log(probs[action] + 1e-8))
        return action, log_prob, value

    def store(self, obs, action, reward, value, log_prob, done, mask):
        self.obs_buf.append(obs)
        self.act_buf.append(action)
        self.rew_buf.append(reward)
        self.val_buf.append(value)
        self.logp_buf.append(log_prob)
        self.done_buf.append(done)
        self.mask_buf.append(mask)

    def finish_episode(self, last_value: float = 0.0):
        self.val_buf.append(last_value)

    def _compute_gae(self) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.rew_buf)
        advantages = np.zeros(n, dtype=np.float32)
        returns    = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0

        for t in reversed(range(n)):
            done_mask = 0.0 if self.done_buf[t] else 1.0
            delta     = (self.rew_buf[t]
                         + self.gamma * self.val_buf[t + 1] * done_mask
                         - self.val_buf[t])
            last_gae  = delta + self.gamma * self.lam * done_mask * last_gae
            advantages[t] = last_gae
            returns[t]    = advantages[t] + self.val_buf[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self) -> Dict[str, float]:
        if len(self.obs_buf) < 2:
            return {}

        advantages, returns = self._compute_gae()
        n = len(self.obs_buf)

        obs_arr  = np.stack(self.obs_buf)
        act_arr  = np.array(self.act_buf,  dtype=np.int32)
        logp_arr = np.array(self.logp_buf, dtype=np.float32)
        mask_arr = np.stack(self.mask_buf)

        metrics  = {"pg_loss": 0, "vf_loss": 0, "entropy": 0}
        n_updates = 0

        for _ in range(self.n_epochs):
            idxs = self.rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = idxs[start:start + self.batch_size]
                if len(batch) < 2:
                    continue
                m = self.net.update_actor_critic(
                    obs_arr[batch], advantages[batch], returns[batch],
                    logp_arr[batch], act_arr[batch], mask_arr[batch],
                    self.lr, self.clip_eps, self.vf_coef, self.ent_coef
                )
                for k in metrics:
                    metrics[k] += m[k]
                n_updates += 1

        if n_updates > 0:
            for k in metrics:
                metrics[k] /= n_updates

        self.obs_buf.clear()
        self.act_buf.clear()
        self.rew_buf.clear()
        self.val_buf.clear()
        self.logp_buf.clear()
        self.done_buf.clear()
        self.mask_buf.clear()
        self.train_steps += 1

        return metrics

    def save(self, path: str):
        np.save(path, self.net.get_params(), allow_pickle=True)

    def load(self, path: str):
        params = np.load(path, allow_pickle=True).item()
        self.net.set_params(params)


# ══════════════════════════════════════════════════════════════════════════════
# DQN AGENT (Starmie — Player 1)
# ══════════════════════════════════════════════════════════════════════════════

class DQNAgent:
    """Off-policy DQN with experience replay, target network, and epsilon-greedy."""

    def __init__(self,
                 player_idx:      int,
                 obs_size:        int   = OBS_SIZE,
                 act_size:        int   = ACT_SIZE,
                 hidden:          int   = 256,
                 lr:              float = 1e-3,
                 gamma:           float = 0.99,
                 eps_start:       float = 1.0,
                 eps_end:         float = 0.05,
                 eps_decay_steps: int   = 30_000,
                 buffer_size:     int   = 20_000,
                 batch_size:      int   = 64,
                 target_update:   int   = 500,
                 min_buffer:      int   = 1_000,
                 seed:            int   = 1):
        self.player_idx    = player_idx
        self.lr            = lr
        self.gamma         = gamma
        self.eps           = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = (eps_start - eps_end) / eps_decay_steps
        self.batch_size    = batch_size
        self.target_update = target_update
        self.min_buffer    = min_buffer

        self.rng   = np.random.default_rng(seed)
        self.qnet  = QNetwork(obs_size, act_size, hidden, self.rng)
        self.tnet  = QNetwork(obs_size, act_size, hidden, self.rng)
        self.tnet.copy_params_from(self.qnet)

        self.buffer = ReplayBuffer(buffer_size, obs_size, self.rng)

        self.steps        = 0
        self.train_steps  = 0
        self.total_losses: List[float] = []

    def act(self, obs: np.ndarray, mask: np.ndarray,
            deterministic: bool = False) -> int:
        if not deterministic and self.rng.random() < self.eps:
            legal = np.where(mask > 0)[0]
            return int(self.rng.choice(legal))
        return self.qnet.masked_greedy(obs, mask)

    def store(self, obs, action, reward, next_obs, done, mask, next_mask):
        self.buffer.push(obs, action, reward, next_obs, done, mask, next_mask)
        self.steps += 1
        self.eps = max(self.eps_end, self.eps - self.eps_decay)

    def update(self) -> Optional[float]:
        if len(self.buffer) < self.min_buffer:
            return None

        obs, actions, rewards, next_obs, dones, masks, next_masks = \
            self.buffer.sample(self.batch_size)

        # Double DQN: online net selects action, target net evaluates
        h0n = next_obs @ self.qnet.net.layers[0].W + self.qnet.net.layers[0].b
        a0n = relu(h0n)
        h1n = a0n @ self.qnet.net.layers[1].W + self.qnet.net.layers[1].b
        a1n = relu(h1n)
        q_next_online = a1n @ self.qnet.net.layers[2].W + self.qnet.net.layers[2].b
        q_next_online = np.where(next_masks > 0, q_next_online, -np.inf)
        best_actions = np.argmax(q_next_online, axis=1)

        h0t = next_obs @ self.tnet.net.layers[0].W + self.tnet.net.layers[0].b
        a0t = relu(h0t)
        h1t = a0t @ self.tnet.net.layers[1].W + self.tnet.net.layers[1].b
        a1t = relu(h1t)
        q_next_target = a1t @ self.tnet.net.layers[2].W + self.tnet.net.layers[2].b
        q_next_vals = q_next_target[np.arange(self.batch_size), best_actions]

        targets = rewards + self.gamma * q_next_vals * (1.0 - dones)

        loss = self.qnet.update(obs, actions, targets, masks, self.lr)
        self.train_steps += 1

        if self.train_steps % self.target_update == 0:
            self.tnet.copy_params_from(self.qnet)

        return loss

    def save(self, path: str):
        np.save(path, self.qnet.get_params(), allow_pickle=True)

    def load(self, path: str):
        params = np.load(path, allow_pickle=True).tolist()
        self.qnet.set_params(params)
        self.tnet.copy_params_from(self.qnet)


# ══════════════════════════════════════════════════════════════════════════════
# BEHAVIORAL CLONING — supervised pre-training from HeuristicAgent
# ══════════════════════════════════════════════════════════════════════════════

class BehavioralCloningTrainer:
    """
    Generates (observation, expert_action) pairs by running HeuristicAgent
    on random games, then trains PPO's actor and DQN's Q-network via supervised
    cross-entropy / regression respectively.
    """

    def __init__(self, n_games: int = 300, seed: int = 7):
        self.n_games = n_games
        self.rng     = np.random.default_rng(seed)

    def collect(self, player_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from ptcg_env import HeuristicAgent, PokemonTCGEnv, compute_legal_mask

        obs_list, act_list, mask_list = [], [], []
        h = HeuristicAgent()

        for g in range(self.n_games):
            env = PokemonTCGEnv(seed=int(self.rng.integers(1e9)))
            env.reset()
            gs  = env.gs
            steps = 0

            while not gs.game_over and steps < 400:
                cp   = gs.current_player
                mask = compute_legal_mask(gs)
                obs  = StateEncoder.encode(gs)

                action = h.act(env)

                if cp == player_idx:
                    obs_list.append(obs)
                    act_list.append(action)
                    mask_list.append(mask)

                env.step(action)
                steps += 1

        obs_arr  = np.stack(obs_list).astype(np.float32)
        act_arr  = np.array(act_list, dtype=np.int32)
        mask_arr = np.stack(mask_list).astype(np.float32)
        return obs_arr, act_arr, mask_arr

    def pretrain_ppo(self, agent: PPOAgent,
                     n_epochs: int = 10,
                     batch_size: int = 128,
                     lr: float = 1e-3,
                     verbose: bool = True) -> List[float]:
        if verbose:
            print(f"  [BC-PPO] Collecting {self.n_games} expert games for P{agent.player_idx}...")
        obs, actions, masks = self.collect(agent.player_idx)
        N = len(obs)
        if verbose:
            print(f"  [BC-PPO] {N} transitions. Training {n_epochs} epochs...")

        losses = []
        for epoch in range(n_epochs):
            idxs       = self.rng.permutation(N)
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, N, batch_size):
                b  = idxs[start:start + batch_size]
                if len(b) < 4:
                    continue
                ob = obs[b].astype(np.float32)
                ac = actions[b]
                mk = masks[b].astype(np.float32)
                B  = len(b)

                pre0 = ob @ agent.net.t0.W + agent.net.t0.b
                h0   = relu(pre0)
                pre1 = h0 @ agent.net.t1.W + agent.net.t1.b
                feat = relu(pre1)
                logits = feat @ agent.net.actor.W + agent.net.actor.b

                logits_m = logits + (mk - 1.0) * 1e9
                logits_m -= logits_m.max(axis=1, keepdims=True)
                exp_l    = np.exp(logits_m)
                probs    = exp_l / (exp_l.sum(axis=1, keepdims=True) + 1e-10)

                log_pa  = np.log(probs[np.arange(B), ac] + 1e-8)
                loss    = -log_pa.mean()
                epoch_loss += loss

                d_logits               = probs.copy()
                d_logits[np.arange(B), ac] -= 1.0
                d_logits               /= B

                dot        = (d_logits * probs).sum(axis=1, keepdims=True)
                d_logits_j = probs * (d_logits - dot)

                agent.net.actor.dW = (feat.T @ d_logits_j).astype(np.float32)
                agent.net.actor.db = d_logits_j.sum(axis=0).astype(np.float32)
                d_feat_a = (d_logits_j @ agent.net.actor.W.T).astype(np.float32)

                d_pre1 = d_feat_a * relu_grad(pre1)
                agent.net.t1.dW = (h0.T  @ d_pre1).astype(np.float32)
                agent.net.t1.db = d_pre1.sum(axis=0).astype(np.float32)
                d_h0   = (d_pre1 @ agent.net.t1.W.T) * relu_grad(pre0)
                agent.net.t0.dW = (ob.T  @ d_h0).astype(np.float32)
                agent.net.t0.db = d_h0.sum(axis=0).astype(np.float32)

                agent.net.actor.adam_update(lr)
                agent.net.t1.adam_update(lr)
                agent.net.t0.adam_update(lr)
                n_batches += 1

            avg = epoch_loss / max(n_batches, 1)
            losses.append(float(avg))
            if verbose:
                print(f"  [BC-PPO] Epoch {epoch+1:>2}/{n_epochs}  CE loss = {avg:.4f}")

        return losses

    def pretrain_dqn(self, agent: DQNAgent,
                     n_epochs: int = 10,
                     batch_size: int = 128,
                     lr: float = 1e-3,
                     verbose: bool = True) -> List[float]:
        if verbose:
            print(f"  [BC-DQN] Collecting {self.n_games} expert games for P{agent.player_idx}...")
        obs, actions, masks = self.collect(agent.player_idx)
        N = len(obs)
        if verbose:
            print(f"  [BC-DQN] {N} transitions. Training {n_epochs} epochs...")

        losses = []
        for epoch in range(n_epochs):
            idxs       = self.rng.permutation(N)
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, N, batch_size):
                b  = idxs[start:start + batch_size]
                if len(b) < 4:
                    continue
                ob = obs[b].astype(np.float32)
                ac = actions[b]
                mk = masks[b].astype(np.float32)
                B  = len(b)

                targets_q = mk * 0.0
                targets_q[np.arange(B), ac] = 1.0

                h0n = ob  @ agent.qnet.net.layers[0].W + agent.qnet.net.layers[0].b
                a0n = relu(h0n)
                h1n = a0n @ agent.qnet.net.layers[1].W + agent.qnet.net.layers[1].b
                a1n = relu(h1n)
                q   = a1n @ agent.qnet.net.layers[2].W + agent.qnet.net.layers[2].b

                diff    = (q - targets_q) * mk
                loss    = 0.5 * (diff ** 2).mean()
                epoch_loss += loss

                d_q  = diff * mk / (mk.sum() + 1e-8)
                d_a1 = d_q  @ agent.qnet.net.layers[2].W.T
                agent.qnet.net.layers[2].dW = (a1n.T @ d_q).astype(np.float32)
                agent.qnet.net.layers[2].db = d_q.sum(axis=0).astype(np.float32)

                d_h1 = d_a1 * relu_grad(h1n)
                d_a0 = d_h1 @ agent.qnet.net.layers[1].W.T
                agent.qnet.net.layers[1].dW = (a0n.T @ d_h1).astype(np.float32)
                agent.qnet.net.layers[1].db = d_h1.sum(axis=0).astype(np.float32)

                d_h0 = d_a0 * relu_grad(h0n)
                agent.qnet.net.layers[0].dW = (ob.T @ d_h0).astype(np.float32)
                agent.qnet.net.layers[0].db = d_h0.sum(axis=0).astype(np.float32)

                for layer in agent.qnet.net.layers:
                    layer.adam_update(lr)
                n_batches += 1

            agent.tnet.copy_params_from(agent.qnet)

            avg = epoch_loss / max(n_batches, 1)
            losses.append(float(avg))
            if verbose:
                print(f"  [BC-DQN] Epoch {epoch+1:>2}/{n_epochs}  MSE loss = {avg:.4f}")

        return losses


def pretrain_agents(ppo: PPOAgent, dqn: DQNAgent,
                    n_games: int = 300, n_epochs: int = 8,
                    verbose: bool = True):
    """Pretrain both agents from HeuristicAgent demonstrations."""
    bc = BehavioralCloningTrainer(n_games=n_games, seed=42)
    if verbose:
        print("\n  Behavioral Cloning pre-training...")
    bc.pretrain_ppo(ppo, n_epochs=n_epochs, lr=5e-4, verbose=verbose)
    bc.pretrain_dqn(dqn, n_epochs=n_epochs, lr=5e-4, verbose=verbose)
    if verbose:
        print("  Pre-training complete.\n")


# ══════════════════════════════════════════════════════════════════════════════
# SELF-PLAY TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_selfplay(
    n_episodes:      int   = 3_000,
    ppo_rollout_len: int   = 512,
    eval_every:      int   = 200,
    eval_games:      int   = 50,
    seed:            int   = 42,
    verbose:         bool  = True,
) -> Tuple[PPOAgent, DQNAgent, Dict]:
    """
    Train PPO (Lucario, P0) vs DQN (Starmie, P1) via self-play.

    Each episode:
      - Both agents play their role until game over.
      - PPO stores transitions in its rollout buffer.
      - DQN stores transitions in its replay buffer.
      - After ppo_rollout_len total PPO turns → PPO updates.
      - After each DQN turn → DQN updates (if buffer ready).
    """
    rng = np.random.default_rng(seed)

    ppo_agent = PPOAgent(player_idx=0, seed=int(rng.integers(1e6)))
    dqn_agent = DQNAgent(player_idx=1, seed=int(rng.integers(1e6)))

    history = {
        "episode":     [],
        "p0_winrate":  [],
        "p1_winrate":  [],
        "avg_turns":   [],
        "ppo_entropy": [],
        "dqn_loss":    [],
        "ppo_pg_loss": [],
    }

    ppo_turns_since_update = 0
    recent_wins  = deque(maxlen=eval_games)
    recent_turns = deque(maxlen=eval_games)

    t_start = time.time()

    for ep in range(n_episodes):
        env_seed = int(rng.integers(1e9))
        senv = SelfPlayEnv(seed=env_seed)
        senv.reset(seed=env_seed)

        ep_turns = 0
        dqn_prev = None

        while not senv.done:
            cp = senv.current_player
            obs, mask = senv.obs_and_mask(cp)

            if cp == 0:
                action, log_prob, value = ppo_agent.act(obs, mask)
                obs0, obs1, r0, r1, done = senv.step(action)
                ppo_agent.store(obs, action, r0, value, log_prob, done, mask)
                ppo_turns_since_update += 1

                if ppo_turns_since_update >= ppo_rollout_len:
                    last_val = 0.0 if done else float(ppo_agent.net.forward(obs0)[1])
                    ppo_agent.finish_episode(last_val)
                    m = ppo_agent.update()
                    ppo_turns_since_update = 0
                    if m:
                        history["ppo_entropy"].append(m.get("entropy", 0))
                        history["ppo_pg_loss"].append(m.get("pg_loss", 0))

                ep_turns += 1

            else:
                action = dqn_agent.act(obs, mask)
                obs0, obs1, r0, r1, done = senv.step(action)

                if dqn_prev is not None:
                    p_obs, p_act, p_rew, p_mask = dqn_prev
                    next_obs1 = obs1 if not done else obs1
                    next_mask1 = senv._get_mask(1) if not done else np.zeros(ACT_SIZE, np.float32)
                    dqn_agent.store(p_obs, p_act, p_rew, obs1, False, p_mask, next_mask1)

                if done:
                    dqn_agent.store(obs, action, r1, obs1, True,
                                    mask, np.zeros(ACT_SIZE, np.float32))
                    dqn_prev = None
                else:
                    dqn_prev = (obs, action, r1, mask)

                loss = dqn_agent.update()
                if loss is not None:
                    history["dqn_loss"].append(loss)

                ep_turns += 1

        # Episode done: flush PPO if it has data
        if len(ppo_agent.obs_buf) > 0:
            ppo_agent.finish_episode(0.0)
            ppo_agent.update()
            ppo_turns_since_update = 0

        # Flush pending DQN transition
        if dqn_prev is not None:
            p_obs, p_act, p_rew, p_mask = dqn_prev
            zero_mask = np.zeros(ACT_SIZE, dtype=np.float32)
            dqn_agent.store(p_obs, p_act, p_rew, p_obs, True, p_mask, zero_mask)

        recent_wins.append(senv.winner)
        recent_turns.append(ep_turns)

        if verbose and (ep + 1) % eval_every == 0:
            n  = len(recent_wins)
            w0 = sum(1 for w in recent_wins if w == 0) / n
            w1 = sum(1 for w in recent_wins if w == 1) / n
            draw = 1.0 - w0 - w1
            avg_t = sum(recent_turns) / len(recent_turns)
            elapsed = time.time() - t_start
            eps = dqn_agent.eps

            ent  = (sum(history["ppo_entropy"][-50:]) /
                    max(1, len(history["ppo_entropy"][-50:])))
            dloss = (sum(history["dqn_loss"][-50:]) /
                     max(1, len(history["dqn_loss"][-50:])))

            print(f"  Ep {ep+1:>5}/{n_episodes} │ "
                  f"P0(PPO/Lucario)={w0:.2%} "
                  f"P1(DQN/Starmie)={w1:.2%} "
                  f"Draw={draw:.2%} │ "
                  f"AvgTurns={avg_t:.1f} │ "
                  f"ε={eps:.3f} │ "
                  f"ent={ent:.3f} │ "
                  f"dqn_loss={dloss:.4f} │ "
                  f"{elapsed:.0f}s")

            history["episode"].append(ep + 1)
            history["p0_winrate"].append(w0)
            history["p1_winrate"].append(w1)
            history["avg_turns"].append(avg_t)

    return ppo_agent, dqn_agent, history


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(ppo_agent: PPOAgent, dqn_agent: DQNAgent,
             n_games: int = 200, seed: int = 9999,
             verbose: bool = True) -> Dict:
    rng = np.random.default_rng(seed)

    class _Random:
        def act(self, senv, cp):
            obs, mask = senv.obs_and_mask(cp)
            legal = np.where(mask > 0)[0]
            return int(rng.choice(legal))

    def run_match(agent0, agent1, n, desc=""):
        wins = [0, 0, 0]
        total_turns = []
        for g in range(n):
            senv = SelfPlayEnv(seed=int(rng.integers(1e9)))
            senv.reset()
            turns = 0
            while not senv.done and turns < 800:
                cp = senv.current_player
                obs, mask = senv.obs_and_mask(cp)
                ag = agent0 if cp == 0 else agent1
                if isinstance(ag, PPOAgent):
                    action, _, _ = ag.act(obs, mask, deterministic=True)
                elif isinstance(ag, DQNAgent):
                    action = ag.act(obs, mask, deterministic=True)
                elif isinstance(ag, HeuristicAgent):
                    action = ag.act(senv.env)
                else:
                    action = ag.act(senv, cp)
                senv.step(action)
                turns += 1
            w = senv.winner
            wins[2 if w < 0 else w] += 1
            total_turns.append(turns)
        if verbose:
            print(f"  {desc:42s} │ "
                  f"P0={wins[0]:3d}/{n} ({wins[0]/n:.1%}) "
                  f"P1={wins[1]:3d}/{n} ({wins[1]/n:.1%}) "
                  f"Draw={wins[2]} │ "
                  f"AvgTurns={sum(total_turns)/len(total_turns):.1f}")
        return wins, total_turns

    if verbose:
        print("\n" + "═" * 72)
        print("EVALUATION RESULTS")
        print("═" * 72)

    heuristic = HeuristicAgent()
    rand      = _Random()
    results   = {}

    w, t = run_match(ppo_agent, dqn_agent, n_games,
                     "PPO(Lucario) vs DQN(Starmie)")
    results["ppo_vs_dqn"] = {"wins": w, "avg_turns": sum(t)/len(t)}

    w, t = run_match(ppo_agent, rand, n_games,
                     "PPO(Lucario) vs Random")
    results["ppo_vs_random"] = {"wins": w, "avg_turns": sum(t)/len(t)}

    w, t = run_match(rand, dqn_agent, n_games,
                     "Random vs DQN(Starmie)")
    results["random_vs_dqn"] = {"wins": w, "avg_turns": sum(t)/len(t)}

    w, t = run_match(heuristic, heuristic, n_games,
                     "Heuristic vs Heuristic (baseline)")
    results["heuristic_vs_heuristic"] = {"wins": w, "avg_turns": sum(t)/len(t)}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ACTION ANALYSIS UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def analyse_agent_policy(agent: PPOAgent | DQNAgent, n_states: int = 100,
                          seed: int = 42):
    rng = np.random.default_rng(seed)
    type_counts = {t: 0 for t in ActionType}
    total = 0

    for _ in range(n_states):
        env = SelfPlayEnv(seed=int(rng.integers(1e9)))
        env.reset()
        for _ in range(int(rng.integers(3, 20))):
            if env.done:
                break
            cp = env.current_player
            obs, mask = env.obs_and_mask(cp)
            legal = np.where(mask > 0)[0]
            env.step(int(rng.choice(legal)))

        if env.done:
            continue

        cp = agent.player_idx
        obs, mask = env.obs_and_mask(cp)
        if isinstance(agent, PPOAgent):
            action, _, _ = agent.act(obs, mask, deterministic=True)
        else:
            action = agent.act(obs, mask, deterministic=True)

        atype, _ = ActionMapper.decode(action)
        type_counts[atype] += 1
        total += 1

    print(f"\n  Agent P{agent.player_idx} action distribution ({total} samples):")
    for atype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(30 * count / max(total, 1))
        print(f"    {atype.name:16s} {count:4d} ({count/max(total,1):5.1%}) {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 70)
    print("POKEMON TCG V4 SELF-PLAY RL TRAINING")
    print("  Agent 0: PPO  (Mega Lucario ex deck)")
    print("  Agent 1: DQN  (Mega Starmie ex deck)")
    print("═" * 70)
    print(f"\nObservation space: {OBS_SIZE}  │  Action space: {ACT_SIZE}")
    print()

    N_EPISODES = 3_000

    print(f"Training for {N_EPISODES} self-play episodes...")
    print(f"{'─'*70}")

    ppo_agent, dqn_agent, history = train_selfplay(
        n_episodes      = N_EPISODES,
        ppo_rollout_len = 512,
        eval_every      = 200,
        eval_games      = 200,
        seed            = 42,
        verbose         = True,
    )

    os.makedirs("v4/models", exist_ok=True)
    ppo_agent.save("v4/models/ppo_lucario.npy")
    dqn_agent.save("v4/models/dqn_starmie.npy")
    print("\n✓ Agents saved to v4/models/")

    results = evaluate(ppo_agent, dqn_agent, n_games=200, verbose=True)

    print()
    analyse_agent_policy(ppo_agent, n_states=200)
    analyse_agent_policy(dqn_agent, n_states=200)

    if history["episode"]:
        print("\n  Training curve (winrates at eval checkpoints):")
        print(f"  {'Episode':>8} │ {'PPO(P0)':>10} {'DQN(P1)':>10} {'AvgTurns':>10}")
        print(f"  {'─'*46}")
        for i, ep in enumerate(history["episode"]):
            print(f"  {ep:>8} │ {history['p0_winrate'][i]:>9.1%} "
                  f"{history['p1_winrate'][i]:>10.1%} "
                  f"{history['avg_turns'][i]:>10.1f}")

    print("\n" + "═" * 70)
    print("DONE")
    print("═" * 70)
