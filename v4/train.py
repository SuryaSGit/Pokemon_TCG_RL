"""
Pokemon TCG v4 — Play & Train (4-Agent)
════════════════════════════════════════
Trains four independent agents (PPO + DQN for each player slot) and crowns
the best for each side as the GUI champion.

  P0 plays Starmie deck
  P1 plays Lucario deck

Usage:
  python play_and_train.py play   [--deck lucario|starmie]
  python play_and_train.py train  [--episodes 500] [--threshold 50]
  python play_and_train.py benchmark [--games 100] [--deck lucario|starmie]
  python play_and_train.py stats
"""

from __future__ import annotations
import sys, os, time, json, textwrap
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, List

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from ptcg_env import (
    PokemonTCGEnv, StateEncoder, ActionMapper, ActionType,
    GameState, PlayerState, PokemonCard, EnergyCard, TrainerCard,
    Stage, CardType, EnergyType, compute_legal_mask,
    build_lucario_deck, build_starmie_deck,
    MAX_BENCH, MAX_ATTACKS, MAX_HAND, HeuristicAgent,
)
from rl_agents import (
    PPOAgent, DQNAgent, SelfPlayEnv, OBS_SIZE, ACT_SIZE,
    evaluate, relu, pretrain_agents,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

def _p(name): return os.path.join(OUT_DIR, name)

# Training weights (latest weights, may not be champion)
PPO_P0_PATH  = _p("ppo_p0_starmie.npy")       # PPO as P0 (Starmie)
DQN_P0_PATH  = _p("dqn_p0_starmie.npy")       # DQN as P0 (Starmie)
PPO_P1_PATH  = _p("ppo_p1_lucario.npy")       # PPO as P1 (Lucario)
DQN_P1_PATH  = _p("dqn_p1_lucario.npy")       # DQN as P1 (Lucario)

# Per-agent champion weights
PPO_P0_CHAMP = _p("ppo_p0_starmie_champion.npy")
DQN_P0_CHAMP = _p("dqn_p0_starmie_champion.npy")
PPO_P1_CHAMP = _p("ppo_p1_lucario_champion.npy")
DQN_P1_CHAMP = _p("dqn_p1_lucario_champion.npy")

# Side champion weights — best agent for each player slot, used by GUI
CHAMP_P0_PATH = _p("champion_p0.npy")         # best P0 agent (PPO or DQN)
CHAMP_P1_PATH = _p("champion_p1.npy")         # best P1 agent (PPO or DQN)

LEADERBOARD   = _p("leaderboard.json")

# Internal registry:  key → (player_idx, train_path, champ_path, agent_type)
_AGENT_REGISTRY = {
    "ppo_p0": (0, PPO_P0_PATH, PPO_P0_CHAMP, "ppo"),
    "dqn_p0": (0, DQN_P0_PATH, DQN_P0_CHAMP, "dqn"),
    "ppo_p1": (1, PPO_P1_PATH, PPO_P1_CHAMP, "ppo"),
    "dqn_p1": (1, DQN_P1_PATH, DQN_P1_CHAMP, "dqn"),
}
_AGENT_SEEDS = {"ppo_p0": 0, "dqn_p0": 1, "ppo_p1": 2, "dqn_p1": 3}

# ── Colours ───────────────────────────────────────────────────────────────────
R  = "\033[91m"; G  = "\033[92m"; Y  = "\033[93m"
B  = "\033[94m"; M  = "\033[95m"; C  = "\033[96m"
W  = "\033[97m"; DIM = "\033[2m";  RST = "\033[0m"
BLD = "\033[1m"

def _c(text, colour): return f"{colour}{text}{RST}"
def header(text): print(f"\n{BLD}{C}{'═'*60}{RST}\n{BLD}{W}  {text}{RST}\n{BLD}{C}{'═'*60}{RST}")
def subheader(text): print(f"\n{BLD}{Y}  ── {text} ──{RST}")
def info(text): print(f"  {DIM}{text}{RST}")
def good(text): print(f"  {G}✓ {text}{RST}")
def bad(text):  print(f"  {R}✗ {text}{RST}")
def ask(prompt): return input(f"\n  {BLD}{M}▶ {prompt}{RST} ").strip()


# ════════════════════════════════════════════════════════════════════════════
# LEADERBOARD
# ════════════════════════════════════════════════════════════════════════════

def _default_lb() -> dict:
    slot = {"champion_winrate": 0.0, "champion_score": 0.0,
            "history": [], "total_episodes": 0}
    return {
        "ppo_p0": dict(slot),
        "dqn_p0": dict(slot),
        "ppo_p1": dict(slot),
        "dqn_p1": dict(slot),
        "champion_p0": {"type": "ppo", "score": 0.0, "winrate": 0.0, "agent": "ppo_p0"},
        "champion_p1": {"type": "dqn", "score": 0.0, "winrate": 0.0, "agent": "dqn_p1"},
    }


def load_leaderboard() -> dict:
    if os.path.exists(LEADERBOARD):
        with open(LEADERBOARD) as f:
            lb = json.load(f)
        # Migrate old 2-agent format
        if "ppo" in lb and "ppo_p0" not in lb:
            info("Migrating leaderboard from 2-agent to 4-agent format…")
            lb = _default_lb()
        return lb
    return _default_lb()


def save_leaderboard(lb: dict):
    with open(LEADERBOARD, "w") as f:
        json.dump(lb, f, indent=2)


def record_result(lb: dict, agent_key: str, winrate: float,
                  episodes: int, promoted: bool,
                  score: float = 0.0, ko_wins: int = 0, deckout_wins: int = 0):
    entry = {
        "timestamp":    time.strftime("%Y-%m-%d %H:%M"),
        "winrate":      round(winrate, 4),
        "score":        round(score, 1),
        "ko_wins":      ko_wins,
        "deckout_wins": deckout_wins,
        "episodes":     episodes,
        "champion":     promoted,
    }
    lb[agent_key]["history"].append(entry)
    lb[agent_key]["total_episodes"] += episodes
    if promoted:
        lb[agent_key]["champion_winrate"] = round(winrate, 4)
        lb[agent_key]["champion_score"]   = round(score, 1)


# ════════════════════════════════════════════════════════════════════════════
# AGENT FACTORY & LOADER
# ════════════════════════════════════════════════════════════════════════════

def _make_agent(agent_type: str, player_idx: int, seed: int) -> PPOAgent | DQNAgent:
    if agent_type == "ppo":
        return PPOAgent(player_idx=player_idx, seed=seed)
    return DQNAgent(player_idx=player_idx, seed=seed)


class RandomPolicyAgent:
    """Stateless random agent for use as a frozen P0 opponent during Lucario training."""
    def __init__(self, player_idx: int = 0, seed: int = 0):
        self.player_idx = player_idx
        self.rng        = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, mask: np.ndarray,
            deterministic: bool = False) -> int:
        legal = np.where(mask > 0)[0]
        if len(legal) == 0:
            return 0
        return int(self.rng.choice(legal))


def load_agent_by_key(key: str, prefer_champion: bool = True) -> PPOAgent | DQNAgent:
    """Load a specific agent by registry key (e.g. 'ppo_p0', 'dqn_p1')."""
    player_idx, train_path, champ_path, agent_type = _AGENT_REGISTRY[key]
    seed  = _AGENT_SEEDS[key]
    agent = _make_agent(agent_type, player_idx, seed)

    paths = [champ_path, train_path] if prefer_champion else [train_path, champ_path]
    for p in paths:
        if os.path.exists(p):
            agent.load(p)
            info(f"Loaded {key} from {os.path.basename(p)}")
            return agent

    info(f"No saved weights for {key} — using random initialisation")
    return agent


def load_champion_for_side(player_idx: int) -> PPOAgent | DQNAgent:
    """Load the best-known champion for a player slot (0=Starmie, 1=Lucario)."""
    lb          = load_leaderboard()
    champ_key   = f"champion_p{player_idx}"
    champ_info  = lb.get(champ_key, {})
    agent_type  = champ_info.get("type", "ppo" if player_idx == 0 else "dqn")
    side_path   = CHAMP_P0_PATH if player_idx == 0 else CHAMP_P1_PATH
    seed        = player_idx * 2
    agent       = _make_agent(agent_type, player_idx, seed)
    deck_label  = "Starmie" if player_idx == 0 else "Lucario"

    if os.path.exists(side_path):
        agent.load(side_path)
        sc = champ_info.get("score", 0.0)
        wr = champ_info.get("winrate", 0.0)
        info(f"Loaded P{player_idx} ({deck_label}) champion "
             f"[{agent_type.upper()}] — score {sc:.0f}  WR {wr:.1%}")
        return agent

    # Fallback: try per-agent champion files, then old naming convention
    fallbacks = [
        (f"ppo_p{player_idx}", *_AGENT_REGISTRY[f"ppo_p{player_idx}"][2:3], "ppo"),
        (f"dqn_p{player_idx}", *_AGENT_REGISTRY[f"dqn_p{player_idx}"][2:3], "dqn"),
        # legacy names from previous 2-agent system
        ("legacy_p0", _p("ppo_lucario_champion.npy"), "ppo"),
        ("legacy_p1", _p("dqn_starmie_champion.npy"), "dqn"),
    ] if player_idx == 0 else [
        (f"ppo_p{player_idx}", *_AGENT_REGISTRY[f"ppo_p{player_idx}"][2:3], "ppo"),
        (f"dqn_p{player_idx}", *_AGENT_REGISTRY[f"dqn_p{player_idx}"][2:3], "dqn"),
        ("legacy_p1", _p("dqn_starmie_champion.npy"), "dqn"),
        ("legacy_p0", _p("ppo_lucario_champion.npy"), "ppo"),
    ]

    for _, fpath, atype in [(x[0], x[1], x[2]) for x in fallbacks]:
        if os.path.exists(fpath):
            a = _make_agent(atype, player_idx, seed)
            a.load(fpath)
            info(f"Loaded P{player_idx} champion from fallback: {os.path.basename(fpath)}")
            return a

    info(f"No champion for P{player_idx} ({deck_label}) — using random initialisation")
    return agent


# ════════════════════════════════════════════════════════════════════════════
# SCORE / BENCHMARK
# ════════════════════════════════════════════════════════════════════════════

def score_agent(agent: PPOAgent | DQNAgent,
                n_games: int = 1_000,
                seed: int = 777_777,
                ko_weight: float = 5.0,
                deckout_weight: float = 1.0,
                verbose: bool = False) -> Dict:
    """Run agent against a random opponent for n_games. Returns scored dict."""
    rng          = np.random.default_rng(seed)
    score        = 0.0
    wins = ko_wins = deckout_wins = losses = draws = 0

    for g in range(n_games):
        senv = SelfPlayEnv(seed=int(rng.integers(1e9)))
        senv.reset()
        turns = 0
        while not senv.done and turns < 600:
            cp        = senv.current_player
            obs, mask = senv.obs_and_mask(cp)
            if cp == agent.player_idx:
                if isinstance(agent, PPOAgent):
                    action, _, _ = agent.act(obs, mask, deterministic=True)
                else:
                    action = agent.act(obs, mask, deterministic=True)
            else:
                legal  = np.where(mask > 0)[0]
                action = int(rng.choice(legal))
            senv.step(action)
            turns += 1

        w      = senv.winner
        reason = senv.env.gs.win_reason

        if w == agent.player_idx:
            wins += 1
            if reason == "ko":
                score   += ko_weight;   ko_wins += 1
            else:
                score   += deckout_weight; deckout_wins += 1
        elif w == -1:
            draws += 1
        else:
            losses += 1

        if verbose and (g + 1) % 100 == 0:
            pct = (g + 1) / n_games
            wr  = wins / (g + 1)
            bar = "█" * int(20 * pct) + "░" * (20 - int(20 * pct))
            print(f"\r  [{bar}] {pct:.0%}  score={score:.0f}  WR={wr:.1%}", end="", flush=True)

    if verbose:
        print()

    return {
        "score":        score,
        "winrate":      wins / n_games,
        "ko_wins":      ko_wins,
        "deckout_wins": deckout_wins,
        "losses":       losses,
        "draws":        draws,
        "n_games":      n_games,
    }


def benchmark(agent: PPOAgent | DQNAgent, n_games: int = 100, seed: int = 9999) -> float:
    return score_agent(agent, n_games=n_games, seed=seed)["winrate"]


# ════════════════════════════════════════════════════════════════════════════
# TRAINING EPISODE HELPER
# ════════════════════════════════════════════════════════════════════════════

def _train_episode(agent0, agent1, senv: SelfPlayEnv) -> int:
    """
    Run one self-play episode with agent0 at P0 and agent1 at P1.
    Each agent accumulates its own transitions.
    RandomPolicyAgent is supported as a non-learning agent.
    Returns winner (0, 1, or -1).
    """
    agents         = [agent0, agent1]
    dqn_prev       = [None, None]   # per-slot DQN bookkeeping
    last_own_obs1  = [None, None]   # per-slot: last obs from that player's perspective

    while not senv.done:
        cp        = senv.current_player
        agent     = agents[cp]
        obs, mask = senv.obs_and_mask(cp)

        if isinstance(agent, PPOAgent):
            action, lp, val = agent.act(obs, mask)
        elif isinstance(agent, HeuristicAgent):
            action = agent.act(senv.env)
        else:
            action = agent.act(obs, mask)

        obs_p0, obs_p1, r0, r1, done = senv.step(action)
        own_obs1 = obs_p0 if cp == 0 else obs_p1
        reward   = r0    if cp == 0 else r1
        last_own_obs1[cp] = own_obs1

        if isinstance(agent, PPOAgent):
            agent.store(obs, action, reward, val, lp, done, mask)
        elif isinstance(agent, DQNAgent):
            if dqn_prev[cp] is not None:
                nm = (senv.obs_and_mask(cp)[1] if not done
                      else np.zeros(ACT_SIZE, np.float32))
                pobs, pact, prew, pmsk = dqn_prev[cp]
                agent.store(pobs, pact, prew, own_obs1, False, pmsk, nm)

            if done:
                agent.store(obs, action, reward, own_obs1, True, mask,
                            np.zeros(ACT_SIZE, np.float32))
                dqn_prev[cp] = None
            else:
                dqn_prev[cp] = (obs, action, reward, mask)
            agent.update()

    # End-of-episode flush
    for i, ag in enumerate(agents):
        if isinstance(ag, PPOAgent):
            if len(ag.obs_buf) > 0:
                ag.finish_episode(0.0)
                ag.update()
        elif isinstance(ag, DQNAgent):
            if dqn_prev[i] is not None:
                pobs, pact, prew, pmsk = dqn_prev[i]
                fb = last_own_obs1[i] if last_own_obs1[i] is not None else pobs
                ag.store(pobs, pact, prew, fb, True, pmsk,
                         np.zeros(ACT_SIZE, np.float32))

    return senv.winner


# ════════════════════════════════════════════════════════════════════════════
# INCREMENTAL TRAINING — 4 AGENTS
# ════════════════════════════════════════════════════════════════════════════

def run_training(n_episodes: int = 5000,
                 improvement_threshold: float = 50.0):
    """
    Train P1 (Lucario) agents only. P0 is a frozen RandomPolicyAgent.
    Alternating pairings:
      even → RandomP0 vs ppo_p1 (Lucario)
      odd  → RandomP0 vs dqn_p1 (Lucario)
    """
    header("LUCARIO TRAINING — V4")
    print(f"  P0 = HeuristicAgent (Starmie deck)\n"
          f"  P1 = Lucario deck  (PPO + DQN)\n"
          f"  Even episodes: Heuristic(P0) vs PPO(P1)\n"
          f"  Odd  episodes: Heuristic(P0) vs DQN(P1)")

    # P0 is a frozen heuristic agent; only P1 agents learn
    rand_p0 = HeuristicAgent()
    ppo_p1  = load_agent_by_key("ppo_p1", prefer_champion=False)
    dqn_p1  = load_agent_by_key("dqn_p1", prefer_champion=False)

    # Hyperparameters
    ppo_p1.ent_coef = 0.03
    ppo_p1.lr       = 2e-4
    ppo_p1.clip_eps = 0.15
    ppo_p1.n_epochs = 10

    dqn_p1.lr       = 5e-4
    effective_dqn_eps = n_episodes // 2
    dqn_p1.eps_decay = (1.0 - dqn_p1.eps_end) / (50 * effective_dqn_eps * 0.8)

    lb = load_leaderboard()

    subheader("Behavioral Cloning warm-start (from HeuristicAgent)")
    pretrain_agents(ppo_p1, dqn_p1, n_games=300, n_epochs=8, verbose=True)

    # Snapshots for rollback if challengers don't improve
    snapshots = {
        "ppo_p1": ppo_p1.net.get_params(),
        "dqn_p1": dqn_p1.qnet.get_params(),
    }

    subheader(f"Training {n_episodes} episodes vs random P0")
    rng         = np.random.default_rng(int(time.time()))
    recent_wins = deque(maxlen=100)
    t0          = time.time()

    for ep in range(n_episodes):
        seed_ep = int(rng.integers(1e9))
        senv    = SelfPlayEnv(seed=seed_ep)
        senv.reset(seed=seed_ep)

        agent1 = ppo_p1 if ep % 2 == 0 else dqn_p1
        winner = _train_episode(rand_p0, agent1, senv)
        recent_wins.append(winner)

        if (ep + 1) % max(1, n_episodes // 10) == 0:
            n   = len(recent_wins)
            w1  = sum(1 for w in recent_wins if w == 1) / n
            pct = (ep + 1) / n_episodes
            bar = "█" * int(20 * pct) + "░" * (20 - int(20 * pct))
            elapsed = time.time() - t0
            eta     = elapsed / pct - elapsed if pct > 0 else 0
            print(f"  [{bar}] {pct:.0%}  P1(Lucario)={w1:.0%}  "
                  f"{elapsed:.0f}s  ETA {eta:.0f}s")

    # ── Evaluation ────────────────────────────────────────────────────────
    EVAL_GAMES = 1_000
    EVAL_SEED  = 777_777
    KO_W       = 5.0
    DECK_W     = 1.0

    subheader(f"Evaluating P1 challengers  ({EVAL_GAMES:,} games · seed {EVAL_SEED})")
    print(f"  Scoring: KO win=×{KO_W:.0f}  Deck-out win=×{DECK_W:.0f}  "
          f"Min improvement={improvement_threshold:.0f} pts\n")

    p1_agents = {"ppo_p1": ppo_p1, "dqn_p1": dqn_p1}

    challenger_results: Dict[str, Dict] = {}
    for key, agent in p1_agents.items():
        print(f"  Scoring {key} (Lucario)…")
        challenger_results[key] = score_agent(
            agent, n_games=EVAL_GAMES, seed=EVAL_SEED,
            ko_weight=KO_W, deckout_weight=DECK_W, verbose=True)

    print(f"\n  {'Agent':8}  {'Challenger':>12}  {'Champion':>12}  {'Δ':>8}  {'Decision'}")
    print(f"  {'─'*62}")

    for key in ("ppo_p1", "dqn_p1"):
        _, train_path, champ_path, agent_type = _AGENT_REGISTRY[key]
        agent     = p1_agents[key]
        new_res   = challenger_results[key]
        new_score = new_res["score"]

        champ_exists = os.path.exists(champ_path)
        if champ_exists:
            champ_agent  = load_agent_by_key(key, prefer_champion=True)
            champ_res    = score_agent(champ_agent, n_games=EVAL_GAMES, seed=EVAL_SEED,
                                       ko_weight=KO_W, deckout_weight=DECK_W, verbose=False)
            champ_score  = champ_res["score"]
        else:
            champ_score = -1.0

        delta    = new_score - champ_score
        improved = delta > improvement_threshold

        col   = G if improved else (Y if delta > 0 else R)
        d_str = _c(f"{delta:+.0f}", col)

        if improved:
            agent.save(champ_path)
            agent.save(train_path)
            record_result(lb, key, new_res["winrate"], n_episodes,
                          promoted=True, score=new_score,
                          ko_wins=new_res["ko_wins"],
                          deckout_wins=new_res["deckout_wins"])
            status = _c("★ NEW CHAMPION", G)
        else:
            if agent_type == "ppo":
                agent.net.set_params(snapshots[key])
            else:
                agent.qnet.set_params(snapshots[key])
                agent.tnet.copy_params_from(agent.qnet)
            record_result(lb, key, new_res["winrate"], n_episodes,
                          promoted=False, score=new_score,
                          ko_wins=new_res["ko_wins"],
                          deckout_wins=new_res["deckout_wins"])
            status = _c("  kept old", DIM)

        champ_str = f"{champ_score:.0f}" if champ_exists else "  (none)"
        print(f"  {key:8}  {new_score:>9.0f} pts  {champ_str:>9} pts  "
              f"{d_str:>8}  {status}")
        print(f"           WR {new_res['winrate']:.1%}  "
              f"KO {new_res['ko_wins']}  Deck {new_res['deckout_wins']}\n")

    # ── Crown P1 champion ─────────────────────────────────────────────────
    subheader("Crowning P1 champion (best Lucario agent for GUI)")
    keys   = ["ppo_p1", "dqn_p1"]
    scores = {k: lb.get(k, {}).get("champion_score", -1.0) for k in keys}
    best_k = max(scores, key=scores.get)
    best_sc = scores[best_k]

    if best_sc > 0:
        best_agent = load_agent_by_key(best_k, prefer_champion=True)
        best_agent.save(CHAMP_P1_PATH)
        best_type = _AGENT_REGISTRY[best_k][3]
        best_wr   = lb.get(best_k, {}).get("champion_winrate", 0.0)
        lb["champion_p1"] = {
            "type":    best_type,
            "score":   round(best_sc, 1),
            "winrate": round(best_wr, 4),
            "agent":   best_k,
        }
        good(f"P1 (Lucario) champion → {best_k.upper()}  "
             f"(score {best_sc:.0f}  WR {best_wr:.1%})")
    else:
        info("P1 (Lucario): no champion yet, skipping side-champion update")

    save_leaderboard(lb)
    print()
    good(f"Leaderboard updated → {LEADERBOARD}")


# ════════════════════════════════════════════════════════════════════════════
# STATS
# ════════════════════════════════════════════════════════════════════════════

def show_stats():
    header("LEADERBOARD & STATS")
    lb = load_leaderboard()

    labels = {
        "ppo_p0": "PPO — P0 (Starmie)",
        "dqn_p0": "DQN — P0 (Starmie)",
        "ppo_p1": "PPO — P1 (Lucario)",
        "dqn_p1": "DQN — P1 (Lucario)",
    }

    for key, label in labels.items():
        data = lb.get(key, {})
        subheader(label)
        wr_str = f"{data.get('champion_winrate', 0):.1%}"
        sc_str = f"{data.get('champion_score', 0):.0f}"
        print(f"  Champion win-rate : {_c(wr_str, G)}")
        print(f"  Champion score    : {_c(sc_str, G)}  (KO×5 + deck-out×1, 10k games)")
        print(f"  Total episodes    : {data.get('total_episodes', 0)}")
        print(f"  Training runs     : {len(data.get('history', []))}")
        hist = data.get("history", [])
        if hist:
            print(f"\n  {'Date':>18}  {'Score':>8}  {'Win%':>7}  {'KO W':>6}  {'Deck W':>7}  "
                  f"{'Eps':>7}  {'Status':>12}")
            print(f"  {'─'*72}")
            for e in hist[-10:]:
                star = _c("★ CHAMPION", G) if e["champion"] else _c("  skipped", DIM)
                print(f"  {e['timestamp']:>18}  {e.get('score',0):>8.0f}  "
                      f"{e['winrate']:>6.1%}  {e.get('ko_wins',0):>6}  "
                      f"{e.get('deckout_wins',0):>7}  {e['episodes']:>7}  {star}")

    subheader("Side Champions (used by GUI)")
    for side in (0, 1):
        deck  = "Starmie" if side == 0 else "Lucario"
        cinfo = lb.get(f"champion_p{side}", {})
        print(f"  P{side} ({deck}): {cinfo.get('agent', '?').upper()}  "
              f"score={cinfo.get('score', 0):.0f}  "
              f"WR={cinfo.get('winrate', 0):.1%}")


# ════════════════════════════════════════════════════════════════════════════
# HUMAN PLAY MODE
# ════════════════════════════════════════════════════════════════════════════

ENERGY_SYMBOLS = {
    EnergyType.COLORLESS: "⬡",
    EnergyType.FIGHTING:  "✊",
    EnergyType.LIGHTNING: "⚡",
    EnergyType.PSYCHIC:   "🔮",
    EnergyType.WATER:     "💧",
    EnergyType.DARKNESS:  "🌑",
}

STAGE_LABELS = {Stage.BASIC: "Basic", Stage.STAGE1: "Stage1", Stage.STAGE2: "Stage2"}


def _hp_bar(current: int, maximum: int, width: int = 14) -> str:
    frac   = max(0, current / maximum)
    filled = int(width * frac)
    col    = G if frac > 0.5 else (Y if frac > 0.25 else R)
    return f"{col}{'█'*filled}{'░'*(width-filled)}{RST} {current}/{maximum}"


def _energy_str(energy: dict) -> str:
    parts = []
    for etype, count in sorted(energy.items()):
        parts.append(f"{ENERGY_SYMBOLS.get(etype, '?')}×{count}")
    return " ".join(parts) if parts else "(none)"


def _pokemon_line(p: PokemonCard, prefix: str = "") -> str:
    stage  = STAGE_LABELS.get(p.stage, "?")
    tool   = f"  Tool:{_c(p.tool_card.name, C)}" if p.tool_card else ""
    status = f"  [{_c(p.status.upper(), Y)}]" if p.status else ""
    ex_tag = ""
    if p.is_mega_ex:
        ex_tag = _c(" [MegaEx]", M)
    elif p.is_ex:
        ex_tag = _c(" [ex]", M)
    return (f"{prefix}{_c(p.name, BLD+W)}{ex_tag} [{stage}]  "
            f"HP: {_hp_bar(p.current_hp, p.hp)}  "
            f"Energy: {_energy_str(p.energy)}{tool}{status}")


def render_board(gs: GameState, human_player: int):
    me  = gs.players[human_player]
    opp = gs.players[1 - human_player]

    stadium_str = (f"  Stadium: {_c(gs.active_stadium.name, B)}"
                   if gs.active_stadium else "")

    print(f"\n{BLD}{C}{'═'*62}{RST}")
    print(f"{BLD}{W}  Turn {gs.turn_number}  │  Your KOs: {_c(me.ko_count, G)}/6  "
          f"│  Opp KOs: {_c(opp.ko_count, R)}/6{stadium_str}{RST}")
    print(f"{BLD}{C}{'─'*62}{RST}")

    print(f"\n  {_c('OPPONENT', R)}")
    print(f"  Deck: {len(opp.deck)} cards   Hand: {len(opp.hand)} cards")
    if opp.active:
        print(f"  {_c('Active:', Y)} {_pokemon_line(opp.active)}")
    else:
        print(f"  Active: {_c('(none)', DIM)}")
    if opp.bench:
        print(f"  Bench:")
        for p in opp.bench:
            print(f"    {_pokemon_line(p, '  ')}")
    else:
        print(f"  Bench: {_c('(empty)', DIM)}")

    print(f"\n  {BLD}{'─'*58}{RST}")

    print(f"\n  {_c('YOU', G)}")
    print(f"  Deck: {len(me.deck)} cards   "
          f"Energy used: {_c('Yes', R) if me.energy_used else _c('No', G)}   "
          f"Supporter used: {_c('Yes', R) if me.supporter_used else _c('No', G)}")
    if me.active:
        print(f"  {_c('Active:', G)} {_pokemon_line(me.active)}")
        for i, atk in enumerate(me.active.attacks):
            can_use = _can_attack(me.active, atk)
            flag    = _c("✓", G) if can_use else _c("✗", R)
            cost_str = _fmt_cost(atk.energy_cost)
            fx = f" ({atk.effect})" if atk.effect else ""
            print(f"    [{i}] {flag} {_c(atk.name, C)} {cost_str} → "
                  f"{_c(f'{atk.damage} dmg', W)}{_c(fx, DIM)}")
    else:
        print(f"  Active: {_c('(none)', DIM)}")

    if me.bench:
        print(f"  Bench:")
        for i, p in enumerate(me.bench):
            print(f"    [{i}] {_pokemon_line(p)}")
    else:
        print(f"  Bench: {_c('(empty)', DIM)}")

    print(f"\n  {_c('Hand', BLD)}:")
    for i, c in enumerate(me.hand[:MAX_HAND]):
        if isinstance(c, PokemonCard):
            ex_tag = " [MegaEx]" if c.is_mega_ex else (" [ex]" if c.is_ex else "")
            extra  = f" [{STAGE_LABELS.get(c.stage,'?')}]{ex_tag}"
            col    = M
        elif isinstance(c, EnergyCard):
            extra = f" [{ENERGY_SYMBOLS.get(c.energy_type, '?')}]"
            col   = Y
        elif isinstance(c, TrainerCard):
            extra = f" [{c.card_type.name.lower()}]"
            col   = C
        else:
            extra = ""; col = W
        print(f"    [{i}] {_c(c.name, col)}{_c(extra, DIM)}")
    if len(me.hand) > MAX_HAND:
        print(f"    ... +{len(me.hand)-MAX_HAND} more")

    print(f"\n{BLD}{C}{'─'*62}{RST}")


def _can_attack(pokemon: PokemonCard, attack) -> bool:
    from ptcg_env import can_pay_cost
    return can_pay_cost(pokemon, attack.energy_cost)


def _fmt_cost(cost: dict) -> str:
    parts = []
    for etype, count in sorted(cost.items()):
        parts.append(f"{ENERGY_SYMBOLS.get(etype,'?')}×{count}")
    return " ".join(parts) if parts else "(free)"


def _describe_action(idx: int, gs: GameState, human_player: int) -> str:
    me = gs.players[human_player]
    try:
        atype, params = ActionMapper.decode(idx)
    except Exception:
        return f"Action {idx}"

    if atype == ActionType.END_TURN:
        return f"{_c('END TURN', Y)}"

    if atype == ActionType.ATTACK:
        ai = params["atk_idx"]
        if me.active and ai < len(me.active.attacks):
            atk      = me.active.attacks[ai]
            cost_str = _fmt_cost(atk.energy_cost)
            return (f"{_c('ATTACK', R)}: {_c(atk.name, W)} "
                    f"{cost_str} → {atk.damage} dmg"
                    + (f" ({atk.effect})" if atk.effect else ""))
        return f"ATTACK {ai}"

    if atype == ActionType.PROMOTE:
        bench_slot = params["bench_slot"]
        for pidx in range(2):
            if gs.players[pidx].pending_promotion:
                p = gs.players[pidx]
                if bench_slot < len(p.bench):
                    poke = p.bench[bench_slot]
                    return (f"{_c('PROMOTE', BLD+G)}: Send "
                            f"{_c(poke.name, W)} to Active "
                            f"({_hp_bar(poke.current_hp, poke.hp)})")
        return f"PROMOTE bench slot {bench_slot}"

    if atype == ActionType.RETREAT:
        bench_slot = params.get("bench_slot", 0)
        if bench_slot < len(me.bench):
            from ptcg_env import _effective_retreat_cost
            tgt      = me.bench[bench_slot]
            cost     = _effective_retreat_cost(me.active, gs, me) if me.active else 0
            cost_str = f"  [costs {cost} energy]" if cost > 0 else "  [free]"
            return (f"{_c('RETREAT', B)}: {_c(me.active.name if me.active else '?', W)} "
                    f"→ {_c(tgt.name, W)} ({_hp_bar(tgt.current_hp, tgt.hp)}){_c(cost_str, DIM)}")
        return f"RETREAT to bench slot {bench_slot}"

    if atype == ActionType.ATTACH_ENERGY:
        hi   = params["hand_idx"]
        slot = params["slot"]
        if slot == MAX_BENCH:
            pos, tgt = "Active", (me.active.name if me.active else "Active")
        elif slot < len(me.bench):
            pos, tgt = f"Bench[{slot}]", me.bench[slot].name
        else:
            pos, tgt = f"bench[{slot}]", f"bench[{slot}]"
        energy_name = me.hand[hi].name if hi < len(me.hand) else "Energy"
        return f"{_c('ATTACH', G)}: {_c(energy_name, Y)} → {_c(pos, DIM)}:{_c(tgt, W)}"

    if atype == ActionType.PLAY_POKEMON:
        hi = params["hand_idx"]
        if hi < len(me.hand):
            return f"{_c('BENCH', M)}: {_c(me.hand[hi].name, W)}"
        return "BENCH pokemon"

    if atype == ActionType.USE_ITEM:
        hi = params["hand_idx"]
        if hi < len(me.hand):
            return f"{_c('ITEM', C)}: {_c(me.hand[hi].name, W)}"
        return "USE item"

    if atype == ActionType.USE_SUPPORTER:
        hi = params["hand_idx"]
        if hi < len(me.hand):
            return f"{_c('SUPPORTER', C)}: {_c(me.hand[hi].name, W)}"
        return "USE supporter"

    if atype == ActionType.USE_STADIUM:
        hi = params["hand_idx"]
        if hi < len(me.hand):
            return f"{_c('STADIUM', B)}: {_c(me.hand[hi].name, W)}"
        return "USE stadium"

    if atype == ActionType.ATTACH_TOOL:
        hi, slot = params["hand_idx"], params["slot"]
        card_name = me.hand[hi].name if hi < len(me.hand) else "?"
        if slot == MAX_BENCH:
            pos, tgt = "Active", (me.active.name if me.active else "Active")
        elif slot < len(me.bench):
            pos, tgt = f"Bench[{slot}]", me.bench[slot].name
        else:
            pos, tgt = f"bench[{slot}]", f"bench[{slot}]"
        return f"{_c('TOOL', C)}: {_c(card_name, W)} → {_c(pos, DIM)}:{_c(tgt, M)}"

    if atype == ActionType.EVOLVE:
        hi, slot = params["hand_idx"], params["slot"]
        card_name = me.hand[hi].name if hi < len(me.hand) else "?"
        if slot == MAX_BENCH:
            pos, tgt = "Active", (me.active.name if me.active else "Active")
        elif slot < len(me.bench):
            pos, tgt = f"Bench[{slot}]", me.bench[slot].name
        else:
            pos, tgt = f"bench[{slot}]", f"bench[{slot}]"
        return f"{_c('EVOLVE', B)}: {_c(card_name, W)} onto {_c(pos, DIM)}:{_c(tgt, M)}"

    if atype == ActionType.USE_ABILITY:
        slot = params.get("slot", 0)
        poke = me.active if slot == 0 else \
               (me.bench[slot-1] if slot-1 < len(me.bench) else None)
        poke_name = poke.name if poke else "?"
        ability   = poke.ability_name if poke else "ability"
        return f"{_c('ABILITY', M)}: {_c(ability, W)} on {_c(poke_name, BLD)}"

    if atype == ActionType.CHOOSE:
        opt_idx = params.get("opt_idx", 0)
        pc = me.pending_choice
        if pc is None:
            return f"CHOOSE option {opt_idx}"
        opts    = pc.get("options", [])
        ptype   = pc["type"]
        opp     = gs.players[1 - human_player]
        label   = f"option {opt_idx}"
        if opt_idx < len(opts):
            v = opts[opt_idx]
            if ptype in ("ultra_ball_discard1", "ultra_ball_discard2"):
                if v < len(me.hand):
                    c = me.hand[v]
                    label = f"discard {c.name}"
            elif ptype == "ultra_ball_pick":
                if v < len(me.deck):
                    label = f"fetch {me.deck[v].name}"
            elif ptype in ("poffin_pick1", "poffin_pick2"):
                if v < len(me.deck):
                    label = f"bench {me.deck[v].name} ({me.deck[v].hp} HP)"
            elif ptype == "night_stretcher_pick":
                if v < len(me.discard):
                    label = f"recover {me.discard[v].name}"
            elif ptype == "boss_orders_pick":
                if v < len(opp.bench):
                    p = opp.bench[v]
                    label = f"drag {p.name} ({_hp_bar(p.current_hp, p.hp)})"
            elif ptype == "switch_target":
                if v < len(me.bench):
                    p = me.bench[v]
                    label = f"switch to {p.name} ({_hp_bar(p.current_hp, p.hp)})"
            elif ptype == "cursed_blast_target":
                if v == 0:
                    p = opp.active
                    label = f"target ACTIVE {p.name if p else '?'}"
                else:
                    bi = v - 1
                    if bi < len(opp.bench):
                        p = opp.bench[bi]
                        label = f"target BENCH {p.name} ({_hp_bar(p.current_hp, p.hp)})"
            elif ptype == "poke_pad_pick":
                if v < len(me.deck):
                    label = f"fetch {me.deck[v].name}"
            elif ptype == "hilda_evo_pick":
                if v < len(me.deck):
                    label = f"take evo {me.deck[v].name}"
            elif ptype == "hilda_energy_pick":
                if v < len(me.deck):
                    label = f"take energy {me.deck[v].name}"
            elif ptype == "last_ditch_catch_pick":
                if v < len(me.deck):
                    label = f"grab supporter {me.deck[v].name}"
            elif ptype == "cruel_arrow_target":
                if v == 0:
                    p = opp.active
                    label = f"target ACTIVE {p.name if p else '?'}"
                else:
                    bi = v - 1
                    if bi < len(opp.bench):
                        p = opp.bench[bi]
                        label = f"target BENCH[{bi}] {p.name} ({_hp_bar(p.current_hp, p.hp)})"
            elif ptype == "aura_jab_energy":
                if v < len(me.bench):
                    p = me.bench[v]
                    label = f"attach Fighting→ Bench[{v}] {p.name}"
            elif ptype == "heave_ho_target":
                if v < len(opp.bench):
                    p = opp.bench[v]
                    label = f"drag {p.name} ({_hp_bar(p.current_hp, p.hp)})"
            elif ptype == "pokegear_pick":
                if v < len(me.deck):
                    label = f"take supporter {me.deck[v].name}"
            elif ptype == "fighting_gong_pick":
                if v < len(me.deck):
                    label = f"take {me.deck[v].name}"
            elif ptype == "petrel_pick":
                if v < len(me.deck):
                    label = f"take trainer {me.deck[v].name}"
        ptype_display = ptype.replace("_", " ").upper()
        return f"{_c('CHOOSE', BLD+Y)} [{ptype_display}]: {_c(label, W)}"

    return f"Action {idx}"


# ── Interactive sub-menu helpers ─────────────────────────────────────────────

def _pick_one(items: list, prompt: str = "Choose", label_fn=None) -> int:
    """Display numbered list, return chosen index (0-based into items)."""
    for i, item in enumerate(items):
        label = label_fn(item) if label_fn else str(item)
        print(f"    {_c(f'[{i}]', BLD+W)} {label}")
    while True:
        raw = ask(f"{prompt} (0-{len(items)-1})")
        if raw.lower() in ("q", "quit", "exit"):
            raise SystemExit(0)
        try:
            c = int(raw)
            if 0 <= c < len(items):
                return c
        except ValueError:
            pass
        print(_c(f"  Enter 0-{len(items)-1}", R))


def _pick_multiple(items: list, count: int, prompt: str = "Choose", label_fn=None) -> List[int]:
    """Pick exactly `count` distinct items; return list of chosen indices into items."""
    if len(items) <= count:
        return list(range(len(items)))
    for i, item in enumerate(items):
        label = label_fn(item) if label_fn else str(item)
        print(f"    {_c(f'[{i}]', BLD+W)} {label}")
    chosen: List[int] = []
    while len(chosen) < count:
        left = count - len(chosen)
        raw = ask(f"{prompt} — pick {left} more (0-{len(items)-1})")
        if raw.lower() in ("q", "quit", "exit"):
            raise SystemExit(0)
        try:
            c = int(raw)
            if 0 <= c < len(items) and c not in chosen:
                chosen.append(c)
                label = label_fn(items[c]) if label_fn else str(items[c])
                print(f"  {_c('✓', G)} {label}")
            elif c in chosen:
                print(_c("  Already picked — choose another", Y))
        except ValueError:
            pass
    return chosen


def _pre_step_choices(action: int, gs: GameState, human_player: int) -> None:
    """Ultra Ball / Poffin / Night Stretcher / Boss's Orders / Switch / Cursed Blast
    now use the pending_choice interrupt system — no pre-step menus needed.
    This function handles only cases that still use gs.ui_choice (bench_50_one, cruel_arrow).
    """
    from ptcg_env import ActionMapper as AM, ActionType as AT
    me  = gs.players[human_player]
    opp = gs.players[1 - human_player]

    try:
        atype, params = AM.decode(action)
    except Exception:
        return

    # ── ATTACK sub-menus (still use ui_choice for non-pending effects) ───────
    if atype == AT.ATTACK:
        if not me.active:
            return
        ai  = params.get("atk_idx", 0)
        eff = me.active.attacks[ai].effect if ai < len(me.active.attacks) else None

        if eff == "bench_50_one" and opp.bench:
            subheader("JETTING BLOW — Choose opponent Bench Pokémon to deal 50 damage to")
            chosen = _pick_one(
                opp.bench, "target",
                label_fn=lambda p: f"{p.name}  {_hp_bar(p.current_hp, p.hp)}")
            gs.ui_choice = {"bench_target": chosen}

        elif eff == "cruel_arrow":
            targets = []
            if opp.active:
                targets.append((-1, f"ACTIVE   {opp.active.name}  "
                                    f"{_hp_bar(opp.active.current_hp, opp.active.hp)}"))
            for bi, bp in enumerate(opp.bench):
                targets.append((bi, f"BENCH[{bi}] {bp.name}  "
                                    f"{_hp_bar(bp.current_hp, bp.hp)}  [no W/R]"))
            if targets:
                subheader("CRUEL ARROW — Choose target (100 dmg; bench targets ignore W/R)")
                chosen = _pick_one(targets, "target", label_fn=lambda t: t[1])
                gs.ui_choice = {"bench_target": targets[chosen][0]}


def pick_action_human(gs: GameState, human_player: int) -> int:
    orig = gs.current_player
    gs.current_player = human_player
    try:
        mask = compute_legal_mask(gs)
    finally:
        gs.current_player = orig

    legal = [i for i, v in enumerate(mask) if v > 0]
    subheader("Your turn — legal actions")
    for display_i, action_idx in enumerate(legal):
        desc = _describe_action(action_idx, gs, human_player)
        print(f"  {_c(f'[{display_i}]', BLD+W)} {desc}")

    while True:
        raw = ask(f"Choose action (0–{len(legal)-1}) or 'q' to quit")
        if raw.lower() in ("q", "quit", "exit"):
            print(_c("\n  Thanks for playing!", G))
            sys.exit(0)
        try:
            choice = int(raw)
            if 0 <= choice < len(legal):
                return legal[choice]
            print(_c(f"  Enter a number between 0 and {len(legal)-1}", R))
        except ValueError:
            print(_c("  Invalid input — enter a number", R))


def _check_and_handle_promotion(gs: GameState, human_player: int,
                                env: PokemonTCGEnv) -> bool:
    from ptcg_env import ActionMapper as AM, ActionType as AT
    handled = False
    for pidx in range(2):
        p = gs.players[pidx]
        if not p.pending_promotion:
            continue
        handled = True
        if pidx == human_player:
            print(f"\n  {_c('YOUR ACTIVE POKÉMON WAS KO\'D!', BLD+R)}")
            print(f"  {_c('Choose a Pokémon to promote to Active:', BLD+Y)}")
            for i, poke in enumerate(p.bench):
                print(f"    {_c(f'[{i}]', BLD+W)} {_pokemon_line(poke)}")
            while True:
                raw = ask(f"Choose (0–{len(p.bench)-1})")
                try:
                    choice = int(raw)
                    if 0 <= choice < len(p.bench):
                        chosen_name = p.bench[choice].name
                        action      = AM.encode(AT.PROMOTE, {"bench_slot": choice})
                        env.step(action)
                        good(f"Promoted {chosen_name} to Active!")
                        break
                    print(_c(f"  Enter 0–{len(p.bench)-1}", R))
                except (ValueError, IndexError):
                    print(_c("  Invalid input", R))
        else:
            best        = max(range(len(p.bench)), key=lambda i: p.bench[i].current_hp)
            chosen_name = p.bench[best].name
            action      = AM.encode(AT.PROMOTE, {"bench_slot": best})
            env.step(action)
            print(f"  {_c('Bot promotes:', Y)} {_c(chosen_name, W)}")
            time.sleep(0.3)
    return handled


def play_game(human_player: int, bot: PPOAgent | DQNAgent,
              seed: int = 0) -> int:
    env = PokemonTCGEnv(seed=seed, debug=False)
    env.reset(seed=seed)
    gs  = env.gs
    rng = np.random.default_rng(seed + 1)

    deck_name = "Mega Starmie ex" if human_player == 0 else "Mega Lucario ex"
    bot_deck  = "Mega Lucario ex" if human_player == 0 else "Mega Starmie ex"
    header(f"NEW GAME — You play {deck_name} (Player {human_player})")
    print(f"  Bot controls: Player {1-human_player} ({bot_deck})")
    print(f"  First player: Player {gs.current_player}")
    print(f"  First to 6 KO points wins!")
    input(_c("\n  Press Enter to start...", DIM))

    while not gs.game_over:
        if _check_and_handle_promotion(gs, human_player, env):
            if gs.game_over:
                break
            continue

        cp = gs.current_player
        render_board(gs, human_player)

        if cp == human_player:
            print(f"\n  {_c('YOUR TURN', BLD+G)}")
            while True:
                if _check_and_handle_promotion(gs, human_player, env):
                    if gs.game_over:
                        break
                    render_board(gs, human_player)

                action   = pick_action_human(gs, human_player)
                atype, _ = ActionMapper.decode(action)

                # Show sub-menus for effects still using ui_choice (bench_50, cruel_arrow)
                _pre_step_choices(action, gs, human_player)

                _, reward, done, info_dict = env.step(action)

                if gs.game_over:
                    break
                if any(gs.players[p].pending_promotion for p in range(2)):
                    _check_and_handle_promotion(gs, human_player, env)
                    if gs.game_over:
                        break

                # Always re-render after every action
                render_board(gs, human_player)

                # End inner loop when the turn has actually changed
                if gs.current_player != human_player:
                    break
                if atype == ActionType.END_TURN:
                    break
        else:
            print(f"\n  {_c('BOT THINKING...', BLD+R)}")
            time.sleep(0.3)

            while gs.current_player == cp and not gs.game_over:
                if _check_and_handle_promotion(gs, human_player, env):
                    if gs.game_over:
                        break
                    continue

                orig = gs.current_player
                gs.current_player = bot.player_idx
                obs  = StateEncoder.encode(gs)
                mask = compute_legal_mask(gs)
                gs.current_player = orig

                if isinstance(bot, PPOAgent):
                    action, _, _ = bot.act(obs, mask, deterministic=True)
                else:
                    action = bot.act(obs, mask, deterministic=True)

                atype, params = ActionMapper.decode(action)
                desc          = _describe_action(action, gs, bot.player_idx)
                print(f"  Bot: {desc}")

                _, reward, done, info_dict = env.step(action)
                time.sleep(0.15)

                if any(gs.players[p].pending_promotion for p in range(2)):
                    _check_and_handle_promotion(gs, human_player, env)
                    if gs.game_over:
                        break

                # Break on END_TURN; for ATTACK only break if no pending choice remains
                if atype == ActionType.END_TURN:
                    break
                if atype == ActionType.ATTACK and gs.players[bot.player_idx].pending_choice is None:
                    break

    render_board(gs, human_player)
    winner = gs.winner
    print()
    if winner == human_player:
        print(_c("  ★ ★ ★  YOU WIN!  ★ ★ ★", BLD+G))
    elif winner == -1:
        print(_c("  DRAW (turn limit reached)", Y))
    else:
        print(_c("  The bot wins this time. Better luck next game!", R))

    ko_you = gs.players[human_player].ko_count
    ko_bot = gs.players[1-human_player].ko_count
    print(f"\n  Final KOs — You: {_c(ko_you, G)}  Bot: {_c(ko_bot, R)}")
    print(f"  Total turns: {gs.turn_number}")
    return winner


def run_play():
    header("POKEMON TCG V4 vs BOT")

    human_player = 0   # always Starmie (P0)
    bot_player   = 1
    bot        = load_champion_for_side(bot_player)
    lb         = load_leaderboard()
    cinfo      = lb.get(f"champion_p{bot_player}", {})
    info(f"Bot champion: {cinfo.get('agent','?').upper()}  "
         f"WR={cinfo.get('winrate',0):.1%}  score={cinfo.get('score',0):.0f}")

    record  = [0, 0, 0]
    game_num = 0
    while True:
        game_num += 1
        seed   = int(time.time()) % 100_000 + game_num
        winner = play_game(human_player, bot, seed=seed)
        if winner == human_player:
            record[0] += 1
        elif winner == bot_player:
            record[1] += 1
        else:
            record[2] += 1

        print(f"\n  {_c('Your record:', BLD)} {_c(record[0], G)} wins "
              f"/ {_c(record[1], R)} losses"
              + (f" / {_c(record[2], Y)} draws" if record[2] else ""))

        again = ask("Play again? [y/n]")
        if again.lower() not in ("y", "yes", ""):
            break

    print()
    good("Thanks for playing!")


def run_benchmark(n_games: int = 100, deck: Optional[str] = None, seed: int = -1):
    header(f"CHAMPION BENCHMARK  ({n_games} games vs random)")
    rng      = np.random.default_rng(seed if seed >= 0 else int(time.time()))
    to_bench = []

    if deck is None:
        for key in ("ppo_p0", "dqn_p0", "ppo_p1", "dqn_p1"):
            player_idx = _AGENT_REGISTRY[key][0]
            deck_label = "Starmie" if player_idx == 0 else "Lucario"
            agent = load_agent_by_key(key, prefer_champion=True)
            to_bench.append((f"{key.upper()} ({deck_label})", agent, player_idx))
    elif deck.lower() in ("starmie", "0"):
        for key in ("ppo_p0", "dqn_p0"):
            agent = load_agent_by_key(key, prefer_champion=True)
            to_bench.append((f"{key.upper()} (Starmie)", agent, 0))
    else:
        for key in ("ppo_p1", "dqn_p1"):
            agent = load_agent_by_key(key, prefer_champion=True)
            to_bench.append((f"{key.upper()} (Lucario)", agent, 1))

    all_results = []
    bar_width   = 40

    for label, agent, player_idx in to_bench:
        subheader(f"{label}  (Player {player_idx}) vs Random")

        wins = losses = draws = ko_wins = deckout_wins = 0
        ko_totals_agent  = []
        ko_totals_random = []
        turn_totals      = []
        outcomes         = []

        for g in range(n_games):
            game_seed = int(rng.integers(1e9))
            senv      = SelfPlayEnv(seed=game_seed)
            senv.reset(seed=game_seed)
            turns = 0

            while not senv.done and turns < 600:
                cp        = senv.current_player
                obs, mask = senv.obs_and_mask(cp)
                if cp == player_idx:
                    if isinstance(agent, PPOAgent):
                        action, _, _ = agent.act(obs, mask, deterministic=True)
                    else:
                        action = agent.act(obs, mask, deterministic=True)
                else:
                    legal  = np.where(mask > 0)[0]
                    action = int(rng.choice(legal))
                senv.step(action)
                turns += 1

            w      = senv.winner
            reason = senv.env.gs.win_reason
            gs     = senv.env.gs

            ko_totals_agent.append(gs.players[player_idx].ko_count)
            ko_totals_random.append(gs.players[1 - player_idx].ko_count)
            turn_totals.append(turns)

            if w == player_idx:
                wins += 1
                ko_wins      += int(reason == "ko")
                deckout_wins += int(reason != "ko")
                outcomes.append("W")
            elif w == -1:
                draws  += 1
                outcomes.append("D")
            else:
                losses += 1
                outcomes.append("L")

            done_pct = (g + 1) / n_games
            filled   = int(bar_width * done_pct)
            wr_live  = wins / (g + 1)
            col      = G if wr_live >= 0.55 else (Y if wr_live >= 0.40 else R)
            bar      = f"{col}{'█' * filled}{RST}{'░' * (bar_width - filled)}"
            print(f"\r  [{bar}] {g+1:>3}/{n_games}  "
                  f"W={_c(wins, G)} L={_c(losses, R)} D={_c(draws, Y)}  "
                  f"WR={_c(f'{wr_live:.1%}', col)}  ",
                  end="", flush=True)

        print()

        winrate = wins / n_games
        col     = G if winrate >= 0.60 else (Y if winrate >= 0.45 else R)

        ribbon = ""
        for o in outcomes:
            ribbon += (_c("W", G) if o == "W" else
                       (_c("L", R) if o == "L" else _c("D", Y)))
        print(f"\n  Outcome ribbon: {ribbon}")

        avg_ko_agent  = sum(ko_totals_agent)  / n_games
        avg_ko_random = sum(ko_totals_random) / n_games
        avg_turns     = sum(turn_totals)      / n_games
        weighted      = ko_wins * 5 + deckout_wins

        print(f"\n  {'Metric':<30} {'Value':>10}")
        print(f"  {'─' * 42}")
        print(f"  {'Win-rate':<30} {_c(f'{winrate:.1%}', col):>10}")
        print(f"  {'Wins / Losses / Draws':<30} {f'{wins} / {losses} / {draws}':>10}")
        print(f"  {'  KO wins (×5 pts each)':<30} {ko_wins:>10}")
        print(f"  {'  Deck-out wins (×1 pt each)':<30} {deckout_wins:>10}")
        print(f"  {'Weighted score':<30} {_c(str(weighted), col):>10}")
        print(f"  {'Avg KOs (champion)':<30} {avg_ko_agent:>10.2f}")
        print(f"  {'Avg KOs (random)':<30} {avg_ko_random:>10.2f}")
        print(f"  {'Avg turns per game':<30} {avg_turns:>10.1f}")

        all_results.append({
            "label":        label,
            "winrate":      winrate,
            "wins":         wins,
            "losses":       losses,
            "draws":        draws,
            "ko_wins":      ko_wins,
            "deckout_wins": deckout_wins,
            "score":        weighted,
            "avg_turns":    avg_turns,
        })

    if len(all_results) > 1:
        subheader("Summary")
        print(f"  {'Agent':<32} {'Win-rate':>10}  {'Score':>8}  "
              f"{'KO W':>6}  {'Deck W':>7}  {'Avg Turns':>10}")
        print(f"  {'─' * 74}")
        for r in all_results:
            col    = G if r["winrate"] >= 0.60 else (Y if r["winrate"] >= 0.45 else R)
            wr_str = f'{r["winrate"]:.1%}'
            print(f"  {r['label']:<32} {_c(wr_str, col):>10}  "
                  f"{r['score']:>8}  {r['ko_wins']:>6}  "
                  f"{r['deckout_wins']:>7}  {r['avg_turns']:>10.1f}")

    return all_results


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    args = sys.argv[1:]
    cmd  = args[0].lower() if args else "help"

    def flag(name, default=None):
        for i, a in enumerate(args):
            if a.startswith(f"--{name}="):
                return a.split("=", 1)[1]
            if a == f"--{name}" and i + 1 < len(args):
                return args[i + 1]
        return default

    if cmd == "play":
        run_play()

    elif cmd == "train":
        episodes  = int(flag("episodes", 5000))
        threshold = float(flag("threshold", 50))
        run_training(n_episodes=episodes, improvement_threshold=threshold)

    elif cmd == "stats":
        show_stats()

    elif cmd == "benchmark":
        n    = int(flag("games", 100))
        deck = flag("deck")
        seed = int(flag("seed", -1))
        run_benchmark(n_games=n, deck=deck, seed=seed)

    else:
        header("POKEMON TCG V4 — PLAY & TRAIN (4-AGENT)")
        print(textwrap.dedent(f"""
          {BLD}DECKS:{RST}
            P0 — Mega Starmie ex  (Water/Psychic)   ← PPO & DQN both trained here
            P1 — Mega Lucario ex  (Fighting)         ← PPO & DQN both trained here

          {BLD}AGENTS:{RST}
            ppo_p0  PPO at P0 (Starmie)    dqn_p0  DQN at P0 (Starmie)
            ppo_p1  PPO at P1 (Lucario)    dqn_p1  DQN at P1 (Lucario)

          {BLD}COMMANDS:{RST}

            {_c('python play_and_train.py play', G)}
              Play against the trained bot (Human vs Bot).
              {DIM}--deck starmie|lucario   choose your deck (default: ask){RST}

            {_c('python play_and_train.py train', G)}
              Train all 4 agents via alternating self-play, then crown
              the best PPO/DQN for each side as the GUI champion.
              {DIM}--episodes 5000    self-play episodes (default: 5000)
              --threshold 50     min score improvement to crown new champion{RST}

            {_c('python play_and_train.py benchmark', G)}
              Run all 4 champions against a random opponent.
              {DIM}--games 100           number of games (default: 100)
              --deck starmie|lucario  test only one side (default: all 4)
              --seed 0               random seed for reproducibility{RST}

            {_c('python play_and_train.py stats', G)}
              Show leaderboard and training history for all 4 agents.

          {BLD}OUTPUT FILES:{RST}
            {PPO_P0_PATH}
            {DQN_P0_PATH}
            {PPO_P1_PATH}
            {DQN_P1_PATH}
            {CHAMP_P0_PATH}   ← used by GUI for bot P0
            {CHAMP_P1_PATH}   ← used by GUI for bot P1
            {LEADERBOARD}
        """))


if __name__ == "__main__":
    main()
