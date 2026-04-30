"""
Pokemon TCG — Play & Train
══════════════════════════
Usage:
  python play_and_train.py play   [--deck lycanroc|raichu] [--model auto]
  python play_and_train.py train  [--episodes 500] [--eval 100]
  python play_and_train.py stats

play  — Human vs the best saved bot for your chosen deck's opponent
train — Continue training from existing .npy files; only keeps a new
        model if it beats the previous champion by >2% win-rate
stats — Show leaderboard and champion info
"""

from __future__ import annotations
import sys, os, time, json, textwrap
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, List

# ── locate siblings ──────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from ptcg_env import (
    PokemonTCGEnv, StateEncoder, ActionMapper, ActionType,
    GameState, PlayerState, PokemonCard, EnergyCard, TrainerCard,
    Stage, CardType, EnergyType, compute_legal_mask,
    build_lycanroc_deck, build_raichu_deck,
    MAX_BENCH, MAX_ATTACKS, MAX_HAND,
)
from rl_agents import (
    PPOAgent, DQNAgent, SelfPlayEnv, OBS_SIZE, ACT_SIZE,
    evaluate, relu, pretrain_agents,
)

# ────────────────────────────────────────────────────────────────────────────
# PATHS
# ────────────────────────────────────────────────────────────────────────────
OUT_DIR         = os.path.join(_DIR, "outputs")
PPO_PATH        = os.path.join(OUT_DIR, "ppo_lycanroc.npy")
DQN_PATH        = os.path.join(OUT_DIR, "dqn_raichu.npy")
PPO_CHAMP_PATH  = os.path.join(OUT_DIR, "ppo_lycanroc_champion.npy")
DQN_CHAMP_PATH  = os.path.join(OUT_DIR, "dqn_raichu_champion.npy")
LEADERBOARD     = os.path.join(OUT_DIR, "leaderboard.json")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colours ──────────────────────────────────────────────────────────────────
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

def load_leaderboard() -> dict:
    if os.path.exists(LEADERBOARD):
        with open(LEADERBOARD) as f:
            return json.load(f)
    return {
        "ppo": {"champion_winrate": 0.0, "history": [], "total_episodes": 0},
        "dqn": {"champion_winrate": 0.0, "history": [], "total_episodes": 0},
    }

def save_leaderboard(lb: dict):
    with open(LEADERBOARD, "w") as f:
        json.dump(lb, f, indent=2)

def record_result(lb: dict, agent_key: str, winrate: float,
                  episodes: int, promoted: bool):
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "winrate":   round(winrate, 4),
        "episodes":  episodes,
        "champion":  promoted,
    }
    lb[agent_key]["history"].append(entry)
    lb[agent_key]["total_episodes"] += episodes
    if promoted:
        lb[agent_key]["champion_winrate"] = round(winrate, 4)

def show_stats():
    header("LEADERBOARD & STATS")
    lb = load_leaderboard()

    for key, label in [("ppo", "PPO — Lycanroc"), ("dqn", "DQN — Alolan Raichu")]:
        data = lb[key]
        subheader(label)
        wr_str = f"{data['champion_winrate']:.1%}"
        print(f"  Champion win-rate : {_c(wr_str, G)}")
        print(f"  Total episodes    : {data['total_episodes']}")
        print(f"  Training runs     : {len(data['history'])}")
        if data["history"]:
            print(f"\n  {'Date':>18}  {'WinRate':>8}  {'Episodes':>9}  {'Status':>10}")
            print(f"  {'─'*52}")
            for e in data["history"][-10:]:
                star = _c("★ CHAMPION", G) if e["champion"] else _c("  skipped", DIM)
                print(f"  {e['timestamp']:>18}  {e['winrate']:>7.1%}  {e['episodes']:>9}  {star}")

# ════════════════════════════════════════════════════════════════════════════
# AGENT LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_agent(player_idx: int, prefer_champion: bool = True) -> PPOAgent | DQNAgent:
    """Load best available saved agent."""
    if player_idx == 0:
        paths = [PPO_CHAMP_PATH, PPO_PATH] if prefer_champion else [PPO_PATH, PPO_CHAMP_PATH]
        agent = PPOAgent(player_idx=0, seed=0)
        cls_name = "PPO"
    else:
        paths = [DQN_CHAMP_PATH, DQN_PATH] if prefer_champion else [DQN_PATH, DQN_CHAMP_PATH]
        agent = DQNAgent(player_idx=1, seed=1)
        cls_name = "DQN"

    for p in paths:
        if os.path.exists(p):
            agent.load(p)
            info(f"Loaded {cls_name} from {os.path.basename(p)}")
            return agent

    info(f"No saved {cls_name} found — using random initialisation")
    return agent

def benchmark(agent: PPOAgent | DQNAgent, n_games: int = 100,
              opponent_is_random: bool = True, seed: int = 9999) -> float:
    """
    Returns win-rate of agent against a random opponent.
    Agent always plays as its player_idx.
    """
    rng = np.random.default_rng(seed)
    wins = 0
    for g in range(n_games):
        senv = SelfPlayEnv(seed=int(rng.integers(1e9)))
        senv.reset()
        turns = 0
        while not senv.done and turns < 500:
            cp = senv.current_player
            obs, mask = senv.obs_and_mask(cp)
            if cp == agent.player_idx:
                if isinstance(agent, PPOAgent):
                    action, _, _ = agent.act(obs, mask, deterministic=True)
                else:
                    action = agent.act(obs, mask, deterministic=True)
            else:
                legal = np.where(mask > 0)[0]
                action = int(rng.choice(legal))
            senv.step(action)
            turns += 1
        if senv.winner == agent.player_idx:
            wins += 1
    return wins / n_games

# ════════════════════════════════════════════════════════════════════════════
# INCREMENTAL TRAINING
# ════════════════════════════════════════════════════════════════════════════

def run_training(n_episodes: int = 500, eval_games: int = 100,
                 improvement_threshold: float = 0.02):
    """
    Load existing champions, train for n_episodes more, benchmark.
    Saves new weights only if winrate improves by > threshold.
    """
    header("INCREMENTAL TRAINING")

    # ── Load existing agents ──────────────────────────────────────────────
    ppo = load_agent(0)
    ppo.ent_coef  = 0.03
    ppo.lr        = 2e-4
    ppo.clip_eps  = 0.15

    dqn = load_agent(1)
    dqn.lr = 5e-4

    # ── Behavioral cloning warm-start (only if agents are fresh/weak) ─────
    lb = load_leaderboard()
    is_fresh = lb["ppo"]["champion_winrate"] < 0.40 or lb["dqn"]["champion_winrate"] < 0.40
    if is_fresh:
        subheader("Behavioral Cloning warm-start (from HeuristicAgent)")
        pretrain_agents(ppo, dqn, n_games=200, n_epochs=6, verbose=True)

    old_ppo_wr = lb["ppo"]["champion_winrate"]
    old_dqn_wr = lb["dqn"]["champion_winrate"]

    subheader("Baseline benchmark (vs random, 100 games)")
    pre_ppo = benchmark(ppo, n_games=eval_games, seed=1111)
    pre_dqn = benchmark(dqn, n_games=eval_games, seed=2222)
    print(f"  PPO pre-train win-rate : {_c(f'{pre_ppo:.1%}', Y)}")
    print(f"  DQN pre-train win-rate : {_c(f'{pre_dqn:.1%}', Y)}")

    # ── Self-play training loop ───────────────────────────────────────────
    subheader(f"Training {n_episodes} self-play episodes")
    rng = np.random.default_rng(int(time.time()))
    ppo_turns = 0
    recent_wins: deque = deque(maxlen=100)
    t0 = time.time()

    # Snapshot weights in case we need to revert
    ppo_snapshot = ppo.net.get_params()
    dqn_snapshot = dqn.qnet.get_params()

    for ep in range(n_episodes):
        seed_ep = int(rng.integers(1e9))
        senv = SelfPlayEnv(seed=seed_ep)
        senv.reset(seed=seed_ep)
        dqn_prev = None

        while not senv.done:
            cp = senv.current_player
            obs, mask = senv.obs_and_mask(cp)
            if cp == 0:
                action, lp, val = ppo.act(obs, mask)
                obs0, obs1, r0, r1, done = senv.step(action)
                ppo.store(obs, action, r0, val, lp, done, mask)
                ppo_turns += 1
                if ppo_turns >= 256:
                    lv = 0.0 if done else ppo.net.forward(obs0)[1]
                    ppo.finish_episode(lv)
                    ppo.update()
                    ppo_turns = 0
            else:
                action = dqn.act(obs, mask)
                obs0, obs1, r0, r1, done = senv.step(action)
                if dqn_prev:
                    nm = (senv.obs_and_mask(1)[1] if not done
                          else np.zeros(ACT_SIZE, np.float32))
                    dqn.store(*dqn_prev[:3], obs1, False, dqn_prev[3], nm)
                if done:
                    dqn.store(obs, action, r1, obs1, True, mask,
                              np.zeros(ACT_SIZE, np.float32))
                    dqn_prev = None
                else:
                    dqn_prev = (obs, action, r1, mask)
                dqn.update()

        if len(ppo.obs_buf) > 0:
            ppo.finish_episode(0.0); ppo.update(); ppo_turns = 0
        if dqn_prev:
            dqn.store(*dqn_prev[:3], dqn_prev[0], True,
                      dqn_prev[3], np.zeros(ACT_SIZE, np.float32))

        recent_wins.append(senv.winner)

        # Progress bar
        if (ep + 1) % max(1, n_episodes // 10) == 0:
            n = len(recent_wins)
            w0 = sum(1 for w in recent_wins if w == 0) / n
            w1 = sum(1 for w in recent_wins if w == 1) / n
            pct = (ep + 1) / n_episodes
            bar = "█" * int(20 * pct) + "░" * (20 - int(20 * pct))
            elapsed = time.time() - t0
            eta = elapsed / pct - elapsed if pct > 0 else 0
            print(f"  [{bar}] {pct:.0%}  P0={w0:.0%} P1={w1:.0%}  "
                  f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s")

    # ── Post-training benchmark ───────────────────────────────────────────
    subheader("Post-training benchmark (vs random, 100 games)")
    post_ppo = benchmark(ppo, n_games=eval_games, seed=3333)
    post_dqn = benchmark(dqn, n_games=eval_games, seed=4444)

    print(f"\n  {'Agent':10s}  {'Before':>8}  {'After':>8}  {'Δ':>8}  {'Decision':>12}")
    print(f"  {'─'*52}")

    for label, pre, post, agent_key, agent, champ_path, snap in [
        ("PPO", pre_ppo, post_ppo, "ppo", ppo, PPO_CHAMP_PATH, ppo_snapshot),
        ("DQN", pre_dqn, post_dqn, "dqn", dqn, DQN_CHAMP_PATH, dqn_snapshot),
    ]:
        delta = post - pre
        old_champ = lb[agent_key]["champion_winrate"]
        improved  = post > old_champ + improvement_threshold

        col    = G if delta > 0.01 else (R if delta < -0.01 else Y)
        d_str  = _c(f"{delta:+.1%}", col)
        status = ""

        if improved:
            # Save new champion
            if agent_key == "ppo":
                agent.save(PPO_CHAMP_PATH)
                agent.save(PPO_PATH)
            else:
                agent.save(DQN_CHAMP_PATH)
                agent.save(DQN_PATH)
            record_result(lb, agent_key, post, n_episodes, promoted=True)
            status = _c("★ NEW CHAMPION", G)
        else:
            # Revert to snapshot (don't overwrite champion)
            if agent_key == "ppo":
                agent.net.set_params(snap)
            else:
                agent.qnet.set_params(snap)
                agent.tnet.copy_params_from(agent.qnet)
            record_result(lb, agent_key, post, n_episodes, promoted=False)
            status = _c("  kept old", DIM)

        print(f"  {label:10s}  {pre:>7.1%}  {post:>7.1%}  {d_str:>8}  {status}")

    save_leaderboard(lb)
    print()
    good(f"Leaderboard updated → {LEADERBOARD}")

# ════════════════════════════════════════════════════════════════════════════
# HUMAN PLAY MODE
# ════════════════════════════════════════════════════════════════════════════

ENERGY_SYMBOLS = {
    EnergyType.COLORLESS: "⬡",
    EnergyType.FIGHTING:  "✊",
    EnergyType.LIGHTNING: "⚡",
    EnergyType.PSYCHIC:   "🔮",
}

STAGE_LABELS = {Stage.BASIC: "Basic", Stage.STAGE1: "Stage1", Stage.STAGE2: "Stage2"}

def _hp_bar(current: int, maximum: int, width: int = 14) -> str:
    frac = max(0, current / maximum)
    filled = int(width * frac)
    col = G if frac > 0.5 else (Y if frac > 0.25 else R)
    return f"{col}{'█'*filled}{'░'*(width-filled)}{RST} {current}/{maximum}"

def _energy_str(energy: dict) -> str:
    parts = []
    for etype, count in sorted(energy.items()):
        sym = ENERGY_SYMBOLS.get(etype, "?")
        parts.append(f"{sym}×{count}")
    return " ".join(parts) if parts else "(none)"

def _card_name(card) -> str:
    return getattr(card, "name", "?")

def render_board(gs: GameState, human_player: int):
    """Print a full board state from the human's perspective."""
    me  = gs.players[human_player]
    opp = gs.players[1 - human_player]

    print(f"\n{BLD}{C}{'═'*60}{RST}")
    print(f"{BLD}{W}  Turn {gs.turn_number}  │  Your KOs: {_c(me.ko_count, G)}/6  "
          f"│  Opp KOs: {_c(opp.ko_count, R)}/6{RST}")
    print(f"{BLD}{C}{'─'*60}{RST}")

    # Opponent side
    print(f"\n  {_c('OPPONENT', R)}")
    print(f"  Deck: {len(opp.deck)} cards   Hand: {len(opp.hand)} cards")

    if opp.active:
        p = opp.active
        print(f"  {_c('Active:', Y)} {_c(p.name, BLD+W)} [{STAGE_LABELS[p.stage]}]  "
              f"HP: {_hp_bar(p.current_hp, p.hp)}  "
              f"Energy: {_energy_str(p.energy)}")
    else:
        print(f"  Active: {_c('(none)', DIM)}")

    if opp.bench:
        bench_str = "  Bench: "
        for p in opp.bench:
            bench_str += (f"{_c(p.name, W)} "
                          f"{_c(f'{p.current_hp}/{p.hp}HP', Y if p.current_hp > p.hp//2 else R)}  ")
        print(bench_str)
    else:
        print(f"  Bench: {_c('(empty)', DIM)}")

    print(f"\n  {BLD}{'─'*56}{RST}")

    # Human side
    print(f"\n  {_c('YOU', G)}")
    print(f"  Deck: {len(me.deck)} cards   "
          f"Energy used: {_c('Yes', R) if me.energy_used else _c('No', G)}   "
          f"Supporter used: {_c('Yes', R) if me.supporter_used else _c('No', G)}")

    if me.active:
        p = me.active
        print(f"  {_c('Active:', G)} {_c(p.name, BLD+W)} [{STAGE_LABELS[p.stage]}]  "
              f"HP: {_hp_bar(p.current_hp, p.hp)}  "
              f"Energy: {_energy_str(p.energy)}")
        # Show attacks
        for i, atk in enumerate(p.attacks):
            can = "✓" if _can_attack(p, atk) else "✗"
            cost_str = _fmt_cost(atk.energy_cost)
            fx = f" ({atk.effect})" if atk.effect else ""
            print(f"    [{i}] {_c(atk.name, C)} {cost_str} → "
                  f"{_c(f'{atk.damage} dmg', W)}{_c(fx, DIM)}")
    else:
        print(f"  Active: {_c('(none)', DIM)}")

    if me.bench:
        bench_str = "  Bench: "
        for i, p in enumerate(me.bench):
            bench_str += (f"[{i}]{_c(p.name, W)} "
                          f"{_c(f'{p.current_hp}/{p.hp}HP', Y if p.current_hp > p.hp//2 else R)}  ")
        print(bench_str)
    else:
        print(f"  Bench: {_c('(empty)', DIM)}")

    # Hand
    print(f"\n  {_c('Hand', BLD)}:")
    for i, c in enumerate(me.hand[:MAX_HAND]):
        if isinstance(c, PokemonCard):
            extra = f" [{STAGE_LABELS[c.stage]}]"
            col = M
        elif isinstance(c, EnergyCard):
            extra = f" [{ENERGY_SYMBOLS.get(c.energy_type, '?')}]"
            col = Y
        elif isinstance(c, TrainerCard):
            extra = f" [{c.card_type.name.lower()}]"
            col = C
        else:
            extra = ""; col = W
        print(f"    [{i}] {_c(c.name, col)}{_c(extra, DIM)}")
    if len(me.hand) > MAX_HAND:
        print(f"    ... +{len(me.hand)-MAX_HAND} more")

    print(f"\n{BLD}{C}{'─'*60}{RST}")

def _can_attack(pokemon: PokemonCard, attack) -> bool:
    from ptcg_env import can_pay_cost
    return can_pay_cost(pokemon, attack.energy_cost)

def _fmt_cost(cost: dict) -> str:
    parts = []
    for etype, count in sorted(cost.items()):
        parts.append(f"{ENERGY_SYMBOLS.get(etype,'?')}×{count}")
    return " ".join(parts) if parts else "(free)"

def _describe_action(idx: int, gs: GameState, human_player: int) -> str:
    """Human-readable description of an action index."""
    me = gs.players[human_player]
    try:
        atype, params = ActionMapper.decode(idx)
    except Exception:
        return f"Action {idx}"

    if atype == ActionType.PROMOTE:
        bench_slot = params["bench_slot"]
        # Find which player is promoting
        for pidx in range(2):
            if gs.players[pidx].pending_promotion:
                p = gs.players[pidx]
                if bench_slot < len(p.bench):
                    poke = p.bench[bench_slot]
                    return (f"{_c('PROMOTE', BLD+G)}: Send "
                            f"{_c(poke.name, W)} to Active "
                            f"({_hp_bar(poke.current_hp, poke.hp)})")
        return f"PROMOTE bench slot {bench_slot}"
    if atype == ActionType.END_TURN:
        return f"{_c('END TURN', Y)}"
    if atype == ActionType.ATTACK:
        ai = params["atk_idx"]
        if me.active and ai < len(me.active.attacks):
            atk = me.active.attacks[ai]
            cost_str = _fmt_cost(atk.energy_cost)
            return (f"{_c('ATTACK', R)}: {_c(atk.name, W)} "
                    f"{cost_str} → {atk.damage} dmg"
                    + (f" ({atk.effect})" if atk.effect else ""))
        return f"ATTACK {ai}"
    if atype == ActionType.ATTACH_ENERGY:
        slot = params["slot"]
        if slot == 0:
            tgt = me.active.name if me.active else "Active"
        else:
            bi = slot - 1
            tgt = me.bench[bi].name if bi < len(me.bench) else f"bench[{bi}]"
        # Find energy card
        energy_name = next(
            (c.name for c in me.hand if isinstance(c, EnergyCard)), "Energy")
        return f"{_c('ATTACH', G)}: {energy_name} → {_c(tgt, W)}"
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
    if atype == ActionType.EVOLVE:
        hi, slot = params["hand_idx"], params["slot"]
        card_name = me.hand[hi].name if hi < len(me.hand) else "?"
        if slot == MAX_BENCH:
            tgt = me.active.name if me.active else "Active"
        else:
            tgt = me.bench[slot].name if slot < len(me.bench) else f"bench[{slot}]"
        return f"{_c('EVOLVE', B)}: {_c(card_name, W)} onto {_c(tgt, M)}"
    return f"Action {idx}"

def pick_action_human(gs: GameState, human_player: int) -> int:
    """Interactive prompt for the human to choose an action."""
    me = gs.players[human_player]

    # Temporarily set current_player to human for mask computation
    orig = gs.current_player
    gs.current_player = human_player
    mask = compute_legal_mask(gs)
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
    """
    If any player has pending_promotion, handle it.
    Human gets an interactive prompt; bot auto-picks highest HP.
    Returns True if a promotion was handled.
    """
    from ptcg_env import ActionMapper as AM, ActionType as AT
    handled = False
    for pidx in range(2):
        p = gs.players[pidx]
        if not p.pending_promotion:
            continue
        handled = True
        if pidx == human_player:
            print(f"\n  {_c('YOUR ACTIVE POKÉMON WAS KO\'D!', BLD+R)}")
            print(f"  {_c('Choose a Pokémon to send to Active:', BLD+Y)}")
            for i, poke in enumerate(p.bench):
                print(f"    {_c(f'[{i}]', BLD+W)} {_c(poke.name, W)}  "
                      f"HP: {_hp_bar(poke.current_hp, poke.hp)}  "
                      f"Energy: {_energy_str(poke.energy)}")
            while True:
                raw = ask(f"Choose (0–{len(p.bench)-1})")
                try:
                    choice = int(raw)
                    if 0 <= choice < len(p.bench):
                        action = AM.encode(AT.PROMOTE, {"bench_slot": choice})
                        env.step(action)
                        good(f"Sent {p.bench[choice].name if p.active is None else p.active.name} to Active!")
                        break
                    print(_c(f"  Enter 0–{len(p.bench)-1}", R))
                except (ValueError, IndexError):
                    print(_c("  Invalid input", R))
        else:
            # Bot: pick highest HP
            best = max(range(len(p.bench)), key=lambda i: p.bench[i].current_hp)
            chosen_name = p.bench[best].name
            action = AM.encode(AT.PROMOTE, {"bench_slot": best})
            env.step(action)
            print(f"  {_c('Bot promotes:', Y)} {_c(chosen_name, W)}")
            time.sleep(0.3)
    return handled


def play_game(human_player: int, bot: PPOAgent | DQNAgent,
              seed: int = 0) -> int:
    """
    Play one full game. human_player is 0 (Lycanroc) or 1 (Raichu).
    Returns winner index.
    """
    env = PokemonTCGEnv(seed=seed, debug=False)
    env.reset(seed=seed)
    gs = env.gs
    rng = np.random.default_rng(seed + 1)

    header(f"NEW GAME — You are {'Lycanroc' if human_player==0 else 'Alolan Raichu'} "
           f"(Player {human_player})")
    print(f"  Bot controls: Player {1-human_player} "
          f"({'DQN/Raichu' if human_player==0 else 'PPO/Lycanroc'})")
    print(f"  First player: Player {gs.current_player}")
    input(_c("\n  Press Enter to start...", DIM))

    while not gs.game_over:
        # Handle any pending promotions first (can arise mid-loop after bot attacks)
        if _check_and_handle_promotion(gs, human_player, env):
            if gs.game_over:
                break
            continue

        cp = gs.current_player
        render_board(gs, human_player)

        if cp == human_player:
            print(f"\n  {_c('YOUR TURN', BLD+G)}")
            while True:
                # Re-check promotions inside the human turn loop
                if _check_and_handle_promotion(gs, human_player, env):
                    if gs.game_over:
                        break
                    render_board(gs, human_player)

                action = pick_action_human(gs, human_player)
                atype, _ = ActionMapper.decode(action)
                _, reward, done, info = env.step(action)

                # Check for promotion after every action
                if gs.game_over:
                    break
                if any(gs.players[p].pending_promotion for p in range(2)):
                    _check_and_handle_promotion(gs, human_player, env)
                    if gs.game_over:
                        break

                if atype == ActionType.ATTACK or atype == ActionType.END_TURN:
                    break

        else:
            print(f"\n  {_c('BOT THINKING...', BLD+R)}")
            time.sleep(0.3)

            # Bot acts until its turn ends (attack or end_turn)
            while gs.current_player == cp and not gs.game_over:
                # Check for pending promotions the bot needs to handle
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
                desc = _describe_action(action, gs, bot.player_idx)
                print(f"  Bot: {desc}")

                _, reward, done, info = env.step(action)
                time.sleep(0.15)

                # Handle any KO promotions that arose from bot's attack
                if any(gs.players[p].pending_promotion for p in range(2)):
                    _check_and_handle_promotion(gs, human_player, env)
                    if gs.game_over:
                        break

                if atype in (ActionType.ATTACK, ActionType.END_TURN):
                    break

    # Game over
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

def run_play(deck_choice: Optional[str] = None):
    header("POKEMON TCG vs BOT")

    # Choose deck
    if deck_choice and deck_choice.lower() in ("lycanroc", "raichu", "0", "1"):
        human_player = 0 if deck_choice.lower() in ("lycanroc", "0") else 1
    else:
        print(f"\n  Choose your deck:")
        print(f"    {_c('[0]', BLD+W)} {_c('Lycanroc', R)} (Fighting — P0)")
        print(f"    {_c('[1]', BLD+W)} {_c('Alolan Raichu', Y)} (Lightning/Psychic — P1)")
        while True:
            choice = ask("Enter 0 or 1")
            if choice in ("0", "1"):
                human_player = int(choice)
                break
            print(_c("  Enter 0 or 1", R))

    # Load bot for opponent slot
    bot_player = 1 - human_player
    bot = load_agent(bot_player, prefer_champion=True)
    lb  = load_leaderboard()
    key = "dqn" if bot_player == 1 else "ppo"
    wr  = lb[key]["champion_winrate"]
    info(f"Bot win-rate (vs random): {wr:.1%}")

    # Play loop
    record = [0, 0]
    game_num = 0
    while True:
        game_num += 1
        seed = int(time.time()) % 100_000 + game_num
        winner = play_game(human_player, bot, seed=seed)
        if winner == human_player:
            record[0] += 1
        elif winner == bot_player:
            record[1] += 1

        print(f"\n  {_c('Your record:', BLD)} {_c(record[0], G)} wins "
              f"/ {_c(record[1], R)} losses")

        again = ask("Play again? [y/n]")
        if again.lower() not in ("y", "yes", ""):
            break

    print()
    good("Thanks for playing!")

def run_benchmark(n_games: int = 100, deck: Optional[str] = None, seed: int = 0):
    """
    Run champion agent(s) against a random opponent for n_games each.
    Shows a live progress bar, per-game outcomes, and a summary table.

    deck: 'lycanroc', 'raichu', or None (both)
    """
    header(f"CHAMPION BENCHMARK  ({n_games} games vs random)")

    rng       = np.random.default_rng(seed if seed else int(time.time()))
    to_bench  = []

    if deck is None or deck.lower() in ("lycanroc", "0"):
        agent = load_agent(0, prefer_champion=True)
        to_bench.append(("PPO — Lycanroc", agent, 0))

    if deck is None or deck.lower() in ("raichu", "1"):
        agent = load_agent(1, prefer_champion=True)
        to_bench.append(("DQN — Alolan Raichu", agent, 1))

    all_results = []

    for label, agent, player_idx in to_bench:
        subheader(f"{label}  (Player {player_idx}) vs Random")

        wins = 0; losses = 0; draws = 0
        ko_totals_agent  = []
        ko_totals_random = []
        turn_totals      = []
        outcomes         = []   # 'W' / 'L' / 'D' per game

        bar_width = 40

        for g in range(n_games):
            game_seed = int(rng.integers(1e9))
            senv      = SelfPlayEnv(seed=game_seed)
            senv.reset(seed=game_seed)
            turns = 0

            while not senv.done and turns < 500:
                cp       = senv.current_player
                obs, mask = senv.obs_and_mask(cp)

                if cp == player_idx:
                    # Champion acts
                    if isinstance(agent, PPOAgent):
                        action, _, _ = agent.act(obs, mask, deterministic=True)
                    else:
                        action = agent.act(obs, mask, deterministic=True)
                else:
                    # Random opponent
                    legal  = np.where(mask > 0)[0]
                    action = int(rng.choice(legal))

                senv.step(action)
                turns += 1

            w = senv.winner
            gs = senv.env.gs

            ko_agent  = gs.players[player_idx].ko_count
            ko_random = gs.players[1 - player_idx].ko_count
            ko_totals_agent.append(ko_agent)
            ko_totals_random.append(ko_random)
            turn_totals.append(turns)

            if w == player_idx:
                wins += 1
                outcomes.append("W")
            elif w == -1:
                draws += 1
                outcomes.append("D")
            else:
                losses += 1
                outcomes.append("L")

            # Live progress bar
            done_pct = (g + 1) / n_games
            filled   = int(bar_width * done_pct)
            wr_live  = wins / (g + 1)
            col      = G if wr_live >= 0.55 else (Y if wr_live >= 0.40 else R)
            bar      = f"{col}{'█' * filled}{RST}{'░' * (bar_width - filled)}"
            print(f"\r  [{bar}] {g+1:>3}/{n_games}  "
                  f"W={_c(wins, G)} L={_c(losses, R)} D={_c(draws, Y)}  "
                  f"WR={_c(f'{wr_live:.1%}', col)}  ", end="", flush=True)

        print()   # newline after progress bar

        winrate = wins / n_games
        col     = G if winrate >= 0.60 else (Y if winrate >= 0.45 else R)

        # Outcome ribbon (compact visual)
        ribbon = ""
        for o in outcomes:
            ribbon += _c("W", G) if o == "W" else (_c("L", R) if o == "L" else _c("D", Y))
        print(f"\n  Outcome ribbon: {ribbon}")

        # Stats table
        avg_ko_agent  = sum(ko_totals_agent)  / n_games
        avg_ko_random = sum(ko_totals_random) / n_games
        avg_turns     = sum(turn_totals)       / n_games
        longest       = max(turn_totals)
        shortest      = min(turn_totals)

        print(f"\n  {'Metric':<28} {'Value':>10}")
        print(f"  {'─' * 40}")
        print(f"  {'Win-rate':<28} {_c(f'{winrate:.1%}', col):>10}")
        print(f"  {'Wins / Losses / Draws':<28} {f'{wins} / {losses} / {draws}':>10}")
        print(f"  {'Avg KOs (champion)':<28} {avg_ko_agent:>10.2f}")
        print(f"  {'Avg KOs (random)':<28} {avg_ko_random:>10.2f}")
        print(f"  {'Avg turns per game':<28} {avg_turns:>10.1f}")
        print(f"  {'Shortest game (turns)':<28} {shortest:>10}")
        print(f"  {'Longest game (turns)':<28} {longest:>10}")

        all_results.append({
            "label":    label,
            "winrate":  winrate,
            "wins":     wins,
            "losses":   losses,
            "draws":    draws,
            "avg_ko_agent":  avg_ko_agent,
            "avg_ko_random": avg_ko_random,
            "avg_turns": avg_turns,
        })

    # Summary if both decks were benchmarked
    if len(all_results) == 2:
        subheader("Summary")
        print(f"  {'Agent':<30} {'Win-rate':>10}  {'Avg KOs':>8}  {'Avg Turns':>10}")
        print(f"  {'─' * 62}")
        for r in all_results:
            col = G if r["winrate"] >= 0.60 else (Y if r["winrate"] >= 0.45 else R)
            wr_str = f'{r["winrate"]:.1%}'
            print(f"  {r['label']:<30} {_c(wr_str, col):>10}  "
                  f"{r['avg_ko_agent']:>8.2f}  {r['avg_turns']:>10.1f}")

    return all_results


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    args = sys.argv[1:]
    cmd  = args[0].lower() if args else "help"

    # Parse simple flags
    def flag(name, default=None):
        for i, a in enumerate(args):
            if a.startswith(f"--{name}="):
                return a.split("=", 1)[1]
            if a == f"--{name}" and i + 1 < len(args):
                return args[i + 1]
        return default

    if cmd == "play":
        deck = flag("deck")
        run_play(deck)

    elif cmd == "train":
        episodes  = int(flag("episodes", 500))
        eval_g    = int(flag("eval",     100))
        threshold = float(flag("threshold", 0.02))
        run_training(n_episodes=episodes, eval_games=eval_g,
                     improvement_threshold=threshold)

    elif cmd == "stats":
        show_stats()

    elif cmd == "benchmark":
        n     = int(flag("games", 100))
        deck  = flag("deck")          # lycanroc | raichu | None → both
        seed  = int(flag("seed", 0))
        run_benchmark(n_games=n, deck=deck, seed=seed)

    else:
        header("POKEMON TCG — PLAY & TRAIN")
        print(textwrap.dedent(f"""
          {BLD}COMMANDS:{RST}

            {_c('python play_and_train.py play', G)}
              Play against the trained bot.
              {DIM}--deck lycanroc|raichu   choose your deck (default: ask){RST}

            {_c('python play_and_train.py train', G)}
              Continue training from saved .npy files.
              Only promotes new weights if they beat the champion.
              {DIM}--episodes 500     how many self-play episodes to run
              --eval 100         games used for benchmarking
              --threshold 0.02   required improvement to crown new champion{RST}

            {_c('python play_and_train.py benchmark', G)}
              Run champion(s) against a random opponent and report win-rate.
              {DIM}--games 100        number of games to play (default: 100)
              --deck lycanroc|raichu  test only one deck (default: both)
              --seed 0           random seed for reproducibility{RST}

            {_c('python play_and_train.py stats', G)}
              Show leaderboard and training history.

          {BLD}FILES:{RST}
            {PPO_PATH}
            {DQN_PATH}
            {PPO_CHAMP_PATH}
            {DQN_CHAMP_PATH}
            {LEADERBOARD}
        """))

if __name__ == "__main__":
    main()