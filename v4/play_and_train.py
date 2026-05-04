"""
Pokemon TCG v4 — Play & Train
══════════════════════════════
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
    MAX_BENCH, MAX_ATTACKS, MAX_HAND,
)
from rl_agents import (
    PPOAgent, DQNAgent, SelfPlayEnv, OBS_SIZE, ACT_SIZE,
    evaluate, relu, pretrain_agents,
)

# ── Paths ────────────────────────────────────────────────────────────────────
OUT_DIR         = os.path.join(_DIR, "outputs")
PPO_PATH        = os.path.join(OUT_DIR, "ppo_lucario.npy")
DQN_PATH        = os.path.join(OUT_DIR, "dqn_starmie.npy")
PPO_CHAMP_PATH  = os.path.join(OUT_DIR, "ppo_lucario_champion.npy")
DQN_CHAMP_PATH  = os.path.join(OUT_DIR, "dqn_starmie_champion.npy")
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
        "ppo": {"champion_winrate": 0.0, "champion_score": 0.0, "history": [], "total_episodes": 0},
        "dqn": {"champion_winrate": 0.0, "champion_score": 0.0, "history": [], "total_episodes": 0},
    }

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

def show_stats():
    header("LEADERBOARD & STATS")
    lb = load_leaderboard()

    for key, label in [("ppo", "PPO — Mega Lucario ex"), ("dqn", "DQN — Mega Starmie ex")]:
        data = lb[key]
        subheader(label)
        wr_str = f"{data['champion_winrate']:.1%}"
        sc_str = f"{data.get('champion_score', 0):.0f}"
        print(f"  Champion win-rate : {_c(wr_str, G)}")
        print(f"  Champion score    : {_c(sc_str, G)}  (KO×5 + deckout×1, 10k games)")
        print(f"  Total episodes    : {data['total_episodes']}")
        print(f"  Training runs     : {len(data['history'])}")
        if data["history"]:
            print(f"\n  {'Date':>18}  {'Score':>8}  {'Win%':>7}  {'KO W':>6}  {'Deck W':>7}  {'Eps':>7}  {'Status':>12}")
            print(f"  {'─'*72}")
            for e in data["history"][-10:]:
                star = _c("★ CHAMPION", G) if e["champion"] else _c("  skipped", DIM)
                sc   = e.get("score", 0)
                ko   = e.get("ko_wins", 0)
                dk   = e.get("deckout_wins", 0)
                print(f"  {e['timestamp']:>18}  {sc:>8.0f}  {e['winrate']:>6.1%}  "
                      f"{ko:>6}  {dk:>7}  {e['episodes']:>7}  {star}")

# ════════════════════════════════════════════════════════════════════════════
# AGENT LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_agent(player_idx: int, prefer_champion: bool = True) -> PPOAgent | DQNAgent:
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


def score_agent(agent: PPOAgent | DQNAgent,
                n_games: int = 10_000,
                seed: int = 777_777,
                ko_weight: float = 5.0,
                deckout_weight: float = 1.0,
                verbose: bool = False) -> Dict:
    """
    Run agent against a random opponent for n_games with a fixed seed.
    KO win → ko_weight points; Deck-out win → deckout_weight points.
    """
    rng          = np.random.default_rng(seed)
    score        = 0.0
    wins         = 0
    ko_wins      = 0
    deckout_wins = 0
    losses       = 0
    draws        = 0

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
                score   += ko_weight
                ko_wins += 1
            else:
                score        += deckout_weight
                deckout_wins += 1
        elif w == -1:
            draws += 1
        else:
            losses += 1

        if verbose and (g + 1) % 1000 == 0:
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
    result = score_agent(agent, n_games=n_games, seed=seed, verbose=False)
    return result["winrate"]


# ════════════════════════════════════════════════════════════════════════════
# INCREMENTAL TRAINING
# ════════════════════════════════════════════════════════════════════════════

def run_training(n_episodes: int = 500, eval_games: int = 100,
                 improvement_threshold: float = 50.0):
    """
    Load existing champions, train for n_episodes self-play episodes, then
    evaluate with weighted scoring (KO win = 5 pts, deck-out win = 1 pt).
    New weights saved only if score beats champion by ≥ improvement_threshold.
    """
    header("INCREMENTAL TRAINING — V4")

    ppo = load_agent(0)
    ppo.ent_coef = 0.03
    ppo.lr       = 2e-4
    ppo.clip_eps = 0.15

    dqn = load_agent(1)
    dqn.lr = 5e-4

    lb = load_leaderboard()
    is_fresh = lb["ppo"]["champion_winrate"] < 0.40 or lb["dqn"]["champion_winrate"] < 0.40
    if is_fresh:
        subheader("Behavioral Cloning warm-start (from HeuristicAgent)")
        pretrain_agents(ppo, dqn, n_games=200, n_epochs=6, verbose=True)

    old_ppo_score = lb["ppo"].get("champion_score", 0.0)
    old_dqn_score = lb["dqn"].get("champion_score", 0.0)
    info(f"Existing PPO champion score: {old_ppo_score:.0f}")
    info(f"Existing DQN champion score: {old_dqn_score:.0f}")

    subheader(f"Training {n_episodes} self-play episodes")
    rng = np.random.default_rng(int(time.time()))
    ppo_turns = 0
    recent_wins: deque = deque(maxlen=100)
    t0 = time.time()

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
            ppo.finish_episode(0.0)
            ppo.update()
            ppo_turns = 0
        if dqn_prev:
            prev_obs, prev_action, prev_reward, prev_mask = dqn_prev
            dqn.store(prev_obs, prev_action, prev_reward,
                      prev_obs, True, prev_mask,
                      np.zeros(ACT_SIZE, np.float32))

        recent_wins.append(senv.winner)

        if (ep + 1) % max(1, n_episodes // 10) == 0:
            n = len(recent_wins)
            w0 = sum(1 for w in recent_wins if w == 0) / n
            w1 = sum(1 for w in recent_wins if w == 1) / n
            pct = (ep + 1) / n_episodes
            bar = "█" * int(20 * pct) + "░" * (20 - int(20 * pct))
            elapsed = time.time() - t0
            eta = elapsed / pct - elapsed if pct > 0 else 0
            print(f"  [{bar}] {pct:.0%}  P0(Lucario)={w0:.0%} P1(Starmie)={w1:.0%}  "
                  f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s")

    EVAL_GAMES = 10_000
    EVAL_SEED  = 777_777
    KO_W       = 5.0
    DECK_W     = 1.0
    MIN_IMPROVEMENT = improvement_threshold

    subheader(f"Champion evaluation  ({EVAL_GAMES:,} games · seed {EVAL_SEED})")
    print(f"  Scoring: KO win = ×{KO_W:.0f} pts  │  Deck-out win = ×{DECK_W:.0f} pt  │  "
          f"Min improvement = {MIN_IMPROVEMENT:.0f} pts\n")
    print(f"  {'Agent':6}  {'Challenger':>12}  {'Champion':>12}  {'Δ':>8}  {'Decision'}")
    print(f"  {'─'*62}")

    for label, agent_key, agent, champ_path, snap in [
        ("PPO", "ppo", ppo, PPO_CHAMP_PATH, ppo_snapshot),
        ("DQN", "dqn", dqn, DQN_CHAMP_PATH, dqn_snapshot),
    ]:
        print(f"  Scoring {label} challenger…")
        new_result = score_agent(agent, n_games=EVAL_GAMES, seed=EVAL_SEED,
                                 ko_weight=KO_W, deckout_weight=DECK_W, verbose=True)
        new_score  = new_result["score"]

        champ_exists = os.path.exists(champ_path)
        if champ_exists:
            print(f"  Scoring {label} champion…")
            champ_agent  = load_agent(0 if agent_key == "ppo" else 1, prefer_champion=True)
            champ_result = score_agent(champ_agent, n_games=EVAL_GAMES, seed=EVAL_SEED,
                                       ko_weight=KO_W, deckout_weight=DECK_W, verbose=True)
            champ_score  = champ_result["score"]
        else:
            champ_score = -1.0
            info(f"No existing {label} champion — challenger auto-promoted")

        delta    = new_score - champ_score
        improved = delta > MIN_IMPROVEMENT
        col      = G if improved else (Y if delta > 0 else R)
        d_str    = _c(f"{delta:+.0f}", col)

        if improved:
            agent.save(champ_path)
            agent.save(PPO_PATH if agent_key == "ppo" else DQN_PATH)
            record_result(lb, agent_key, new_result["winrate"], n_episodes,
                          promoted=True, score=new_score,
                          ko_wins=new_result["ko_wins"],
                          deckout_wins=new_result["deckout_wins"])
            status = _c("★ NEW CHAMPION", G)
        else:
            if agent_key == "ppo":
                agent.net.set_params(snap)
            else:
                agent.qnet.set_params(snap)
                agent.tnet.copy_params_from(agent.qnet)
            record_result(lb, agent_key, new_result["winrate"], n_episodes,
                          promoted=False, score=new_score,
                          ko_wins=new_result["ko_wins"],
                          deckout_wins=new_result["deckout_wins"])
            status = _c("  kept old", DIM)

        champ_str = f"{champ_score:.0f}" if champ_exists else "  (none)"
        print(f"  {label:6}  "
              f"{new_score:>9.0f} pts  "
              f"{champ_str:>9} pts  "
              f"{d_str:>8}  "
              f"{status}")
        print(f"         WR {new_result['winrate']:.1%}  "
              f"KO {new_result['ko_wins']}  "
              f"Deck {new_result['deckout_wins']}\n")

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
    EnergyType.WATER:     "💧",
    EnergyType.DARKNESS:  "🌑",
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

def _pokemon_line(p: PokemonCard, prefix: str = "") -> str:
    stage = STAGE_LABELS.get(p.stage, "?")
    tool  = f"  Tool:{_c(p.tool_card.name, C)}" if p.tool_card else ""
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

    stadium_str = (f"  Stadium: {_c(gs.active_stadium.name, B)}" if gs.active_stadium
                   else "")

    print(f"\n{BLD}{C}{'═'*62}{RST}")
    print(f"{BLD}{W}  Turn {gs.turn_number}  │  Your KOs: {_c(me.ko_count, G)}/6  "
          f"│  Opp KOs: {_c(opp.ko_count, R)}/6{stadium_str}{RST}")
    print(f"{BLD}{C}{'─'*62}{RST}")

    # Opponent
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

    # Human
    print(f"\n  {_c('YOU', G)}")
    print(f"  Deck: {len(me.deck)} cards   "
          f"Energy used: {_c('Yes', R) if me.energy_used else _c('No', G)}   "
          f"Supporter used: {_c('Yes', R) if me.supporter_used else _c('No', G)}")
    if me.active:
        print(f"  {_c('Active:', G)} {_pokemon_line(me.active)}")
        for i, atk in enumerate(me.active.attacks):
            can_use = _can_attack(me.active, atk)
            flag = _c("✓", G) if can_use else _c("✗", R)
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
            extra = f" [{STAGE_LABELS.get(c.stage,'?')}]{ex_tag}"
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
            atk = me.active.attacks[ai]
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
            tgt = me.bench[bench_slot]
            return (f"{_c('RETREAT', B)}: {_c(me.active.name if me.active else '?', W)} "
                    f"→ {_c(tgt.name, W)} ({_hp_bar(tgt.current_hp, tgt.hp)})")
        return f"RETREAT to bench slot {bench_slot}"

    if atype == ActionType.ATTACH_ENERGY:
        slot = params["slot"]
        tgt = (me.active.name if me.active else "Active") if slot == 0 else \
              (me.bench[slot-1].name if slot-1 < len(me.bench) else f"bench[{slot-1}]")
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

    if atype == ActionType.USE_STADIUM:
        hi = params["hand_idx"]
        if hi < len(me.hand):
            return f"{_c('STADIUM', B)}: {_c(me.hand[hi].name, W)}"
        return "USE stadium"

    if atype == ActionType.ATTACH_TOOL:
        hi, slot = params["hand_idx"], params["slot"]
        card_name = me.hand[hi].name if hi < len(me.hand) else "?"
        if slot == 0:
            tgt = me.active.name if me.active else "Active"
        else:
            tgt = me.bench[slot-1].name if slot-1 < len(me.bench) else f"bench[{slot-1}]"
        return f"{_c('TOOL', C)}: {_c(card_name, W)} → {_c(tgt, M)}"

    if atype == ActionType.EVOLVE:
        hi, slot = params["hand_idx"], params["slot"]
        card_name = me.hand[hi].name if hi < len(me.hand) else "?"
        if slot == MAX_BENCH:
            tgt = me.active.name if me.active else "Active"
        else:
            tgt = me.bench[slot].name if slot < len(me.bench) else f"bench[{slot}]"
        return f"{_c('EVOLVE', B)}: {_c(card_name, W)} onto {_c(tgt, M)}"

    if atype == ActionType.USE_ABILITY:
        slot = params.get("slot", 0)
        if slot == 0:
            poke = me.active
        else:
            bi = slot - 1
            poke = me.bench[bi] if bi < len(me.bench) else None
        poke_name = poke.name if poke else "?"
        ability   = poke.ability_name if poke else "ability"
        return f"{_c('ABILITY', M)}: {_c(ability, W)} on {_c(poke_name, BLD)}"

    return f"Action {idx}"


def pick_action_human(gs: GameState, human_player: int) -> int:
    me = gs.players[human_player]
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
                        action = AM.encode(AT.PROMOTE, {"bench_slot": choice})
                        env.step(action)
                        good(f"Promoted {chosen_name} to Active!")
                        break
                    print(_c(f"  Enter 0–{len(p.bench)-1}", R))
                except (ValueError, IndexError):
                    print(_c("  Invalid input", R))
        else:
            best = max(range(len(p.bench)), key=lambda i: p.bench[i].current_hp)
            chosen_name = p.bench[best].name
            action = AM.encode(AT.PROMOTE, {"bench_slot": best})
            env.step(action)
            print(f"  {_c('Bot promotes:', Y)} {_c(chosen_name, W)}")
            time.sleep(0.3)
    return handled


def play_game(human_player: int, bot: PPOAgent | DQNAgent,
              seed: int = 0) -> int:
    env = PokemonTCGEnv(seed=seed, debug=False)
    env.reset(seed=seed)
    gs = env.gs
    rng = np.random.default_rng(seed + 1)

    deck_name = "Mega Lucario ex" if human_player == 0 else "Mega Starmie ex"
    bot_deck  = "Mega Starmie ex" if human_player == 0 else "Mega Lucario ex"
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

                action = pick_action_human(gs, human_player)
                atype, _ = ActionMapper.decode(action)
                _, reward, done, info_dict = env.step(action)

                if gs.game_over:
                    break
                if any(gs.players[p].pending_promotion for p in range(2)):
                    _check_and_handle_promotion(gs, human_player, env)
                    if gs.game_over:
                        break

                if atype in (ActionType.ATTACK, ActionType.END_TURN):
                    break
                # Mega Evolution also ends turn
                if gs.mega_evolved_this_turn:
                    print(_c("  Mega Evolution ends your turn!", Y))
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
                desc = _describe_action(action, gs, bot.player_idx)
                print(f"  Bot: {desc}")

                _, reward, done, info_dict = env.step(action)
                time.sleep(0.15)

                if any(gs.players[p].pending_promotion for p in range(2)):
                    _check_and_handle_promotion(gs, human_player, env)
                    if gs.game_over:
                        break

                if atype in (ActionType.ATTACK, ActionType.END_TURN):
                    break
                if gs.mega_evolved_this_turn:
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


def run_play(deck_choice: Optional[str] = None):
    header("POKEMON TCG V4 vs BOT")

    if deck_choice and deck_choice.lower() in ("lucario", "starmie", "0", "1"):
        human_player = 0 if deck_choice.lower() in ("lucario", "0") else 1
    else:
        print(f"\n  Choose your deck:")
        print(f"    {_c('[0]', BLD+W)} {_c('Mega Lucario ex', R)} (Fighting — P0)")
        print(f"    {_c('[1]', BLD+W)} {_c('Mega Starmie ex', C)} (Water/Psychic — P1)")
        while True:
            choice = ask("Enter 0 or 1")
            if choice in ("0", "1"):
                human_player = int(choice)
                break
            print(_c("  Enter 0 or 1", R))

    bot_player = 1 - human_player
    bot = load_agent(bot_player, prefer_champion=True)
    lb  = load_leaderboard()
    key = "dqn" if bot_player == 1 else "ppo"
    wr  = lb[key]["champion_winrate"]
    info(f"Bot win-rate (vs random): {wr:.1%}")

    record = [0, 0, 0]
    game_num = 0
    while True:
        game_num += 1
        seed = int(time.time()) % 100_000 + game_num
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

    if deck is None or deck.lower() in ("lucario", "0"):
        agent = load_agent(0, prefer_champion=True)
        to_bench.append(("PPO — Mega Lucario ex", agent, 0))

    if deck is None or deck.lower() in ("starmie", "1"):
        agent = load_agent(1, prefer_champion=True)
        to_bench.append(("DQN — Mega Starmie ex", agent, 1))

    all_results = []

    for label, agent, player_idx in to_bench:
        subheader(f"{label}  (Player {player_idx}) vs Random")

        wins = 0; losses = 0; draws = 0
        ko_wins = 0; deckout_wins = 0
        ko_totals_agent  = []
        ko_totals_random = []
        turn_totals      = []
        outcomes         = []

        bar_width = 40

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

            ko_agent  = gs.players[player_idx].ko_count
            ko_random = gs.players[1 - player_idx].ko_count
            ko_totals_agent.append(ko_agent)
            ko_totals_random.append(ko_random)
            turn_totals.append(turns)

            if w == player_idx:
                wins += 1
                ko_wins += int(reason == "ko")
                deckout_wins += int(reason != "ko")
                outcomes.append("W")
            elif w == -1:
                draws += 1
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
            ribbon += _c("W", G) if o == "W" else (_c("L", R) if o == "L" else _c("D", Y))
        print(f"\n  Outcome ribbon: {ribbon}")

        avg_ko_agent  = sum(ko_totals_agent)  / n_games
        avg_ko_random = sum(ko_totals_random) / n_games
        avg_turns     = sum(turn_totals)       / n_games

        weighted = ko_wins * 5 + deckout_wins

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
            "avg_ko_agent": avg_ko_agent,
            "avg_turns":    avg_turns,
        })

    if len(all_results) == 2:
        subheader("Summary")
        print(f"  {'Agent':<32} {'Win-rate':>10}  {'Score':>8}  {'KO W':>6}  {'Deck W':>7}  {'Avg Turns':>10}")
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
        deck = flag("deck")
        run_play(deck)

    elif cmd == "train":
        episodes  = int(flag("episodes", 500))
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
        header("POKEMON TCG V4 — PLAY & TRAIN")
        print(textwrap.dedent(f"""
          {BLD}COMMANDS:{RST}

            {_c('python play_and_train.py play', G)}
              Play against the trained bot (Human vs Bot).
              {DIM}--deck lucario|starmie   choose your deck (default: ask){RST}

            {_c('python play_and_train.py train', G)}
              Continue training from saved .npy files via self-play.
              Champion decided by 10,000-game weighted score (KO×5, deck-out×1).
              {DIM}--episodes 500     self-play episodes (default: 500)
              --threshold 50     min score improvement to crown new champion{RST}

            {_c('python play_and_train.py benchmark', G)}
              Run champion(s) against a random opponent.
              {DIM}--games 100           number of games (default: 100)
              --deck lucario|starmie  test only one deck (default: both)
              --seed 0               random seed for reproducibility{RST}

            {_c('python play_and_train.py stats', G)}
              Show leaderboard and training history.

          {BLD}FILES:{RST}
            {PPO_PATH}
            {DQN_PATH}
            {PPO_CHAMP_PATH}
            {DQN_CHAMP_PATH}
            {LEADERBOARD}

          {BLD}DECKS:{RST}
            P0 — Mega Lucario ex  (Fighting)   trained with PPO
            P1 — Mega Starmie ex  (Water/Dark)  trained with DQN
        """))

if __name__ == "__main__":
    main()
