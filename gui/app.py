"""
Pokemon TCG — Web GUI Backend
Flask server that bridges the game engine and champion RL agents to a browser UI.

Run:
    pip install flask
    python app.py

Then open http://localhost:5000
"""

from __future__ import annotations
import sys, os, time, threading, copy
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, render_template

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from ptcg_env import (
    PokemonTCGEnv, ActionMapper, ActionType, EnergyType, Stage, CardType,
    PokemonCard, EnergyCard, TrainerCard, compute_legal_mask,
    GameState, PlayerState, MAX_BENCH, MAX_ATTACKS, KO_TO_WIN,
    StateEncoder,
)
from rl_agents import PPOAgent, DQNAgent, ACT_SIZE

# ── Paths ──────────────────────────────────────────────────────────────────
OUT_DIR        = os.path.join(_DIR, "outputs")
PPO_CHAMP      = os.path.join(OUT_DIR, "ppo_lycanroc_champion.npy")
DQN_CHAMP      = os.path.join(OUT_DIR, "dqn_raichu_champion.npy")
IMAGES_DIR     = os.path.join(_DIR, "pokemon_images")

app = Flask(__name__, template_folder=os.path.join(_DIR, "templates"))

# Effects that flag a card as a healer. Add new effect names here when
# extending the engine — avoids hardcoding strings in multiple places.
HEAL_EFFECTS = {"heal_active_20", "heal_active_30"}

# Cap stored log size to avoid unbounded memory growth in long games.
LOG_MAX = 200

# ── Global game state (single session) ────────────────────────────────────
_game: dict = {
    "env":          None,   # PokemonTCGEnv
    "human_player": None,   # 0 or 1
    "bot":          None,   # PPOAgent | DQNAgent
    "bot_player":   None,   # 0 or 1
    "log":          [],     # list of message strings
    "pending":      False,  # True while bot is "thinking"
    "mode":         "idle", # "idle" | "playing" | "over"
}
_lock = threading.Lock()


def _log(msg: str) -> None:
    """Append a log line, trimming history to LOG_MAX."""
    g = _game
    g["log"].append(msg)
    if len(g["log"]) > LOG_MAX:
        del g["log"][:-LOG_MAX]


# ══════════════════════════════════════════════════════════════════════════
# SERIALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════

ENERGY_NAMES = {
    EnergyType.COLORLESS: "Colorless",
    EnergyType.FIGHTING:  "Fighting",
    EnergyType.LIGHTNING: "Lightning",
    EnergyType.PSYCHIC:   "Psychic",
}

ENERGY_COLORS = {
    EnergyType.COLORLESS: "#bbb",
    EnergyType.FIGHTING:  "#c84",
    EnergyType.LIGHTNING: "#fe4",
    EnergyType.PSYCHIC:   "#c8f",
}

STAGE_NAMES = {Stage.BASIC: "Basic", Stage.STAGE1: "Stage 1", Stage.STAGE2: "Stage 2"}


def _img(name: str) -> str:
    """Return URL for a card image, checking .jpg then .png, or empty string if missing."""
    base = name.replace(" ", "_")
    for ext in (".jpg", ".jpeg", ".png"):
        path = os.path.join(IMAGES_DIR, base + ext)
        if os.path.exists(path):
            return f"/images/{base}{ext}"
    return ""


def _serialize_energy(energy: dict) -> list:
    out = []
    for etype, count in sorted(energy.items()):
        out.append({
            "type":  ENERGY_NAMES.get(etype, "?"),
            "color": ENERGY_COLORS.get(etype, "#888"),
            "count": count,
        })
    return out


def _serialize_attack(atk) -> dict:
    cost = []
    for etype, count in sorted(atk.energy_cost.items()):
        for _ in range(count):
            cost.append({
                "type":  ENERGY_NAMES.get(etype, "?"),
                "color": ENERGY_COLORS.get(etype, "#888"),
            })
    return {
        "name":   atk.name,
        "damage": atk.damage,
        "cost":   cost,
        "effect": atk.effect or "",
    }


def _serialize_pokemon(p: PokemonCard | None, index: int = -1) -> dict | None:
    if p is None:
        return None
    hp_max = max(p.hp, 1)   # guard against div-by-zero
    hp_pct = round(max(0, min(p.current_hp, hp_max)) / hp_max * 100)
    return {
        "name":     p.name,
        "hp":       p.hp,
        "current_hp": p.current_hp,
        "hp_pct":   hp_pct,
        "stage":    STAGE_NAMES.get(p.stage, "?"),
        "evolves_from": p.evolves_from or "",
        "energy":   _serialize_energy(p.energy),
        "total_energy": p.total_energy(),
        "attacks":  [_serialize_attack(a) for a in p.attacks],
        "img":      _img(p.name),
        "index":    index,
    }


def _serialize_card(card, hand_idx: int) -> dict:
    if isinstance(card, PokemonCard):
        d = _serialize_pokemon(card, index=hand_idx)
        d["card_type"] = "pokemon"
        return d
    if isinstance(card, EnergyCard):
        return {
            "card_type": "energy",
            "name":      card.name,
            "energy_type": ENERGY_NAMES.get(card.energy_type, "?"),
            "color":     ENERGY_COLORS.get(card.energy_type, "#888"),
            "img":       _img(card.name),
            "index":     hand_idx,
        }
    if isinstance(card, TrainerCard):
        ct = "item" if card.card_type == CardType.ITEM else "supporter"
        return {
            "card_type": ct,
            "name":      card.name,
            "effect":    card.effect,
            "img":       _img(card.name),
            "index":     hand_idx,
        }
    return {"card_type": "unknown", "name": "?", "index": hand_idx}


def _get_legal_actions_for(env: PokemonTCGEnv, player_idx: int) -> list[int]:
    gs   = env.gs
    orig = gs.current_player
    gs.current_player = player_idx
    mask = compute_legal_mask(gs)
    gs.current_player = orig
    return [int(i) for i, v in enumerate(mask) if v > 0]


def _build_legal_set(env: PokemonTCGEnv, human: int) -> set[int]:
    return set(_get_legal_actions_for(env, human))


def _promotion_pending(gs: GameState) -> dict | None:
    """If any player has pending_promotion, return info about who and what bench they have."""
    for pidx in range(2):
        p = gs.players[pidx]
        if p.pending_promotion:
            return {
                "player_idx": pidx,
                "bench": [_serialize_pokemon(pk, i) for i, pk in enumerate(p.bench)],
            }
    return None


def build_state_json() -> dict:
    """Serialise the full visible game state for the frontend.

    Output contract:
      legal_hand[hand_idx] = list of action records for that card.
      Each record is {action: int, type: str, params: dict, target?: dict}
      where `target` describes the target Pokémon (slot, name) for cards
      that need a target picker (energy attach, heal items).
    """
    g   = _game
    env = g["env"]
    if env is None:
        return {"mode": "idle"}

    gs   = env.gs
    h    = g["human_player"]
    bot  = g["bot_player"]
    me   = gs.players[h]
    opp  = gs.players[bot]

    legal = _build_legal_set(env, h) if not g["pending"] else set()

    # ── Bucket actions by action type ────────────────────────────────
    by_type: dict[ActionType, list[tuple[int, dict]]] = {t: [] for t in ActionType}
    for aidx in legal:
        atype, params = ActionMapper.decode(aidx)
        by_type[atype].append((aidx, params))

    # ── Helper: describe a target Pokémon from a slot index ─────────
    def target_desc(slot: int) -> dict:
        """slot 0 = active; 1..5 = bench[slot-1]"""
        if slot == 0:
            return {"slot": 0, "name": me.active.name if me.active else "?", "where": "active"}
        bi = slot - 1
        name = me.bench[bi].name if 0 <= bi < len(me.bench) else "?"
        return {"slot": slot, "name": name, "where": "bench", "bench_idx": bi}

    # ── Build legal_hand: each hand card → list of actions ───────────
    legal_hand: dict[int, list] = {}

    # Hand-indexed actions (PLAY_POKEMON, USE_ITEM, USE_SUPPORTER, EVOLVE)
    for atype in (ActionType.PLAY_POKEMON, ActionType.USE_ITEM,
                  ActionType.USE_SUPPORTER, ActionType.EVOLVE):
        for aidx, params in by_type[atype]:
            hi = params.get("hand_idx", -1)
            if hi < 0:
                continue
            entry = {"action": aidx, "type": atype.name, "params": params}
            if atype == ActionType.EVOLVE:
                # Evolve target slot: 0..4 = bench, 5 = active. Convert to our slot scheme.
                slot = params["slot"]
                ui_slot = 0 if slot == 5 else slot + 1   # active=0, bench[i]=i+1
                entry["target"] = target_desc(ui_slot)
            legal_hand.setdefault(hi, []).append(entry)

    # Energy cards: ATTACH_ENERGY is keyed by slot, not hand_idx.
    # Map all attach actions onto every energy card in hand so any of them
    # is a valid click target visually.  Engine consumes whichever Energy is found first.
    attach_actions = by_type[ActionType.ATTACH_ENERGY]
    if attach_actions:
        for hi, card in enumerate(me.hand):
            if not isinstance(card, EnergyCard):
                continue
            for aidx, params in attach_actions:
                slot = params["slot"]
                legal_hand.setdefault(hi, []).append({
                    "action": aidx,
                    "type":   "ATTACH_ENERGY",
                    "params": params,
                    "target": target_desc(slot),
                })

    # ── Slot-keyed lookups for direct clicks on Pokémon ──────────────
    legal_attack: dict[int, int]  = {}
    legal_attach: dict[int, int]  = {}
    legal_promote: dict[int, int] = {}

    for aidx, params in by_type[ActionType.ATTACK]:
        legal_attack[params["atk_idx"]] = aidx
    for aidx, params in by_type[ActionType.ATTACH_ENERGY]:
        legal_attach[params["slot"]] = aidx
    for aidx, params in by_type[ActionType.PROMOTE]:
        legal_promote[params["bench_slot"]] = aidx

    # ── Discard pile (full list, newest last) ────────────────────────
    discard_all     = [_serialize_card(c, i) for i, c in enumerate(me.discard)]
    opp_discard_all = [_serialize_card(c, i) for i, c in enumerate(opp.discard)]
    discard_top     = discard_all[-1]     if discard_all     else None
    opp_discard_top = opp_discard_all[-1] if opp_discard_all else None

    # ── Hand-card categorisation flags ───────────────────────────────
    # Energy cards in hand (index list)
    energy_hand_indices = [i for i, c in enumerate(me.hand) if isinstance(c, EnergyCard)]
    # Heal items currently target only the active in this engine; we still flag
    # them for the UI so it can show a clear "tap active to heal" affordance.
    heal_hand_indices: list[int] = []
    for aidx, params in by_type[ActionType.USE_ITEM]:
        hi = params.get("hand_idx", -1)
        if hi < 0 or hi >= len(me.hand):
            continue
        card = me.hand[hi]
        if isinstance(card, TrainerCard) and card.effect in HEAL_EFFECTS:
            if hi not in heal_hand_indices:
                heal_hand_indices.append(hi)

    return {
        "mode":          g["mode"],
        "turn":          gs.turn_number,
        "current_player": gs.current_player,
        "human_player":  h,
        "bot_player":    bot,
        "is_my_turn":    gs.current_player == h and not g["pending"],
        "pending_bot":   g["pending"],
        "energy_used":   me.energy_used,
        "supporter_used": me.supporter_used,
        "winner":        gs.winner,
        "win_reason":    gs.win_reason,
        "log":           g["log"][-30:],

        "my": {
            "ko_count":     me.ko_count,
            "deck_size":    len(me.deck),
            "hand":         [_serialize_card(c, i) for i, c in enumerate(me.hand)],
            "active":       _serialize_pokemon(me.active),
            "bench":        [_serialize_pokemon(p, i) for i, p in enumerate(me.bench)],
            "discard_size": len(me.discard),
            "discard_top":  discard_top,
            "discard_all":  discard_all,
        },
        "opp": {
            "ko_count":     opp.ko_count,
            "deck_size":    len(opp.deck),
            "hand_size":    len(opp.hand),
            "active":       _serialize_pokemon(opp.active),
            "bench":        [_serialize_pokemon(p, i) for i, p in enumerate(opp.bench)],
            "discard_size": len(opp.discard),
            "discard_top":  opp_discard_top,
            "discard_all":  opp_discard_all,
        },

        "legal_hand":           {k: v for k, v in legal_hand.items()},
        "legal_attacks":        legal_attack,
        "legal_attach":         legal_attach,
        "legal_promote":        legal_promote,
        "energy_hand_indices":  energy_hand_indices,
        "heal_hand_indices":    heal_hand_indices,
        "can_end_turn":         (ActionMapper.END_TURN_IDX in legal),
        "promotion":            _promotion_pending(gs),
    }


# ══════════════════════════════════════════════════════════════════════════
# BOT LOGIC
# ══════════════════════════════════════════════════════════════════════════

def _bot_act(env: PokemonTCGEnv, bot_agent, bot_player: int) -> int:
    """Get the bot's chosen action."""
    gs   = env.gs
    orig = gs.current_player
    gs.current_player = bot_player
    obs  = StateEncoder.encode(gs)
    mask = compute_legal_mask(gs)
    gs.current_player = orig

    if isinstance(bot_agent, PPOAgent):
        action, _, _ = bot_agent.act(obs, mask, deterministic=True)
    else:
        action = bot_agent.act(obs, mask, deterministic=True)
    return int(action)


def _run_bot_turn():
    """Run the bot's full turn in a background thread with natural per-action delays."""
    time.sleep(0.6)  # initial pause so human can see the board

    g   = _game
    env = g["env"]
    bot = g["bot"]
    bot_player = g["bot_player"]
    gs  = env.gs

    # Natural delays per action type (seconds)
    ACTION_DELAYS = {
        ActionType.ATTACK:       0.7,   # attacking feels significant
        ActionType.END_TURN:     0.3,
        ActionType.EVOLVE:       0.5,   # evolution is a moment
        ActionType.ATTACH_ENERGY:0.35,
        ActionType.PLAY_POKEMON: 0.35,
        ActionType.USE_SUPPORTER:0.45,  # Hau drawing 3 cards — slight pause
        ActionType.USE_ITEM:     0.35,
        ActionType.PROMOTE:      0.4,
    }

    while (not gs.game_over
           and gs.current_player == bot_player
           and not any(gs.players[p].pending_promotion for p in range(2))):

        try:
            action = _bot_act(env, bot, bot_player)
        except Exception as e:
            with _lock:
                _log(f"Bot error during action selection: {e}")
            break
        atype, params = ActionMapper.decode(action)

        # For non-attack actions: log BEFORE step (the action references hand
        # indices that will shift after step). For attacks: defer logging so
        # we can include actual damage dealt.
        if atype != ActionType.ATTACK:
            msg = _action_to_log(action, gs, bot_player, prefix="Bot")
            with _lock:
                _log(msg)
                try:
                    env.step(action)
                except Exception as e:
                    _log(f"Bot step failed: {e}")
                    break
        else:
            # Capture attack name BEFORE step (the active card might be KO'd)
            atk_name = ""
            if me_bot := gs.players[bot_player].active:
                ai = params.get("atk_idx", 0)
                if 0 <= ai < len(me_bot.attacks):
                    atk_name = me_bot.attacks[ai].name
            with _lock:
                try:
                    env.step(action)
                except Exception as e:
                    _log(f"Bot step failed: {e}")
                    break
                dmg = gs.last_attack_damage
                _log(f"Bot used {atk_name}!" + (f" ({dmg} dmg)" if dmg else ""))
                if gs.last_damage_prevented:
                    _log("✓ Your protection blocked all damage!")
                elif gs.last_damage_reduced:
                    _log("✓ Your effect reduced bot's damage.")

        delay = ACTION_DELAYS.get(atype, 0.35)
        time.sleep(delay)

        if atype in (ActionType.ATTACK, ActionType.END_TURN):
            break
        if gs.game_over:
            break

    # Handle bot promotion if needed
    _handle_bot_promotion()

    with _lock:
        g["pending"] = False
        if gs.game_over:
            g["mode"] = "over"
            winner_label = "You win! 🎉" if gs.winner == g["human_player"] else "Bot wins!"
            _log(f"--- GAME OVER: {winner_label} (by {gs.win_reason}) ---")


def _handle_bot_promotion():
    """If bot has pending_promotion, resolve it automatically."""
    g   = _game
    env = g["env"]
    gs  = env.gs
    bot_player = g["bot_player"]

    p = gs.players[bot_player]
    if not p.pending_promotion:
        return
    # Defensive: if the bench is empty here, the engine should have already
    # ended the game in _handle_ko. Clear the flag and bail to avoid a crash.
    if not p.bench:
        p.pending_promotion = False
        return

    # Bot picks highest HP bench Pokémon
    best = max(range(len(p.bench)), key=lambda i: p.bench[i].current_hp)
    chosen_name = p.bench[best].name
    action = ActionMapper.encode(ActionType.PROMOTE, {"bench_slot": best})
    with _lock:
        _log(f"Bot promotes {chosen_name} to Active.")
    try:
        env.step(action)
    except Exception as e:
        with _lock:
            _log(f"Bot promotion failed: {e}")
    time.sleep(0.3)


def _action_to_log(action_idx: int, gs: GameState, player_idx: int, prefix: str) -> str:
    """Human-readable log entry for an action."""
    me = gs.players[player_idx]
    try:
        atype, params = ActionMapper.decode(action_idx)
    except Exception:
        return f"{prefix}: unknown action"

    # Guarded hand-index lookup: never accept negative or out-of-range values.
    def hand_name(hi: int) -> str:
        return me.hand[hi].name if 0 <= hi < len(me.hand) else "?"

    if atype == ActionType.END_TURN:
        return f"{prefix} ended their turn."
    if atype == ActionType.ATTACK:
        ai  = params.get("atk_idx", 0)
        atk = me.active.attacks[ai] if (me.active and 0 <= ai < len(me.active.attacks)) else None
        name = atk.name if atk else "?"
        dmg  = atk.damage if atk else 0
        return f"{prefix} used {name}!" + (f" ({dmg} dmg)" if dmg else "")
    if atype == ActionType.ATTACH_ENERGY:
        slot = params.get("slot", 0)
        tgt  = "Active" if slot == 0 else f"bench[{slot-1}]"
        ename = next((c.name for c in me.hand if isinstance(c, EnergyCard)), "Energy")
        return f"{prefix} attached {ename} to {tgt}."
    if atype == ActionType.PLAY_POKEMON:
        return f"{prefix} played {hand_name(params.get('hand_idx', -1))} to bench."
    if atype == ActionType.EVOLVE:
        slot = params.get("slot", 0)
        tgt  = "Active" if slot == 5 else f"bench[{slot}]"
        return f"{prefix} evolved {tgt} into {hand_name(params.get('hand_idx', -1))}."
    if atype == ActionType.USE_ITEM:
        return f"{prefix} used item: {hand_name(params.get('hand_idx', -1))}."
    if atype == ActionType.USE_SUPPORTER:
        return f"{prefix} used supporter: {hand_name(params.get('hand_idx', -1))}."
    if atype == ActionType.PROMOTE:
        bs   = params.get("bench_slot", 0)
        if 0 <= bs < len(me.bench):
            return f"{prefix} promoted {me.bench[bs].name} to Active."
        return f"{prefix} promoted a Pokémon to Active."
    return f"{prefix}: {atype.name}"


# ══════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/images/<path:fname>")
def serve_image(fname):
    return send_from_directory(IMAGES_DIR, fname)


@app.route("/api/new_game", methods=["POST"])
def new_game():
    try:
        data = request.get_json(force=True, silent=False) or {}
    except Exception:
        return jsonify({"error": "Malformed JSON"}), 400
    if not isinstance(data, dict):
        return jsonify({"error": "Body must be a JSON object"}), 400

    deck_choice = str(data.get("deck", "lycanroc")).lower()
    if deck_choice not in ("lycanroc", "raichu"):
        return jsonify({"error": "Invalid 'deck' value (use 'lycanroc' or 'raichu')"}), 400

    human_player = 0 if deck_choice == "lycanroc" else 1
    bot_player   = 1 - human_player

    # Load champion agent for bot
    if bot_player == 0:
        bot = PPOAgent(player_idx=0, seed=0)
        champ_path = PPO_CHAMP
    else:
        bot = DQNAgent(player_idx=1, seed=1)
        champ_path = DQN_CHAMP

    if os.path.exists(champ_path):
        try:
            bot.load(champ_path)
        except Exception as e:
            print(f"Failed to load champion {champ_path}: {e}")

    seed = int(time.time()) % 100_000
    env  = PokemonTCGEnv(seed=seed)
    env.reset(seed=seed)

    with _lock:
        _game["env"]          = env
        _game["human_player"] = human_player
        _game["bot"]          = bot
        _game["bot_player"]   = bot_player
        _game["log"]          = [f"New game started! You are playing {deck_choice.title()}."]
        _game["pending"]      = False
        _game["mode"]         = "playing"

        gs = env.gs
        # If bot goes first, kick off its turn (under the lock)
        if gs.current_player == bot_player:
            _game["pending"] = True
            _log("Bot goes first...")
            t = threading.Thread(target=_run_bot_turn, daemon=True)
            t.start()

    return jsonify(build_state_json())


@app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify(build_state_json())


@app.route("/api/action", methods=["POST"])
def do_action():
    # ── Parse + validate request body ──────────────────────────────
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Malformed JSON"}), 400
    if not isinstance(data, dict):
        return jsonify({"error": "Body must be a JSON object"}), 400
    try:
        action_idx = int(data.get("action", -1))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid 'action' value"}), 400
    if not (0 <= action_idx < ActionMapper.TOTAL_ACTIONS):
        return jsonify({"error": "Action index out of range"}), 400

    # Optional: which energy card the user clicked (for ATTACH_ENERGY).
    raw_attach = data.get("attach_hand_idx", None)
    attach_hand_idx: int | None = None
    if raw_attach is not None:
        try:
            attach_hand_idx = int(raw_attach)
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid 'attach_hand_idx' value"}), 400

    g   = _game
    env = g["env"]
    if env is None:
        return jsonify({"error": "No active game"}), 400

    gs = env.gs
    h  = g["human_player"]

    # ── Locked critical section: validate legality & step atomically ──
    # This prevents races between the read of "pending"/"current_player"
    # and the env.step() call, and between competing requests.
    with _lock:
        if gs.game_over:
            return jsonify({"error": "Game is over"}), 400
        if g["pending"]:
            return jsonify({"error": "Bot is thinking"}), 400
        if (gs.current_player != h
                and not any(gs.players[p].pending_promotion for p in range(2))):
            return jsonify({"error": "Not your turn"}), 400

        legal = _build_legal_set(env, h)
        if action_idx not in legal:
            return jsonify({"error": f"Illegal action {action_idx}"}), 400

        atype, params = ActionMapper.decode(action_idx)

        # Energy-card preference (only honoured if hand_idx is in range
        # and points at an energy card; engine clears it after use).
        if atype == ActionType.ATTACH_ENERGY and attach_hand_idx is not None:
            me_h = gs.players[h]
            if 0 <= attach_hand_idx < len(me_h.hand):
                me_h.preferred_attach_hand_idx = attach_hand_idx

        # Step the engine. On non-attack: log first; on attack: log after.
        try:
            if atype != ActionType.ATTACK:
                _log(_action_to_log(action_idx, gs, h, prefix="You"))
                env.step(action_idx)
            else:
                atk_name = ""
                if me_h := gs.players[h].active:
                    ai = params.get("atk_idx", 0)
                    if 0 <= ai < len(me_h.attacks):
                        atk_name = me_h.attacks[ai].name
                env.step(action_idx)
                dmg = gs.last_attack_damage
                _log(f"You used {atk_name}!" + (f" ({dmg} dmg)" if dmg else ""))
                if gs.last_damage_prevented:
                    _log("⚠️ Bot's protection prevented all damage!")
                elif gs.last_damage_reduced:
                    _log("⚠️ Bot's effect reduced your damage.")
        except Exception as e:
            _log(f"Action failed: {e}")
            return jsonify({"error": "Engine error", "detail": str(e)}), 500

        if gs.game_over:
            g["mode"] = "over"
            winner_label = "You win! 🎉" if gs.winner == h else "Bot wins!"
            _log(f"--- GAME OVER: {winner_label} (by {gs.win_reason}) ---")
            return jsonify(build_state_json())

        # If the bot now has pending_promotion (because human KO'd it),
        # resolve it inside the same lock so the bot's promotion is atomic.
        bot_player = g["bot_player"]
        if gs.players[bot_player].pending_promotion:
            # _handle_bot_promotion takes the lock itself — release here briefly
            pass   # handled outside the with block below

    # Run promotion + spawn bot turn outside the main lock to avoid nesting
    bot_player = g["bot_player"]
    if gs.players[bot_player].pending_promotion:
        _handle_bot_promotion()

    if (not gs.game_over
            and gs.current_player == bot_player
            and not any(gs.players[p].pending_promotion for p in range(2))):
        with _lock:
            if not g["pending"]:           # guard against double-spawn
                g["pending"] = True
                t = threading.Thread(target=_run_bot_turn, daemon=True)
                t.start()

    return jsonify(build_state_json())


@app.route("/api/skip_turn", methods=["POST"])
def skip_turn():
    """End the human's turn immediately."""
    g   = _game
    env = g["env"]
    if env is None:
        return jsonify({"error": "No active game"}), 400

    gs = env.gs
    h  = g["human_player"]

    with _lock:
        if gs.game_over or gs.current_player != h or g["pending"]:
            return jsonify({"error": "Cannot skip right now"}), 400
        _log("You ended your turn.")
        try:
            env.step(ActionMapper.END_TURN_IDX)
        except Exception as e:
            _log(f"End-turn failed: {e}")
            return jsonify({"error": "Engine error", "detail": str(e)}), 500

        if gs.game_over:
            g["mode"] = "over"
            return jsonify(build_state_json())

    bot_player = g["bot_player"]
    if (gs.current_player == bot_player
            and not any(gs.players[p].pending_promotion for p in range(2))):
        with _lock:
            if not g["pending"]:
                g["pending"] = True
                t = threading.Thread(target=_run_bot_turn, daemon=True)
                t.start()

    return jsonify(build_state_json())


if __name__ == "__main__":
    print("Starting Pokemon TCG GUI server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=False, threaded=True, port=5000)
