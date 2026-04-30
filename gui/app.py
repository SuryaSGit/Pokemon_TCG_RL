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

# ── Global game state (single session) ────────────────────────────────────
_game: dict = {
    "env":          None,   # PokemonTCGEnv
    "human_player": None,   # 0 or 1
    "bot":          None,   # PPOAgent | DQNAgent
    "bot_player":   None,   # 0 or 1
    "log":          [],     # list of message strings
    "pending":      False,  # True while bot is "thinking"
    "selected":     None,   # hand card index selected by human (for two-step actions)
    "mode":         "idle", # "idle" | "playing" | "over"
}
_lock = threading.Lock()


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
    return {
        "name":     p.name,
        "hp":       p.hp,
        "current_hp": p.current_hp,
        "hp_pct":   round(p.current_hp / p.hp * 100),
        "stage":    STAGE_NAMES[p.stage],
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
    """Serialise the full visible game state for the frontend."""
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

    # Work out which hand cards are legal for each action type
    legal_hand = {}   # hand_idx -> list of action descriptions
    for action_idx in legal:
        atype, params = ActionMapper.decode(action_idx)
        hi = params.get("hand_idx", -1)
        if hi >= 0:
            if hi not in legal_hand:
                legal_hand[hi] = []
            legal_hand[hi].append({"action": action_idx, "type": atype.name, "params": params})

    # Legal attack indices
    legal_attacks = {}
    for action_idx in legal:
        atype, params = ActionMapper.decode(action_idx)
        if atype == ActionType.ATTACK:
            legal_attacks[params["atk_idx"]] = action_idx

    # Legal attach slots (0=active, 1-5=bench)
    legal_attach = {}
    for action_idx in legal:
        atype, params = ActionMapper.decode(action_idx)
        if atype == ActionType.ATTACH_ENERGY:
            legal_attach[params["slot"]] = action_idx

    # Legal promote slots
    legal_promote = {}
    for action_idx in legal:
        atype, params = ActionMapper.decode(action_idx)
        if atype == ActionType.PROMOTE:
            legal_promote[params["bench_slot"]] = action_idx

    # Discard top card (last element)
    discard_top = _serialize_card(me.discard[-1], -1) if me.discard else None
    opp_discard_top = _serialize_card(opp.discard[-1], -1) if opp.discard else None

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
        "selected":      g["selected"],

        "my": {
            "ko_count":    me.ko_count,
            "deck_size":   len(me.deck),
            "hand":        [_serialize_card(c, i) for i, c in enumerate(me.hand)],
            "active":      _serialize_pokemon(me.active),
            "bench":       [_serialize_pokemon(p, i) for i, p in enumerate(me.bench)],
            "discard_size": len(me.discard),
            "discard_top": discard_top,
        },
        "opp": {
            "ko_count":    opp.ko_count,
            "deck_size":   len(opp.deck),
            "hand_size":   len(opp.hand),
            "active":      _serialize_pokemon(opp.active),
            "bench":       [_serialize_pokemon(p, i) for i, p in enumerate(opp.bench)],
            "discard_size": len(opp.discard),
            "discard_top": opp_discard_top,
        },

        "legal_hand":    legal_hand,
        "legal_attacks": legal_attacks,
        "legal_attach":  legal_attach,
        "legal_promote": legal_promote,
        "can_end_turn":  (ActionMapper.END_TURN_IDX in legal),
        "promotion":     _promotion_pending(gs),
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
    """Run the bot's full turn in a background thread with delays."""
    time.sleep(0.5)  # initial pause so human can see the board

    g   = _game
    env = g["env"]
    bot = g["bot"]
    bot_player = g["bot_player"]
    gs  = env.gs

    while (not gs.game_over
           and gs.current_player == bot_player
           and not any(gs.players[p].pending_promotion for p in range(2))):

        action = _bot_act(env, bot, bot_player)
        atype, params = ActionMapper.decode(action)

        # Build log message
        msg = _action_to_log(action, gs, bot_player, prefix="Bot")
        with _lock:
            g["log"].append(msg)

        env.step(action)
        time.sleep(0.45)  # pause between bot actions so they're readable

        # After an attack or end_turn, the turn switches — stop
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
            g["log"].append(f"--- GAME OVER: {winner_label} (by {gs.win_reason}) ---")


def _handle_bot_promotion():
    """If bot has pending_promotion, resolve it automatically."""
    g   = _game
    env = g["env"]
    gs  = env.gs
    bot_player = g["bot_player"]

    p = gs.players[bot_player]
    if not p.pending_promotion:
        return

    # Bot picks highest HP bench pokemon
    best = max(range(len(p.bench)), key=lambda i: p.bench[i].current_hp)
    chosen_name = p.bench[best].name
    action = ActionMapper.encode(ActionType.PROMOTE, {"bench_slot": best})
    with _lock:
        g["log"].append(f"Bot promotes {chosen_name} to Active.")
    env.step(action)
    time.sleep(0.3)


def _action_to_log(action_idx: int, gs: GameState, player_idx: int, prefix: str) -> str:
    """Human-readable log entry for an action."""
    me = gs.players[player_idx]
    try:
        atype, params = ActionMapper.decode(action_idx)
    except Exception:
        return f"{prefix}: unknown action"

    if atype == ActionType.END_TURN:
        return f"{prefix} ended their turn."
    if atype == ActionType.ATTACK:
        ai  = params.get("atk_idx", 0)
        atk = me.active.attacks[ai] if me.active and ai < len(me.active.attacks) else None
        name = atk.name if atk else "?"
        dmg  = atk.damage if atk else 0
        return f"{prefix} used {name}!" + (f" ({dmg} dmg)" if dmg else "")
    if atype == ActionType.ATTACH_ENERGY:
        slot = params.get("slot", 0)
        tgt  = "Active" if slot == 0 else f"bench[{slot-1}]"
        # Find energy in hand
        ename = next((c.name for c in me.hand if isinstance(c, EnergyCard)), "Energy")
        return f"{prefix} attached {ename} to {tgt}."
    if atype == ActionType.PLAY_POKEMON:
        hi = params.get("hand_idx", 0)
        name = me.hand[hi].name if hi < len(me.hand) else "?"
        return f"{prefix} played {name} to bench."
    if atype == ActionType.EVOLVE:
        hi   = params.get("hand_idx", 0)
        name = me.hand[hi].name if hi < len(me.hand) else "?"
        slot = params.get("slot", 0)
        tgt  = "Active" if slot == 5 else f"bench[{slot}]"
        return f"{prefix} evolved {tgt} into {name}."
    if atype == ActionType.USE_ITEM:
        hi   = params.get("hand_idx", 0)
        name = me.hand[hi].name if hi < len(me.hand) else "?"
        return f"{prefix} used item: {name}."
    if atype == ActionType.USE_SUPPORTER:
        hi   = params.get("hand_idx", 0)
        name = me.hand[hi].name if hi < len(me.hand) else "?"
        return f"{prefix} used supporter: {name}."
    if atype == ActionType.PROMOTE:
        bs   = params.get("bench_slot", 0)
        if bs < len(me.bench):
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
    data        = request.get_json(force=True)
    deck_choice = data.get("deck", "lycanroc").lower()  # "lycanroc" | "raichu"
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
        bot.load(champ_path)

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
        _game["selected"]     = None
        _game["mode"]         = "playing"

    gs = env.gs
    # If bot goes first, kick off its turn
    if gs.current_player == bot_player:
        _game["pending"] = True
        _game["log"].append("Bot goes first...")
        t = threading.Thread(target=_run_bot_turn, daemon=True)
        t.start()

    return jsonify(build_state_json())


@app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify(build_state_json())


@app.route("/api/action", methods=["POST"])
def do_action():
    data       = request.get_json(force=True)
    action_idx = int(data.get("action", -1))

    g   = _game
    env = g["env"]
    if env is None:
        return jsonify({"error": "No active game"}), 400

    gs = env.gs
    h  = g["human_player"]

    if gs.game_over:
        return jsonify({"error": "Game is over"}), 400
    if gs.current_player != h and not any(gs.players[p].pending_promotion for p in range(2)):
        return jsonify({"error": "Not your turn"}), 400
    if g["pending"]:
        return jsonify({"error": "Bot is thinking"}), 400

    # Validate action is legal
    legal = _build_legal_set(env, h)
    if action_idx not in legal:
        return jsonify({"error": f"Illegal action {action_idx}"}), 400

    atype, params = ActionMapper.decode(action_idx)

    # Log the human's action
    msg = _action_to_log(action_idx, gs, h, prefix="You")
    with _lock:
        g["log"].append(msg)
        g["selected"] = None

    env.step(action_idx)

    if gs.game_over:
        with _lock:
            g["mode"]  = "over"
            winner_label = "You win! 🎉" if gs.winner == h else "Bot wins!"
            g["log"].append(f"--- GAME OVER: {winner_label} (by {gs.win_reason}) ---")
        return jsonify(build_state_json())

    # Check if bot needs to make a promotion after human's attack
    bot_player = g["bot_player"]
    if gs.players[bot_player].pending_promotion:
        _handle_bot_promotion()

    # If turn switched to bot (after attack/end_turn), run bot turn
    if (not gs.game_over
            and gs.current_player == bot_player
            and not any(gs.players[p].pending_promotion for p in range(2))):
        with _lock:
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

    if gs.game_over or gs.current_player != h or g["pending"]:
        return jsonify({"error": "Cannot skip right now"}), 400

    with _lock:
        g["log"].append("You ended your turn.")
        g["selected"] = None

    env.step(ActionMapper.END_TURN_IDX)

    if gs.game_over:
        with _lock:
            g["mode"] = "over"
        return jsonify(build_state_json())

    bot_player = g["bot_player"]
    if gs.current_player == bot_player:
        with _lock:
            g["pending"] = True
        t = threading.Thread(target=_run_bot_turn, daemon=True)
        t.start()

    return jsonify(build_state_json())


@app.route("/api/select", methods=["POST"])
def select_card():
    """Track which hand card the human has clicked (for two-step actions)."""
    data = request.get_json(force=True)
    idx  = data.get("index", None)
    with _lock:
        _game["selected"] = idx
    return jsonify({"selected": idx})


if __name__ == "__main__":
    print("Starting Pokemon TCG GUI server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=False, threaded=True, port=5000)
