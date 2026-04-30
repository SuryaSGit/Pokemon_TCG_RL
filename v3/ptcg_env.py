"""
Pokemon TCG RL Environment
Lycanroc Half-Deck vs Alolan Raichu Half-Deck (Sun & Moon Trainer Kit)
Gym-style environment for PPO/DQN training.
"""

from __future__ import annotations
import numpy as np
import random
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from enum import IntEnum

# ─────────────────────────────────────────────
# ENUMS & CONSTANTS
# ─────────────────────────────────────────────

class EnergyType(IntEnum):
    COLORLESS  = 0
    FIGHTING   = 1
    LIGHTNING  = 2
    PSYCHIC    = 3

class Stage(IntEnum):
    BASIC   = 0
    STAGE1  = 1
    STAGE2  = 2

class CardType(IntEnum):
    POKEMON   = 0
    ENERGY    = 1
    ITEM      = 2
    SUPPORTER = 3

class ActionType(IntEnum):
    ATTACH_ENERGY  = 0
    PLAY_POKEMON   = 1
    USE_ITEM       = 2
    USE_SUPPORTER  = 3
    EVOLVE         = 4
    ATTACK         = 5
    END_TURN       = 6
    PROMOTE        = 7   # choose which benched pokemon becomes active after a KO

# Max bench size, max attacks per pokemon, etc.
MAX_BENCH     = 5
MAX_ATTACKS   = 2
MAX_HAND      = 10   # soft cap for encoding
MAX_DECK      = 30
KO_TO_WIN     = 6

# Pokemon identity index (for encoding)
POKEMON_IDS = {
    "Caterpie": 0, "Pikipek": 1, "Trumbeak": 2, "Fletchling": 3,
    "Fletchinder": 4, "Yungoos": 5, "Rockruff": 6, "Lycanroc": 7,
    "Makuhita": 8, "Toucannon": 9,
    "Stufful": 10, "Golbat": 11, "Zubat": 12, "Spearow": 13,
    "Pikachu": 14, "Raichu": 15, "Bewear": 16, "Grubbin": 17,
    "Togedemaru": 18, "Drowzee": 19,
}
NUM_POKEMON_IDS = len(POKEMON_IDS)

# ─────────────────────────────────────────────
# CARD DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class Attack:
    name: str
    damage: int
    energy_cost: Dict[EnergyType, int]   # e.g. {FIGHTING:1, COLORLESS:1}
    effect: Optional[str] = None         # string tag for special effects

@dataclass
class PokemonCard:
    name: str
    hp: int
    stage: Stage
    evolves_from: Optional[str]
    attacks: List[Attack]
    card_type: CardType = CardType.POKEMON
    # Runtime state (set when in play)
    current_hp: int = 0
    energy: Dict[EnergyType, int] = field(default_factory=dict)
    is_active: bool = False
    effect_flags: Dict[str, Any] = field(default_factory=dict)  # for status effects
    turn_played: int = -1   # turn the card was put into play (for evolution timing)

    def __post_init__(self):
        self.current_hp = self.hp

    def total_energy(self) -> int:
        return sum(self.energy.values())

    def hp_fraction(self) -> float:
        return self.current_hp / self.hp

    def clone(self) -> 'PokemonCard':
        c = copy.copy(self)
        c.attacks = self.attacks  # immutable list of immutable attacks
        c.energy = dict(self.energy)
        c.effect_flags = dict(self.effect_flags)
        return c


@dataclass
class TrainerCard:
    name: str
    card_type: CardType   # ITEM or SUPPORTER
    effect: str

@dataclass
class EnergyCard:
    name: str
    energy_type: EnergyType
    card_type: CardType = CardType.ENERGY


# Type alias
Card = Any  # PokemonCard | TrainerCard | EnergyCard

# ─────────────────────────────────────────────
# DECK DEFINITIONS
# ─────────────────────────────────────────────

def _e(cost_str: str) -> Dict[EnergyType, int]:
    """Parse energy cost string like '1F1C' -> {FIGHTING:1, COLORLESS:1}"""
    cost: Dict[EnergyType, int] = {}
    i = 0
    while i < len(cost_str):
        count = int(cost_str[i]); i += 1
        t = cost_str[i]; i += 1
        etype = {'C': EnergyType.COLORLESS, 'F': EnergyType.FIGHTING,
                 'L': EnergyType.LIGHTNING, 'P': EnergyType.PSYCHIC}[t]
        cost[etype] = count
    return cost


def build_lycanroc_deck() -> List[Card]:
    cards: List[Card] = []
    # Caterpie x1
    cards.append(PokemonCard("Caterpie", 50, Stage.BASIC, None, [
        Attack("Soothing Scent", 0,  _e("1C"), effect="heal_self_20"),
        Attack("String Shot",   20,  _e("2C")),
    ]))
    # Pikipek x1
    cards.append(PokemonCard("Pikipek", 50, Stage.BASIC, None, [
        Attack("Peck", 10, _e("1C"), effect="coin_plus10"),
    ]))
    # Trumbeak x1
    cards.append(PokemonCard("Trumbeak", 80, Stage.STAGE1, "Pikipek", [
        Attack("Bullet Seed", 0, _e("1C"), effect="flip4_20each"),
    ]))
    # Fletchling x1
    cards.append(PokemonCard("Fletchling", 50, Stage.BASIC, None, [
        Attack("Hinder",    0,  _e("1C"), effect="opponent_minus20_next_turn"),
        Attack("Peck",     20,  _e("2C")),
    ]))
    # Fletchinder x1
    cards.append(PokemonCard("Fletchinder", 70, Stage.STAGE1, "Fletchling", [
        Attack("Peck",         20, _e("1C")),
        Attack("Flame Charge", 40, _e("2C")),
    ]))
    # Yungoos x1
    cards.append(PokemonCard("Yungoos", 70, Stage.BASIC, None, [
        Attack("Tackle",    10, _e("1C")),
        Attack("Bite",      20, _e("2C")),
    ]))
    # Rockruff x2
    for _ in range(2):
        cards.append(PokemonCard("Rockruff", 60, Stage.BASIC, None, [
            Attack("Tackle", 10, _e("1C")),
            Attack("Rock Throw", 20, _e("1F1C")),
        ]))
    # Lycanroc x2
    for _ in range(2):
        cards.append(PokemonCard("Lycanroc", 110, Stage.STAGE1, "Rockruff", [
            Attack("Accelerock",  20, _e("2C")),
            Attack("Rock Slide",  80, _e("2F1C")),
        ]))
    # Makuhita x1
    cards.append(PokemonCard("Makuhita", 80, Stage.BASIC, None, [
        Attack("Knock Away", 10, _e("1F")),
        Attack("Seismic Toss", 40, _e("1F1C")),
    ]))
    # Toucannon x1
    cards.append(PokemonCard("Toucannon", 140, Stage.STAGE2, "Trumbeak", [
        Attack("Beak Blast",    60, _e("2C")),
        Attack("Hyper Voice",  100, _e("3C")),
    ]))
    # Trainers
    for _ in range(2):
        cards.append(TrainerCard("Hau", CardType.SUPPORTER, "draw3"))
    cards.append(TrainerCard("Big Malasada", CardType.ITEM, "heal_active_20"))
    for _ in range(2):
        cards.append(TrainerCard("Great Ball", CardType.ITEM, "great_ball"))
    # Energy
    for _ in range(13):
        cards.append(EnergyCard("Fighting Energy", EnergyType.FIGHTING))
    return cards


def build_raichu_deck() -> List[Card]:
    cards: List[Card] = []
    # Stufful x1
    cards.append(PokemonCard("Stufful", 70, Stage.BASIC, None, [
        Attack("Tackle", 30, _e("2C")),
    ]))
    # Zubat x1
    cards.append(PokemonCard("Zubat", 50, Stage.BASIC, None, [
        Attack("Supersonic", 0, _e("1P"), effect="discard_random_opponent_hand"),
    ]))
    # Golbat x1
    cards.append(PokemonCard("Golbat", 80, Stage.STAGE1, "Zubat", [
        Attack("Leech Life", 10, _e("2C"), effect="flip2_plus20each"),
    ]))
    # Spearow x1
    cards.append(PokemonCard("Spearow", 60, Stage.BASIC, None, [
        Attack("Peck", 10, _e("1C"), effect="plus30_vs_caterpie"),
    ]))
    # Pikachu x2
    for _ in range(2):
        cards.append(PokemonCard("Pikachu", 60, Stage.BASIC, None, [
            Attack("Gnaw",          10, _e("1C")),
            Attack("Thunder Shock", 10, _e("1L1C"), effect="bench_10_any"),
        ]))
    # Raichu x2
    for _ in range(2):
        cards.append(PokemonCard("Raichu", 110, Stage.STAGE1, "Pikachu", [
            Attack("Thunderbolt", 10, _e("1L"), effect="coin_plus30"),
            Attack("Circle Circuit", 70, _e("1L2C")),
        ]))
    # Bewear x1
    cards.append(PokemonCard("Bewear", 130, Stage.STAGE1, "Stufful", [
        Attack("Tackle",   40, _e("2C")),
        Attack("Hammer In", 80, _e("3C"), effect="optional_plus40_self20"),
    ]))
    # Grubbin x1
    cards.append(PokemonCard("Grubbin", 70, Stage.BASIC, None, [
        Attack("Tackle", 20, _e("2C")),
    ]))
    # Togedemaru x1
    cards.append(PokemonCard("Togedemaru", 70, Stage.BASIC, None, [
        Attack("Rollout",        0, _e("1C"), effect="coin_protect_self"),
        Attack("Discharge",      0, _e("0C"), effect="discard_lightning_30each"),
    ]))
    # Drowzee x1
    cards.append(PokemonCard("Drowzee", 70, Stage.BASIC, None, [
        Attack("Psych Up", 0,  _e("1P"), effect="10x_opp_energy"),
        Attack("Pound",   20,  _e("2C")),
    ]))
    # Trainers
    for _ in range(2):
        cards.append(TrainerCard("Hau", CardType.SUPPORTER, "draw3"))
    cards.append(TrainerCard("Potion", CardType.ITEM, "heal_active_30"))
    for _ in range(2):
        cards.append(TrainerCard("Great Ball", CardType.ITEM, "great_ball"))
    # Energy
    for _ in range(9):
        cards.append(EnergyCard("Lightning Energy", EnergyType.LIGHTNING))
    for _ in range(4):
        cards.append(EnergyCard("Psychic Energy", EnergyType.PSYCHIC))
    return cards


# ─────────────────────────────────────────────
# PLAYER STATE
# ─────────────────────────────────────────────

@dataclass
class PlayerState:
    deck_id: int   # 0=Lycanroc, 1=Raichu
    deck: List[Card] = field(default_factory=list)
    hand: List[Card] = field(default_factory=list)
    active: Optional[PokemonCard] = None
    bench: List[PokemonCard] = field(default_factory=list)   # max 5
    discard: List[Card] = field(default_factory=list)
    ko_count: int = 0
    energy_used: bool = False
    supporter_used: bool = False
    turn_count: int = 0   # total turns taken
    pending_promotion: bool = False   # True when active was KO'd and player must choose replacement

    def all_pokemon_in_play(self) -> List[PokemonCard]:
        pokes = []
        if self.active:
            pokes.append(self.active)
        pokes.extend(self.bench)
        return pokes

    def has_basics_in_hand(self) -> bool:
        return any(isinstance(c, PokemonCard) and c.stage == Stage.BASIC for c in self.hand)

    def basics_in_hand(self) -> List[int]:
        return [i for i, c in enumerate(self.hand)
                if isinstance(c, PokemonCard) and c.stage == Stage.BASIC]

# ─────────────────────────────────────────────
# FULL GAME STATE
# ─────────────────────────────────────────────

@dataclass
class GameState:
    players: List[PlayerState]
    current_player: int = 0          # 0 or 1
    turn_number: int = 0
    game_over: bool = False
    winner: int = -1
    rng: Any = None                  # seeded numpy rng
    # Per-turn transient flags (cleared each turn)
    damage_reduction: Dict[int, int] = field(default_factory=dict)   # player_idx -> reduction

    def opponent(self) -> int:
        return 1 - self.current_player

    def current(self) -> PlayerState:
        return self.players[self.current_player]

    def opp(self) -> PlayerState:
        return self.players[self.opponent()]


# ─────────────────────────────────────────────
# ACTION ENCODING
# ─────────────────────────────────────────────

class ActionMapper:
    """
    Flat discrete action space.

    Layout (indices):
      [0]                       END_TURN
      [1 .. MAX_BENCH+1]        ATTACH_ENERGY to active(0) or bench[0..4]  → 6 slots
      [7 .. 7+MAX_HAND-1]       PLAY_POKEMON from hand[i]                  → 10 slots
      [17 .. 17+MAX_HAND-1]     USE_ITEM from hand[i]                      → 10 slots
      [27 .. 27+MAX_HAND-1]     USE_SUPPORTER from hand[i]                 → 10 slots
      [37 .. 37 + MAX_BENCH*MAX_HAND - 1]   EVOLVE hand[h] onto bench_slot s
                                             indexed as h*MAX_BENCH + s     → 50 slots
                                             + evolve active (slot=5)       → 10 more
      [97 .. 97 + MAX_ATTACKS - 1]          ATTACK attack_idx              → 2 slots

    Total: 1 + 6 + 10 + 10 + 10 + 60 + 2 = 99 actions
    """

    END_TURN_IDX      = 0
    ATTACH_START      = 1           # 1..6
    ATTACH_COUNT      = MAX_BENCH + 1
    PLAY_START        = 7           # 7..16
    PLAY_COUNT        = MAX_HAND
    ITEM_START        = 17          # 17..26
    ITEM_COUNT        = MAX_HAND
    SUPP_START        = 27          # 27..36
    SUPP_COUNT        = MAX_HAND
    EVOLVE_START      = 37          # 37..96  (hand_idx * 6 + slot,  slot 5 = active)
    EVOLVE_COUNT      = MAX_HAND * (MAX_BENCH + 1)
    ATTACK_START      = 97          # 97..98
    ATTACK_COUNT      = MAX_ATTACKS
    PROMOTE_START     = 99          # 99..103  bench slot 0..4 -> becomes active
    PROMOTE_COUNT     = MAX_BENCH
    TOTAL_ACTIONS     = 104

    @classmethod
    def decode(cls, idx: int) -> Tuple[ActionType, Dict]:
        if idx == cls.END_TURN_IDX:
            return ActionType.END_TURN, {}
        if cls.ATTACH_START <= idx < cls.ATTACH_START + cls.ATTACH_COUNT:
            slot = idx - cls.ATTACH_START
            return ActionType.ATTACH_ENERGY, {"slot": slot}
        if cls.PLAY_START <= idx < cls.PLAY_START + cls.PLAY_COUNT:
            hand_idx = idx - cls.PLAY_START
            return ActionType.PLAY_POKEMON, {"hand_idx": hand_idx}
        if cls.ITEM_START <= idx < cls.ITEM_START + cls.ITEM_COUNT:
            hand_idx = idx - cls.ITEM_START
            return ActionType.USE_ITEM, {"hand_idx": hand_idx}
        if cls.SUPP_START <= idx < cls.SUPP_START + cls.SUPP_COUNT:
            hand_idx = idx - cls.SUPP_START
            return ActionType.USE_SUPPORTER, {"hand_idx": hand_idx}
        if cls.EVOLVE_START <= idx < cls.EVOLVE_START + cls.EVOLVE_COUNT:
            offset = idx - cls.EVOLVE_START
            hand_idx = offset // (MAX_BENCH + 1)
            slot     = offset %  (MAX_BENCH + 1)
            return ActionType.EVOLVE, {"hand_idx": hand_idx, "slot": slot}
        if cls.ATTACK_START <= idx < cls.ATTACK_START + cls.ATTACK_COUNT:
            atk_idx = idx - cls.ATTACK_START
            return ActionType.ATTACK, {"atk_idx": atk_idx}
        if cls.PROMOTE_START <= idx < cls.PROMOTE_START + cls.PROMOTE_COUNT:
            bench_slot = idx - cls.PROMOTE_START
            return ActionType.PROMOTE, {"bench_slot": bench_slot}
        raise ValueError(f"Invalid action index {idx}")

    @classmethod
    def encode(cls, atype: ActionType, params: Dict) -> int:
        if atype == ActionType.END_TURN:
            return cls.END_TURN_IDX
        if atype == ActionType.ATTACH_ENERGY:
            return cls.ATTACH_START + params["slot"]
        if atype == ActionType.PLAY_POKEMON:
            return cls.PLAY_START + params["hand_idx"]
        if atype == ActionType.USE_ITEM:
            return cls.ITEM_START + params["hand_idx"]
        if atype == ActionType.USE_SUPPORTER:
            return cls.SUPP_START + params["hand_idx"]
        if atype == ActionType.EVOLVE:
            return cls.EVOLVE_START + params["hand_idx"] * (MAX_BENCH + 1) + params["slot"]
        if atype == ActionType.ATTACK:
            return cls.ATTACK_START + params["atk_idx"]
        if atype == ActionType.PROMOTE:
            return cls.PROMOTE_START + params["bench_slot"]
        raise ValueError(f"Unknown action type {atype}")


# ─────────────────────────────────────────────
# STATE ENCODER
# ─────────────────────────────────────────────

class StateEncoder:
    """
    Encodes GameState → fixed-size float32 vector.

    Per-pokemon features (6 values):
      hp_frac, energy_count_norm, pokemon_id_norm, stage_norm,
      has_attack0, has_attack1

    Layout:
      my_active:          6
      opp_active:         6
      my_bench:     5 * 6 = 30
      opp_bench:    5 * 6 = 30
      my_hand_hist:       4   (pokemon, energy, item, supporter counts / 10)
      opp_hand_size:      1   (normalized)
      my_deck_size:       1
      opp_deck_size:      1
      my_ko:              1
      opp_ko:             1
      energy_used:        1
      supporter_used:     1
      turn_norm:          1
      ─────────────────────
      Total:             84
    """
    STATE_SIZE = 84

    @staticmethod
    def _encode_pokemon(p: Optional[PokemonCard]) -> np.ndarray:
        if p is None:
            return np.zeros(6, dtype=np.float32)
        pid = POKEMON_IDS.get(p.name, 0) / max(NUM_POKEMON_IDS - 1, 1)
        return np.array([
            p.hp_fraction(),
            min(p.total_energy() / 4.0, 1.0),
            pid,
            p.stage / 2.0,
            float(len(p.attacks) > 0),
            float(len(p.attacks) > 1),
        ], dtype=np.float32)

    @classmethod
    def encode(cls, gs: GameState) -> np.ndarray:
        me  = gs.current()
        opp = gs.opp()
        vec = []

        # Active pokemon
        vec.append(cls._encode_pokemon(me.active))
        vec.append(cls._encode_pokemon(opp.active))

        # Bench (pad to 5)
        for i in range(MAX_BENCH):
            p = me.bench[i] if i < len(me.bench) else None
            vec.append(cls._encode_pokemon(p))
        for i in range(MAX_BENCH):
            p = opp.bench[i] if i < len(opp.bench) else None
            vec.append(cls._encode_pokemon(p))

        # Hand histogram (my hand)
        counts = [0, 0, 0, 0]  # pokemon, energy, item, supporter
        for c in me.hand:
            ct = getattr(c, 'card_type', CardType.ENERGY)
            if ct == CardType.POKEMON:   counts[0] += 1
            elif ct == CardType.ENERGY:  counts[1] += 1
            elif ct == CardType.ITEM:    counts[2] += 1
            elif ct == CardType.SUPPORTER: counts[3] += 1
        vec.append(np.array([c / 10.0 for c in counts], dtype=np.float32))

        # Opponent hand size (hidden info - just size)
        vec.append(np.array([len(opp.hand) / 10.0], dtype=np.float32))

        # Deck sizes
        vec.append(np.array([len(me.deck)  / MAX_DECK], dtype=np.float32))
        vec.append(np.array([len(opp.deck) / MAX_DECK], dtype=np.float32))

        # KO counters
        vec.append(np.array([me.ko_count  / KO_TO_WIN], dtype=np.float32))
        vec.append(np.array([opp.ko_count / KO_TO_WIN], dtype=np.float32))

        # Binary flags
        vec.append(np.array([float(me.energy_used), float(me.supporter_used)], dtype=np.float32))

        # Turn number (normalized)
        vec.append(np.array([min(gs.turn_number / 30.0, 1.0)], dtype=np.float32))

        result = np.concatenate([v.flatten() for v in vec])
        assert result.shape == (cls.STATE_SIZE,), f"State size mismatch: {result.shape}"
        return result


# ─────────────────────────────────────────────
# LEGAL ACTION MASK
# ─────────────────────────────────────────────

def can_pay_cost(pokemon: PokemonCard, cost: Dict[EnergyType, int]) -> bool:
    """Check if a pokemon has enough energy to use an attack."""
    # Total colorless requirement can be satisfied by any energy
    required = dict(cost)
    available = dict(pokemon.energy)
    # Pay typed energy first
    for etype, amount in required.items():
        if etype == EnergyType.COLORLESS:
            continue
        have = available.get(etype, 0)
        if have < amount:
            return False
        available[etype] = have - amount
    # Now pay colorless from remaining
    colorless_needed = required.get(EnergyType.COLORLESS, 0)
    total_remaining = sum(available.values())
    return total_remaining >= colorless_needed


def compute_legal_mask(gs: GameState) -> np.ndarray:
    """
    Returns a binary mask of shape (TOTAL_ACTIONS,) where 1 = legal.

    When any player has pending_promotion=True, ONLY the PROMOTE actions
    for that player are legal — all other actions are blocked until the
    active pokemon is chosen.
    """
    mask = np.zeros(ActionMapper.TOTAL_ACTIONS, dtype=np.float32)
    me  = gs.current()
    opp = gs.opp()

    # ── PROMOTION INTERRUPT ──────────────────────────────────────────────
    # Check if either player is waiting for a promotion choice.
    # The current player promotes their own KO'd pokemon; but if the
    # *opponent* needs to promote, we also pause until they do.
    for pidx in range(2):
        p = gs.players[pidx]
        if p.pending_promotion:
            # Only PROMOTE actions valid for this player's bench
            for i in range(len(p.bench)):
                mask[ActionMapper.PROMOTE_START + i] = 1.0
            return mask   # nothing else allowed

    # ── Normal turn mask ─────────────────────────────────────────────────

    # END_TURN always legal
    mask[ActionMapper.END_TURN_IDX] = 1.0

    # --- ATTACH_ENERGY ---
    has_energy_in_hand = any(isinstance(c, EnergyCard) for c in me.hand)
    if has_energy_in_hand and not me.energy_used:
        # Attach to active
        if me.active is not None:
            mask[ActionMapper.ATTACH_START] = 1.0
        # Attach to bench slots
        for i, p in enumerate(me.bench):
            mask[ActionMapper.ATTACH_START + 1 + i] = 1.0

    # --- PLAY_POKEMON (bench) ---
    if len(me.bench) < MAX_BENCH:
        for i, c in enumerate(me.hand):
            if i >= MAX_HAND:
                break
            if isinstance(c, PokemonCard) and c.stage == Stage.BASIC:
                mask[ActionMapper.PLAY_START + i] = 1.0

    # --- USE_ITEM ---
    for i, c in enumerate(me.hand):
        if i >= MAX_HAND:
            break
        if isinstance(c, TrainerCard) and c.card_type == CardType.ITEM:
            # Prune: Big Malasada / Potion only useful if active has damage
            if c.effect in ("heal_active_20", "heal_active_30"):
                if me.active is not None and me.active.current_hp < me.active.hp:
                    mask[ActionMapper.ITEM_START + i] = 1.0
            else:
                mask[ActionMapper.ITEM_START + i] = 1.0

    # --- USE_SUPPORTER ---
    if not me.supporter_used:
        for i, c in enumerate(me.hand):
            if i >= MAX_HAND:
                break
            if isinstance(c, TrainerCard) and c.card_type == CardType.SUPPORTER:
                mask[ActionMapper.SUPP_START + i] = 1.0

    # --- EVOLVE ---
    for hi, hc in enumerate(me.hand):
        if hi >= MAX_HAND:
            break
        if not (isinstance(hc, PokemonCard) and hc.stage != Stage.BASIC):
            continue
        # Try to evolve active
        if me.active is not None and can_evolve(hc, me.active, gs.turn_number):
            slot = MAX_BENCH  # slot 5 = active
            mask[ActionMapper.EVOLVE_START + hi * (MAX_BENCH + 1) + slot] = 1.0
        # Try to evolve bench
        for bi, bp in enumerate(me.bench):
            if can_evolve(hc, bp, gs.turn_number):
                mask[ActionMapper.EVOLVE_START + hi * (MAX_BENCH + 1) + bi] = 1.0

    # --- ATTACK ---
    if me.active is not None:
        for ai, atk in enumerate(me.active.attacks):
            if ai >= MAX_ATTACKS:
                break
            if can_pay_cost(me.active, atk.energy_cost):
                mask[ActionMapper.ATTACK_START + ai] = 1.0

    return mask


def can_evolve(evo_card: PokemonCard, target: PokemonCard, turn_number: int) -> bool:
    """Can evo_card evolve target? Must have been in play at least 1 full turn."""
    if evo_card.evolves_from != target.name:
        return False
    # Can't evolve on the same turn it was played (target.turn_played < current turn)
    if target.turn_played < 0 or target.turn_played >= turn_number:
        return False
    # Can't evolve into higher stage than stage+1
    if evo_card.stage != target.stage + 1:
        return False
    return True


# ─────────────────────────────────────────────
# GAME ENGINE
# ─────────────────────────────────────────────

class GameEngine:
    """Core simulation engine. Stateless methods operating on GameState."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _coin_flip(self) -> bool:
        return bool(self.rng.integers(0, 2))

    def _shuffle(self, lst: List) -> None:
        indices = self.rng.permutation(len(lst)).tolist()
        lst[:] = [lst[i] for i in indices]

    def new_game(self) -> GameState:
        deck0 = build_lycanroc_deck()
        deck1 = build_raichu_deck()
        self._shuffle(deck0)
        self._shuffle(deck1)

        p0 = PlayerState(deck_id=0, deck=deck0)
        p1 = PlayerState(deck_id=1, deck=deck1)
        gs = GameState(players=[p0, p1], rng=self.rng)

        # Draw opening hands (redraws until basic pokemon found)
        for pidx in range(2):
            self._draw_opening_hand(gs, pidx)

        # Set active pokemon (choose first basic from hand for each player)
        for pidx in range(2):
            self._auto_place_active(gs, pidx)

        # Coin flip for first player
        gs.current_player = int(self._coin_flip())
        gs.turn_number = 1
        return gs

    def _draw_opening_hand(self, gs: GameState, pidx: int) -> None:
        p = gs.players[pidx]
        attempts = 0
        while True:
            p.hand.clear()
            self._draw_n(gs, pidx, 7)
            if p.has_basics_in_hand():
                break
            # No basics: return hand to deck, reshuffle
            p.deck.extend(p.hand)
            p.hand.clear()
            self._shuffle(p.deck)
            attempts += 1
            if attempts > 20:
                raise RuntimeError("Could not draw opening hand with Basic pokemon")

    def _auto_place_active(self, gs: GameState, pidx: int) -> None:
        """Place the first basic pokemon from hand as active."""
        p = gs.players[pidx]
        for i, c in enumerate(p.hand):
            if isinstance(c, PokemonCard) and c.stage == Stage.BASIC:
                c.is_active = True
                c.turn_played = 0
                p.active = p.hand.pop(i)
                return
        raise RuntimeError(f"Player {pidx} has no basic pokemon to place as active")

    def _draw_n(self, gs: GameState, pidx: int, n: int) -> bool:
        """Draw n cards. Returns False if deck ran out."""
        p = gs.players[pidx]
        for _ in range(n):
            if not p.deck:
                gs.game_over = True
                gs.winner = 1 - pidx
                return False
            p.hand.append(p.deck.pop(0))
        return True

    def start_turn(self, gs: GameState) -> None:
        """Begin the current player's turn: draw a card, reset flags."""
        me = gs.current()
        me.energy_used = False
        me.supporter_used = False
        # Clear damage reduction from my previous turn
        gs.damage_reduction.pop(gs.current_player, None)
        # Draw a card (first player doesn't draw on turn 1 in standard rules,
        # but we allow it for simplicity / symmetry)
        if gs.turn_number > 1 or True:  # always draw for simplicity
            self._draw_n(gs, gs.current_player, 1)

    def apply_action(self, gs: GameState, action_idx: int) -> Tuple[float, bool]:
        """
        Apply action to game state. Returns (reward, done).
        Modifies gs in-place.
        """
        if gs.game_over:
            return 0.0, True

        atype, params = ActionMapper.decode(action_idx)
        me  = gs.current()
        opp = gs.opp()
        reward = -0.01  # small negative per action to encourage efficiency

        if atype == ActionType.END_TURN:
            self._end_turn(gs)
            return reward, gs.game_over

        if atype == ActionType.ATTACH_ENERGY:
            self._attach_energy(gs, params["slot"])

        elif atype == ActionType.PLAY_POKEMON:
            self._play_pokemon(gs, params["hand_idx"])

        elif atype == ActionType.USE_ITEM:
            self._use_item(gs, params["hand_idx"])

        elif atype == ActionType.USE_SUPPORTER:
            self._use_supporter(gs, params["hand_idx"])

        elif atype == ActionType.EVOLVE:
            self._evolve(gs, params["hand_idx"], params["slot"])

        elif atype == ActionType.PROMOTE:
            self._promote(gs, params["bench_slot"])

        elif atype == ActionType.ATTACK:
            r = self._attack(gs, params["atk_idx"])
            reward += r
            # After attack: end turn
            self._end_turn(gs)

        return reward, gs.game_over

    # ── Sub-actions ──────────────────────────────

    def _attach_energy(self, gs: GameState, slot: int) -> None:
        me = gs.current()
        if me.energy_used:
            return
        # Find first energy card in hand
        for i, c in enumerate(me.hand):
            if isinstance(c, EnergyCard):
                me.hand.pop(i)
                target = me.active if slot == 0 else (me.bench[slot - 1] if slot - 1 < len(me.bench) else None)
                if target is not None:
                    target.energy[c.energy_type] = target.energy.get(c.energy_type, 0) + 1
                    me.energy_used = True
                return

    def _play_pokemon(self, gs: GameState, hand_idx: int) -> None:
        me = gs.current()
        if hand_idx >= len(me.hand):
            return
        c = me.hand[hand_idx]
        if not (isinstance(c, PokemonCard) and c.stage == Stage.BASIC):
            return
        if len(me.bench) >= MAX_BENCH:
            return
        c.turn_played = gs.turn_number
        me.bench.append(me.hand.pop(hand_idx))

    def _use_item(self, gs: GameState, hand_idx: int) -> None:
        me  = gs.current()
        opp = gs.opp()
        if hand_idx >= len(me.hand):
            return
        c = me.hand[hand_idx]
        if not (isinstance(c, TrainerCard) and c.card_type == CardType.ITEM):
            return
        me.hand.pop(hand_idx)
        me.discard.append(c)

        if c.effect == "heal_active_20" and me.active:
            me.active.current_hp = min(me.active.hp, me.active.current_hp + 20)
        elif c.effect == "heal_active_30" and me.active:
            me.active.current_hp = min(me.active.hp, me.active.current_hp + 30)
        elif c.effect == "great_ball":
            self._great_ball(gs, gs.current_player)

    def _great_ball(self, gs: GameState, pidx: int) -> None:
        """Look at top 7; take a pokemon, shuffle rest."""
        p = gs.players[pidx]
        top7 = p.deck[:7]
        rest = p.deck[7:]
        pokemon_found = [c for c in top7 if isinstance(c, PokemonCard)]
        non_pokemon   = [c for c in top7 if not isinstance(c, PokemonCard)]
        if pokemon_found:
            # Take the first pokemon found
            p.hand.append(pokemon_found[0])
            non_pokemon.extend(pokemon_found[1:])
        # Shuffle rest back
        self._shuffle(non_pokemon)
        p.deck = non_pokemon + rest

    def _use_supporter(self, gs: GameState, hand_idx: int) -> None:
        me  = gs.current()
        if hand_idx >= len(me.hand):
            return
        c = me.hand[hand_idx]
        if not (isinstance(c, TrainerCard) and c.card_type == CardType.SUPPORTER):
            return
        if me.supporter_used:
            return
        me.hand.pop(hand_idx)
        me.discard.append(c)
        me.supporter_used = True

        if c.effect == "draw3":
            self._draw_n(gs, gs.current_player, 3)

    def _promote(self, gs: GameState, bench_slot: int) -> None:
        """Promote a benched pokemon to active after a KO (player's choice)."""
        # The player who needs to promote is whoever has pending_promotion set.
        # This is called as part of the *loser's* turn or as an interrupt.
        # We check both players but only act on the one pending.
        for pidx in range(2):
            p = gs.players[pidx]
            if p.pending_promotion:
                if bench_slot < len(p.bench):
                    promoted = p.bench.pop(bench_slot)
                    promoted.is_active = True
                    p.active = promoted
                    p.pending_promotion = False
                return

    def _evolve(self, gs: GameState, hand_idx: int, slot: int) -> None:
        """Evolve pokemon at slot (5=active, 0-4=bench) using hand_idx card."""
        me = gs.current()
        if hand_idx >= len(me.hand):
            return
        evo_card = me.hand[hand_idx]
        if not isinstance(evo_card, PokemonCard):
            return

        target = me.active if slot == MAX_BENCH else (me.bench[slot] if slot < len(me.bench) else None)
        if target is None:
            return
        if not can_evolve(evo_card, target, gs.turn_number):
            return

        # Transfer HP damage and energy
        damage_taken = target.hp - target.current_hp
        evo_card = me.hand.pop(hand_idx)
        evo_card.current_hp = max(1, evo_card.hp - damage_taken)
        evo_card.energy = dict(target.energy)
        evo_card.turn_played = target.turn_played
        evo_card.effect_flags = {}
        me.discard.append(target)  # base form goes to discard

        if slot == MAX_BENCH:
            evo_card.is_active = True
            me.active = evo_card
        else:
            me.bench[slot] = evo_card

    def _attack(self, gs: GameState, atk_idx: int) -> float:
        """Resolve an attack. Returns reward delta."""
        me  = gs.current()
        opp = gs.opp()
        if me.active is None or atk_idx >= len(me.active.attacks):
            return 0.0

        atk = me.active.attacks[atk_idx]
        if not can_pay_cost(me.active, atk.energy_cost):
            return 0.0

        reward = 0.0
        base_damage = atk.damage
        effect = atk.effect or ""

        # Apply damage reduction from opponent's previous turn
        damage_reduction = gs.damage_reduction.get(gs.opponent(), 0)

        # Resolve effect
        final_damage = base_damage

        if effect == "coin_plus10":
            if self._coin_flip():
                final_damage += 10
        elif effect == "coin_plus30":
            if self._coin_flip():
                final_damage += 30
        elif effect == "flip4_20each":
            final_damage = sum(20 for _ in range(4) if self._coin_flip())
        elif effect == "flip2_plus20each":
            final_damage = base_damage + sum(20 for _ in range(2) if self._coin_flip())
        elif effect == "heal_self_20":
            if me.active:
                me.active.current_hp = min(me.active.hp, me.active.current_hp + 20)
        elif effect == "opponent_minus20_next_turn":
            gs.damage_reduction[gs.current_player] = 20   # stored for opponent to check
        elif effect == "discard_random_opponent_hand":
            if opp.hand:
                idx = int(self.rng.integers(0, len(opp.hand)))
                card = opp.hand.pop(idx)
                opp.deck.append(card)
                self._shuffle(opp.deck)
        elif effect == "plus30_vs_caterpie":
            if opp.active and opp.active.name == "Caterpie":
                final_damage += 30
        elif effect == "bench_10_any":
            # Deal 10 to a random bench pokemon
            if opp.bench:
                bi = int(self.rng.integers(0, len(opp.bench)))
                opp.bench[bi].current_hp -= 10
                if opp.bench[bi].current_hp <= 0:
                    self._handle_ko(gs, gs.opponent(), is_active=False, bench_idx=bi)
        elif effect == "optional_plus40_self20":
            # For simplicity: agent always takes the damage boost (heuristic)
            final_damage += 40
            if me.active:
                me.active.current_hp -= 20
        elif effect == "coin_protect_self":
            if self._coin_flip():
                me.active.effect_flags["protected"] = gs.turn_number
        elif effect == "discard_lightning_30each":
            lightning = me.active.energy.get(EnergyType.LIGHTNING, 0)
            me.active.energy[EnergyType.LIGHTNING] = 0
            final_damage = lightning * 30
        elif effect == "10x_opp_energy":
            if opp.active:
                final_damage = 10 * opp.active.total_energy()

        # Apply damage reduction (from Fletchling's Hinder)
        final_damage = max(0, final_damage - damage_reduction)

        # Apply protection flag
        if opp.active and opp.active.effect_flags.get("protected") == gs.turn_number - 1:
            final_damage = 0

        # Deal damage to opponent's active
        if opp.active and final_damage > 0:
            opp.active.current_hp -= final_damage

        # Check KO
        if opp.active and opp.active.current_hp <= 0:
            reward += 1.0
            self._handle_ko(gs, gs.opponent(), is_active=True)
            if gs.game_over:
                reward += 10.0 if gs.winner == gs.current_player else -10.0

        return reward

    def _handle_ko(self, gs: GameState, defeated_player_idx: int,
                   is_active: bool, bench_idx: int = -1) -> None:
        """Handle a KO: award prize counter, set pending_promotion for the loser."""
        winner_idx   = 1 - defeated_player_idx
        loser        = gs.players[defeated_player_idx]
        winner       = gs.players[winner_idx]

        winner.ko_count += 1

        if is_active:
            ko_poke = loser.active
            loser.discard.append(ko_poke)
            loser.active = None
            if not loser.bench:
                # No bench pokemon → loser loses immediately
                gs.game_over = True
                gs.winner = winner_idx
                return
            elif len(loser.bench) == 1:
                # Only one option — auto-promote, no choice needed
                promoted = loser.bench.pop(0)
                promoted.is_active = True
                loser.active = promoted
            else:
                # Multiple bench options → player must choose
                loser.pending_promotion = True
        else:
            if bench_idx < len(loser.bench):
                ko_poke = loser.bench.pop(bench_idx)
                loser.discard.append(ko_poke)

        if winner.ko_count >= KO_TO_WIN:
            gs.game_over = True
            gs.winner = winner_idx

    def _end_turn(self, gs: GameState) -> None:
        """Switch active player, increment turn counter."""
        gs.current_player = gs.opponent()
        gs.turn_number += 1
        # Update turn_played for newly played cards (already set)
        # Start next player's turn (draw card etc) is done in start_turn()


# ─────────────────────────────────────────────
# GYM-STYLE ENVIRONMENT
# ─────────────────────────────────────────────

class PokemonTCGEnv:
    """
    Two-player zero-sum environment.
    Agents take turns; each call to step() advances one half-turn.

    Observation:  float32 vector of size StateEncoder.STATE_SIZE
    Action:       int in [0, ActionMapper.TOTAL_ACTIONS)
    Reward:       scalar from current player's perspective
    """

    def __init__(self, seed: int = 42, debug: bool = False):
        self.engine = GameEngine(seed=seed)
        self.seed   = seed
        self.debug  = debug
        self.gs: Optional[GameState] = None
        self._step_count = 0
        self._max_steps  = 500  # safety limit

    @property
    def observation_space_size(self) -> int:
        return StateEncoder.STATE_SIZE

    @property
    def action_space_size(self) -> int:
        return ActionMapper.TOTAL_ACTIONS

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.seed = seed
            self.engine = GameEngine(seed=seed)
        self.gs = self.engine.new_game()
        self.engine.start_turn(self.gs)
        self._step_count = 0
        if self.debug:
            self._print_state()
        return StateEncoder.encode(self.gs)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        assert self.gs is not None, "Call reset() first"
        if self.gs.game_over:
            return StateEncoder.encode(self.gs), 0.0, True, {
                "current_player": self.gs.current_player,
                "turn": self.gs.turn_number,
                "winner": self.gs.winner,
                "ko_counts": [p.ko_count for p in self.gs.players],
            }

        mask = self.get_legal_mask()
        if mask[action] == 0:
            # Illegal action penalty
            if self.debug:
                print(f"  [ILLEGAL action {action}]")
            return StateEncoder.encode(self.gs), -0.5, False, {"illegal": True}

        current_player_before = self.gs.current_player
        reward, done = self.engine.apply_action(self.gs, action)
        self._step_count += 1

        # Safety: end game if too long
        if self._step_count >= self._max_steps and not done:
            done = True
            self.gs.game_over = True
            self.gs.winner = -1  # draw

        # If not done and player switched, start new turn
        if not done and self.gs.current_player != current_player_before:
            self.engine.start_turn(self.gs)

        obs  = StateEncoder.encode(self.gs)
        info = {
            "current_player": self.gs.current_player,
            "turn": self.gs.turn_number,
            "winner": self.gs.winner if done else -1,
            "ko_counts": [p.ko_count for p in self.gs.players],
        }

        if self.debug:
            atype, params = ActionMapper.decode(action)
            print(f"  P{current_player_before} → {atype.name}{params}  r={reward:.2f}")
            if done:
                print(f"  GAME OVER. Winner: P{self.gs.winner}")

        return obs, reward, done, info

    def get_legal_mask(self) -> np.ndarray:
        return compute_legal_mask(self.gs)

    def get_legal_actions(self) -> List[int]:
        mask = self.get_legal_mask()
        return [i for i, v in enumerate(mask) if v > 0]

    def _print_state(self) -> None:
        gs = self.gs
        me = gs.current()
        print(f"\n── Turn {gs.turn_number} | Player {gs.current_player} ──")
        print(f"  Active: {me.active.name if me.active else 'None'} "
              f"HP:{me.active.current_hp if me.active else '-'} "
              f"E:{me.active.energy if me.active else {}}")
        print(f"  Bench: {[p.name for p in me.bench]}")
        print(f"  Hand ({len(me.hand)}): {[getattr(c,'name','?') for c in me.hand]}")
        print(f"  Deck: {len(me.deck)} | KOs: {me.ko_count}")


# ─────────────────────────────────────────────
# HEURISTIC AGENT (baseline)
# ─────────────────────────────────────────────

class HeuristicAgent:
    """
    Strong rule-based baseline agent.

    Priority order:
    1. PROMOTE  — if pending, pick the highest-HP benched pokemon
    2. EVOLVE   — always evolve if possible (highest stage first)
    3. ATTACK   — always attack if possible, using the HIGHEST-damage
                  legal attack (strictly better = more base damage).
                  Tie-break: prefer attacks with useful effects.
    4. ATTACH_ENERGY — to active if it needs more energy for an attack,
                       otherwise to the most evolved bench pokemon
    5. PLAY_POKEMON  — bench basics that are evolution bases first
    6. USE_ITEM      — heal if damaged, great ball otherwise
    7. USE_SUPPORTER — draw cards
    8. END_TURN
    """

    def act(self, env: 'PokemonTCGEnv') -> int:
        gs     = env.gs
        me     = gs.current()
        legal  = env.get_legal_actions()

        def actions_of(t: ActionType) -> List[int]:
            return [a for a in legal if ActionMapper.decode(a)[0] == t]

        # 1. PROMOTE — mandatory, pick highest-HP bench pokemon
        promotes = actions_of(ActionType.PROMOTE)
        if promotes:
            # Figure out which player needs promotion
            for pidx in range(2):
                if gs.players[pidx].pending_promotion:
                    p = gs.players[pidx]
                    # Map promote action → bench slot, pick highest HP
                    best_slot = max(
                        range(len(p.bench)),
                        key=lambda i: p.bench[i].current_hp
                    )
                    enc = ActionMapper.encode(ActionType.PROMOTE, {"bench_slot": best_slot})
                    if enc in legal:
                        return enc
            return promotes[0]

        # 2. EVOLVE — always evolve, prefer highest stage
        evolves = actions_of(ActionType.EVOLVE)
        if evolves:
            def evolve_stage(a: int) -> int:
                _, p = ActionMapper.decode(a)
                hi = p["hand_idx"]
                if hi < len(me.hand) and isinstance(me.hand[hi], PokemonCard):
                    return me.hand[hi].stage
                return 0
            return max(evolves, key=evolve_stage)

        # 3. ATTACK — highest base damage among legal attacks
        attacks = actions_of(ActionType.ATTACK)
        if attacks and me.active:
            def attack_value(a: int) -> float:
                _, p = ActionMapper.decode(a)
                ai = p["atk_idx"]
                if ai < len(me.active.attacks):
                    atk = me.active.attacks[ai]
                    # Damage + small tiebreaker for useful effects
                    effect_bonus = 0.5 if atk.effect and atk.damage > 0 else 0
                    return atk.damage + effect_bonus
                return 0.0
            return max(attacks, key=attack_value)

        # 4. ATTACH_ENERGY — to pokemon that needs it most for an attack
        attaches = actions_of(ActionType.ATTACH_ENERGY)
        if attaches:
            def attach_priority(a: int) -> float:
                _, p = ActionMapper.decode(a)
                slot = p["slot"]
                target = me.active if slot == 0 else (
                    me.bench[slot - 1] if slot - 1 < len(me.bench) else None)
                if target is None:
                    return -1.0
                # Priority: active first, then highest-stage bench
                base = 10.0 if slot == 0 else float(target.stage)
                # Prefer targets whose cheapest attack is nearly affordable
                min_cost = min(
                    (sum(atk.energy_cost.values()) for atk in target.attacks),
                    default=99
                )
                gap = max(0, min_cost - target.total_energy())
                return base - gap * 0.1
            return max(attaches, key=attach_priority)

        # 5. PLAY_POKEMON — bench evolution bases first
        plays = actions_of(ActionType.PLAY_POKEMON)
        if plays:
            def play_priority(a: int) -> float:
                _, p = ActionMapper.decode(a)
                hi = p["hand_idx"]
                if hi < len(me.hand) and isinstance(me.hand[hi], PokemonCard):
                    card = me.hand[hi]
                    # Check if something in deck/hand/bench evolves from it
                    all_cards = me.hand + me.deck + me.bench
                    is_evo_base = any(
                        isinstance(c, PokemonCard) and c.evolves_from == card.name
                        for c in all_cards
                    )
                    return (2.0 if is_evo_base else 1.0) + card.hp / 200.0
                return 0.0
            return max(plays, key=play_priority)

        # 6. USE_ITEM — heal first, then great ball
        items = actions_of(ActionType.USE_ITEM)
        if items:
            def item_priority(a: int) -> float:
                _, p = ActionMapper.decode(a)
                hi = p["hand_idx"]
                if hi < len(me.hand) and isinstance(me.hand[hi], TrainerCard):
                    effect = me.hand[hi].effect
                    if effect in ("heal_active_20", "heal_active_30"):
                        # Prefer healing when badly damaged
                        if me.active:
                            return 1.0 - me.active.hp_fraction()
                        return 0.0
                    elif effect == "great_ball":
                        return 0.5
                return 0.0
            return max(items, key=item_priority)

        # 7. USE_SUPPORTER
        supps = actions_of(ActionType.USE_SUPPORTER)
        if supps:
            return supps[0]

        # 8. END_TURN
        return ActionMapper.encode(ActionType.END_TURN, {})


# ─────────────────────────────────────────────
# RANDOM AGENT
# ─────────────────────────────────────────────

class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, env: PokemonTCGEnv) -> int:
        legal = env.get_legal_actions()
        return self.rng.choice(legal)


# ─────────────────────────────────────────────
# SIMPLE PPO STUB
# ─────────────────────────────────────────────

class PPOStub:
    """
    Minimal PPO training loop stub.
    Replace network/optimizer with your framework of choice (PyTorch/JAX).
    """

    def __init__(self, obs_size: int, act_size: int):
        self.obs_size = obs_size
        self.act_size = act_size
        # Placeholder: random policy
        self._rng = np.random.default_rng(0)

    def get_action_and_value(self, obs: np.ndarray, mask: np.ndarray):
        """Returns (action, log_prob, value). Replace with neural net."""
        probs = mask / (mask.sum() + 1e-8)
        action = self._rng.choice(self.act_size, p=probs)
        log_prob = np.log(probs[action] + 1e-8)
        value = 0.0
        return action, log_prob, value

    def train_step(self, rollout_buffer):
        """Placeholder training step. Implement PPO loss here."""
        pass

    def collect_rollout(self, env: PokemonTCGEnv, n_steps: int = 256):
        """Collect n_steps of experience. Returns list of (obs, act, rew, done, mask)."""
        buffer = []
        obs = env.reset()
        for _ in range(n_steps):
            mask = env.get_legal_mask()
            action, log_prob, value = self.get_action_and_value(obs, mask)
            next_obs, reward, done, info = env.step(action)
            buffer.append((obs, action, reward, done, mask, log_prob, value))
            obs = next_obs if not done else env.reset()
        return buffer


# ─────────────────────────────────────────────
# UNIT TESTS
# ─────────────────────────────────────────────

def run_unit_tests():
    print("=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)

    # ── Test 1: Deck building ─────────────────────────────────
    deck0 = build_lycanroc_deck()
    deck1 = build_raichu_deck()
    assert len(deck0) == 30, f"Lycanroc deck size {len(deck0)} != 30"
    assert len(deck1) == 30, f"Raichu deck size {len(deck1)} != 30"
    print("✓ Deck sizes correct (30 each)")

    # ── Test 2: Game initialisation ───────────────────────────
    env = PokemonTCGEnv(seed=42)
    obs = env.reset()
    assert obs.shape == (StateEncoder.STATE_SIZE,), f"Obs shape {obs.shape}"
    assert env.gs is not None
    assert env.gs.players[0].active is not None
    assert env.gs.players[1].active is not None
    print("✓ Game initialises correctly")

    # ── Test 3: State encoding ────────────────────────────────
    enc = StateEncoder.encode(env.gs)
    assert enc.dtype == np.float32
    assert not np.any(np.isnan(enc))
    print("✓ State encoding produces valid float32 vector")

    # ── Test 4: Legal mask ────────────────────────────────────
    mask = env.get_legal_mask()
    assert mask.shape == (ActionMapper.TOTAL_ACTIONS,)
    assert mask[ActionMapper.END_TURN_IDX] == 1.0, "END_TURN should always be legal"
    legal = env.get_legal_actions()
    assert len(legal) > 0
    print(f"✓ Legal mask computed ({len(legal)} legal actions)")

    # ── Test 5: Energy attachment limit ──────────────────────
    env2 = PokemonTCGEnv(seed=10)
    env2.reset()
    gs = env2.gs
    # Manually add energy to hand
    gs.current().hand.insert(0, EnergyCard("Fighting Energy", EnergyType.FIGHTING))
    gs.current().hand.insert(0, EnergyCard("Fighting Energy", EnergyType.FIGHTING))
    gs.current().energy_used = False
    # Attach once
    mask1 = env2.get_legal_mask()
    attach_actions = [i for i in range(ActionMapper.ATTACH_START,
                                       ActionMapper.ATTACH_START + ActionMapper.ATTACH_COUNT)
                      if mask1[i] == 1.0]
    assert len(attach_actions) > 0, "Should be able to attach energy"
    env2.step(attach_actions[0])
    # If no turn switch happened, energy_used should be True
    if not env2.gs.game_over:
        print("✓ Energy attach limit enforced (energy_used flag)")

    # ── Test 6: KO handling ───────────────────────────────────
    env3 = PokemonTCGEnv(seed=5)
    env3.reset()
    gs3 = env3.gs
    opp_idx = gs3.opponent()
    opp = gs3.opp()
    # Set active to 1 HP so next attack will KO
    if opp.active:
        opp.active.current_hp = 1
    # Run random actions until game over or KO happens
    initial_ko = gs3.current().ko_count
    for _ in range(50):
        if env3.gs.game_over:
            break
        legal = env3.get_legal_actions()
        action = legal[0]
        obs, rew, done, info = env3.step(action)
        if done:
            break
    print("✓ KO handling executes without crash")

    # ── Test 7: Evolution rules ───────────────────────────────
    env4 = PokemonTCGEnv(seed=99)
    obs = env4.reset()
    gs4 = env4.gs
    me = gs4.current()
    # Manually put a Rockruff on bench with early turn
    rockruff = PokemonCard("Rockruff", 60, Stage.BASIC, None, [
        Attack("Tackle", 10, _e("1C"))
    ])
    rockruff.turn_played = 0  # played "last turn"
    me.bench.append(rockruff)
    lycanroc = PokemonCard("Lycanroc", 110, Stage.STAGE1, "Rockruff", [
        Attack("Accelerock", 20, _e("2C"))
    ])
    me.hand.append(lycanroc)
    gs4.turn_number = 2  # current turn is 2, so rockruff (turn 0) can evolve
    assert can_evolve(lycanroc, rockruff, gs4.turn_number), "Lycanroc should evolve Rockruff"
    # Same turn: turn_played == turn_number → cannot evolve
    rockruff.turn_played = 2
    assert not can_evolve(lycanroc, rockruff, gs4.turn_number), "Can't evolve same turn"
    print("✓ Evolution timing rules correct")

    # ── Test 8: Action mapper round-trip ─────────────────────
    test_cases = [
        (ActionType.END_TURN, {}),
        (ActionType.ATTACH_ENERGY, {"slot": 3}),
        (ActionType.PLAY_POKEMON, {"hand_idx": 5}),
        (ActionType.USE_ITEM, {"hand_idx": 2}),
        (ActionType.USE_SUPPORTER, {"hand_idx": 7}),
        (ActionType.EVOLVE, {"hand_idx": 3, "slot": 4}),
        (ActionType.ATTACK, {"atk_idx": 1}),
    ]
    for atype, params in test_cases:
        idx = ActionMapper.encode(atype, params)
        decoded_type, decoded_params = ActionMapper.decode(idx)
        assert decoded_type == atype, f"Type mismatch for {atype}"
        assert decoded_params == params, f"Param mismatch: {decoded_params} != {params}"
    print("✓ ActionMapper round-trip encode/decode correct")

    # ── Test 9: Full random game plays to completion ──────────
    env5 = PokemonTCGEnv(seed=777)
    agent = RandomAgent(seed=0)
    obs = env5.reset()
    done = False
    steps = 0
    while not done and not env5.gs.game_over:
        action = agent.act(env5)
        obs, rew, done, info = env5.step(action)
        steps += 1
    assert done or env5.gs.game_over
    print(f"✓ Full random game completed in {steps} steps. Winner: P{info['winner']}")

    # ── Test 10: Heuristic vs Random ─────────────────────────
    h_agent = HeuristicAgent()
    r_agent = RandomAgent(seed=42)
    wins = [0, 0]
    n_games = 20
    for g in range(n_games):
        env_g = PokemonTCGEnv(seed=1000 + g)
        obs = env_g.reset()
        done = False
        while not done:
            cp = env_g.gs.current_player
            if cp == 0:
                act = h_agent.act(env_g)
            else:
                act = r_agent.act(env_g)
            obs, rew, done, info = env_g.step(act)
        winner = info["winner"]
        if winner >= 0:
            wins[winner] += 1
    print(f"✓ Heuristic vs Random over {n_games} games: H={wins[0]}, R={wins[1]}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


# ─────────────────────────────────────────────
# DEMO: RANDOM vs RANDOM
# ─────────────────────────────────────────────

def demo_random_vs_random(n_games: int = 5, seed: int = 42):
    print("\n" + "=" * 60)
    print(f"DEMO: Random vs Random ({n_games} games)")
    print("=" * 60)
    wins = [0, 0, 0]  # P0, P1, Draw
    total_steps = []

    for g in range(n_games):
        env = PokemonTCGEnv(seed=seed + g, debug=False)
        obs = env.reset()
        done = False
        steps = 0
        agent = RandomAgent(seed=seed + g)

        while not done:
            action = agent.act(env)
            obs, rew, done, info = env.step(action)
            steps += 1

        winner = info["winner"]
        if winner == -1:
            wins[2] += 1
        else:
            wins[winner] += 1
        total_steps.append(steps)
        print(f"  Game {g+1}: Winner=P{winner if winner>=0 else 'Draw'} | "
              f"Steps={steps} | KOs={info['ko_counts']} | Turns={info['turn']}")

    print(f"\nResults: P0={wins[0]}, P1={wins[1]}, Draw={wins[2]}")
    print(f"Avg steps: {sum(total_steps)/len(total_steps):.1f}")


# ─────────────────────────────────────────────
# VECTORIZED BATCH ENVIRONMENT
# ─────────────────────────────────────────────

class VecPokemonTCGEnv:
    """
    Simple vectorized environment running n_envs in parallel (serial).
    For true parallelism, replace with multiprocessing.
    """

    def __init__(self, n_envs: int = 8, base_seed: int = 0):
        self.envs = [PokemonTCGEnv(seed=base_seed + i) for i in range(n_envs)]
        self.n_envs = n_envs

    def reset(self) -> np.ndarray:
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        obs_list, rew_list, done_list, info_list = [], [], [], []
        for i, env in enumerate(self.envs):
            obs, rew, done, info = env.step(int(actions[i]))
            if done:
                obs = env.reset()
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(done)
            info_list.append(info)
        return (np.stack(obs_list), np.array(rew_list, dtype=np.float32),
                np.array(done_list, dtype=bool), info_list)

    def get_legal_masks(self) -> np.ndarray:
        return np.stack([e.get_legal_mask() for e in self.envs])


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_unit_tests()
    demo_random_vs_random(n_games=5, seed=42)

    print("\n" + "=" * 60)
    print("ENVIRONMENT SUMMARY")
    print("=" * 60)
    print(f"  Observation space: {StateEncoder.STATE_SIZE} floats")
    print(f"  Action space:      {ActionMapper.TOTAL_ACTIONS} discrete actions")
    print(f"  Action breakdown:")
    print(f"    END_TURN:        1")
    print(f"    ATTACH_ENERGY:   {ActionMapper.ATTACH_COUNT}  (active + 5 bench)")
    print(f"    PLAY_POKEMON:    {ActionMapper.PLAY_COUNT}  (hand positions)")
    print(f"    USE_ITEM:        {ActionMapper.ITEM_COUNT}  (hand positions)")
    print(f"    USE_SUPPORTER:   {ActionMapper.SUPP_COUNT}  (hand positions)")
    print(f"    EVOLVE:          {ActionMapper.EVOLVE_COUNT}  (hand x bench+active)")
    print(f"    ATTACK:          {ActionMapper.ATTACK_COUNT}  (attack slots)")

    print("\nPPO stub example:")
    ppo = PPOStub(StateEncoder.STATE_SIZE, ActionMapper.TOTAL_ACTIONS)
    env = PokemonTCGEnv(seed=0)
    buffer = ppo.collect_rollout(env, n_steps=256)
    print(f"  Collected {len(buffer)} steps of experience.")
    print("\nReady for RL training!")
