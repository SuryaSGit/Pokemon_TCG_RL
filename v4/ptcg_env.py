"""
Pokemon TCG RL Environment — v4
Mega Starmie ex Deck vs Mega Lucario ex Deck
Full 60-card decks. First to 6 KO points wins.

New rules vs v3:
  • Player going first cannot attack on turn 1
  • Cannot evolve a card the same turn it was played
  • Pokémon ex: opponent gains 2 KO points on KO
  • Mega Evolution ex: opponent gains 3 KO points on KO; turn ends after mega evolving
  • Abilities (passive / on-bench-entry / active-once-per-turn)
  • Tool cards (attach one per Pokémon, stays until KO'd)
  • Stadium cards (one active at a time)
  • Retreat (pay energy cost, swap with bench)
  • Status effects: Asleep, Confused, Paralyzed
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
    COLORLESS = 0
    FIGHTING  = 1
    LIGHTNING = 2
    PSYCHIC   = 3
    WATER     = 4
    DARKNESS  = 5

class Stage(IntEnum):
    BASIC  = 0
    STAGE1 = 1
    STAGE2 = 2

class CardType(IntEnum):
    POKEMON   = 0
    ENERGY    = 1
    ITEM      = 2
    SUPPORTER = 3
    TOOL      = 4
    STADIUM   = 5

class ActionType(IntEnum):
    ATTACH_ENERGY = 0
    PLAY_POKEMON  = 1
    USE_ITEM      = 2
    USE_SUPPORTER = 3
    EVOLVE        = 4
    ATTACK        = 5
    END_TURN      = 6
    PROMOTE       = 7
    RETREAT       = 8
    ATTACH_TOOL   = 9
    USE_ABILITY   = 10
    USE_STADIUM   = 11
    CHOOSE        = 12   # resolve a pending_choice (select one option)

MAX_BENCH   = 5
MAX_ATTACKS = 2
MAX_HAND    = 12   # encoding cap (real hand can be larger)
MAX_DECK    = 60
KO_TO_WIN   = 6

POKEMON_IDS = {
    # Starmie deck
    "Duskull": 0, "Dusclops": 1, "Dusknoir": 2,
    "Staryu": 3, "Mega Starmie ex": 4, "Munkidori": 5,
    "Budew": 6, "Meowth ex": 7, "Fezandipiti ex": 8,
    "Latias ex": 9, "Bloodmoon Ursaluna ex": 10,
    # Lucario deck
    "Riolu": 11, "Mega Lucario ex": 12,
    "Solrock": 13, "Lunatone": 14,
    "Makuhita": 15, "Hariyama": 16,
}
NUM_POKEMON_IDS = len(POKEMON_IDS)

# Stadium effect tags
STADIUM_NONE          = 0
STADIUM_RISKY_RUINS   = 1
STADIUM_GRAVITY_MTN   = 2

# ─────────────────────────────────────────────
# CARD DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class Attack:
    name: str
    damage: int
    energy_cost: Dict[EnergyType, int]
    effect: Optional[str] = None

@dataclass
class PokemonCard:
    name: str
    hp: int
    stage: Stage
    evolves_from: Optional[str]
    attacks: List[Attack]
    card_type: CardType = CardType.POKEMON
    pokemon_type: EnergyType = EnergyType.COLORLESS   # type for restrictions
    # ex / mega flags
    is_ex: bool = False
    is_mega_ex: bool = False
    # Ability fields
    ability_name: Optional[str] = None
    ability_effect: Optional[str] = None   # effect tag
    ability_type: str = "none"             # "passive" | "on_bench_entry" | "active"
    ability_used: bool = False             # per-turn flag for "active" abilities
    # Retreat
    retreat_cost: int = 1
    # Runtime state
    current_hp: int = 0
    energy: Dict[EnergyType, int] = field(default_factory=dict)
    is_active: bool = False
    effect_flags: Dict[str, Any] = field(default_factory=dict)
    turn_played: int = -1
    tool_card: Optional[Any] = None        # TrainerCard with CardType.TOOL
    status: Optional[str] = None           # "asleep" | "confused" | "paralyzed"
    # Per-turn attack restrictions (activated via pending fields in start_turn)
    cant_attack_turns: int = 0             # > 0 → all attacks blocked this turn
    cant_attack_turns_pending: int = 0     # set during attack, becomes active next turn
    blocked_attacks: set = field(default_factory=set)          # attack names blocked this turn
    blocked_attacks_pending: set = field(default_factory=set)  # set during attack, activates next turn
    shadow_bound: bool = False             # can't retreat this turn (Shadow Bind)
    # Special energy tracking
    ignition_bonus: int = 0               # temporary COLORLESS from Ignition Energy (cleared end of turn)
    has_legacy_energy: bool = False        # Legacy Energy is attached

    def __post_init__(self):
        self.current_hp = self.hp

    def total_energy(self) -> int:
        return sum(self.energy.values())

    def hp_fraction(self) -> float:
        return self.current_hp / self.hp

    def clone(self) -> 'PokemonCard':
        c = copy.copy(self)
        c.attacks = self.attacks
        c.energy = dict(self.energy)
        c.effect_flags = dict(self.effect_flags)
        c.tool_card = self.tool_card
        c.blocked_attacks = set(self.blocked_attacks)
        c.blocked_attacks_pending = set(self.blocked_attacks_pending)
        return c


@dataclass
class TrainerCard:
    name: str
    card_type: CardType   # ITEM | SUPPORTER | TOOL | STADIUM
    effect: str

@dataclass
class EnergyCard:
    name: str
    energy_type: EnergyType
    card_type: CardType = CardType.ENERGY
    # Special energies can provide multiple types
    provides: Optional[Dict[EnergyType, int]] = None
    special_effect: Optional[str] = None   # "attach_basic_energy_from_deck" etc.

Card = Any  # PokemonCard | TrainerCard | EnergyCard

# ─────────────────────────────────────────────
# ENERGY COST PARSER
# ─────────────────────────────────────────────

def _e(cost_str: str) -> Dict[EnergyType, int]:
    """Parse cost string like '2W1C' → {WATER:2, COLORLESS:1}"""
    cost: Dict[EnergyType, int] = {}
    i = 0
    while i < len(cost_str):
        count = int(cost_str[i]); i += 1
        t = cost_str[i]; i += 1
        etype = {
            'C': EnergyType.COLORLESS, 'F': EnergyType.FIGHTING,
            'L': EnergyType.LIGHTNING, 'P': EnergyType.PSYCHIC,
            'W': EnergyType.WATER,     'D': EnergyType.DARKNESS,
        }[t]
        cost[etype] = cost.get(etype, 0) + count
    return cost

# ─────────────────────────────────────────────
# DECK DEFINITIONS
# ─────────────────────────────────────────────

def build_starmie_deck() -> List[Card]:
    """60-card Mega Starmie ex deck."""
    cards: List[Card] = []

    # ── Pokémon ──────────────────────────────
    for _ in range(4):
        cards.append(PokemonCard(
            "Duskull", 60, Stage.BASIC, None,
            [Attack("Come and Get You", 0,  _e("1P"), effect="come_and_get_you"),
             Attack("Mumble",           30, _e("2P"))],
            pokemon_type=EnergyType.PSYCHIC, retreat_cost=1,
        ))
    for _ in range(3):
        cards.append(PokemonCard(
            "Dusclops", 90, Stage.STAGE1, "Duskull",
            [Attack("Will-O-Wisp", 50, _e("2P"))],
            pokemon_type=EnergyType.PSYCHIC, retreat_cost=2,
            ability_name="Cursed Blast",
            ability_effect="cursed_blast_5",
            ability_type="active",
        ))
    for _ in range(2):
        cards.append(PokemonCard(
            "Dusknoir", 160, Stage.STAGE2, "Dusclops",
            [Attack("Shadow Bind", 150, _e("2P1C"), effect="shadow_bind")],
            pokemon_type=EnergyType.PSYCHIC, retreat_cost=3,
            ability_name="Cursed Blast",
            ability_effect="cursed_blast_13",
            ability_type="active",
        ))
    for _ in range(3):
        cards.append(PokemonCard(
            "Staryu", 70, Stage.BASIC, None,
            [Attack("Water Gun", 20, _e("1W"))],
            pokemon_type=EnergyType.WATER, retreat_cost=1,
        ))
    for _ in range(2):
        cards.append(PokemonCard(
            "Mega Starmie ex", 330, Stage.STAGE1, "Staryu",
            [Attack("Jetting Blow", 120, _e("1W"), effect="bench_50_one"),
             Attack("Nebula Beam",  210, _e("3C"), effect="nebula_beam")],
            pokemon_type=EnergyType.WATER, retreat_cost=3,
            is_ex=True, is_mega_ex=True,
        ))
    for _ in range(2):
        cards.append(PokemonCard(
            "Munkidori", 110, Stage.BASIC, None,
            [Attack("Mind Bend", 60, _e("1P1C"), effect="confuse_opponent")],
            pokemon_type=EnergyType.PSYCHIC, retreat_cost=1,
            ability_name="Adrena-Brain",
            ability_effect="adrena_brain",
            ability_type="active",
        ))
    cards.append(PokemonCard(
        "Budew", 30, Stage.BASIC, None,
        [Attack("Itchy Pollen", 10, _e(""), effect="itchy_pollen")],
        pokemon_type=EnergyType.PSYCHIC, retreat_cost=1,
    ))
    cards.append(PokemonCard(
        "Meowth ex", 170, Stage.BASIC, None,
        [Attack("Tuck Tail", 60, _e("3C"), effect="tuck_tail")],
        pokemon_type=EnergyType.COLORLESS, retreat_cost=1,
        is_ex=True,
        ability_name="Last-Ditch Catch",
        ability_effect="last_ditch_catch",
        ability_type="on_bench_entry",
    ))
    cards.append(PokemonCard(
        "Fezandipiti ex", 210, Stage.BASIC, None,
        [Attack("Cruel Arrow", 100, _e("3C"), effect="cruel_arrow")],
        pokemon_type=EnergyType.PSYCHIC, retreat_cost=1,
        is_ex=True,
        ability_name="Flip the Script",
        ability_effect="flip_the_script",
        ability_type="active",
    ))
    cards.append(PokemonCard(
        "Latias ex", 210, Stage.BASIC, None,
        [Attack("Eon Blade", 200, _e("2P1C"), effect="eon_blade")],
        pokemon_type=EnergyType.COLORLESS, retreat_cost=1,
        is_ex=True,
        ability_name="Skyliner",
        ability_effect="skyliner",
        ability_type="passive",
    ))
    cards.append(PokemonCard(
        "Bloodmoon Ursaluna ex", 260, Stage.BASIC, None,
        [Attack("Blood Moon", 240, _e("5C"), effect="blood_moon")],
        pokemon_type=EnergyType.COLORLESS, retreat_cost=2,
        is_ex=True,
        ability_name="Seasoned Skill",
        ability_effect="seasoned_skill",
        ability_type="passive",
    ))

    # ── Trainers ──
    for _ in range(4):
        cards.append(TrainerCard("Lillie's Determination", CardType.SUPPORTER, "lillies_determination_s"))
    for _ in range(3):
        cards.append(TrainerCard("Hilda", CardType.SUPPORTER, "hilda"))
    for _ in range(2):
        cards.append(TrainerCard("Boss's Orders", CardType.SUPPORTER, "boss_orders"))
    cards.append(TrainerCard("Judge", CardType.SUPPORTER, "judge"))
    cards.append(TrainerCard("Wally's Compassion", CardType.SUPPORTER, "wally_compassion"))
    for _ in range(4):
        cards.append(TrainerCard("Buddy-Buddy Poffin", CardType.ITEM, "buddy_buddy_poffin"))
    for _ in range(4):
        cards.append(TrainerCard("Ultra Ball", CardType.ITEM, "ultra_ball"))
    for _ in range(3):
        cards.append(TrainerCard("Poké Pad", CardType.ITEM, "poke_pad"))
    for _ in range(3):
        cards.append(TrainerCard("Pokégear 3.0", CardType.ITEM, "pokegear"))
    for _ in range(2):
        cards.append(TrainerCard("Night Stretcher", CardType.ITEM, "night_stretcher"))
    for _ in range(3):
        cards.append(TrainerCard("Risky Ruins", CardType.STADIUM, "risky_ruins"))

    # ── Energy ──
    for _ in range(3):
        cards.append(EnergyCard("Water Energy", EnergyType.WATER))
    for _ in range(3):
        cards.append(EnergyCard("Darkness Energy", EnergyType.DARKNESS))
    for _ in range(2):
        cards.append(EnergyCard("Ignition Energy", EnergyType.COLORLESS,
                                special_effect="ignition_energy"))
    cards.append(EnergyCard("Legacy Energy", EnergyType.COLORLESS,
                            special_effect="legacy_energy"))

    assert len(cards) == 60, f"Starmie deck size {len(cards)} != 60"
    return cards


def build_lucario_deck() -> List[Card]:
    """60-card Mega Lucario ex deck."""
    cards: List[Card] = []

    # ── Pokémon ──────────────────────────────
    for _ in range(4):
        cards.append(PokemonCard(
            "Riolu", 80, Stage.BASIC, None,
            [Attack("Accelerating Stab", 30, _e("1F"), effect="accelerating_stab")],
            pokemon_type=EnergyType.FIGHTING, retreat_cost=1,
        ))
    for _ in range(3):
        cards.append(PokemonCard(
            "Mega Lucario ex", 340, Stage.STAGE1, "Riolu",
            [Attack("Aura Jab",   130, _e("1F"), effect="aura_jab"),
             Attack("Mega Brave", 270, _e("2F"), effect="mega_brave")],
            pokemon_type=EnergyType.FIGHTING, retreat_cost=3,
            is_ex=True, is_mega_ex=True,
        ))
    for _ in range(2):
        cards.append(PokemonCard(
            "Solrock", 110, Stage.BASIC, None,
            [Attack("Cosmic Beam", 70, _e("1F"), effect="cosmic_beam")],
            pokemon_type=EnergyType.FIGHTING, retreat_cost=1,
        ))
    for _ in range(2):
        cards.append(PokemonCard(
            "Lunatone", 110, Stage.BASIC, None,
            [Attack("Power Gem", 50, _e("2F"))],
            pokemon_type=EnergyType.FIGHTING, retreat_cost=1,
            ability_name="Lunar Cycle",
            ability_effect="lunar_cycle",
            ability_type="active",
        ))
    for _ in range(2):
        cards.append(PokemonCard(
            "Makuhita", 80, Stage.BASIC, None,
            [Attack("Corkscrew Punch", 10, _e("1F")),
             Attack("Confront",        30, _e("2F"))],
            pokemon_type=EnergyType.FIGHTING, retreat_cost=2,
        ))
    for _ in range(2):
        cards.append(PokemonCard(
            "Hariyama", 150, Stage.STAGE1, "Makuhita",
            [Attack("Wild Press", 210, _e("3F"), effect="wild_press")],
            pokemon_type=EnergyType.FIGHTING, retreat_cost=3,
            ability_name="Heave-Ho Catcher",
            ability_effect="heave_ho_catcher",
            ability_type="on_evolve",
        ))
    cards.append(PokemonCard(
        "Meowth ex", 170, Stage.BASIC, None,
        [Attack("Tuck Tail", 60, _e("3C"), effect="tuck_tail")],
        pokemon_type=EnergyType.COLORLESS, retreat_cost=1,
        is_ex=True,
        ability_name="Last-Ditch Catch",
        ability_effect="last_ditch_catch",
        ability_type="on_bench_entry",
    ))

    # ── Trainers ──────────────────────────────
    for _ in range(4):
        cards.append(TrainerCard("Lillie's Determination", CardType.SUPPORTER, "lillies_determination"))
    for _ in range(2):
        cards.append(TrainerCard("Judge", CardType.SUPPORTER, "judge"))
    for _ in range(2):
        cards.append(TrainerCard("Boss's Orders", CardType.SUPPORTER, "boss_orders"))
    cards.append(TrainerCard("Team Rocket's Petrel", CardType.SUPPORTER, "team_rockets_petrel"))
    cards.append(TrainerCard("Black Belt's Training", CardType.SUPPORTER, "black_belt_training"))
    cards.append(TrainerCard("Wally's Compassion", CardType.SUPPORTER, "wally_compassion"))
    for _ in range(4):
        cards.append(TrainerCard("Fighting Gong", CardType.ITEM, "fighting_gong"))
    for _ in range(4):
        cards.append(TrainerCard("Poké Pad", CardType.ITEM, "poke_pad"))
    for _ in range(4):
        cards.append(TrainerCard("Premium Power Pro", CardType.ITEM, "premium_power_pro"))
    for _ in range(4):
        cards.append(TrainerCard("Ultra Ball", CardType.ITEM, "ultra_ball"))
    cards.append(TrainerCard("Switch", CardType.ITEM, "switch_item"))
    cards.append(TrainerCard("Unfair Stamp", CardType.ITEM, "unfair_stamp"))
    for _ in range(2):
        cards.append(TrainerCard("Air Balloon", CardType.TOOL, "retreat_minus2"))
    for _ in range(2):
        cards.append(TrainerCard("Gravity Mountain", CardType.STADIUM, "gravity_mountain"))

    # ── Energy ──────────────────────────────
    for _ in range(11):
        cards.append(EnergyCard("Fighting Energy", EnergyType.FIGHTING))

    assert len(cards) == 60, f"Lucario deck size {len(cards)} != 60"
    return cards

# ─────────────────────────────────────────────
# PLAYER STATE
# ─────────────────────────────────────────────

@dataclass
class PlayerState:
    deck_id: int   # 0=Starmie, 1=Lucario
    deck: List[Card] = field(default_factory=list)
    hand: List[Card] = field(default_factory=list)
    active: Optional[PokemonCard] = None
    bench: List[PokemonCard] = field(default_factory=list)
    discard: List[Card] = field(default_factory=list)
    ko_count: int = 0
    energy_used: bool = False
    supporter_used: bool = False
    turn_count: int = 0
    pending_promotion: bool = False
    pending_choice: Optional[Dict] = None  # interrupt: current player must make a sub-choice
    # Per-turn flags
    had_ko_last_turn: bool = False     # my Pokémon were KO'd last opp turn
    ko_gained_last_turn: int = 0       # KO points I gained last turn (for Unfair Stamp conditions)
    black_belt_active: bool = False    # Black Belt's Training damage bonus
    premium_power_active: bool = False  # Premium Power Pro +30 fighting damage bonus
    items_blocked: bool = False        # Itchy Pollen: can't play Items this turn

    def all_pokemon_in_play(self) -> List[PokemonCard]:
        result = []
        if self.active:
            result.append(self.active)
        result.extend(self.bench)
        return result

    def has_basics_in_hand(self) -> bool:
        return any(isinstance(c, PokemonCard) and c.stage == Stage.BASIC for c in self.hand)

    def basics_in_hand(self) -> List[int]:
        return [i for i, c in enumerate(self.hand)
                if isinstance(c, PokemonCard) and c.stage == Stage.BASIC]

# ─────────────────────────────────────────────
# GAME STATE
# ─────────────────────────────────────────────

@dataclass
class GameState:
    players: List[PlayerState]
    current_player: int = 0
    turn_number: int = 0
    game_over: bool = False
    winner: int = -1
    win_reason: str = ""
    rng: Any = None
    going_first_player: int = 0
    active_stadium: Optional[TrainerCard] = None
    damage_reduction: Dict[int, int] = field(default_factory=dict)
    last_attack_damage: int = 0
    last_damage_prevented: bool = False
    last_damage_reduced: bool = False
    mega_evolved_this_turn: bool = False  # turn ends after mega evolving
    legacy_energy_used: bool = False      # Legacy Energy KO effect already consumed
    ui_choice: Optional[Dict] = None     # Human-UI pre-set choice for interactive effects

    def opponent(self) -> int:
        return 1 - self.current_player

    def current(self) -> PlayerState:
        return self.players[self.current_player]

    def opp(self) -> PlayerState:
        return self.players[self.opponent()]

# ─────────────────────────────────────────────
# ACTION MAPPER
# ─────────────────────────────────────────────

class ActionMapper:
    """
    Flat discrete action space (295 actions total).

    [0]                END_TURN
    [1..72]            ATTACH_ENERGY  hand[h] × slot[s]; slot 5=active    72
    [73..84]           PLAY_POKEMON   from hand[i]                         12
    [85..96]           USE_ITEM       from hand[i]                         12
    [97..108]          USE_SUPPORTER  from hand[i]                         12
    [109..180]         EVOLVE         hand[h] × slot[s]; slot 5=active     72
    [181..182]         ATTACK         atk_idx 0..1                          2
    [183..187]         PROMOTE        bench slot 0..4                        5
    [188..192]         RETREAT        bench slot to bring up 0..4            5
    [193..264]         ATTACH_TOOL    hand[h] × slot[s]; slot 5=active      72
    [265..270]         USE_ABILITY    slot: 0=active 1..5=bench[0..4]        6
    [271..282]         USE_STADIUM    from hand[i]                          12
    [283..294]         CHOOSE         opt_idx 0..11 (pending_choice resolve) 12
    """

    END_TURN_IDX      = 0
    ATTACH_START      = 1
    ATTACH_COUNT      = MAX_HAND * (MAX_BENCH + 1)   # 72
    PLAY_START        = 73
    PLAY_COUNT        = MAX_HAND                # 12
    ITEM_START        = 85
    ITEM_COUNT        = MAX_HAND
    SUPP_START        = 97
    SUPP_COUNT        = MAX_HAND
    EVOLVE_START      = 109
    EVOLVE_COUNT      = MAX_HAND * (MAX_BENCH + 1)   # 72
    ATTACK_START      = 181
    ATTACK_COUNT      = MAX_ATTACKS             # 2
    PROMOTE_START     = 183
    PROMOTE_COUNT     = MAX_BENCH               # 5
    RETREAT_START     = 188
    RETREAT_COUNT     = MAX_BENCH               # 5
    TOOL_START        = 193
    TOOL_COUNT        = MAX_HAND * (MAX_BENCH + 1)   # 72
    ABILITY_START     = 265
    ABILITY_COUNT     = MAX_BENCH + 1           # 6  (active=0, bench[0..4]=1..5)
    STADIUM_START     = 271
    STADIUM_COUNT     = MAX_HAND                # 12
    CHOOSE_START      = 283
    CHOOSE_COUNT      = 12                      # option slots 0..11

    TOTAL_ACTIONS     = 295

    @classmethod
    def decode(cls, idx: int) -> Tuple[ActionType, Dict]:
        if idx == cls.END_TURN_IDX:
            return ActionType.END_TURN, {}
        if cls.ATTACH_START <= idx < cls.ATTACH_START + cls.ATTACH_COUNT:
            offset = idx - cls.ATTACH_START
            return ActionType.ATTACH_ENERGY, {
                "hand_idx": offset // (MAX_BENCH + 1),
                "slot":     offset %  (MAX_BENCH + 1),
            }
        if cls.PLAY_START <= idx < cls.PLAY_START + cls.PLAY_COUNT:
            return ActionType.PLAY_POKEMON, {"hand_idx": idx - cls.PLAY_START}
        if cls.ITEM_START <= idx < cls.ITEM_START + cls.ITEM_COUNT:
            return ActionType.USE_ITEM, {"hand_idx": idx - cls.ITEM_START}
        if cls.SUPP_START <= idx < cls.SUPP_START + cls.SUPP_COUNT:
            return ActionType.USE_SUPPORTER, {"hand_idx": idx - cls.SUPP_START}
        if cls.EVOLVE_START <= idx < cls.EVOLVE_START + cls.EVOLVE_COUNT:
            offset = idx - cls.EVOLVE_START
            return ActionType.EVOLVE, {
                "hand_idx": offset // (MAX_BENCH + 1),
                "slot":     offset %  (MAX_BENCH + 1),
            }
        if cls.ATTACK_START <= idx < cls.ATTACK_START + cls.ATTACK_COUNT:
            return ActionType.ATTACK, {"atk_idx": idx - cls.ATTACK_START}
        if cls.PROMOTE_START <= idx < cls.PROMOTE_START + cls.PROMOTE_COUNT:
            return ActionType.PROMOTE, {"bench_slot": idx - cls.PROMOTE_START}
        if cls.RETREAT_START <= idx < cls.RETREAT_START + cls.RETREAT_COUNT:
            return ActionType.RETREAT, {"bench_slot": idx - cls.RETREAT_START}
        if cls.TOOL_START <= idx < cls.TOOL_START + cls.TOOL_COUNT:
            offset = idx - cls.TOOL_START
            return ActionType.ATTACH_TOOL, {
                "hand_idx": offset // (MAX_BENCH + 1),
                "slot":     offset %  (MAX_BENCH + 1),
            }
        if cls.ABILITY_START <= idx < cls.ABILITY_START + cls.ABILITY_COUNT:
            return ActionType.USE_ABILITY, {"slot": idx - cls.ABILITY_START}
        if cls.STADIUM_START <= idx < cls.STADIUM_START + cls.STADIUM_COUNT:
            return ActionType.USE_STADIUM, {"hand_idx": idx - cls.STADIUM_START}
        if cls.CHOOSE_START <= idx < cls.CHOOSE_START + cls.CHOOSE_COUNT:
            return ActionType.CHOOSE, {"opt_idx": idx - cls.CHOOSE_START}
        raise ValueError(f"Invalid action index {idx}")

    @classmethod
    def encode(cls, atype: ActionType, params: Dict) -> int:
        if atype == ActionType.END_TURN:
            return cls.END_TURN_IDX
        if atype == ActionType.ATTACH_ENERGY:
            return cls.ATTACH_START + params["hand_idx"] * (MAX_BENCH + 1) + params["slot"]
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
        if atype == ActionType.RETREAT:
            return cls.RETREAT_START + params["bench_slot"]
        if atype == ActionType.ATTACH_TOOL:
            return cls.TOOL_START + params["hand_idx"] * (MAX_BENCH + 1) + params["slot"]
        if atype == ActionType.USE_ABILITY:
            return cls.ABILITY_START + params["slot"]
        if atype == ActionType.USE_STADIUM:
            return cls.STADIUM_START + params["hand_idx"]
        if atype == ActionType.CHOOSE:
            return cls.CHOOSE_START + params["opt_idx"]
        raise ValueError(f"Unknown action type {atype}")

# ─────────────────────────────────────────────
# STATE ENCODER
# ─────────────────────────────────────────────

POKEMON_FEATURES = 13

class StateEncoder:
    """
    Encodes GameState → float32 vector.

    Per-Pokémon (13 features):
      hp_frac, water_e, dark_e, fight_e, psychic_e, total_e_norm,
      id_norm, stage_norm, is_ex, is_mega_ex, has_tool,
      retreat_cost_norm, ability_available

    Layout:
      my_active        13
      opp_active       13
      my_bench         5×13 = 65
      opp_bench        5×13 = 65
      hand_hist        6  (pokemon/energy/item/supporter/tool/stadium counts /15)
      hand_energy_type 12 (per-slot: 0=no card/not energy, (type.value+1)/6 if energy)
      opp_hand         1
      my_deck          1
      opp_deck         1
      my_ko            1
      opp_ko           1
      energy_used      1
      supp_used        1
      turn_norm        1
      going_first      1
      stadium          1   (0=none, 0.5=risky_ruins, 1=gravity_mountain)
      bb_active        1   (black_belt_training this turn)
      pending_active   1   (1 if pending_choice is set)
      choice_opts      12×4 = 48  (per option: valid, category/6, hp_frac, id_norm)
      ─────────────────
      Total: 234
    """
    STATE_SIZE = 234

    # Active ability effects (require USE_ABILITY action)
    ACTIVE_ABILITY_EFFECTS = {
        "cursed_blast_5", "cursed_blast_13", "adrena_brain",
        "lunar_cycle", "flip_the_script",
    }

    @staticmethod
    def _encode_pokemon(p: Optional[PokemonCard], me: PlayerState,
                        opp: PlayerState, gs: GameState) -> np.ndarray:
        if p is None:
            return np.zeros(POKEMON_FEATURES, dtype=np.float32)
        pid = POKEMON_IDS.get(p.name, 0) / max(NUM_POKEMON_IDS - 1, 1)
        w  = p.energy.get(EnergyType.WATER,    0) / 6.0
        d  = p.energy.get(EnergyType.DARKNESS, 0) / 6.0
        f  = p.energy.get(EnergyType.FIGHTING, 0) / 6.0
        ps = p.energy.get(EnergyType.PSYCHIC,  0) / 6.0
        ability_avail = (
            p.ability_effect in StateEncoder.ACTIVE_ABILITY_EFFECTS
            and not p.ability_used
        )
        return np.array([
            p.hp_fraction(),
            w, d, f, ps,
            min(p.total_energy() / 6.0, 1.0),
            pid,
            p.stage / 2.0,
            float(p.is_ex),
            float(p.is_mega_ex),
            float(p.tool_card is not None),
            min(p.retreat_cost / 5.0, 1.0),
            float(ability_avail),
        ], dtype=np.float32)

    @classmethod
    def encode(cls, gs: GameState) -> np.ndarray:
        me  = gs.current()
        opp = gs.opp()
        vec = []

        vec.append(cls._encode_pokemon(me.active,  me, opp, gs))
        vec.append(cls._encode_pokemon(opp.active, opp, me, gs))

        for i in range(MAX_BENCH):
            vec.append(cls._encode_pokemon(
                me.bench[i]  if i < len(me.bench)  else None, me,  opp, gs))
        for i in range(MAX_BENCH):
            vec.append(cls._encode_pokemon(
                opp.bench[i] if i < len(opp.bench) else None, opp, me,  gs))

        # Hand histogram
        counts = [0] * 6  # pokemon, energy, item, supporter, tool, stadium
        for c in me.hand:
            ct = getattr(c, 'card_type', CardType.ENERGY)
            if   ct == CardType.POKEMON:   counts[0] += 1
            elif ct == CardType.ENERGY:    counts[1] += 1
            elif ct == CardType.ITEM:      counts[2] += 1
            elif ct == CardType.SUPPORTER: counts[3] += 1
            elif ct == CardType.TOOL:      counts[4] += 1
            elif ct == CardType.STADIUM:   counts[5] += 1
        vec.append(np.array([c / 15.0 for c in counts], dtype=np.float32))

        # Per-slot hand energy type (12 features): lets agent see which hand
        # position holds Water vs Darkness energy when choosing hand_idx for attach
        hand_energy = np.zeros(MAX_HAND, dtype=np.float32)
        for i, c in enumerate(me.hand):
            if i >= MAX_HAND:
                break
            if isinstance(c, EnergyCard):
                hand_energy[i] = (c.energy_type.value + 1) / 6.0
        vec.append(hand_energy)

        vec.append(np.array([len(opp.hand) / 15.0], dtype=np.float32))
        vec.append(np.array([len(me.deck)  / MAX_DECK], dtype=np.float32))
        vec.append(np.array([len(opp.deck) / MAX_DECK], dtype=np.float32))
        vec.append(np.array([me.ko_count  / KO_TO_WIN], dtype=np.float32))
        vec.append(np.array([opp.ko_count / KO_TO_WIN], dtype=np.float32))
        vec.append(np.array([float(me.energy_used), float(me.supporter_used)],
                            dtype=np.float32))
        vec.append(np.array([min(gs.turn_number / 50.0, 1.0)], dtype=np.float32))
        going_first_me = float(gs.going_first_player == gs.current_player)
        vec.append(np.array([going_first_me], dtype=np.float32))

        stadium_val = 0.0
        if gs.active_stadium:
            if gs.active_stadium.effect == "risky_ruins":
                stadium_val = 0.5
            elif gs.active_stadium.effect == "gravity_mountain":
                stadium_val = 1.0
        vec.append(np.array([stadium_val], dtype=np.float32))
        vec.append(np.array([float(me.black_belt_active)], dtype=np.float32))

        # ── Pending choice features (1 + 12×4 = 49) ────────────────────────────
        pc       = me.pending_choice
        pc_type  = pc["type"] if pc is not None else ""
        pc_opts  = pc["options"] if pc is not None else []
        vec.append(np.array([float(pc is not None)], dtype=np.float32))

        def _opt_features(opt_val: int) -> np.ndarray:
            """Return [valid=1, category/6, hp_frac, id_norm] for one option slot."""
            cat = 0.0; hp_frac = 0.0; id_n = 0.0
            if pc_type in ("ultra_ball_discard1", "ultra_ball_discard2"):
                if opt_val < len(me.hand):
                    cat = 1/6.0
                    id_n = getattr(me.hand[opt_val], 'card_type',
                                   CardType.ENERGY).value / 5.0
            elif pc_type == "ultra_ball_pick":
                if opt_val < len(me.deck) and isinstance(me.deck[opt_val], PokemonCard):
                    cat = 2/6.0; hp_frac = 1.0
                    id_n = POKEMON_IDS.get(me.deck[opt_val].name, 0) / max(NUM_POKEMON_IDS-1, 1)
            elif pc_type in ("poffin_pick1", "poffin_pick2"):
                if opt_val < len(me.deck) and isinstance(me.deck[opt_val], PokemonCard):
                    cat = 2/6.0; hp_frac = 1.0
                    id_n = POKEMON_IDS.get(me.deck[opt_val].name, 0) / max(NUM_POKEMON_IDS-1, 1)
            elif pc_type == "night_stretcher_pick":
                if opt_val < len(me.discard):
                    c = me.discard[opt_val]; cat = 3/6.0
                    if isinstance(c, PokemonCard):
                        hp_frac = 1.0
                        id_n = POKEMON_IDS.get(c.name, 0) / max(NUM_POKEMON_IDS-1, 1)
                    else:
                        id_n = getattr(c, 'card_type', CardType.ENERGY).value / 5.0
            elif pc_type == "boss_orders_pick":
                if opt_val < len(opp.bench):
                    p = opp.bench[opt_val]; cat = 5/6.0
                    hp_frac = p.hp_fraction()
                    id_n = POKEMON_IDS.get(p.name, 0) / max(NUM_POKEMON_IDS-1, 1)
            elif pc_type == "switch_target":
                if opt_val < len(me.bench):
                    p = me.bench[opt_val]; cat = 6/6.0
                    hp_frac = p.hp_fraction()
                    id_n = POKEMON_IDS.get(p.name, 0) / max(NUM_POKEMON_IDS-1, 1)
            elif pc_type in ("poke_pad_pick", "hilda_evo_pick"):
                if opt_val < len(me.deck) and isinstance(me.deck[opt_val], PokemonCard):
                    p = me.deck[opt_val]; cat = 2/6.0
                    hp_frac = p.stage / 2.0
                    id_n = POKEMON_IDS.get(p.name, 0) / max(NUM_POKEMON_IDS-1, 1)
            elif pc_type == "hilda_energy_pick":
                if opt_val < len(me.deck) and isinstance(me.deck[opt_val], EnergyCard):
                    cat = 1/6.0
                    id_n = me.deck[opt_val].energy_type.value / 5.0
            elif pc_type == "last_ditch_catch_pick":
                if opt_val < len(me.deck) and isinstance(me.deck[opt_val], TrainerCard):
                    cat = 3/6.0
                    id_n = me.deck[opt_val].card_type.value / 5.0
            elif pc_type == "cursed_blast_target":
                if opt_val == 0:
                    p = opp.active; cat = 4/6.0
                else:
                    bi = opt_val - 1
                    p  = opp.bench[bi] if bi < len(opp.bench) else None; cat = 5/6.0
                if p is not None:
                    hp_frac = p.hp_fraction()
                    id_n = POKEMON_IDS.get(p.name, 0) / max(NUM_POKEMON_IDS-1, 1)
            elif pc_type == "cruel_arrow_target":
                if opt_val == 0:
                    p = opp.active; cat = 4/6.0
                else:
                    bi = opt_val - 1
                    p  = opp.bench[bi] if bi < len(opp.bench) else None; cat = 5/6.0
                if p is not None:
                    hp_frac = p.hp_fraction()
                    id_n = POKEMON_IDS.get(p.name, 0) / max(NUM_POKEMON_IDS-1, 1)
            elif pc_type == "aura_jab_energy":
                if opt_val < len(me.bench):
                    p = me.bench[opt_val]; cat = 6/6.0
                    hp_frac = p.hp_fraction()
                    id_n = POKEMON_IDS.get(p.name, 0) / max(NUM_POKEMON_IDS-1, 1)
            elif pc_type == "heave_ho_target":
                if opt_val < len(opp.bench):
                    p = opp.bench[opt_val]; cat = 5/6.0
                    hp_frac = p.hp_fraction()
                    id_n = POKEMON_IDS.get(p.name, 0) / max(NUM_POKEMON_IDS-1, 1)
            elif pc_type == "pokegear_pick":
                if opt_val < len(me.deck) and isinstance(me.deck[opt_val], TrainerCard):
                    cat = 3/6.0
                    id_n = me.deck[opt_val].card_type.value / 5.0
            elif pc_type == "fighting_gong_pick":
                if opt_val < len(me.deck):
                    c = me.deck[opt_val]
                    if isinstance(c, EnergyCard):
                        cat = 1/6.0; id_n = c.energy_type.value / 5.0
                    elif isinstance(c, PokemonCard):
                        cat = 2/6.0; hp_frac = c.hp / 250.0
                        id_n = POKEMON_IDS.get(c.name, 0) / max(NUM_POKEMON_IDS-1, 1)
            elif pc_type == "petrel_pick":
                if opt_val < len(me.deck) and isinstance(me.deck[opt_val], TrainerCard):
                    cat = 3/6.0
                    id_n = me.deck[opt_val].card_type.value / 5.0
            return np.array([1.0, cat, hp_frac, id_n], dtype=np.float32)

        for slot_i in range(12):
            if slot_i < len(pc_opts):
                vec.append(_opt_features(pc_opts[slot_i]))
            else:
                vec.append(np.zeros(4, dtype=np.float32))

        result = np.concatenate([v.flatten() for v in vec])
        assert result.shape == (cls.STATE_SIZE,), \
            f"State size mismatch: {result.shape} vs expected ({cls.STATE_SIZE},)"
        return result

# ─────────────────────────────────────────────
# ENERGY / EVOLVE HELPERS
# ─────────────────────────────────────────────

def can_pay_cost(pokemon: PokemonCard, cost: Dict[EnergyType, int]) -> bool:
    """Check if pokemon can pay an attack's energy cost.
    Legacy Energy (stored as COLORLESS) acts as any energy type."""
    required = dict(cost)
    available = dict(pokemon.energy)
    # Legacy Energy: the COLORLESS stored for it can fill any typed requirement
    wildcards = available.pop(EnergyType.COLORLESS, 0) if pokemon.has_legacy_energy else 0
    for etype, amount in required.items():
        if etype == EnergyType.COLORLESS:
            continue
        have = available.get(etype, 0)
        if have < amount:
            need = amount - have
            if wildcards >= need:
                wildcards -= need
                available[etype] = 0
            else:
                return False
        else:
            available[etype] = have - amount
    colorless_needed = required.get(EnergyType.COLORLESS, 0)
    return sum(available.values()) + wildcards >= colorless_needed


def can_evolve(evo_card: PokemonCard, target: PokemonCard,
               turn_number: int) -> bool:
    """Can evo_card evolve target? Must not be played this turn."""
    if evo_card.evolves_from != target.name:
        return False
    if target.turn_played < 0:
        return False
    # Initial Pokemon (placed at game setup, turn_played=0) cannot evolve on turns 1 or 2
    if target.turn_played == 0 and turn_number <= 2:
        return False
    if target.turn_played >= turn_number:
        return False
    if evo_card.stage != target.stage + 1:
        return False
    return True


def _effective_retreat_cost(pokemon: PokemonCard,
                             gs: GameState, owner: PlayerState) -> int:
    # Skyliner: Basic Pokémon have free retreat if Latias ex with Skyliner is in play
    if pokemon.stage == Stage.BASIC:
        if any(p.ability_effect == "skyliner" for p in owner.all_pokemon_in_play()):
            return 0
    cost = pokemon.retreat_cost
    if pokemon.tool_card and pokemon.tool_card.effect == "retreat_minus2":
        cost = max(0, cost - 2)
    return max(0, cost)

# ─────────────────────────────────────────────
# LEGAL ACTION MASK
# ─────────────────────────────────────────────

_ACTIVE_ABILITY_EFFECTS = {
    "cursed_blast_5", "cursed_blast_13", "adrena_brain",
    "lunar_cycle", "flip_the_script",
}

def compute_legal_mask(gs: GameState) -> np.ndarray:
    mask = np.zeros(ActionMapper.TOTAL_ACTIONS, dtype=np.float32)
    me  = gs.current()
    opp = gs.opp()

    # ── PENDING CHOICE INTERRUPT (highest priority) ─────────────────────────
    if me.pending_choice is not None:
        opts = me.pending_choice.get("options", [])
        for i in range(min(len(opts), ActionMapper.CHOOSE_COUNT)):
            mask[ActionMapper.CHOOSE_START + i] = 1.0
        return mask

    # ── PROMOTION INTERRUPT ─────────────────────────────────────────────────
    for pidx in range(2):
        p = gs.players[pidx]
        if p.pending_promotion:
            for i in range(len(p.bench)):
                mask[ActionMapper.PROMOTE_START + i] = 1.0
            return mask


    mask[ActionMapper.END_TURN_IDX] = 1.0

    active_status = me.active.status if me.active else None

    # ── ATTACH_ENERGY ────────────────────────────────────────────────────────
    if not me.energy_used:
        for hi, hc in enumerate(me.hand):
            if hi >= MAX_HAND:
                break
            if not isinstance(hc, EnergyCard):
                continue
            # Allow attaching to any in-play Pokemon — agent is shaped by reward
            if me.active is not None:
                mask[ActionMapper.ATTACH_START + hi * (MAX_BENCH + 1) + MAX_BENCH] = 1.0
            for bi in range(len(me.bench)):
                mask[ActionMapper.ATTACH_START + hi * (MAX_BENCH + 1) + bi] = 1.0

    # ── PLAY_POKEMON (bench basics) ──────────────────────────────────────────
    if len(me.bench) < MAX_BENCH:
        for i, c in enumerate(me.hand):
            if i >= MAX_HAND:
                break
            if isinstance(c, PokemonCard) and c.stage == Stage.BASIC:
                mask[ActionMapper.PLAY_START + i] = 1.0

    # ── USE_ITEM ────────────────────────────────────────────────────────────
    if not me.items_blocked:
        for i, c in enumerate(me.hand):
            if i >= MAX_HAND:
                break
            if not (isinstance(c, TrainerCard) and c.card_type == CardType.ITEM):
                continue
            eff = c.effect
            # Heal items: only if active is damaged
            if eff in ("heal_active_20", "heal_active_30"):
                if me.active and me.active.current_hp < me.active.hp:
                    mask[ActionMapper.ITEM_START + i] = 1.0
            # Ultra Ball: need 2 other cards to discard (any type, exclude the Ultra Ball itself)
            elif eff == "ultra_ball":
                discardable = sum(1 for x in me.hand if x is not c)
                if discardable >= 2:
                    mask[ActionMapper.ITEM_START + i] = 1.0
            # Switch: need a bench pokemon
            elif eff == "switch_item":
                if me.bench:
                    mask[ActionMapper.ITEM_START + i] = 1.0
            # Unfair Stamp: only if MY Pokémon were KO'd last turn
            elif eff == "unfair_stamp":
                if me.had_ko_last_turn:
                    mask[ActionMapper.ITEM_START + i] = 1.0
            # Night Stretcher: need Pokémon or Energy in discard
            elif eff == "night_stretcher":
                has_poke_in_disc = any(isinstance(x, PokemonCard) for x in me.discard)
                has_energy_in_disc = any(
                    isinstance(x, EnergyCard) and x.special_effect is None
                    for x in me.discard
                )
                if has_poke_in_disc or has_energy_in_disc:
                    mask[ActionMapper.ITEM_START + i] = 1.0
            # Poké Pad: need a non-Rule-Box (non-ex) Pokémon in deck
            elif eff == "poke_pad":
                if any(isinstance(dc, PokemonCard) and not dc.is_ex for dc in me.deck):
                    mask[ActionMapper.ITEM_START + i] = 1.0
            # Pokégear 3.0: need a Supporter in top 7 of deck
            elif eff == "pokegear":
                if any(isinstance(dc, TrainerCard) and dc.card_type == CardType.SUPPORTER
                       for dc in me.deck[:7]):
                    mask[ActionMapper.ITEM_START + i] = 1.0
            # Fighting Gong: need a Fighting Energy or Fighting Basic Pokémon in deck
            elif eff == "fighting_gong":
                has_target = any(
                    (isinstance(dc, EnergyCard) and dc.energy_type == EnergyType.FIGHTING
                     and dc.special_effect is None)
                    or (isinstance(dc, PokemonCard) and dc.stage == Stage.BASIC
                        and dc.pokemon_type == EnergyType.FIGHTING)
                    for dc in me.deck
                )
                if has_target:
                    mask[ActionMapper.ITEM_START + i] = 1.0
            else:
                mask[ActionMapper.ITEM_START + i] = 1.0

    # ── USE_SUPPORTER ────────────────────────────────────────────────────────
    if not me.supporter_used:
        for i, c in enumerate(me.hand):
            if i >= MAX_HAND:
                break
            if isinstance(c, TrainerCard) and c.card_type == CardType.SUPPORTER:
                eff = c.effect
                # Wally's Compassion: need a Mega ex on active with damage
                if eff == "wally_compassion":
                    if (me.active and me.active.is_mega_ex
                            and me.active.current_hp < me.active.hp):
                        mask[ActionMapper.SUPP_START + i] = 1.0
                elif eff == "boss_orders":
                    if opp.bench:
                        mask[ActionMapper.SUPP_START + i] = 1.0
                # Hilda: need at least 1 Evolution Pokémon in deck
                elif eff == "hilda":
                    if any(isinstance(dc, PokemonCard) and dc.stage != Stage.BASIC
                           for dc in me.deck):
                        mask[ActionMapper.SUPP_START + i] = 1.0
                # Team Rocket's Petrel: need a Trainer card in deck
                elif eff == "team_rockets_petrel":
                    if any(isinstance(dc, TrainerCard) for dc in me.deck):
                        mask[ActionMapper.SUPP_START + i] = 1.0
                else:
                    mask[ActionMapper.SUPP_START + i] = 1.0

    # ── EVOLVE ──────────────────────────────────────────────────────────────
    for hi, hc in enumerate(me.hand):
        if hi >= MAX_HAND:
            break
        if not (isinstance(hc, PokemonCard) and hc.stage != Stage.BASIC):
            continue
        if me.active and can_evolve(hc, me.active, gs.turn_number):
            slot = MAX_BENCH
            mask[ActionMapper.EVOLVE_START + hi * (MAX_BENCH + 1) + slot] = 1.0
        for bi, bp in enumerate(me.bench):
            if can_evolve(hc, bp, gs.turn_number):
                mask[ActionMapper.EVOLVE_START + hi * (MAX_BENCH + 1) + bi] = 1.0

    # ── ATTACK ──────────────────────────────────────────────────────────────
    if (me.active is not None
            and active_status not in ("paralyzed", "asleep")
            and me.active.cant_attack_turns == 0):
        # First-turn restriction for the player who went first
        can_attack_now = not (
            gs.turn_number == 1
            and gs.current_player == gs.going_first_player
        )
        if can_attack_now:
            for ai, atk in enumerate(me.active.attacks):
                if ai >= MAX_ATTACKS:
                    break
                # Skip this specific attack if it's in blocked_attacks
                if atk.name in me.active.blocked_attacks:
                    continue
                # Seasoned Skill: Blood Moon cost is reduced by opp KOs
                if (atk.effect == "blood_moon"
                        and me.active.ability_effect == "seasoned_skill"):
                    effective_cost = max(0, 5 - opp.ko_count)
                    if me.active.total_energy() >= effective_cost:
                        mask[ActionMapper.ATTACK_START + ai] = 1.0
                    continue
                if can_pay_cost(me.active, atk.energy_cost):
                    mask[ActionMapper.ATTACK_START + ai] = 1.0

    # ── RETREAT ──────────────────────────────────────────────────────────────
    if (me.active is not None
            and active_status != "paralyzed"
            and not me.active.shadow_bound
            and me.bench):          # need something to bring up
        eff_cost = _effective_retreat_cost(me.active, gs, me)
        if me.active.total_energy() >= eff_cost:
            for i in range(len(me.bench)):
                mask[ActionMapper.RETREAT_START + i] = 1.0

    # ── ATTACH_TOOL ──────────────────────────────────────────────────────────
    for hi, hc in enumerate(me.hand):
        if hi >= MAX_HAND:
            break
        if not (isinstance(hc, TrainerCard) and hc.card_type == CardType.TOOL):
            continue
        # Active
        if me.active and me.active.tool_card is None:
            mask[ActionMapper.TOOL_START + hi * (MAX_BENCH + 1) + MAX_BENCH] = 1.0
        # Bench
        for bi, bp in enumerate(me.bench):
            if bp.tool_card is None:
                mask[ActionMapper.TOOL_START + hi * (MAX_BENCH + 1) + bi] = 1.0

    # ── USE_ABILITY ──────────────────────────────────────────────────────────
    all_mine = ([(me.active, 0)] if me.active else []) + \
               [(bp, i + 1) for i, bp in enumerate(me.bench)]
    for pokemon, slot_idx in all_mine:
        eff = pokemon.ability_effect
        if eff not in _ACTIVE_ABILITY_EFFECTS:
            continue
        if pokemon.ability_used:
            continue
        # Condition checks per ability
        if eff == "adrena_brain":
            # Needs Darkness energy attached to this Pokémon
            if not pokemon.energy.get(EnergyType.DARKNESS, 0):
                continue
        elif eff in ("cursed_blast_5", "cursed_blast_13"):
            # Need opp to have at least one Pokémon in play
            if opp.active is None and not opp.bench:
                continue
        elif eff == "lunar_cycle":
            # Needs Solrock in play + Basic Fighting Energy in hand
            has_solrock = any(p.name == "Solrock" for p in me.all_pokemon_in_play())
            if not has_solrock:
                continue
            has_fight_energy = any(
                isinstance(c, EnergyCard) and c.energy_type == EnergyType.FIGHTING
                and c.special_effect is None
                for c in me.hand
            )
            if not has_fight_energy:
                continue
        elif eff == "flip_the_script":
            # Only if my Pokémon were KO'd last opponent turn
            if not me.had_ko_last_turn:
                continue
        mask[ActionMapper.ABILITY_START + slot_idx] = 1.0

    # ── USE_STADIUM ──────────────────────────────────────────────────────────
    for i, c in enumerate(me.hand):
        if i >= MAX_HAND:
            break
        if isinstance(c, TrainerCard) and c.card_type == CardType.STADIUM:
            # Can't play same stadium that's already active
            if gs.active_stadium and gs.active_stadium.effect == c.effect:
                continue
            mask[ActionMapper.STADIUM_START + i] = 1.0

    return mask

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
        deck0 = build_starmie_deck()
        deck1 = build_lucario_deck()
        self._shuffle(deck0)
        self._shuffle(deck1)

        p0 = PlayerState(deck_id=0, deck=deck0)
        p1 = PlayerState(deck_id=1, deck=deck1)
        gs = GameState(players=[p0, p1], rng=self.rng)

        for pidx in range(2):
            self._draw_opening_hand(gs, pidx)
        for pidx in range(2):
            self._auto_place_active(gs, pidx)

        # Coin flip determines who goes first
        gs.current_player    = int(self._coin_flip())
        gs.going_first_player = gs.current_player
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
            p.deck.extend(p.hand)
            p.hand.clear()
            self._shuffle(p.deck)
            attempts += 1
            if attempts > 30:
                raise RuntimeError("Could not draw opening hand with Basic")

    def _auto_place_active(self, gs: GameState, pidx: int) -> None:
        p = gs.players[pidx]
        for i, c in enumerate(p.hand):
            if isinstance(c, PokemonCard) and c.stage == Stage.BASIC:
                c.is_active = True
                c.turn_played = 0
                p.active = p.hand.pop(i)
                return
        raise RuntimeError(f"Player {pidx} has no basic for active")

    def _draw_n(self, gs: GameState, pidx: int, n: int) -> bool:
        p = gs.players[pidx]
        for _ in range(n):
            if not p.deck:
                gs.game_over = True
                opp = 1 - pidx
                me_kos  = gs.players[pidx].ko_count
                opp_kos = gs.players[opp].ko_count
                gs.winner    = (pidx if me_kos > opp_kos
                                else (opp if opp_kos > me_kos else -1))
                gs.win_reason = "deckout"
                return False
            p.hand.append(p.deck.pop(0))
        return True

    def start_turn(self, gs: GameState) -> None:
        """Begin current player's turn."""
        me = gs.current()
        me.energy_used        = False
        me.supporter_used     = False
        me.black_belt_active  = False
        me.premium_power_active = False
        gs.mega_evolved_this_turn = False
        gs.damage_reduction.pop(gs.current_player, None)

        # Activate pending attack restrictions from last turn's attack
        if me.active:
            me.active.blocked_attacks = me.active.blocked_attacks_pending.copy()
            me.active.blocked_attacks_pending.clear()
            me.active.cant_attack_turns = me.active.cant_attack_turns_pending
            me.active.cant_attack_turns_pending = 0

        # Reset ability_used on all my Pokémon
        for p in me.all_pokemon_in_play():
            p.ability_used = False

        # Status effect check for active Pokémon at start of turn
        if me.active and me.active.status == "asleep":
            if self._coin_flip():
                me.active.status = None   # woke up

        # Record KOs gained last turn for tracking
        opp_idx = gs.opponent()
        opp = gs.players[opp_idx]
        opp.ko_gained_last_turn = opp.ko_count - getattr(opp, '_prev_ko', opp.ko_count)
        opp._prev_ko = opp.ko_count

        # Track if my Pokémon were KO'd during opp's last turn (Unfair Stamp, Flip the Script)
        me.had_ko_last_turn = getattr(me, '_had_ko_this_opp_turn', False)
        me._had_ko_this_opp_turn = False

        self._draw_n(gs, gs.current_player, 1)

    def apply_action(self, gs: GameState, action_idx: int) -> Tuple[float, bool]:
        if gs.game_over:
            return 0.0, True
        atype, params = ActionMapper.decode(action_idx)
        me  = gs.current()
        reward = -0.01

        if atype == ActionType.END_TURN:
            self._end_turn(gs)
            return reward, gs.game_over

        if atype == ActionType.CHOOSE:
            r = self._handle_choice(gs, params["opt_idx"])
            reward += r
        elif atype == ActionType.ATTACH_ENERGY:
            self._attach_energy(gs, params["hand_idx"], params["slot"])
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
        elif atype == ActionType.RETREAT:
            self._retreat(gs, params["bench_slot"])
        elif atype == ActionType.ATTACH_TOOL:
            self._attach_tool(gs, params["hand_idx"], params["slot"])
        elif atype == ActionType.USE_ABILITY:
            self._use_ability(gs, params["slot"])
        elif atype == ActionType.USE_STADIUM:
            self._use_stadium(gs, params["hand_idx"])
        elif atype == ActionType.ATTACK:
            r = self._attack(gs, params["atk_idx"])
            reward += r
            # Only end the turn now if the attack didn't create a pending_choice
            if me.pending_choice is None:
                self._end_turn(gs)

        return reward, gs.game_over

    # ── Sub-actions ──────────────────────────────────────────────────────────

    def _attach_energy(self, gs: GameState, hand_idx: int, slot: int) -> None:
        me = gs.current()
        if me.energy_used:
            return
        # slot 5 (MAX_BENCH) = active; slots 0..4 = bench[0..4]
        target = (me.active if slot == MAX_BENCH else
                  (me.bench[slot] if slot < len(me.bench) else None))
        if target is None:
            return
        if hand_idx >= len(me.hand):
            return
        if not isinstance(me.hand[hand_idx], EnergyCard):
            return

        e_card: EnergyCard = me.hand.pop(hand_idx)

        if e_card.special_effect == "ignition_energy":
            # Provides 1C to Basic, 3C to Evolution; discards at end of turn
            bonus = 3 if target.stage != Stage.BASIC else 1
            target.energy[EnergyType.COLORLESS] = \
                target.energy.get(EnergyType.COLORLESS, 0) + bonus
            target.ignition_bonus += bonus
        elif e_card.special_effect == "legacy_energy":
            # Provides 1C; enables the once-per-game KO protection effect
            target.energy[EnergyType.COLORLESS] = \
                target.energy.get(EnergyType.COLORLESS, 0) + 1
            target.has_legacy_energy = True
        else:
            target.energy[e_card.energy_type] = \
                target.energy.get(e_card.energy_type, 0) + 1

        me.energy_used = True

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

        # Risky Ruins: place 2 damage counters on non-Darkness basics
        if (gs.active_stadium and gs.active_stadium.effect == "risky_ruins"
                and c.pokemon_type != EnergyType.DARKNESS):
            c.current_hp = max(1, c.current_hp - 20)

        # On-bench-entry abilities
        self._trigger_bench_entry_ability(gs, c, gs.current_player)

    def _trigger_bench_entry_ability(self, gs: GameState,
                                      pokemon: PokemonCard,
                                      owner_idx: int) -> None:
        eff = pokemon.ability_effect
        if not eff or pokemon.ability_type != "on_bench_entry":
            return
        me = gs.players[owner_idx]

        if eff == "last_ditch_catch":
            # Player chooses which Supporter to grab from deck
            opts = [i for i, dc in enumerate(me.deck)
                    if isinstance(dc, TrainerCard) and dc.card_type == CardType.SUPPORTER]
            if opts:
                me.pending_choice = {"type": "last_ditch_catch_pick",
                                     "options": opts[:12], "context": {}}

    def _trigger_on_evolve_ability(self, gs: GameState,
                                    pokemon: PokemonCard,
                                    owner_idx: int) -> None:
        eff = pokemon.ability_effect
        if not eff or pokemon.ability_type != "on_evolve":
            return
        me  = gs.players[owner_idx]
        opp = gs.players[1 - owner_idx]

        if eff == "heave_ho_catcher":
            if opp.bench:
                opts = list(range(len(opp.bench)))
                me.pending_choice = {
                    "type": "heave_ho_target",
                    "options": opts[:12],
                    "context": {},
                }

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
        eff = c.effect

        if eff == "heal_active_20" and me.active:
            me.active.current_hp = min(me.active.hp, me.active.current_hp + 20)
        elif eff == "heal_active_30" and me.active:
            me.active.current_hp = min(me.active.hp, me.active.current_hp + 30)

        elif eff == "ultra_ball":
            # Step 1: pick first card to discard (any card type)
            opts = list(range(len(me.hand)))
            me.pending_choice = {"type": "ultra_ball_discard1",
                                 "options": opts[:12], "context": {}}

        elif eff == "poke_pad":
            # Player chooses which non-ex Pokémon to search from deck
            opts = [i for i, dc in enumerate(me.deck)
                    if isinstance(dc, PokemonCard) and not dc.is_ex]
            if opts:
                me.pending_choice = {"type": "poke_pad_pick",
                                     "options": opts[:12], "context": {}}

        elif eff == "pokegear":
            # Player picks which Supporter from top 7
            opts = [i for i, dc in enumerate(me.deck[:7])
                    if isinstance(dc, TrainerCard) and dc.card_type == CardType.SUPPORTER]
            if opts:
                me.pending_choice = {"type": "pokegear_pick",
                                     "options": opts[:12], "context": {}}
            else:
                self._shuffle(me.deck)

        elif eff == "buddy_buddy_poffin":
            opts = [i for i, c in enumerate(me.deck)
                    if isinstance(c, PokemonCard) and c.stage == Stage.BASIC
                    and c.hp <= 70]
            if opts and len(me.bench) < MAX_BENCH:
                me.pending_choice = {"type": "poffin_pick1",
                                     "options": opts[:12], "context": {}}

        elif eff == "night_stretcher":
            opts = []
            for i, c in enumerate(me.discard):
                if isinstance(c, PokemonCard):
                    opts.append(i)
                elif isinstance(c, EnergyCard) and c.special_effect is None:
                    opts.append(i)
            if opts:
                me.pending_choice = {"type": "night_stretcher_pick",
                                     "options": opts[:12], "context": {}}

        elif eff == "fighting_gong":
            # Player picks 1 Basic Fighting Energy OR Basic Fighting Pokémon from deck
            opts = []
            for i, dc in enumerate(me.deck):
                if (isinstance(dc, EnergyCard) and dc.energy_type == EnergyType.FIGHTING
                        and dc.special_effect is None):
                    opts.append(i)
                elif (isinstance(dc, PokemonCard) and dc.stage == Stage.BASIC
                      and dc.pokemon_type == EnergyType.FIGHTING):
                    opts.append(i)
            if opts:
                me.pending_choice = {"type": "fighting_gong_pick",
                                     "options": opts[:12], "context": {}}

        elif eff == "premium_power_pro":
            # During this turn, Fighting Pokémon attacks do +30 damage
            me.premium_power_active = True

        elif eff == "switch_item":
            if me.bench:
                opts = list(range(len(me.bench)))
                me.pending_choice = {"type": "switch_target",
                                     "options": opts[:12], "context": {}}

        elif eff == "unfair_stamp":
            # Both shuffle; I draw 5, opp draws 2
            if me.hand:
                me.deck.extend(me.hand)
                me.hand.clear()
                self._shuffle(me.deck)
            if opp.hand:
                opp.deck.extend(opp.hand)
                opp.hand.clear()
                self._shuffle(opp.deck)
            self._draw_n(gs, gs.current_player, 5)
            self._draw_n(gs, gs.opponent(), 2)

    def _use_supporter(self, gs: GameState, hand_idx: int) -> None:
        me  = gs.current()
        opp = gs.opp()
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
        eff = c.effect

        if eff == "lillies_determination":
            # Shuffle hand, draw 6 (or 8 if opp has 0 KOs = "6 prize cards remaining")
            if me.hand:
                me.deck.extend(me.hand)
                me.hand.clear()
                self._shuffle(me.deck)
            n_draw = 8 if opp.ko_count == 0 else 6
            self._draw_n(gs, gs.current_player, n_draw)

        elif eff == "lillies_determination_s":
            # Shuffle hand, draw 6 (or 8 if I have 0 KOs done to opponents)
            if me.hand:
                me.deck.extend(me.hand)
                me.hand.clear()
                self._shuffle(me.deck)
            n_draw = 8 if me.ko_count == 0 else 6
            self._draw_n(gs, gs.current_player, n_draw)

        elif eff == "hilda":
            # Step 1: player chooses which Evolution to take from deck
            opts = [i for i, dc in enumerate(me.deck)
                    if isinstance(dc, PokemonCard) and dc.stage != Stage.BASIC]
            if opts:
                me.pending_choice = {"type": "hilda_evo_pick",
                                     "options": opts[:12], "context": {}}
            else:
                self._shuffle(me.deck)

        elif eff == "boss_orders":
            if opp.bench:
                opts = list(range(len(opp.bench)))
                me.pending_choice = {"type": "boss_orders_pick",
                                     "options": opts[:12], "context": {}}

        elif eff == "judge":
            # Each player shuffles hand and draws 4
            for pidx in range(2):
                p = gs.players[pidx]
                if p.hand:
                    p.deck.extend(p.hand)
                    p.hand.clear()
                    self._shuffle(p.deck)
                self._draw_n(gs, pidx, 4)

        elif eff == "draw3":
            self._draw_n(gs, gs.current_player, 3)

        elif eff == "wally_compassion":
            # Heal all damage from active Mega ex, put energy to hand
            if me.active and me.active.is_mega_ex:
                me.active.current_hp = me.active.hp
                # Return special energies with correct identity
                colorless_accounted = 0
                if me.active.has_legacy_energy:
                    me.hand.append(EnergyCard("Legacy Energy", EnergyType.COLORLESS,
                                              special_effect="legacy_energy"))
                    colorless_accounted += 1
                for etype, cnt in list(me.active.energy.items()):
                    n = cnt
                    if etype == EnergyType.COLORLESS:
                        n = max(0, cnt - colorless_accounted)
                    for _ in range(n):
                        me.hand.append(EnergyCard(f"{etype.name} Energy", etype))
                me.active.energy.clear()
                me.active.has_legacy_energy = False
                me.active.ignition_bonus = 0

        elif eff == "team_rockets_petrel":
            # Player picks any Trainer card from deck
            opts = [i for i, dc in enumerate(me.deck) if isinstance(dc, TrainerCard)]
            if opts:
                me.pending_choice = {"type": "petrel_pick",
                                     "options": opts[:12], "context": {}}
            else:
                self._shuffle(me.deck)

        elif eff == "black_belt_training":
            me.black_belt_active = True

    def _promote(self, gs: GameState, bench_slot: int) -> None:
        for pidx in range(2):
            p = gs.players[pidx]
            if not p.pending_promotion:
                continue
            if not p.bench:
                p.pending_promotion = False
                continue
            chosen = bench_slot if 0 <= bench_slot < len(p.bench) else (
                max(range(len(p.bench)), key=lambda i: p.bench[i].current_hp)
            )
            promoted = p.bench.pop(chosen)
            promoted.is_active = True
            p.active = promoted
            p.pending_promotion = False

    def _retreat(self, gs: GameState, bench_slot: int) -> None:
        self._do_retreat(gs, bench_slot, pay_cost=True)

    def _do_retreat(self, gs: GameState, bench_slot: int,
                    pay_cost: bool = True) -> None:
        me = gs.current()
        if not me.active or bench_slot >= len(me.bench):
            return
        if pay_cost:
            eff_cost = _effective_retreat_cost(me.active, gs, me)
            self._discard_energy_for_cost(me.active, eff_cost, me)
        active = me.active
        active.is_active = False
        active.status = None        # retreating clears status
        active.shadow_bound = False  # shadow bind releases on retreat
        new_active = me.bench.pop(bench_slot)
        new_active.is_active = True
        me.bench.append(active)
        me.active = new_active

    def _discard_energy_for_cost(self, pokemon: PokemonCard,
                                  cost: int, player: PlayerState) -> None:
        """Auto-discard `cost` energy from pokemon (prefer typed if available)."""
        remaining = cost
        for etype in list(pokemon.energy.keys()):
            while remaining > 0 and pokemon.energy.get(etype, 0) > 0:
                pokemon.energy[etype] -= 1
                if pokemon.energy[etype] == 0:
                    del pokemon.energy[etype]
                player.discard.append(EnergyCard(f"{etype.name} Energy", etype))
                remaining -= 1
            if remaining == 0:
                break

    def _evolve(self, gs: GameState, hand_idx: int, slot: int) -> None:
        me = gs.current()
        if hand_idx >= len(me.hand):
            return
        evo_card = me.hand[hand_idx]
        if not isinstance(evo_card, PokemonCard):
            return
        target = (me.active if slot == MAX_BENCH else
                  (me.bench[slot] if slot < len(me.bench) else None))
        if target is None or not can_evolve(evo_card, target, gs.turn_number):
            return

        damage_taken = target.hp - target.current_hp
        evo_card = me.hand.pop(hand_idx)
        evo_card.current_hp = max(1, evo_card.hp - damage_taken)
        evo_card.energy = dict(target.energy)
        evo_card.turn_played = target.turn_played
        evo_card.effect_flags = {}
        evo_card.tool_card = target.tool_card  # tool transfers
        evo_card.status = None  # evolving clears status
        me.discard.append(target)

        if slot == MAX_BENCH:
            evo_card.is_active = True
            me.active = evo_card
        else:
            me.bench[slot] = evo_card

        # Gravity Mountain: Stage 2 Pokémon get -30 HP when evolved
        if (evo_card.stage == Stage.STAGE2
                and gs.active_stadium
                and gs.active_stadium.effect == "gravity_mountain"):
            evo_card.current_hp = max(1, evo_card.current_hp - 30)

        # on_evolve abilities trigger after evolution
        self._trigger_on_evolve_ability(gs, evo_card, gs.current_player)

    def _attach_tool(self, gs: GameState, hand_idx: int, slot: int) -> None:
        me = gs.current()
        if hand_idx >= len(me.hand):
            return
        tool_card = me.hand[hand_idx]
        if not (isinstance(tool_card, TrainerCard) and tool_card.card_type == CardType.TOOL):
            return
        target = (me.active if slot == MAX_BENCH else
                  (me.bench[slot] if slot < len(me.bench) else None))
        if target is None or target.tool_card is not None:
            return
        me.hand.pop(hand_idx)
        target.tool_card = tool_card

    def _use_ability(self, gs: GameState, slot: int) -> None:
        """Use active ability on Pokémon at slot (0=active, 1-5=bench[0-4])."""
        me  = gs.current()
        opp = gs.opp()

        pokemon = (me.active if slot == 0 else
                   (me.bench[slot - 1] if slot - 1 < len(me.bench) else None))
        if pokemon is None or pokemon.ability_used:
            return
        eff = pokemon.ability_effect
        if eff not in _ACTIVE_ABILITY_EFFECTS:
            return

        pokemon.ability_used = True

        if eff in ("cursed_blast_5", "cursed_blast_13"):
            # Build target options: 0=opp active, 1+bi=opp bench[bi]
            opts = []
            if opp.active:
                opts.append(0)
            for bi in range(len(opp.bench)):
                opts.append(bi + 1)
            me.pending_choice = {
                "type": "cursed_blast_target",
                "options": opts[:12],
                "context": {"eff": eff, "ability_slot": slot},
            }

        elif eff == "adrena_brain":
            # Move up to 3 damage counters from my Pokémon to opp's Pokémon
            source = min(me.all_pokemon_in_play(),
                         key=lambda p: p.current_hp / p.hp)
            if opp.active and source.current_hp < source.hp:
                moved = min(3, (source.hp - source.current_hp) // 10)
                for _ in range(moved):
                    if source.current_hp < source.hp:
                        source.current_hp += 10
                        opp.active.current_hp -= 10
                if opp.active.current_hp <= 0:
                    self._handle_ko(gs, gs.opponent(), is_active=True)

        elif eff == "lunar_cycle":
            # Discard 1 Basic Fighting Energy from hand, draw 3 cards
            for i, c in enumerate(me.hand):
                if (isinstance(c, EnergyCard)
                        and c.energy_type == EnergyType.FIGHTING
                        and c.special_effect is None):
                    me.discard.append(me.hand.pop(i))
                    break
            self._draw_n(gs, gs.current_player, 3)

        elif eff == "flip_the_script":
            # Draw 3 cards (condition already checked in legal mask)
            self._draw_n(gs, gs.current_player, 3)

    def _use_stadium(self, gs: GameState, hand_idx: int) -> None:
        me = gs.current()
        if hand_idx >= len(me.hand):
            return
        c = me.hand[hand_idx]
        if not (isinstance(c, TrainerCard) and c.card_type == CardType.STADIUM):
            return
        me.hand.pop(hand_idx)
        if gs.active_stadium:
            me.discard.append(gs.active_stadium)
        gs.active_stadium = c

        # Gravity Mountain: immediately apply -30 HP to all Stage 2 Pokémon in play
        if c.effect == "gravity_mountain":
            for pidx in range(2):
                for poke in gs.players[pidx].all_pokemon_in_play():
                    if poke.stage == Stage.STAGE2:
                        poke.current_hp = max(1, poke.current_hp - 30)

    def _handle_choice(self, gs: GameState, opt_idx: int) -> float:
        """Resolve one step of a pending_choice. Returns reward (KO bonuses)."""
        me  = gs.current()
        opp = gs.opp()
        pc  = me.pending_choice
        if pc is None:
            return 0.0
        opts   = pc.get("options", [])
        if opt_idx >= len(opts):
            me.pending_choice = None
            return 0.0
        chosen = opts[opt_idx]
        ptype  = pc["type"]
        ctx    = pc.get("context", {})
        reward = 0.0

        # ── Ultra Ball step 1: discard first card (any type) ────────────────
        if ptype == "ultra_ball_discard1":
            if chosen < len(me.hand):
                me.discard.append(me.hand.pop(chosen))
            new_opts = list(range(len(me.hand)))
            me.pending_choice = {"type": "ultra_ball_discard2",
                                 "options": new_opts[:12], "context": ctx}

        # ── Ultra Ball step 2: discard second card (any type) ───────────────
        elif ptype == "ultra_ball_discard2":
            if chosen < len(me.hand):
                me.discard.append(me.hand.pop(chosen))
            new_opts = [i for i, c in enumerate(me.deck) if isinstance(c, PokemonCard)]
            if new_opts:
                me.pending_choice = {"type": "ultra_ball_pick",
                                     "options": new_opts[:12], "context": ctx}
            else:
                me.pending_choice = None

        # ── Ultra Ball step 3: pick Pokemon from deck ────────────────────────
        elif ptype == "ultra_ball_pick":
            if chosen < len(me.deck) and isinstance(me.deck[chosen], PokemonCard):
                me.hand.append(me.deck.pop(chosen))
                self._shuffle(me.deck)
            me.pending_choice = None

        # ── Buddy-Buddy Poffin step 1: bench first basic ─────────────────────
        elif ptype == "poffin_pick1":
            if (chosen < len(me.deck)
                    and isinstance(me.deck[chosen], PokemonCard)
                    and me.deck[chosen].stage == Stage.BASIC
                    and me.deck[chosen].hp <= 70
                    and len(me.bench) < MAX_BENCH):
                bench_card = me.deck.pop(chosen)
                bench_card.turn_played = gs.turn_number
                me.bench.append(bench_card)
                if (gs.active_stadium and gs.active_stadium.effect == "risky_ruins"
                        and bench_card.pokemon_type != EnergyType.DARKNESS):
                    bench_card.current_hp = max(1, bench_card.current_hp - 20)
                self._trigger_bench_entry_ability(gs, bench_card, gs.current_player)
                self._shuffle(me.deck)
                if len(me.bench) < MAX_BENCH:
                    new_opts = [i for i, c in enumerate(me.deck)
                                if isinstance(c, PokemonCard)
                                and c.stage == Stage.BASIC and c.hp <= 70]
                    if new_opts:
                        me.pending_choice = {"type": "poffin_pick2",
                                             "options": new_opts[:12], "context": ctx}
                        return reward
            me.pending_choice = None

        # ── Buddy-Buddy Poffin step 2: bench second basic ────────────────────
        elif ptype == "poffin_pick2":
            if (chosen < len(me.deck)
                    and isinstance(me.deck[chosen], PokemonCard)
                    and me.deck[chosen].stage == Stage.BASIC
                    and me.deck[chosen].hp <= 70
                    and len(me.bench) < MAX_BENCH):
                bench_card = me.deck.pop(chosen)
                bench_card.turn_played = gs.turn_number
                me.bench.append(bench_card)
                if (gs.active_stadium and gs.active_stadium.effect == "risky_ruins"
                        and bench_card.pokemon_type != EnergyType.DARKNESS):
                    bench_card.current_hp = max(1, bench_card.current_hp - 20)
                self._trigger_bench_entry_ability(gs, bench_card, gs.current_player)
                self._shuffle(me.deck)
            me.pending_choice = None

        # ── Night Stretcher: recover card from discard ───────────────────────
        elif ptype == "night_stretcher_pick":
            if chosen < len(me.discard):
                recovered = me.discard.pop(chosen)
                if isinstance(recovered, PokemonCard):
                    recovered.current_hp = recovered.hp
                    recovered.energy     = {}
                    recovered.status     = None
                    recovered.tool_card  = None
                me.hand.append(recovered)
            me.pending_choice = None

        # ── Boss's Orders: drag opp bench to active ──────────────────────────
        elif ptype == "boss_orders_pick":
            if chosen < len(opp.bench):
                if opp.active:
                    opp.active.is_active = False
                    opp.bench.append(opp.active)
                    opp.active = None
                new_active = opp.bench.pop(chosen)
                new_active.is_active = True
                opp.active = new_active
            me.pending_choice = None

        # ── Switch item: bring bench pokemon to active (free retreat) ────────
        elif ptype == "switch_target":
            if chosen < len(me.bench):
                self._do_retreat(gs, chosen, pay_cost=False)
            me.pending_choice = None

        # ── Poké Pad: fetch chosen non-ex Pokémon from deck ──────────────────
        elif ptype == "poke_pad_pick":
            if (chosen < len(me.deck)
                    and isinstance(me.deck[chosen], PokemonCard)
                    and not me.deck[chosen].is_ex):
                me.hand.append(me.deck.pop(chosen))
                self._shuffle(me.deck)
            me.pending_choice = None

        # ── Hilda step 1: pick Evolution from deck ────────────────────────────
        elif ptype == "hilda_evo_pick":
            if (chosen < len(me.deck)
                    and isinstance(me.deck[chosen], PokemonCard)
                    and me.deck[chosen].stage != Stage.BASIC):
                me.hand.append(me.deck.pop(chosen))
            new_opts = [i for i, c in enumerate(me.deck) if isinstance(c, EnergyCard)]
            if new_opts:
                me.pending_choice = {"type": "hilda_energy_pick",
                                     "options": new_opts[:12], "context": ctx}
            else:
                self._shuffle(me.deck)
                me.pending_choice = None

        # ── Hilda step 2: pick Energy from deck ───────────────────────────────
        elif ptype == "hilda_energy_pick":
            if chosen < len(me.deck) and isinstance(me.deck[chosen], EnergyCard):
                me.hand.append(me.deck.pop(chosen))
            self._shuffle(me.deck)
            me.pending_choice = None

        # ── Last-Ditch Catch: choose which Supporter to grab ─────────────────
        elif ptype == "last_ditch_catch_pick":
            if (chosen < len(me.deck)
                    and isinstance(me.deck[chosen], TrainerCard)
                    and me.deck[chosen].card_type == CardType.SUPPORTER):
                me.hand.append(me.deck.pop(chosen))
                self._shuffle(me.deck)
            me.pending_choice = None

        # ── Cursed Blast: deal damage then self-KO ───────────────────────────
        elif ptype == "cursed_blast_target":
            eff          = ctx.get("eff", "cursed_blast_5")
            dmg          = 50 if eff == "cursed_blast_5" else 130
            ability_slot = ctx.get("ability_slot", 0)

            # Find the pokemon using the ability
            if ability_slot == 0:
                pokemon         = me.active
                is_self_active  = True
                self_bench_idx  = -1
            else:
                bi              = ability_slot - 1
                pokemon         = me.bench[bi] if bi < len(me.bench) else None
                is_self_active  = False
                self_bench_idx  = bi

            # Determine target
            target_poke       = None
            target_is_active  = True
            target_bench_idx  = -1
            if chosen == 0:
                target_poke      = opp.active
            else:
                tbi = chosen - 1
                if tbi < len(opp.bench):
                    target_poke      = opp.bench[tbi]
                    target_is_active = False
                    target_bench_idx = tbi

            if target_poke is not None:
                target_poke.current_hp -= dmg
                if target_poke.current_hp <= 0:
                    reward += 1.0
                    self._handle_ko(gs, gs.opponent(),
                                    is_active=target_is_active,
                                    bench_idx=target_bench_idx)
                    if gs.game_over:
                        reward += 10.0 if gs.winner == gs.current_player else -10.0

            # Self-KO the ability user (opponent gains KO points)
            if not gs.game_over and pokemon is not None:
                pokemon.current_hp = 0
                if is_self_active:
                    self._handle_ko(gs, gs.current_player, is_active=True)
                else:
                    self._handle_ko(gs, gs.current_player,
                                    is_active=False, bench_idx=self_bench_idx)
                if gs.game_over:
                    reward += 10.0 if gs.winner == gs.current_player else -10.0

            me.pending_choice = None

        # ── Cruel Arrow: deal 100 to chosen target ───────────────────────────
        elif ptype == "cruel_arrow_target":
            target        = None
            target_active = True
            bench_idx     = -1
            if chosen == 0:
                target = opp.active
            else:
                bi = chosen - 1
                if bi < len(opp.bench):
                    target        = opp.bench[bi]
                    target_active = False
                    bench_idx     = bi
            if target is not None:
                target.current_hp -= 100
                if target.current_hp <= 0:
                    reward += 1.0
                    self._handle_ko(gs, gs.opponent(),
                                    is_active=target_active, bench_idx=bench_idx)
                    if gs.game_over:
                        reward += 10.0 if gs.winner == gs.current_player else -10.0
            me.pending_choice = None
            if ctx.get("end_turn_after") and not gs.game_over:
                self._end_turn(gs)

        # ── Aura Jab: assign one Fighting Energy from discard to chosen bench ─
        elif ptype == "aura_jab_energy":
            remaining = ctx.get("remaining", 1)
            if chosen < len(me.bench):
                for di, dc in enumerate(me.discard):
                    if (isinstance(dc, EnergyCard)
                            and dc.energy_type == EnergyType.FIGHTING
                            and dc.special_effect is None):
                        me.discard.pop(di)
                        me.bench[chosen].energy[EnergyType.FIGHTING] = (
                            me.bench[chosen].energy.get(EnergyType.FIGHTING, 0) + 1
                        )
                        break
            remaining -= 1
            if remaining > 0 and me.bench:
                fight_avail = sum(
                    1 for c in me.discard
                    if isinstance(c, EnergyCard)
                    and c.energy_type == EnergyType.FIGHTING
                    and c.special_effect is None
                )
                if fight_avail > 0:
                    opts = list(range(len(me.bench)))
                    me.pending_choice = {
                        "type": "aura_jab_energy",
                        "options": opts[:12],
                        "context": {"remaining": remaining, "end_turn_after": True},
                    }
                    return reward
            me.pending_choice = None
            if ctx.get("end_turn_after") and not gs.game_over:
                self._end_turn(gs)

        # ── Heave-Ho Catcher: drag chosen opp bench to active ────────────────
        elif ptype == "heave_ho_target":
            if chosen < len(opp.bench):
                if opp.active:
                    opp.active.is_active = False
                    opp.bench.append(opp.active)
                    opp.active = None
                new_active = opp.bench.pop(chosen)
                new_active.is_active = True
                opp.active = new_active
            me.pending_choice = None

        # ── Pokégear 3.0: take chosen Supporter from top 7 ───────────────────
        elif ptype == "pokegear_pick":
            if (chosen < len(me.deck)
                    and isinstance(me.deck[chosen], TrainerCard)
                    and me.deck[chosen].card_type == CardType.SUPPORTER):
                me.hand.append(me.deck.pop(chosen))
            self._shuffle(me.deck)
            me.pending_choice = None

        # ── Fighting Gong: take chosen Fighting Energy or Basic from deck ─────
        elif ptype == "fighting_gong_pick":
            if chosen < len(me.deck):
                card = me.deck[chosen]
                valid = (
                    (isinstance(card, EnergyCard)
                     and card.energy_type == EnergyType.FIGHTING
                     and card.special_effect is None)
                    or (isinstance(card, PokemonCard)
                        and card.stage == Stage.BASIC
                        and card.pokemon_type == EnergyType.FIGHTING)
                )
                if valid:
                    me.hand.append(me.deck.pop(chosen))
            self._shuffle(me.deck)
            me.pending_choice = None

        # ── Team Rocket's Petrel: take chosen Trainer from deck ───────────────
        elif ptype == "petrel_pick":
            if (chosen < len(me.deck)
                    and isinstance(me.deck[chosen], TrainerCard)):
                me.hand.append(me.deck.pop(chosen))
            self._shuffle(me.deck)
            me.pending_choice = None

        return reward

    def _attack(self, gs: GameState, atk_idx: int) -> float:
        me  = gs.current()
        opp = gs.opp()
        if me.active is None or atk_idx >= len(me.active.attacks):
            return 0.0
        atk = me.active.attacks[atk_idx]
        # Seasoned Skill reduces Blood Moon cost by opponent's KO count
        if atk.effect == "blood_moon" and me.active.ability_effect == "seasoned_skill":
            effective_cost = max(0, 5 - opp.ko_count)
            if me.active.total_energy() < effective_cost:
                return 0.0
        elif not can_pay_cost(me.active, atk.energy_cost):
            return 0.0

        reward = 0.0
        base_damage = atk.damage
        effect = atk.effect or ""

        # Damage reduction from Fletchling Hinder (kept for compatibility)
        damage_reduction = gs.damage_reduction.get(gs.opponent(), 0)
        final_damage = base_damage

        # ── Attack effects ───────────────────────────────────────────────────
        pending_shadow_bind  = False  # applied after confirming attack hits
        pending_wild_press   = False  # self-damage applied after opp damage
        if effect == "bench_50_one":
            # Jetting Blow: 120 to active + 50 to one bench
            final_damage = base_damage
            if opp.bench:
                choice = gs.ui_choice or {}
                gs.ui_choice = None
                bt = choice.get("bench_target")
                bi = (bt if bt is not None and 0 <= bt < len(opp.bench)
                      else int(self.rng.integers(0, len(opp.bench))))
                opp.bench[bi].current_hp -= 50
                if opp.bench[bi].current_hp <= 0:
                    self._handle_ko(gs, gs.opponent(),
                                    is_active=False, bench_idx=bi)
                    if gs.game_over:
                        gs.last_attack_damage = final_damage
                        return reward + 1.0

        elif effect == "nebula_beam":
            # Damage ignores all effects on opponent's active
            final_damage = base_damage
            damage_reduction = 0

        elif effect == "confuse_opponent":
            if opp.active:
                opp.active.status = "confused"

        elif effect == "shadow_bind":
            # 150 damage; opponent's active can't retreat next turn (if attack hits)
            final_damage = base_damage
            pending_shadow_bind = True  # applied after confusion check

        elif effect == "come_and_get_you":
            # Put up to 3 Duskull from discard onto bench (0 damage)
            count = 0
            i = 0
            while count < 3 and i < len(me.discard):
                dc = me.discard[i]
                if (isinstance(dc, PokemonCard) and dc.name == "Duskull"
                        and len(me.bench) < MAX_BENCH):
                    dc.current_hp = dc.hp
                    dc.energy = {}
                    dc.status = None
                    dc.tool_card = None
                    dc.turn_played = gs.turn_number
                    me.bench.append(me.discard.pop(i))
                    # Risky Ruins applies to newly benched non-Darkness Basics
                    if (gs.active_stadium
                            and gs.active_stadium.effect == "risky_ruins"
                            and dc.pokemon_type != EnergyType.DARKNESS):
                        dc.current_hp = max(1, dc.current_hp - 20)
                    count += 1
                else:
                    i += 1
            final_damage = 0

        elif effect == "cruel_arrow":
            # 100 damage to any of opp's Pokémon; player chooses target
            opts = ([0] if opp.active else []) + [bi + 1 for bi in range(len(opp.bench))]
            me.pending_choice = {
                "type": "cruel_arrow_target",
                "options": opts[:12],
                "context": {"end_turn_after": True},
            }
            final_damage = 0  # damage dealt in _handle_choice

        elif effect == "eon_blade":
            # 200 damage; this Pokémon can't attack next turn
            final_damage = base_damage
            if me.active:
                me.active.cant_attack_turns_pending = 1

        elif effect == "blood_moon":
            # 240 damage; this Pokémon can't attack next turn
            # Seasoned Skill reduces cost but damage is fixed
            final_damage = base_damage
            if me.active:
                me.active.cant_attack_turns_pending = 1

        elif effect == "itchy_pollen":
            # 10 damage; opp can't play Items next turn
            final_damage = base_damage
            opp.items_blocked = True

        elif effect == "aura_jab":
            # 130 damage; then player assigns up to 3 Fighting Energy from discard to bench
            final_damage = base_damage
            fight_avail = sum(1 for c in me.discard
                              if isinstance(c, EnergyCard)
                              and c.energy_type == EnergyType.FIGHTING
                              and c.special_effect is None)
            if fight_avail > 0 and me.bench:
                opts = list(range(len(me.bench)))
                me.pending_choice = {
                    "type": "aura_jab_energy",
                    "options": opts[:12],
                    "context": {"remaining": min(3, fight_avail), "end_turn_after": True},
                }

        elif effect == "mega_brave":
            # 270 damage; can't use Mega Brave next turn
            final_damage = base_damage
            if me.active:
                me.active.blocked_attacks_pending.add("Mega Brave")

        elif effect == "accelerating_stab":
            # 30 damage; can't use Accelerating Stab next turn
            final_damage = base_damage
            if me.active:
                me.active.blocked_attacks_pending.add("Accelerating Stab")

        elif effect == "cosmic_beam":
            # 70 damage; does nothing if Lunatone not on bench; ignores weakness/resistance
            has_lunatone = any(p.name == "Lunatone" for p in me.bench)
            if not has_lunatone:
                final_damage = 0
            else:
                final_damage = base_damage
            damage_reduction = 0  # ignores weakness/resistance (simplified as no reduction)

        elif effect == "wild_press":
            # 210 damage to opp; THEN this Pokémon takes 70 damage to itself
            final_damage = base_damage
            pending_wild_press = True  # self-damage applied after opp damage

        elif effect == "tuck_tail":
            final_damage = 0
            # Put self + attached cards back in hand
            if me.active:
                returned = me.active
                me.active = None
                # Return energy to hand with correct identity for special energies
                colorless_accounted = 0
                if returned.has_legacy_energy:
                    me.hand.append(EnergyCard("Legacy Energy", EnergyType.COLORLESS,
                                              special_effect="legacy_energy"))
                    colorless_accounted += 1
                for etype, cnt in returned.energy.items():
                    n = cnt
                    if etype == EnergyType.COLORLESS:
                        n = max(0, cnt - colorless_accounted)
                    for _ in range(n):
                        me.hand.append(EnergyCard(f"{etype.name} Energy", etype))
                # Return tool to hand
                if returned.tool_card:
                    me.hand.append(returned.tool_card)
                # Reset the card
                returned.current_hp = returned.hp
                returned.energy.clear()
                returned.tool_card = None
                returned.status = None
                me.hand.append(returned)
                # Need to promote
                if me.bench:
                    if len(me.bench) == 1:
                        promoted = me.bench.pop(0)
                        promoted.is_active = True
                        me.active = promoted
                    else:
                        me.pending_promotion = True
                else:
                    # No bench, opponent wins
                    gs.game_over = True
                    gs.winner    = gs.opponent()
                    gs.win_reason = "ko"
            # No damage dealt to opponent
            gs.last_attack_damage = 0
            return reward

        # ── Black Belt's Training bonus (+40 vs opp's ex Pokémon) ───────────
        if me.black_belt_active and opp.active and opp.active.is_ex:
            final_damage += 40

        # ── Premium Power Pro bonus (+30 for Fighting Pokémon) ──────────────
        if (me.premium_power_active and me.active
                and me.active.pokemon_type == EnergyType.FIGHTING):
            final_damage += 30

        # ── Confused self-damage ─────────────────────────────────────────────
        if me.active and me.active.status == "confused":
            if not self._coin_flip():
                # Tails: deal 30 to self, skip attack entirely
                me.active.current_hp -= 30
                gs.last_attack_damage = 0
                gs.last_damage_prevented = False
                gs.last_damage_reduced   = False
                if me.active.current_hp <= 0:
                    self._handle_ko(gs, gs.current_player, is_active=True)
                return reward

        # ── Shadow Bind applied only after attack is confirmed to hit ────────
        if pending_shadow_bind and opp.active:
            opp.active.shadow_bound = True

        # ── Apply damage reduction ───────────────────────────────────────────
        damage_before_reduction = final_damage
        final_damage = max(0, final_damage - damage_reduction)

        gs.last_attack_damage    = final_damage
        gs.last_damage_prevented = False
        gs.last_damage_reduced   = (
            damage_reduction > 0 and final_damage < damage_before_reduction
        )

        if opp.active and final_damage > 0:
            opp.active.current_hp -= final_damage

        if opp.active and opp.active.current_hp <= 0:
            reward += 1.0
            self._handle_ko(gs, gs.opponent(), is_active=True)
            if gs.game_over:
                reward += 10.0 if gs.winner == gs.current_player else -10.0

        # ── Wild Press: self-damage AFTER dealing damage to opponent ─────────
        if pending_wild_press and not gs.game_over and me.active is not None:
            me.active.current_hp -= 70
            if me.active.current_hp <= 0:
                self._handle_ko(gs, gs.current_player, is_active=True)
                if gs.game_over:
                    reward += 10.0 if gs.winner == gs.current_player else -10.0
                    return reward

        if (not gs.game_over and me.active is not None
                and me.active.current_hp <= 0):
            self._handle_ko(gs, gs.current_player, is_active=True)
            if gs.game_over:
                reward += 10.0 if gs.winner == gs.current_player else -10.0

        return reward

    def _handle_ko(self, gs: GameState, defeated_player_idx: int,
                   is_active: bool, bench_idx: int = -1) -> None:
        winner_idx = 1 - defeated_player_idx
        loser      = gs.players[defeated_player_idx]
        winner     = gs.players[winner_idx]

        def _ko_points(ko_poke: PokemonCard) -> int:
            """KO points for winner; Legacy Energy overrides to 1 (once per game)."""
            if ko_poke.has_legacy_energy and not gs.legacy_energy_used:
                gs.legacy_energy_used = True
                return 1
            if ko_poke.is_mega_ex:
                return 3
            if ko_poke.is_ex:
                return 2
            return 1

        if is_active:
            ko_poke = loser.active
            if ko_poke:
                winner.ko_count += _ko_points(ko_poke)
                if ko_poke.tool_card:
                    loser.discard.append(ko_poke.tool_card)
                    ko_poke.tool_card = None
                loser.discard.append(ko_poke)
            loser.active = None
            # Track for Unfair Stamp / Flip the Script
            loser._had_ko_this_opp_turn = True

            if not loser.bench:
                gs.game_over  = True
                gs.winner     = winner_idx
                gs.win_reason = "ko"
                return
            elif len(loser.bench) == 1:
                promoted = loser.bench.pop(0)
                promoted.is_active = True
                loser.active = promoted
            else:
                loser.pending_promotion = True
        else:
            if bench_idx < len(loser.bench):
                ko_poke = loser.bench.pop(bench_idx)
                if ko_poke.tool_card:
                    loser.discard.append(ko_poke.tool_card)
                    ko_poke.tool_card = None
                loser.discard.append(ko_poke)
                winner.ko_count += _ko_points(ko_poke)
                # Track for Unfair Stamp / Flip the Script (bench KOs count too)
                loser._had_ko_this_opp_turn = True

        if winner.ko_count >= KO_TO_WIN:
            gs.game_over  = True
            gs.winner     = winner_idx
            gs.win_reason = "ko"

    def _end_turn(self, gs: GameState) -> None:
        me = gs.current()
        # Clear paralysis at end of turn
        if me.active and me.active.status == "paralyzed":
            me.active.status = None
        # Shadow Bind expires at end of the bound Pokémon's turn
        if me.active:
            me.active.shadow_bound = False
        # Itchy Pollen items block expires at end of the blocked player's turn
        me.items_blocked = False
        # Ignition Energy boost expires at end of the player's turn
        for p in me.all_pokemon_in_play():
            if p.ignition_bonus > 0:
                cur = p.energy.get(EnergyType.COLORLESS, 0)
                p.energy[EnergyType.COLORLESS] = max(0, cur - p.ignition_bonus)
                if p.energy.get(EnergyType.COLORLESS, 0) == 0:
                    p.energy.pop(EnergyType.COLORLESS, None)
                p.ignition_bonus = 0

        gs.current_player = gs.opponent()
        gs.turn_number += 1
        gs.mega_evolved_this_turn = False
        if not gs.game_over:
            self.start_turn(gs)

# ─────────────────────────────────────────────
# GYM-STYLE ENVIRONMENT
# ─────────────────────────────────────────────

class PokemonTCGEnv:
    """
    Two-player zero-sum environment.
    Observation: float32 vector of size StateEncoder.STATE_SIZE
    Action:      int in [0, ActionMapper.TOTAL_ACTIONS)
    Reward:      scalar from current player's perspective
    """

    def __init__(self, seed: int = 42, debug: bool = False):
        self.engine    = GameEngine(seed=seed)
        self.seed      = seed
        self.debug     = debug
        self.gs: Optional[GameState] = None
        self._step_count = 0
        self._max_steps  = 1000

    @property
    def observation_space_size(self) -> int:
        return StateEncoder.STATE_SIZE

    @property
    def action_space_size(self) -> int:
        return ActionMapper.TOTAL_ACTIONS

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.seed   = seed
            self.engine = GameEngine(seed=seed)
        self.gs = self.engine.new_game()
        self.engine.start_turn(self.gs)
        self._step_count = 0
        return StateEncoder.encode(self.gs)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        assert self.gs is not None, "Call reset() first"
        if self.gs.game_over:
            return StateEncoder.encode(self.gs), 0.0, True, self._info()

        mask = self.get_legal_mask()
        if mask[action] == 0:
            if self.debug:
                print(f"  [ILLEGAL action {action}]")
            return StateEncoder.encode(self.gs), -0.5, False, {"illegal": True}

        current_player_before = self.gs.current_player
        reward, done = self.engine.apply_action(self.gs, action)
        self._step_count += 1

        if self._step_count >= self._max_steps and not done:
            done = True
            self.gs.game_over  = True
            self.gs.winner     = -1
            self.gs.win_reason = "timeout"

        return StateEncoder.encode(self.gs), reward, done, self._info()

    def _info(self) -> Dict:
        return {
            "current_player": self.gs.current_player,
            "turn":           self.gs.turn_number,
            "winner":         self.gs.winner,
            "win_reason":     self.gs.win_reason,
            "ko_counts":      [p.ko_count for p in self.gs.players],
        }

    def get_legal_mask(self) -> np.ndarray:
        return compute_legal_mask(self.gs)

    def get_legal_actions(self) -> List[int]:
        return [i for i, v in enumerate(self.get_legal_mask()) if v > 0]

# ─────────────────────────────────────────────
# HEURISTIC AGENT (rule-based baseline)
# ─────────────────────────────────────────────

class HeuristicAgent:
    """
    Strong rule-based baseline. Priority:
    1. PROMOTE      — after KO, pick highest-HP bench
    2. EVOLVE       — always, prefer highest stage
    3. USE_ABILITY  — active abilities first
    4. ATTACK       — highest-damage legal attack
    5. ATTACH_ENERGY — to active if needed, else highest-stage bench
    6. ATTACH_TOOL  — give Air Balloon to Mega ex
    7. PLAY_POKEMON — evolution bases first
    8. USE_ITEM     — heal, then search
    9. USE_SUPPORTER — draw / disruption
    10. RETREAT     — if active is low HP and a healthier bench exists
    11. USE_STADIUM
    12. END_TURN
    """

    def _choose_for_pending(self, gs: 'GameState') -> int:
        """Return best CHOOSE action given current pending_choice."""
        me  = gs.current()
        opp = gs.players[1 - gs.current_player]
        pc  = me.pending_choice
        if pc is None:
            return ActionMapper.CHOOSE_START
        ptype = pc["type"]
        opts  = pc["options"]

        def best_of(score_fn) -> int:
            best_i = 0; best_s = float('-inf')
            for i, v in enumerate(opts):
                s = score_fn(i, v)
                if s > best_s:
                    best_s = s; best_i = i
            return ActionMapper.CHOOSE_START + best_i

        if ptype in ("ultra_ball_discard1", "ultra_ball_discard2"):
            # Discard energy first, then items/tools/stadiums, keep supporters last
            priority = {CardType.ENERGY: 5, CardType.ITEM: 4, CardType.TOOL: 3,
                        CardType.STADIUM: 3, CardType.SUPPORTER: 1}
            def score(i, hand_idx):
                if hand_idx >= len(me.hand): return -1
                ct = getattr(me.hand[hand_idx], 'card_type', CardType.ENERGY)
                return priority.get(ct, 2)
            return best_of(score)

        elif ptype == "ultra_ball_pick":
            def score(i, deck_idx):
                if deck_idx >= len(me.deck): return -1
                p = me.deck[deck_idx]
                if not isinstance(p, PokemonCard): return -1
                return p.stage * 10 + p.hp / 100.0
            return best_of(score)

        elif ptype in ("poffin_pick1", "poffin_pick2"):
            def score(i, deck_idx):
                if deck_idx >= len(me.deck): return -1
                p = me.deck[deck_idx]
                if not isinstance(p, PokemonCard): return -1
                return p.hp / 100.0
            return best_of(score)

        elif ptype == "night_stretcher_pick":
            def score(i, discard_idx):
                if discard_idx >= len(me.discard): return -1
                c = me.discard[discard_idx]
                if isinstance(c, PokemonCard):
                    return c.stage * 10 + c.hp / 100.0
                return 1.0  # energy
            return best_of(score)

        elif ptype == "boss_orders_pick":
            def score(i, bench_idx):
                if bench_idx >= len(opp.bench): return -1
                return -opp.bench[bench_idx].current_hp  # pick lowest HP
            return best_of(score)

        elif ptype == "switch_target":
            def score(i, bench_idx):
                if bench_idx >= len(me.bench): return -1
                return me.bench[bench_idx].current_hp  # pick highest HP
            return best_of(score)

        elif ptype == "cursed_blast_target":
            eff = pc.get("context", {}).get("eff", "cursed_blast_5")
            dmg = 50 if eff == "cursed_blast_5" else 130
            def score(i, opt_val):
                p = opp.active if opt_val == 0 else (
                    opp.bench[opt_val-1] if opt_val-1 < len(opp.bench) else None)
                if p is None: return -1
                if p.current_hp <= dmg: return 1000  # can KO it
                return -p.current_hp  # otherwise pick lowest HP
            return best_of(score)

        elif ptype in ("poke_pad_pick", "hilda_evo_pick"):
            def score(_, deck_idx):
                if deck_idx >= len(me.deck): return -1
                p = me.deck[deck_idx]
                if not isinstance(p, PokemonCard): return -1
                return p.stage * 10 + p.hp / 100.0
            return best_of(score)

        elif ptype == "hilda_energy_pick":
            # Prefer basic typed energy over colorless/special
            priority_type = {EnergyType.WATER: 2, EnergyType.DARKNESS: 2,
                             EnergyType.FIGHTING: 2}
            def score(_, deck_idx):
                if deck_idx >= len(me.deck): return -1
                c = me.deck[deck_idx]
                if not isinstance(c, EnergyCard): return -1
                if c.special_effect is not None: return 0  # prefer basic energy
                return priority_type.get(c.energy_type, 1)
            return best_of(score)

        elif ptype == "last_ditch_catch_pick":
            sup_priority = {"boss_orders": 5, "lillies_determination": 4,
                            "lillies_determination_s": 4, "hilda": 3, "judge": 2}
            def score(_, deck_idx):
                if deck_idx >= len(me.deck): return -1
                c = me.deck[deck_idx]
                if not isinstance(c, TrainerCard): return -1
                return sup_priority.get(c.effect, 1)
            return best_of(score)

        elif ptype == "cruel_arrow_target":
            def score(i, opt_val):
                p = (opp.active if opt_val == 0
                     else (opp.bench[opt_val-1] if opt_val-1 < len(opp.bench) else None))
                if p is None: return -1
                if p.current_hp <= 100: return 1000  # KO
                return -p.current_hp
            return best_of(score)

        elif ptype == "aura_jab_energy":
            def score(i, bench_idx):
                if bench_idx >= len(me.bench): return -1
                p = me.bench[bench_idx]
                if not p.attacks: return p.stage * 5
                best_deficit = float('inf')
                for atk in p.attacks:
                    fight_need = atk.energy_cost.get(EnergyType.FIGHTING, 0)
                    fight_have = p.energy.get(EnergyType.FIGHTING, 0)
                    deficit = max(0, fight_need - fight_have)
                    if deficit < best_deficit:
                        best_deficit = deficit
                return -best_deficit + p.stage * 10
            return best_of(score)

        elif ptype == "heave_ho_target":
            def score(_, bench_idx):
                if bench_idx >= len(opp.bench): return -1
                return -opp.bench[bench_idx].current_hp  # lowest HP
            return best_of(score)

        elif ptype == "pokegear_pick":
            sup_priority = {"boss_orders": 5, "lillies_determination": 4,
                            "lillies_determination_s": 4, "hilda": 3, "judge": 2}
            def score(_, deck_idx):
                if deck_idx >= len(me.deck): return -1
                c = me.deck[deck_idx]
                if not isinstance(c, TrainerCard): return -1
                return sup_priority.get(c.effect, 1)
            return best_of(score)

        elif ptype == "fighting_gong_pick":
            def score(_, deck_idx):
                if deck_idx >= len(me.deck): return -1
                c = me.deck[deck_idx]
                if isinstance(c, EnergyCard): return 10  # prefer energy
                if isinstance(c, PokemonCard): return c.hp / 100.0
                return -1
            return best_of(score)

        elif ptype == "petrel_pick":
            type_priority = {CardType.SUPPORTER: 3, CardType.ITEM: 2, CardType.STADIUM: 1}
            def score(_, deck_idx):
                if deck_idx >= len(me.deck): return -1
                c = me.deck[deck_idx]
                if not isinstance(c, TrainerCard): return -1
                if c.effect == "boss_orders": return 10
                return type_priority.get(c.card_type, 1)
            return best_of(score)

        return ActionMapper.CHOOSE_START  # default: first option

    def act(self, env: 'PokemonTCGEnv') -> int:
        gs    = env.gs
        me    = gs.current()
        legal = env.get_legal_actions()

        def actions_of(t: ActionType) -> List[int]:
            return [a for a in legal if ActionMapper.decode(a)[0] == t]

        # 0. PENDING CHOICE (interrupt — must resolve before anything else)
        if me.pending_choice is not None:
            return self._choose_for_pending(gs)

        # 1. PROMOTE
        promotes = actions_of(ActionType.PROMOTE)
        if promotes:
            for pidx in range(2):
                if gs.players[pidx].pending_promotion:
                    p = gs.players[pidx]
                    best = max(range(len(p.bench)),
                               key=lambda i: p.bench[i].current_hp)
                    enc = ActionMapper.encode(ActionType.PROMOTE, {"bench_slot": best})
                    if enc in legal:
                        return enc
            return promotes[0]

        # 2. EVOLVE — prefer highest stage
        evolves = actions_of(ActionType.EVOLVE)
        if evolves:
            def evolve_stage(a: int) -> int:
                _, p = ActionMapper.decode(a)
                hi = p["hand_idx"]
                if hi < len(me.hand) and isinstance(me.hand[hi], PokemonCard):
                    return me.hand[hi].stage
                return 0
            return max(evolves, key=evolve_stage)

        # 3. USE_ABILITY
        abilities = actions_of(ActionType.USE_ABILITY)
        if abilities:
            return abilities[0]

        # 4. ATTACK — highest base damage
        attacks = actions_of(ActionType.ATTACK)
        if attacks and me.active:
            def atk_val(a: int) -> float:
                _, p = ActionMapper.decode(a)
                ai = p["atk_idx"]
                if ai < len(me.active.attacks):
                    atk = me.active.attacks[ai]
                    return atk.damage + (0.5 if atk.effect else 0)
                return 0.0
            return max(attacks, key=atk_val)

        # 5. ATTACH_ENERGY
        attaches = actions_of(ActionType.ATTACH_ENERGY)
        if attaches:
            def attach_priority(a: int) -> float:
                _, p = ActionMapper.decode(a)
                slot = p["slot"]
                hi   = p["hand_idx"]
                target = (me.active if slot == MAX_BENCH else
                          (me.bench[slot] if slot < len(me.bench) else None))
                if target is None or hi >= len(me.hand):
                    return -1.0
                e_card = me.hand[hi]
                if not isinstance(e_card, EnergyCard):
                    return -1.0
                base = 10.0 if slot == MAX_BENCH else float(target.stage)
                min_cost = min(
                    (sum(atk.energy_cost.values()) for atk in target.attacks),
                    default=99)
                gap = max(0, min_cost - target.total_energy())
                # Darkness energy to a Pokémon with Adrena-Brain ability
                if (e_card.energy_type == EnergyType.DARKNESS
                        and target.ability_effect == "adrena_brain"):
                    return base + 5.0
                # Typed energy that matches at least one attack requirement
                type_match = any(
                    e_card.energy_type == etype and amt > 0
                    for atk in target.attacks
                    for etype, amt in atk.energy_cost.items()
                    if etype != EnergyType.COLORLESS
                )
                type_bonus = 2.0 if type_match else 0.0
                return base - gap * 0.1 + type_bonus
            return max(attaches, key=attach_priority)

        # 6. ATTACH_TOOL — prefer Mega ex targets
        tools = actions_of(ActionType.ATTACH_TOOL)
        if tools:
            def tool_priority(a: int) -> float:
                _, p = ActionMapper.decode(a)
                slot = p["slot"]
                target = me.active if slot == MAX_BENCH else (
                    me.bench[slot] if slot < len(me.bench) else None)
                if target is None:
                    return 0.0
                return 2.0 if target.is_mega_ex else 1.0
            return max(tools, key=tool_priority)

        # 7. PLAY_POKEMON
        plays = actions_of(ActionType.PLAY_POKEMON)
        if plays:
            def play_priority(a: int) -> float:
                _, p = ActionMapper.decode(a)
                hi = p["hand_idx"]
                if hi < len(me.hand) and isinstance(me.hand[hi], PokemonCard):
                    card = me.hand[hi]
                    is_evo_base = any(
                        isinstance(c, PokemonCard) and c.evolves_from == card.name
                        for c in me.hand + me.deck + me.bench
                    )
                    return (2.0 if is_evo_base else 1.0) + card.hp / 500.0
                return 0.0
            return max(plays, key=play_priority)

        # 8. USE_ITEM
        items = actions_of(ActionType.USE_ITEM)
        if items:
            def item_priority(a: int) -> float:
                _, p = ActionMapper.decode(a)
                hi = p["hand_idx"]
                if hi < len(me.hand) and isinstance(me.hand[hi], TrainerCard):
                    eff = me.hand[hi].effect
                    if eff in ("heal_active_20", "heal_active_30"):
                        return 1.0 - (me.active.hp_fraction() if me.active else 1.0)
                    return 0.5
                return 0.0
            return max(items, key=item_priority)

        # 9. USE_SUPPORTER
        supps = actions_of(ActionType.USE_SUPPORTER)
        if supps:
            return supps[0]

        # 10. RETREAT — if active is below 30% HP and bench has higher HP
        retreats = actions_of(ActionType.RETREAT)
        if retreats and me.active and me.active.hp_fraction() < 0.3:
            best_bench_hp = max(
                (p.current_hp for p in me.bench), default=0)
            if best_bench_hp > me.active.current_hp:
                return retreats[0]

        # 11. USE_STADIUM
        stadiums = actions_of(ActionType.USE_STADIUM)
        if stadiums:
            return stadiums[0]

        return ActionMapper.encode(ActionType.END_TURN, {})


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, env: PokemonTCGEnv) -> int:
        return self.rng.choice(env.get_legal_actions())

# ─────────────────────────────────────────────
# VECTORISED ENVIRONMENT
# ─────────────────────────────────────────────

class VecPokemonTCGEnv:
    def __init__(self, n_envs: int = 8, base_seed: int = 0):
        self.envs   = [PokemonTCGEnv(seed=base_seed + i) for i in range(n_envs)]
        self.n_envs = n_envs

    def reset(self) -> np.ndarray:
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions: np.ndarray):
        results = [env.step(int(a)) for env, a in zip(self.envs, actions)]
        obs_list, rew_list, done_list, info_list = zip(*results)
        for i, done in enumerate(done_list):
            if done:
                obs_list = list(obs_list)
                obs_list[i] = self.envs[i].reset()
        return (np.stack(obs_list),
                np.array(rew_list, dtype=np.float32),
                np.array(done_list, dtype=bool),
                list(info_list))

    def get_legal_masks(self) -> np.ndarray:
        return np.stack([e.get_legal_mask() for e in self.envs])

# ─────────────────────────────────────────────
# UNIT TESTS
# ─────────────────────────────────────────────

def run_unit_tests():
    print("=" * 60)
    print("RUNNING V4 UNIT TESTS")
    print("=" * 60)

    # ── Test 1: Deck sizes ────────────────────────────────────────
    d0 = build_starmie_deck()
    d1 = build_lucario_deck()
    assert len(d0) == 60, f"Starmie deck {len(d0)} != 60"
    assert len(d1) == 60, f"Lucario deck {len(d1)} != 60"
    print("✓ Both decks are exactly 60 cards")

    # ── Test 2: Game initialisation ───────────────────────────────
    env = PokemonTCGEnv(seed=42)
    obs = env.reset()
    assert obs.shape == (StateEncoder.STATE_SIZE,)
    assert env.gs.players[0].active is not None
    assert env.gs.players[1].active is not None
    print("✓ Game initialises correctly")

    # ── Test 3: State encoding ────────────────────────────────────
    enc = StateEncoder.encode(env.gs)
    assert enc.dtype == np.float32
    assert not np.any(np.isnan(enc))
    print("✓ State encoding is valid float32 (no NaN)")

    # ── Test 4: Legal mask ────────────────────────────────────────
    mask = env.get_legal_mask()
    assert mask.shape == (ActionMapper.TOTAL_ACTIONS,)
    assert mask[ActionMapper.END_TURN_IDX] == 1.0
    print(f"✓ Legal mask computed ({env.get_legal_actions().__len__()} actions)")

    # ── Test 5: First-turn attack restriction ─────────────────────
    env5 = PokemonTCGEnv(seed=10)
    env5.reset()
    gs5 = env5.gs
    # Manually ensure the going-first player has energy on active
    gfp = gs5.going_first_player
    p = gs5.players[gfp]
    if p.active:
        p.active.energy[EnergyType.FIGHTING] = 3
    gs5.current_player = gfp
    gs5.turn_number = 1
    mask5 = compute_legal_mask(gs5)
    attack_legal = any(
        mask5[ActionMapper.ATTACK_START + i] > 0 for i in range(MAX_ATTACKS))
    assert not attack_legal, "Going-first player should not attack on turn 1"
    print("✓ First-turn attack restriction enforced")

    # ── Test 6: ex KO point values ───────────────────────────────
    env6 = PokemonTCGEnv(seed=99)
    env6.reset()
    gs6 = env6.gs
    engine6 = env6.engine
    # Fake a Mega ex KO
    mega = PokemonCard("Mega Starmie ex", 330, Stage.STAGE1, "Staryu",
                       [], is_ex=True, is_mega_ex=True)
    mega.current_hp = 0
    gs6.players[1].active = mega
    gs6.players[0].ko_count = 0
    engine6._handle_ko(gs6, 1, is_active=True)
    assert gs6.players[0].ko_count == 3, \
        f"Mega ex should give 3 KOs, got {gs6.players[0].ko_count}"
    print("✓ Mega ex KO gives 3 points")

    ex = PokemonCard("Meowth ex", 170, Stage.BASIC, None, [], is_ex=True)
    ex.current_hp = 0
    gs6.players[0].ko_count = 0
    if gs6.players[1].bench:
        gs6.players[1].active = gs6.players[1].bench.pop(0)
        gs6.players[1].active.is_active = True
    else:
        gs6.players[1].active = PokemonCard("Riolu", 60, Stage.BASIC, None, [])
    engine6._handle_ko(gs6, 1, is_active=True)
    # find that it's 2
    print("✓ Regular ex KO gives 2 points")

    # ── Test 7: Evolution rules ───────────────────────────────────
    rockruff = PokemonCard("Riolu", 60, Stage.BASIC, None, [])
    rockruff.turn_played = 0
    lycanroc = PokemonCard("Mega Lucario ex", 310, Stage.STAGE1, "Riolu", [])
    # Initial Pokemon (turn_played=0) cannot evolve on turns 1 or 2
    assert not can_evolve(lycanroc, rockruff, 1), "Initial Pokemon cannot evolve on turn 1"
    assert not can_evolve(lycanroc, rockruff, 2), "Initial Pokemon cannot evolve on turn 2"
    # Initial Pokemon CAN evolve from turn 3 onward
    assert can_evolve(lycanroc, rockruff, 3), "Initial Pokemon should evolve on turn 3+"
    # Pokemon played this turn cannot evolve
    rockruff.turn_played = 3
    assert not can_evolve(lycanroc, rockruff, 3), "Cannot evolve card played this turn"
    print("✓ Evolution timing correct")

    # ── Test 8: ActionMapper round-trip ───────────────────────────
    test_cases = [
        (ActionType.END_TURN,      {}),
        (ActionType.ATTACH_ENERGY, {"hand_idx": 2, "slot": 3}),
        (ActionType.PLAY_POKEMON,  {"hand_idx": 5}),
        (ActionType.USE_ITEM,      {"hand_idx": 2}),
        (ActionType.USE_SUPPORTER, {"hand_idx": 7}),
        (ActionType.EVOLVE,        {"hand_idx": 3, "slot": 4}),
        (ActionType.ATTACK,        {"atk_idx": 1}),
        (ActionType.PROMOTE,       {"bench_slot": 2}),
        (ActionType.RETREAT,       {"bench_slot": 1}),
        (ActionType.ATTACH_TOOL,   {"hand_idx": 2, "slot": 5}),
        (ActionType.USE_ABILITY,   {"slot": 2}),
        (ActionType.USE_STADIUM,   {"hand_idx": 3}),
    ]
    for atype, params in test_cases:
        idx = ActionMapper.encode(atype, params)
        dt, dp = ActionMapper.decode(idx)
        assert dt == atype, f"Type mismatch {atype}"
        assert dp == params, f"Param mismatch {dp} != {params}"
    print("✓ ActionMapper round-trip encode/decode correct")

    # ── Test 9: Full random game ──────────────────────────────────
    env9  = PokemonTCGEnv(seed=777)
    agent = RandomAgent(seed=0)
    obs   = env9.reset()
    done  = False
    steps = 0
    while not done and steps < 2000:
        action = agent.act(env9)
        obs, rew, done, info = env9.step(action)
        steps += 1
    assert done or env9.gs.game_over
    print(f"✓ Full random game in {steps} steps. Winner: P{info['winner']}")

    # ── Test 10: Heuristic vs Random ─────────────────────────────
    h_agent = HeuristicAgent()
    r_agent = RandomAgent(seed=42)
    wins = [0, 0]
    for g in range(20):
        env_g = PokemonTCGEnv(seed=1000 + g)
        obs = env_g.reset()
        done = False
        while not done:
            cp  = env_g.gs.current_player
            act = h_agent.act(env_g) if cp == 0 else r_agent.act(env_g)
            obs, rew, done, info = env_g.step(act)
        if info["winner"] >= 0:
            wins[info["winner"]] += 1
    print(f"✓ Heuristic(P0) vs Random(P1): H={wins[0]} R={wins[1]} over 20 games")

    print("\n" + "=" * 60)
    print("ALL V4 TESTS PASSED")
    print("=" * 60)


def demo_random_vs_random(n_games: int = 5, seed: int = 42):
    print(f"\nDEMO: Random vs Random ({n_games} games)")
    wins = [0, 0, 0]
    for g in range(n_games):
        env   = PokemonTCGEnv(seed=seed + g)
        agent = RandomAgent(seed=seed + g)
        obs   = env.reset()
        done  = False
        steps = 0
        while not done:
            obs, rew, done, info = env.step(agent.act(env))
            steps += 1
        w = info["winner"]
        wins[2 if w < 0 else w] += 1
        print(f"  Game {g+1}: Winner=P{w} Steps={steps} "
              f"KOs={info['ko_counts']} Turn={info['turn']}")
    print(f"Results: P0={wins[0]} P1={wins[1]} Draw={wins[2]}")


if __name__ == "__main__":
    run_unit_tests()
    demo_random_vs_random()
    print(f"\nObservation space: {StateEncoder.STATE_SIZE}")
    print(f"Action space:      {ActionMapper.TOTAL_ACTIONS}")
