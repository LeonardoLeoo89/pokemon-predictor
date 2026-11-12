"""Read species names, types, base stats and HP from battle records."""

from typing import Any, Dict, List


def _get_species_name(poke_dict: Dict[str, Any]) -> str:
    """Return a lowercased species name."""
    if not isinstance(poke_dict, dict):
        return ""
    val = poke_dict.get("name")
    if val and isinstance(val, str):
        return val.strip().lower()
    return ""


def _get_pokemon_types(poke_dict: Dict[str, Any]) -> List[str]:
    """Read recorded types, excluding the 'notype' placeholder."""
    if not isinstance(poke_dict, dict):
        return []
    return [t for t in poke_dict.get("types", []) if t != "notype"]


def _get_pokemon_stat(poke_dict: Dict[str, Any], stat: str) -> float:
    """Read a base stat from nested or flat fields."""
    if not isinstance(poke_dict, dict):
        return 0.0
    b_stats = poke_dict.get("base_stats")
    if isinstance(b_stats, dict) and stat in b_stats:
        return float(b_stats[stat])
    stat_aliases = {
        "hp": ["base_hp", "hp"],
        "attack": ["base_atk", "base_attack", "attack"],
        "defense": ["base_def", "base_defense", "defense"],
        "special-attack": ["base_spa", "base_special_attack", "special-attack"],
        "special-defense": ["base_spd", "base_special_defense", "special-defense"],
        "speed": ["base_spe", "base_speed", "speed"],
    }
    for alias in stat_aliases.get(stat, [stat]):
        if alias in poke_dict:
            return float(poke_dict[alias])
    return 0.0


def _extract_hp(state: Dict[str, Any], default: float = 100.0) -> float:
    """Read HP on a 0–100 scale from fractional or percentage fields."""
    if not isinstance(state, dict):
        return default
    val = state.get("hp_pct", state.get("hp_percent"))
    if val is None:
        return default
    val = float(val)
    if 0.0 < val <= 1.0:
        return val * 100.0
    return max(0.0, min(100.0, val))
