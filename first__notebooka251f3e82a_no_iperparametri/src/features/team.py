"""Team-stat summaries and type coverage."""

from typing import Any, Dict, List
import numpy as np
from .schema import TYPE_CHART
from .parsing import _get_pokemon_stat, _get_pokemon_types


def compute_type_multiplier(
    attacker_types: List[str],
    defender_types: List[str],
) -> float:
    """Return the largest matchup multiplier for the supplied types."""

    if not attacker_types or not defender_types:
        return 1.0

    max_mult = None
    for atk_t in attacker_types:
        atk_t = atk_t.lower()
        if atk_t not in TYPE_CHART:
            continue
        chart_entry = TYPE_CHART[atk_t]

        current_mult = 1.0
        for def_t in defender_types:
            def_t = def_t.lower()
            if isinstance(chart_entry, dict) and def_t in chart_entry:
                current_mult *= chart_entry[def_t]
            else:
                current_mult *= 1.0

        if max_mult is None or current_mult > max_mult:
            max_mult = current_mult

    return max_mult if max_mult is not None else 1.0


def compute_structural_type_synergy(
    team: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Summarize team resistances, immunities and weakness coverage."""
    all_chart_types = list(TYPE_CHART.keys())
    n_poke = len(team)

    if n_poke == 0 or not all_chart_types:
        return {
            "p1_total_resistances": 0.0,
            "p1_total_immunities": 0.0,
            "p1_covered_weaknesses_ratio": 0.0,
        }

    team_types_list = [_get_pokemon_types(p) for p in team]

    mult_matrix = np.ones((n_poke, len(all_chart_types)), dtype=float)
    for i, p_types in enumerate(team_types_list):
        if not p_types:
            continue
        for t_idx, atk_t in enumerate(all_chart_types):
            mult_matrix[i, t_idx] = compute_type_multiplier([atk_t], p_types)

    total_weaknesses = int(np.sum(mult_matrix > 1.0))
    total_resistances = int(np.sum((mult_matrix > 0.0) & (mult_matrix < 1.0)))
    total_immunities = int(np.sum(mult_matrix == 0.0))

    covered_weaknesses = 0
    for i in range(n_poke):
        for t_idx in range(len(all_chart_types)):
            if mult_matrix[i, t_idx] > 1.0:
                teammates_mults = np.delete(mult_matrix[:, t_idx], i)
                if np.any(teammates_mults < 1.0):
                    covered_weaknesses += 1

    covered_ratio = (
        float(covered_weaknesses / total_weaknesses) if total_weaknesses > 0 else 1.0
    )

    return {
        "p1_total_resistances": float(total_resistances),
        "p1_total_immunities": float(total_immunities),
        "p1_covered_weaknesses_ratio": covered_ratio,
    }


def _statistic(operation, values):
    """Reduce a stat list, returning zero when it is empty."""
    return float(operation(values)) if values else 0.0


def _team_stat_features(p1_team):
    """Summarize P1 base stats and held items."""
    feats = {}
    p1_spec_bulk_list = []
    p1_leftovers_count = 0

    hp_vals = [_get_pokemon_stat(p, "hp") for p in p1_team]
    atk_vals = [_get_pokemon_stat(p, "attack") for p in p1_team]
    def_vals = [_get_pokemon_stat(p, "defense") for p in p1_team]
    spd_vals = [_get_pokemon_stat(p, "special-defense") for p in p1_team]
    spe_vals = [_get_pokemon_stat(p, "speed") for p in p1_team]

    feats["p1_mean_hp"] = _statistic(np.mean, hp_vals)
    feats["p1_std_attack"] = _statistic(np.std, atk_vals)
    feats["p1_max_attack"] = _statistic(np.max, atk_vals)
    feats["p1_min_attack"] = _statistic(np.min, atk_vals)
    feats["p1_mean_defense"] = _statistic(np.mean, def_vals)
    feats["p1_std_defense"] = _statistic(np.std, def_vals)
    feats["p1_max_defense"] = _statistic(np.max, def_vals)
    feats["p1_min_defense"] = _statistic(np.min, def_vals)
    feats["p1_std_special-defense"] = _statistic(np.std, spd_vals)
    feats["p1_min_special-defense"] = _statistic(np.min, spd_vals)
    feats["p1_std_speed"] = _statistic(np.std, spe_vals)
    feats["p1_min_speed"] = _statistic(np.min, spe_vals)

    for p in p1_team:
        hp = _get_pokemon_stat(p, "hp")
        spd = _get_pokemon_stat(p, "special-defense")
        p1_spec_bulk_list.append((hp * spd) / 100.0)
        item = str(p.get("item", "")).lower()
        if "leftovers" in item:
            p1_leftovers_count += 1

    feats["p1_mean_spec_bulk"] = _statistic(np.mean, p1_spec_bulk_list)
    feats["p1_leftovers_count"] = p1_leftovers_count

    return feats
