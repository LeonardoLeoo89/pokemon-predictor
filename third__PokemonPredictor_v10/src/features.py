from itertools import combinations
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

STATS = ["hp", "attack", "defense", "special-attack", "special-defense", "speed"]
STATUS_CONDITIONS = ["frz", "slp", "tox", "psn", "par", "brn"]

STATUS_WEIGHTS: Dict[str, float] = {
    "frz": 3.0,
    "slp": 2.0,
    "tox": 1.8,
    "par": 1.6,
    "brn": 1.5,
    "psn": 1.0,
}

TYPE_CHART: Dict[str, Dict[str, float]] = {
    "normal": {
        "rock": 0.5,
        "ghost": 0.0,
        "steel": 0.5,
    },
    "fire": {
        "fire": 0.5,
        "water": 0.5,
        "grass": 2.0,
        "ice": 2.0,
        "bug": 2.0,
        "rock": 0.5,
        "dragon": 0.5,
        "steel": 2.0,
    },
    "water": {
        "fire": 2.0,
        "water": 0.5,
        "grass": 0.5,
        "ground": 2.0,
        "rock": 2.0,
        "dragon": 0.5,
    },
    "electric": {
        "water": 2.0,
        "electric": 0.5,
        "grass": 0.5,
        "ground": 0.0,
        "flying": 2.0,
        "dragon": 0.5,
    },
    "grass": {
        "fire": 0.5,
        "water": 2.0,
        "grass": 0.5,
        "poison": 0.5,
        "ground": 2.0,
        "flying": 0.5,
        "bug": 0.5,
        "rock": 2.0,
        "dragon": 0.5,
        "steel": 0.5,
    },
    "ice": {
        "fire": 0.5,
        "water": 0.5,
        "grass": 2.0,
        "ice": 0.5,
        "ground": 2.0,
        "flying": 2.0,
        "dragon": 2.0,
        "steel": 0.5,
    },
    "fighting": {
        "normal": 2.0,
        "ice": 2.0,
        "poison": 0.5,
        "flying": 0.5,
        "psychic": 0.5,
        "bug": 0.5,
        "rock": 2.0,
        "ghost": 0.0,
        "dark": 2.0,
        "steel": 2.0,
    },
    "poison": {
        "grass": 2.0,
        "poison": 0.5,
        "ground": 0.5,
        "rock": 0.5,
        "ghost": 0.5,
        "steel": 0.0,
    },
    "ground": {
        "fire": 2.0,
        "electric": 2.0,
        "grass": 0.5,
        "poison": 2.0,
        "flying": 0.0,
        "bug": 0.5,
        "rock": 2.0,
        "steel": 2.0,
    },
    "flying": {
        "electric": 0.5,
        "grass": 2.0,
        "fighting": 2.0,
        "bug": 2.0,
        "rock": 0.5,
        "steel": 0.5,
    },
    "psychic": {
        "fighting": 2.0,
        "poison": 2.0,
        "psychic": 0.5,
        "dark": 0.0,
        "steel": 0.5,
    },
    "bug": {
        "fire": 0.5,
        "grass": 2.0,
        "fighting": 0.5,
        "poison": 0.5,
        "flying": 0.5,
        "psychic": 2.0,
        "ghost": 0.5,
        "dark": 2.0,
        "steel": 0.5,
    },
    "rock": {
        "fire": 2.0,
        "ice": 2.0,
        "fighting": 0.5,
        "ground": 0.5,
        "flying": 2.0,
        "bug": 2.0,
        "steel": 0.5,
    },
    "ghost": {
        "normal": 0.0,
        "psychic": 2.0,
        "ghost": 2.0,
        "dark": 0.5,
        "steel": 0.5,
    },
    "dragon": {
        "dragon": 2.0,
        "steel": 0.5,
    },
    "steel": {
        "fire": 0.5,
        "water": 0.5,
        "electric": 0.5,
        "ice": 2.0,
        "rock": 2.0,
        "steel": 0.5,
    },
    "dark": {
        "fighting": 0.5,
        "psychic": 2.0,
        "ghost": 2.0,
        "dark": 0.5,
        "steel": 0.5,
    },
}

def _get_species_name(poke_dict: Dict[str, Any]) -> str:

    if not isinstance(poke_dict, dict):
        return ""
    val = poke_dict.get("name")
    if val and isinstance(val, str):
        return val.strip().lower()
    return ""

def _get_pokemon_types(poke_dict: Dict[str, Any]) -> List[str]:

    if not isinstance(poke_dict, dict):
        return []
    return [t for t in poke_dict.get("types", []) if t != "notype"]

def _get_pokemon_stat(poke_dict: Dict[str, Any], stat: str) -> float:

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

def compute_type_multiplier(
    attacker_types: List[str],
    defender_types: List[str],
    type_chart: Optional[Dict[str, Any]] = None,
) -> float:

    chart = type_chart if type_chart is not None else TYPE_CHART
    if not chart or not attacker_types or not defender_types:
        return 1.0

    max_mult = None
    for atk_t in attacker_types:
        atk_t = atk_t.lower()
        if atk_t not in chart:
            continue
        chart_entry = chart[atk_t]

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
    type_chart: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:

    chart = type_chart if type_chart is not None else TYPE_CHART
    all_chart_types = list(chart.keys())
    n_poke = len(team)

    if n_poke == 0 or not all_chart_types:
        return {
            "p1_total_weaknesses": 0.0,
            "p1_total_resistances": 0.0,
            "p1_total_immunities": 0.0,
            "p1_covered_weaknesses_ratio": 0.0,
            "p1_shared_uncovered_weaknesses": 0.0,
            "p1_stab_coverage_ratio": 0.0,
            "p1_pairwise_type_complementarity": 0.0,
        }

    team_types_list = [_get_pokemon_types(p) for p in team]

    mult_matrix = np.ones((n_poke, len(all_chart_types)), dtype=float)
    for i, p_types in enumerate(team_types_list):
        if not p_types:
            continue
        for t_idx, atk_t in enumerate(all_chart_types):
            mult_matrix[i, t_idx] = compute_type_multiplier([atk_t], p_types, chart)

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

    covered_ratio = float(covered_weaknesses / total_weaknesses) if total_weaknesses > 0 else 1.0

    shared_uncovered = 0
    for t_idx in range(len(all_chart_types)):
        col = mult_matrix[:, t_idx]
        if np.sum(col > 1.0) >= 2 and not np.any(col < 1.0):
            shared_uncovered += 1

    pair_synergy_scores = []
    for i, j in combinations(range(n_poke), 2):
        i_weak_j_res = np.sum((mult_matrix[i] > 1.0) & (mult_matrix[j] < 1.0))
        j_weak_i_res = np.sum((mult_matrix[j] > 1.0) & (mult_matrix[i] < 1.0))
        shared_weak = np.sum((mult_matrix[i] > 1.0) & (mult_matrix[j] > 1.0))
        pair_synergy_scores.append(float(i_weak_j_res + j_weak_i_res - shared_weak))

    pairwise_comp = float(np.mean(pair_synergy_scores)) if pair_synergy_scores else 0.0

    unique_team_types = list(set(t for p_t in team_types_list for t in p_t))
    types_hit_super_effectively = 0
    for def_t in all_chart_types:
        if compute_type_multiplier(unique_team_types, [def_t], chart) >= 2.0:
            types_hit_super_effectively += 1

    stab_coverage_ratio = float(types_hit_super_effectively / len(all_chart_types))

    return {
        "p1_total_weaknesses": float(total_weaknesses),
        "p1_total_resistances": float(total_resistances),
        "p1_total_immunities": float(total_immunities),
        "p1_covered_weaknesses_ratio": covered_ratio,
        "p1_shared_uncovered_weaknesses": float(shared_uncovered),
        "p1_stab_coverage_ratio": stab_coverage_ratio,
        "p1_pairwise_type_complementarity": pairwise_comp,
    }

def fit_historical_stats(
    train_records: List[Dict[str, Any]],
    smoothing_m: float = 20.0,
    pair_smoothing_m: float = 15.0,
) -> Dict[str, Any]:

    wins_list = [int(b["player_won"]) for b in train_records if "player_won" in b]
    global_prior = float(np.mean(wins_list)) if wins_list else 0.5

    p1_species_stats: Dict[str, List[int]] = {}
    p2_species_stats: Dict[str, List[int]] = {}
    p1_species_pair_stats: Dict[Tuple[str, str], List[int]] = {}
    p1_k_tuple_stats: Dict[int, Dict[Tuple[str, ...], List[int]]] = {k: {} for k in range(1, 7)}
    p1_type_stats: Dict[str, List[int]] = {}
    p1_type_pair_stats: Dict[Tuple[str, str], List[int]] = {}
    lead_matchup_stats: Dict[Tuple[str, str], List[int]] = {}
    species_dex: Dict[str, Dict[str, Any]] = {}

    for battle in train_records:
        if "player_won" not in battle:
            continue
        won = int(battle["player_won"])

        p1_team = battle.get("p1_team_details", [])
        for p in p1_team:
            sp_name = _get_species_name(p)
            if sp_name and sp_name not in species_dex:
                species_dex[sp_name] = {
                    "types": _get_pokemon_types(p),
                    "stats": {s: _get_pokemon_stat(p, s) for s in STATS},
                }

        p2_lead = battle.get("p2_lead_details", {})
        sp2_lead = _get_species_name(p2_lead)
        if sp2_lead and sp2_lead not in species_dex:
            species_dex[sp2_lead] = {
                "types": _get_pokemon_types(p2_lead),
                "stats": {s: _get_pokemon_stat(p2_lead, s) for s in STATS},
            }

        p1_species = sorted(set(filter(None, (_get_species_name(p) for p in p1_team))))
        p1_types = sorted(set(t for p in p1_team for t in _get_pokemon_types(p)))

        for k in range(1, min(7, len(p1_species) + 1)):
            for combo in combinations(p1_species, k):
                if combo not in p1_k_tuple_stats[k]:
                    p1_k_tuple_stats[k][combo] = [0, 0]
                p1_k_tuple_stats[k][combo][0] += won
                p1_k_tuple_stats[k][combo][1] += 1

        for sp in p1_species:
            if sp not in p1_species_stats:
                p1_species_stats[sp] = [0, 0]
            p1_species_stats[sp][0] += won
            p1_species_stats[sp][1] += 1

        for tp in p1_types:
            if tp not in p1_type_stats:
                p1_type_stats[tp] = [0, 0]
            p1_type_stats[tp][0] += won
            p1_type_stats[tp][1] += 1

        for tp_a, tp_b in combinations(p1_types, 2):
            t_pair = (tp_a, tp_b)
            if t_pair not in p1_type_pair_stats:
                p1_type_pair_stats[t_pair] = [0, 0]
            p1_type_pair_stats[t_pair][0] += won
            p1_type_pair_stats[t_pair][1] += 1

        p2_lead_sp = _get_species_name(battle.get("p2_lead_details", {}))
        p2_seen_species = set()
        if p2_lead_sp:
            p2_seen_species.add(p2_lead_sp)

        for turn_data in battle.get("battle_timeline", []):
            p2_state = turn_data.get("p2_pokemon_state", turn_data.get("p2_state", {})) or {}
            sp_turn = _get_species_name(p2_state)
            if sp_turn:
                p2_seen_species.add(sp_turn)

        for sp2 in p2_seen_species:
            if sp2 not in p2_species_stats:
                p2_species_stats[sp2] = [0, 0]
            p2_species_stats[sp2][0] += won
            p2_species_stats[sp2][1] += 1

        p1_lead_sp = _get_species_name(p1_team[0]) if p1_team else ""
        if p1_lead_sp and p2_lead_sp:
            m_key = (p1_lead_sp, p2_lead_sp)
            if m_key not in lead_matchup_stats:
                lead_matchup_stats[m_key] = [0, 0]
            lead_matchup_stats[m_key][0] += won
            lead_matchup_stats[m_key][1] += 1

    def _smooth(w: int, c: int, m: float) -> float:
        return float((w + m * global_prior) / (c + m))

    p1_sp_wr = {sp: _smooth(w, c, smoothing_m) for sp, (w, c) in p1_species_stats.items()}

    p2_sp_wr = {sp: 1.0 - _smooth(w, c, smoothing_m) for sp, (w, c) in p2_species_stats.items()}

    k_smoothing_m = {1: smoothing_m, 2: pair_smoothing_m, 3: 10.0, 4: 8.0, 5: 5.0, 6: 3.0}
    p1_k_wr: Dict[int, Dict[Tuple[str, ...], float]] = {}
    p1_k_synergy: Dict[int, Dict[Tuple[str, ...], float]] = {}

    for k in range(1, 7):
        m_k = k_smoothing_m.get(k, 5.0)
        p1_k_wr[k] = {combo: _smooth(w, c, m_k) for combo, (w, c) in p1_k_tuple_stats[k].items()}
        if k >= 2:
            p1_k_synergy[k] = {}
            for combo, combo_wr in p1_k_wr[k].items():
                expected_wr = float(np.mean([p1_sp_wr.get(sp, global_prior) for sp in combo]))
                p1_k_synergy[k][combo] = float(combo_wr - expected_wr)

    p1_pair_wr = p1_k_wr[2]
    p1_pair_synergy = p1_k_synergy[2]

    p1_tp_wr = {tp: _smooth(w, c, smoothing_m) for tp, (w, c) in p1_type_stats.items()}
    p1_tp_pair_wr = {}
    p1_tp_pair_synergy = {}
    for (tp_a, tp_b), (w, c) in p1_type_pair_stats.items():
        t_wr = _smooth(w, c, pair_smoothing_m)
        p1_tp_pair_wr[(tp_a, tp_b)] = t_wr
        expected_t_wr = 0.5 * (p1_tp_wr.get(tp_a, global_prior) + p1_tp_wr.get(tp_b, global_prior))
        p1_tp_pair_synergy[(tp_a, tp_b)] = float(t_wr - expected_t_wr)

    lead_matchup_wr = {
        k: _smooth(w, c, pair_smoothing_m) for k, (w, c) in lead_matchup_stats.items()
    }

    return {
        "global_prior": global_prior,
        "p1_species_wr": p1_sp_wr,
        "p2_species_wr": p2_sp_wr,
        "p1_k_wr": p1_k_wr,
        "p1_k_synergy": p1_k_synergy,
        "p1_pair_wr": p1_pair_wr,
        "p1_pair_synergy": p1_pair_synergy,
        "p1_type_wr": p1_tp_wr,
        "p1_type_pair_wr": p1_tp_pair_wr,
        "p1_type_pair_synergy": p1_tp_pair_synergy,
        "lead_matchup_wr": lead_matchup_wr,
        "species_dex": species_dex,
    }

def extract_empirical_synergy_features(
    battle: Dict[str, Any],
    historical_stats: Optional[Dict[str, Any]],
) -> Dict[str, float]:

    default_feats: Dict[str, float] = {
        "p1_team_mean_species_wr": 0.5,
        "p1_team_max_species_wr": 0.5,
        "p1_team_min_species_wr": 0.5,
        "p1_team_std_species_wr": 0.0,
        "p1_lead_species_wr": 0.5,
        "p2_lead_species_wr": 0.5,
        "p2_revealed_mean_species_wr": 0.5,
        "species_wr_advantage": 0.0,
        "lead_matchup_empirical_wr": 0.5,
        "p1_species_pair_mean_wr": 0.5,
        "p1_species_pair_max_wr": 0.5,
        "p1_species_pair_synergy_lift_mean": 0.0,
        "p1_species_pair_synergy_lift_max": 0.0,
        "p1_type_mean_empirical_wr": 0.5,
        "p1_type_pair_mean_wr": 0.5,
        "p1_type_pair_synergy_lift_mean": 0.0,
    }
    for k in range(1, 7):
        default_feats[f"p1_k{k}_mean_wr"] = 0.5
        default_feats[f"p1_k{k}_max_wr"] = 0.5
        default_feats[f"p1_k{k}_min_wr"] = 0.5
        if k >= 2:
            default_feats[f"p1_k{k}_synergy_lift_mean"] = 0.0
            default_feats[f"p1_k{k}_synergy_lift_max"] = 0.0

    if not historical_stats:
        return default_feats

    prior = historical_stats.get("global_prior", 0.5)
    p1_sp_map = historical_stats.get("p1_species_wr", {})
    p2_sp_map = historical_stats.get("p2_species_wr", {})
    p1_k_wr_map = historical_stats.get("p1_k_wr", {})
    p1_k_syn_map = historical_stats.get("p1_k_synergy", {})
    p1_tp_map = historical_stats.get("p1_type_wr", {})
    p1_tp_pair_wr_map = historical_stats.get("p1_type_pair_wr", {})
    p1_tp_pair_syn_map = historical_stats.get("p1_type_pair_synergy", {})
    lead_matchup_map = historical_stats.get("lead_matchup_wr", {})

    p1_team = battle.get("p1_team_details", [])
    p1_species_ordered = [_get_species_name(p) for p in p1_team if _get_species_name(p)]
    p1_species_unique = sorted(set(p1_species_ordered))
    p1_types_unique = sorted(set(t for p in p1_team for t in _get_pokemon_types(p)))

    p1_wrs = [p1_sp_map.get(sp, prior) for sp in p1_species_ordered] if p1_species_ordered else [prior]
    p1_lead_sp = p1_species_ordered[0] if p1_species_ordered else ""
    p1_lead_wr = p1_sp_map.get(p1_lead_sp, prior)

    p2_lead_sp = _get_species_name(battle.get("p2_lead_details", {}))
    p2_lead_wr = p2_sp_map.get(p2_lead_sp, 1.0 - prior)

    p2_seen_species = set()
    if p2_lead_sp:
        p2_seen_species.add(p2_lead_sp)
    for turn_data in battle.get("battle_timeline", []):
        p2_state = turn_data.get("p2_pokemon_state", turn_data.get("p2_state", {})) or {}
        sp_turn = _get_species_name(p2_state)
        if sp_turn:
            p2_seen_species.add(sp_turn)

    p2_revealed_wrs = [p2_sp_map.get(sp, 1.0 - prior) for sp in p2_seen_species] if p2_seen_species else [1.0 - prior]

    feats: Dict[str, float] = {}

    for k in range(1, 7):
        k_combos = list(combinations(p1_species_unique, k))
        if k_combos:
            k_table = p1_k_wr_map.get(k, {})
            k_syn_table = p1_k_syn_map.get(k, {})
            k_wrs = [k_table.get(c, prior) for c in k_combos]
            feats[f"p1_k{k}_mean_wr"] = float(np.mean(k_wrs))
            feats[f"p1_k{k}_max_wr"] = float(np.max(k_wrs))
            feats[f"p1_k{k}_min_wr"] = float(np.min(k_wrs))
            if k >= 2:
                k_lifts = [k_syn_table.get(c, 0.0) for c in k_combos]
                feats[f"p1_k{k}_synergy_lift_mean"] = float(np.mean(k_lifts))
                feats[f"p1_k{k}_synergy_lift_max"] = float(np.max(k_lifts))
        else:
            feats[f"p1_k{k}_mean_wr"] = prior
            feats[f"p1_k{k}_max_wr"] = prior
            feats[f"p1_k{k}_min_wr"] = prior
            if k >= 2:
                feats[f"p1_k{k}_synergy_lift_mean"] = 0.0
                feats[f"p1_k{k}_synergy_lift_max"] = 0.0

    feats["p1_species_pair_mean_wr"] = feats["p1_k2_mean_wr"]
    feats["p1_species_pair_max_wr"] = feats["p1_k2_max_wr"]
    feats["p1_species_pair_synergy_lift_mean"] = feats["p1_k2_synergy_lift_mean"]
    feats["p1_species_pair_synergy_lift_max"] = feats["p1_k2_synergy_lift_max"]

    type_wrs = [p1_tp_map.get(tp, prior) for tp in p1_types_unique] if p1_types_unique else [prior]
    type_pair_wrs = []
    type_pair_lifts = []
    for tp_a, tp_b in combinations(p1_types_unique, 2):
        type_pair_wrs.append(p1_tp_pair_wr_map.get((tp_a, tp_b), prior))
        type_pair_lifts.append(p1_tp_pair_syn_map.get((tp_a, tp_b), 0.0))

    lead_matchup_wr = lead_matchup_map.get((p1_lead_sp, p2_lead_sp), prior) if (p1_lead_sp and p2_lead_sp) else prior

    mean_p1_wr = float(np.mean(p1_wrs))
    mean_p2_wr = float(np.mean(p2_revealed_wrs))

    feats["p1_team_mean_species_wr"] = mean_p1_wr
    feats["p1_team_max_species_wr"] = float(np.max(p1_wrs))
    feats["p1_team_min_species_wr"] = float(np.min(p1_wrs))
    feats["p1_team_std_species_wr"] = float(np.std(p1_wrs))
    feats["p1_lead_species_wr"] = float(p1_lead_wr)
    feats["p2_lead_species_wr"] = float(p2_lead_wr)
    feats["p2_revealed_mean_species_wr"] = mean_p2_wr
    feats["species_wr_advantage"] = float(mean_p1_wr - mean_p2_wr)
    feats["lead_matchup_empirical_wr"] = float(lead_matchup_wr)
    feats["p1_type_mean_empirical_wr"] = float(np.mean(type_wrs))
    feats["p1_type_pair_mean_wr"] = float(np.mean(type_pair_wrs)) if type_pair_wrs else prior
    feats["p1_type_pair_synergy_lift_mean"] = float(np.mean(type_pair_lifts)) if type_pair_lifts else 0.0

    return feats

def _extract_hp(state: Dict[str, Any], default: float = 100.0) -> float:

    if not isinstance(state, dict):
        return default
    val = state.get("hp_pct", state.get("hp_percent"))
    if val is None:
        return default
    val = float(val)
    if 0.0 < val <= 1.0:
        return val * 100.0
    return max(0.0, min(100.0, val))

def extract_timeline_dynamics(battle_or_timeline: Any) -> Dict[str, Any]:

    if isinstance(battle_or_timeline, dict):
        timeline = battle_or_timeline.get("battle_timeline", [])
    elif isinstance(battle_or_timeline, list):
        timeline = battle_or_timeline
    else:
        timeline = []

    feats: Dict[str, Any] = {"turns_observed": len(timeline)}

    p1_fainted_set = set()
    p2_fainted_set = set()
    p1_fainted_early = 0
    p2_fainted_early = 0
    p1_fainted_late = 0
    p2_fainted_late = 0

    p1_switches = 0
    p2_switches = 0
    p1_status_turns = 0
    p2_status_turns = 0
    p1_active_hp = 100.0
    p2_active_hp = 100.0
    p1_hp_history = []
    p2_hp_history = []

    p1_revealed_roster: Dict[str, Dict[str, Any]] = {}
    p2_revealed_roster: Dict[str, Dict[str, Any]] = {}

    last_p1_sp = None
    last_p2_sp = None
    p1_damage_dealt = 0.0
    p2_damage_dealt = 0.0
    p1_spikes = 0
    p2_spikes = 0
    p1_spikes_live = False
    p2_spikes_live = False
    p1_hazard_chip = 0.0
    p2_hazard_chip = 0.0
    p1_spins = 0
    p2_spins = 0
    p1_booms = 0
    p2_booms = 0
    p1_rests = 0
    p2_rests = 0

    for idx, turn_data in enumerate(timeline):
        is_early = idx < 15
        p1_state = turn_data.get("p1_pokemon_state", turn_data.get("p1_state", {})) or {}
        p2_state = turn_data.get("p2_pokemon_state", turn_data.get("p2_state", {})) or {}

        p1_sp = _get_species_name(p1_state) or f"p1_slot_{len(p1_revealed_roster)}"
        p2_sp = _get_species_name(p2_state) or f"p2_slot_{len(p2_revealed_roster)}"

        action_p1_switch = str(turn_data.get("p1_action", "")).lower() == "switch"
        action_p2_switch = str(turn_data.get("p2_action", "")).lower() == "switch"
        name_p1_switch = (last_p1_sp is not None and p1_sp != last_p1_sp)
        name_p2_switch = (last_p2_sp is not None and p2_sp != last_p2_sp)

        if action_p1_switch or name_p1_switch:
            p1_switches += 1
            if p2_spikes_live:
                p1_hazard_chip += 12.5
        if action_p2_switch or name_p2_switch:
            p2_switches += 1
            if p1_spikes_live:
                p2_hazard_chip += 12.5

        last_p1_sp = p1_sp
        last_p2_sp = p2_sp

        p1_hp = _extract_hp(p1_state, default=p1_active_hp)
        p2_hp = _extract_hp(p2_state, default=p2_active_hp)

        p1_status = str(p1_state.get("status", "")).lower()
        p2_status = str(p2_state.get("status", "")).lower()

        p1_ko = (
            bool(turn_data.get("p1_fainted", p1_state.get("fainted", False)))
            or p1_status == "fnt"
            or p1_hp <= 0.0
            or p1_sp in p1_fainted_set
        )
        p2_ko = (
            bool(turn_data.get("p2_fainted", p2_state.get("fainted", False)))
            or p2_status == "fnt"
            or p2_hp <= 0.0
            or p2_sp in p2_fainted_set
        )

        if p1_ko:
            p1_hp = 0.0
            if p1_sp not in p1_fainted_set:
                p1_fainted_set.add(p1_sp)
                if is_early:
                    p1_fainted_early += 1
                else:
                    p1_fainted_late += 1

        if p2_ko:
            p2_hp = 0.0
            if p2_sp not in p2_fainted_set:
                p2_fainted_set.add(p2_sp)
                if is_early:
                    p2_fainted_early += 1
                else:
                    p2_fainted_late += 1

        p1_active_hp = p1_hp
        p2_active_hp = p2_hp
        p1_hp_history.append(p1_active_hp)
        p2_hp_history.append(p2_active_hp)

        if p1_status in STATUS_CONDITIONS:
            p1_status_turns += 1
        if p2_status in STATUS_CONDITIONS:
            p2_status_turns += 1

        p1_revealed_roster[p1_sp] = {
            "hp": 0.0 if p1_ko else p1_active_hp,
            "status": "" if p1_ko else p1_status,
            "fainted": p1_ko,
        }
        p2_revealed_roster[p2_sp] = {
            "hp": 0.0 if p2_ko else p2_active_hp,
            "status": "" if p2_ko else p2_status,
            "fainted": p2_ko,
        }

        if idx > 0:
            prev_turn = timeline[idx - 1]
            prev_p1 = prev_turn.get("p1_pokemon_state", prev_turn.get("p1_state", {})) or {}
            prev_p2 = prev_turn.get("p2_pokemon_state", prev_turn.get("p2_state", {})) or {}
            prev_sp1 = _get_species_name(prev_p1)
            prev_sp2 = _get_species_name(prev_p2)
            prev_hp1 = _extract_hp(prev_p1)
            prev_hp2 = _extract_hp(prev_p2)
            if prev_sp1 == p1_sp and prev_hp1 > p1_hp:
                p2_damage_dealt += (prev_hp1 - p1_hp)
            if prev_sp2 == p2_sp and prev_hp2 > p2_hp:
                p1_damage_dealt += (prev_hp2 - p2_hp)

        m1 = turn_data.get("p1_move_details")
        m2 = turn_data.get("p2_move_details")
        if isinstance(m1, dict):
            m1_name = str(m1.get("name", "")).lower()
            if m1_name == "spikes":
                p1_spikes += 1
                p1_spikes_live = True
            elif "spin" in m1_name:
                p1_spins += 1
                p2_spikes_live = False
            elif "explosion" in m1_name or "self-destruct" in m1_name:
                p1_booms += 1
            elif m1_name == "rest":
                p1_rests += 1
        if isinstance(m2, dict):
            m2_name = str(m2.get("name", "")).lower()
            if m2_name == "spikes":
                p2_spikes += 1
                p2_spikes_live = True
            elif "spin" in m2_name:
                p2_spins += 1
                p1_spikes_live = False
            elif "explosion" in m2_name or "self-destruct" in m2_name:
                p2_booms += 1
            elif m2_name == "rest":
                p2_rests += 1

    p1_fainted_final = min(6, len(p1_fainted_set))
    p2_fainted_final = min(6, len(p2_fainted_set))

    feats["p1_fainted_count"] = p1_fainted_final
    feats["p2_fainted_count"] = p2_fainted_final
    feats["fainted_diff"] = p2_fainted_final - p1_fainted_final
    feats["fainted_diff_early"] = p2_fainted_early - p1_fainted_early
    feats["fainted_diff_late"] = p2_fainted_late - p1_fainted_late
    feats["fainted_momentum_shift"] = feats["fainted_diff_late"] - feats["fainted_diff_early"]

    feats["p1_remaining_estimated"] = 6 - p1_fainted_final
    feats["p2_remaining_estimated"] = 6 - p2_fainted_final
    feats["remaining_diff"] = feats["p1_remaining_estimated"] - feats["p2_remaining_estimated"]

    p1_unrev = max(0, 6 - len(p1_revealed_roster))
    p2_unrev = max(0, 6 - len(p2_revealed_roster))
    p1_total_hp = sum(v["hp"] for v in p1_revealed_roster.values() if not v["fainted"]) + p1_unrev * 100.0
    p2_total_hp = sum(v["hp"] for v in p2_revealed_roster.values() if not v["fainted"]) + p2_unrev * 100.0

    feats["p1_total_hp_pool"] = float(p1_total_hp)
    feats["p2_total_hp_pool"] = float(p2_total_hp)
    feats["total_hp_pool_diff"] = float(p1_total_hp - p2_total_hp)
    feats["total_hp_pool_ratio"] = float(p1_total_hp / (p2_total_hp + 1.0))

    feats["p1_active_hp_percent"] = p1_active_hp
    feats["p2_active_hp_percent"] = p2_active_hp
    feats["hp_diff_end"] = p1_active_hp - p2_active_hp

    feats["p1_switches"] = p1_switches
    feats["p2_switches"] = p2_switches
    feats["switch_diff"] = p1_switches - p2_switches

    feats["p1_status_turns"] = p1_status_turns
    feats["p2_status_turns"] = p2_status_turns
    feats["status_diff"] = p2_status_turns - p1_status_turns

    p1_sev = sum(STATUS_WEIGHTS.get(v["status"], 0.0) for v in p1_revealed_roster.values() if not v["fainted"])
    p2_sev = sum(STATUS_WEIGHTS.get(v["status"], 0.0) for v in p2_revealed_roster.values() if not v["fainted"])
    feats["status_severity_diff"] = float(p2_sev - p1_sev)

    feats["p1_damage_dealt"] = float(p1_damage_dealt)
    feats["p2_damage_dealt"] = float(p2_damage_dealt)
    feats["damage_dealt_diff"] = float(p1_damage_dealt - p2_damage_dealt)
    feats["p1_spikes_set"] = p1_spikes
    feats["p2_spikes_set"] = p2_spikes
    feats["spikes_diff"] = p1_spikes - p2_spikes
    feats["p1_spikes_chip_damage"] = float(p1_spikes * p2_switches * 12.5)
    feats["p2_spikes_chip_damage"] = float(p2_spikes * p1_switches * 12.5)
    feats["spikes_chip_damage_diff"] = feats["p1_spikes_chip_damage"] - feats["p2_spikes_chip_damage"]
    feats["p1_rests_used"] = p1_rests
    feats["p2_rests_used"] = p2_rests

    end_p1 = timeline[-1].get("p1_pokemon_state", timeline[-1].get("p1_state", {})) if timeline else {}
    end_p2 = timeline[-1].get("p2_pokemon_state", timeline[-1].get("p2_state", {})) if timeline else {}
    b1_dict = end_p1.get("boosts", {}) if isinstance(end_p1.get("boosts"), dict) else {}
    b2_dict = end_p2.get("boosts", {}) if isinstance(end_p2.get("boosts"), dict) else {}
    b1_net = sum(b1_dict.values())
    b2_net = sum(b2_dict.values())
    feats["p1_active_boosts_net"] = float(b1_net)
    feats["p2_active_boosts_net"] = float(b2_net)
    feats["active_boosts_diff"] = float(b1_net - b2_net)
    feats["active_stat_boost_advantage"] = float(b1_net - b2_net)

    if len(timeline) >= 20:
        t20 = timeline[19]
        st1_20 = t20.get("p1_pokemon_state", t20.get("p1_state", {})) or {}
        st2_20 = t20.get("p2_pokemon_state", t20.get("p2_state", {})) or {}
        hp1_20 = _extract_hp(st1_20)
        hp2_20 = _extract_hp(st2_20)
        feats["late_game_hp_momentum"] = float((p1_active_hp - hp1_20) - (p2_active_hp - hp2_20))
    else:
        feats["late_game_hp_momentum"] = 0.0

    st1_status = str(end_p1.get("status", "")).lower()
    st2_status = str(end_p2.get("status", "")).lower()
    incap1 = 1.0 if st1_status in ["frz", "slp"] else 0.0
    incap2 = 1.0 if st2_status in ["frz", "slp"] else 0.0
    feats["active_incapacitation_advantage"] = float(incap2 - incap1)

    feats["p1_hazard_chip_damage"] = float(p1_hazard_chip)
    feats["p2_hazard_chip_damage"] = float(p2_hazard_chip)
    feats["true_hazard_chip_damage_diff"] = float(p2_hazard_chip - p1_hazard_chip)
    feats["active_spikes_at_turn30_diff"] = float(p1_spikes_live) - float(p2_spikes_live)
    feats["rapid_spins_diff"] = float(p1_spins - p2_spins)
    feats["explosions_diff"] = float(p1_booms - p2_booms)

    eff1 = [str(e).lower() for e in (end_p1.get("effects", []) or [])]
    eff2 = [str(e).lower() for e in (end_p2.get("effects", []) or [])]
    sub1 = 1.0 if "substitute" in eff1 else 0.0
    sub2 = 1.0 if "substitute" in eff2 else 0.0
    feats["active_substitute_advantage"] = float(sub1 - sub2)

    hp_act_diff = p1_active_hp - p2_active_hp
    feats["active_tempo_advantage"] = float((hp_act_diff / 100.0) + (b1_net - b2_net) * 0.5)

    p1_surv_count = max(1, 6 - p1_fainted_final)
    p2_surv_count = max(1, 6 - p2_fainted_final)
    feats["hp_per_surv_diff"] = float((p1_total_hp / p1_surv_count) - (p2_total_hp / p2_surv_count))

    if len(p1_hp_history) >= 2:
        feats["p1_hp_slope"] = float((p1_hp_history[-1] - p1_hp_history[0]) / len(p1_hp_history))
    else:
        feats["p1_hp_slope"] = 0.0

    return feats

MEDIAN_BATTLE_LENGTH_HP: float = 80.27
MEDIAN_FIRST_FAINT_TURN: float = 17.00
MEDIAN_BATTLE_LENGTH_FAINT: float = 180.00

def compute_battle_length_and_offensive_advantage(
    p1_team: List[Dict[str, Any]],
    timeline: List[Dict[str, Any]],
    extracted_timeline_feats: Optional[Dict[str, Any]] = None,
    median_battle_length: float = MEDIAN_BATTLE_LENGTH_HP,
    median_first_faint: float = MEDIAN_FIRST_FAINT_TURN,
) -> Dict[str, float]:

    atk_list = [_get_pokemon_stat(p, "attack") for p in p1_team]
    spa_list = [_get_pokemon_stat(p, "special-attack") for p in p1_team]
    spe_list = [_get_pokemon_stat(p, "speed") for p in p1_team]
    hp_list = [_get_pokemon_stat(p, "hp") for p in p1_team]
    def_list = [_get_pokemon_stat(p, "defense") for p in p1_team]
    spd_list = [_get_pokemon_stat(p, "special-defense") for p in p1_team]

    mean_atk = float(np.mean(atk_list)) if atk_list else 70.0
    mean_spa = float(np.mean(spa_list)) if spa_list else 70.0
    mean_spe = float(np.mean(spe_list)) if spe_list else 70.0
    mean_hp = float(np.mean(hp_list)) if hp_list else 70.0
    mean_def = float(np.mean(def_list)) if def_list else 70.0
    mean_spd = float(np.mean(spd_list)) if spd_list else 70.0

    p1_off_stat = (mean_atk + mean_spa + mean_spe) / (mean_hp + mean_def + mean_spd + 1e-5)

    moves = [t.get("p1_move_details", {}) for t in timeline if t.get("p1_move_details")]
    damaging_moves = sum(1 for m in moves if float(m.get("base_power", 0) or 0) > 0)
    move_off_ratio = float(damaging_moves / max(1, len(moves)))

    p1_offensiveness = float(0.7 * p1_off_stat + 0.3 * move_off_ratio)

    if extracted_timeline_feats:
        p1_hp_pool = float(extracted_timeline_feats.get("p1_total_hp_pool", 600.0))
        p2_hp_pool = float(extracted_timeline_feats.get("p2_total_hp_pool", 600.0))
        p1_f = int(extracted_timeline_feats.get("p1_fainted_count", 0))
        p2_f = int(extracted_timeline_feats.get("p2_fainted_count", 0))
    else:
        p1_hp_pool = 600.0
        p2_hp_pool = 600.0
        p1_f = 0
        p2_f = 0

    tot_hp_lost = max(1.0, (600.0 - p1_hp_pool) + (600.0 - p2_hp_pool))
    hp_rate = tot_hp_lost / 30.0
    b_len_hp = float(min(600.0, 1200.0 / hp_rate))

    first_faint_turn = 30.0
    for t_idx, t in enumerate(timeline):
        st1 = t.get("p1_pokemon_state", t.get("p1_state", {})) or {}
        st2 = t.get("p2_pokemon_state", t.get("p2_state", {})) or {}
        if st1.get("status") == "fnt" or _extract_hp(st1) <= 0.0 or st2.get("status") == "fnt" or _extract_hp(st2) <= 0.0:
            first_faint_turn = float(t.get("turn", t_idx + 1))
            break

    offensive_tempo_adv = float(p1_offensiveness * (median_battle_length - b_len_hp) / median_battle_length)

    offensive_first_faint_adv = float(p1_offensiveness * (median_first_faint - first_faint_turn) / median_first_faint)

    p1_surv_len = float(30.0 * 6.0 / (p1_f + 0.25))
    p2_surv_len = float(30.0 * 6.0 / (p2_f + 0.25))
    offensive_survival_leverage = float(p1_offensiveness * (p1_surv_len - p2_surv_len) / median_battle_length)

    return {
        "median_battle_length": float(median_battle_length),
        "battle_length_est": b_len_hp,
        "battle_length_diff_from_median": float(median_battle_length - b_len_hp),
        "p1_offensiveness_score": p1_offensiveness,
        "first_faint_turn": first_faint_turn,
        "first_faint_diff_from_median": float(median_first_faint - first_faint_turn),
        "offensive_tempo_advantage": offensive_tempo_adv,
        "offensive_first_faint_advantage": offensive_first_faint_adv,
        "offensive_survival_leverage": offensive_survival_leverage,
    }

INTERNAL_SPECIES_CACHE: Dict[str, Dict[str, Any]] = {}

def register_species_from_battle(battle: Dict[str, Any]) -> None:

    for p in battle.get("p1_team_details", []):
        sp = _get_species_name(p)
        if sp and sp not in INTERNAL_SPECIES_CACHE:
            INTERNAL_SPECIES_CACHE[sp] = {
                "types": _get_pokemon_types(p),
                "stats": {s: _get_pokemon_stat(p, s) for s in STATS},
            }
    p2_lead = battle.get("p2_lead_details", {})
    sp2 = _get_species_name(p2_lead)
    if sp2 and sp2 not in INTERNAL_SPECIES_CACHE:
        INTERNAL_SPECIES_CACHE[sp2] = {
            "types": _get_pokemon_types(p2_lead),
            "stats": {s: _get_pokemon_stat(p2_lead, s) for s in STATS},
        }

def get_internal_species_info(
    species_name: str,
    species_dex: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    sp = str(species_name).lower().strip().replace(" ", "").replace("-", "")
    if species_dex and sp in species_dex:
        return species_dex[sp]
    if sp in INTERNAL_SPECIES_CACHE:
        return INTERNAL_SPECIES_CACHE[sp]
    for k, v in INTERNAL_SPECIES_CACHE.items():
        if k in sp or sp in k:
            return v
    return {
        "types": ["normal"],
        "stats": {s: 70.0 for s in STATS},
    }

def trace_opponent_pokemon_and_features(
    battle: Dict[str, Any],
    type_chart: Optional[Dict[str, Any]] = None,
    species_dex: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:

    chart = type_chart if type_chart is not None else TYPE_CHART
    register_species_from_battle(battle)

    p1_team = battle.get("p1_team_details", [])
    p2_lead = battle.get("p2_lead_details", {})
    timeline = battle.get("battle_timeline", [])

    p2_revealed: Dict[str, Dict[str, Any]] = {}
    lead_sp = _get_species_name(p2_lead)
    if lead_sp:
        lead_types = _get_pokemon_types(p2_lead)
        lead_stats = {s: _get_pokemon_stat(p2_lead, s) for s in STATS}
        p2_revealed[lead_sp] = {
            "types": lead_types,
            "stats": lead_stats,
            "fainted": False,
            "hp": 100.0,
            "turns_active": 0,
        }

    p1_fainted = set()
    p1_hp_map: Dict[str, float] = {}
    end_p1_sp = ""
    end_p2_sp = ""

    for t_idx, t in enumerate(timeline):
        st1 = t.get("p1_pokemon_state", t.get("p1_state", {})) or {}
        st2 = t.get("p2_pokemon_state", t.get("p2_state", {})) or {}
        s1 = _get_species_name(st1)
        s2 = _get_species_name(st2)

        if s1:
            end_p1_sp = s1
            hp1 = _extract_hp(st1)
            p1_hp_map[s1] = 0.0 if st1.get("status") == "fnt" or hp1 <= 0.0 else hp1
            if st1.get("status") == "fnt" or hp1 <= 0.0:
                p1_fainted.add(s1)

        if s2:
            end_p2_sp = s2
            hp2 = _extract_hp(st2)
            if s2 not in p2_revealed:
                dex_info = get_internal_species_info(s2, species_dex)
                p2_revealed[s2] = {
                    "types": dex_info.get("types", ["normal"]),
                    "stats": dex_info.get("stats", {s: 70.0 for s in STATS}),
                    "fainted": False,
                    "hp": hp2,
                    "turns_active": 1,
                }
            else:
                p2_revealed[s2]["hp"] = hp2
                p2_revealed[s2]["turns_active"] += 1

            if st2.get("status") == "fnt" or hp2 <= 0.0:
                p2_revealed[s2]["fainted"] = True

    n_p2_rev = len(p2_revealed)
    n_p2_fnt = sum(1 for v in p2_revealed.values() if v["fainted"])
    n_p2_surv = max(0, n_p2_rev - n_p2_fnt)
    p2_faint_rate = float(n_p2_fnt / max(1, n_p2_rev))

    p2_bsts = [sum(v["stats"].values()) for v in p2_revealed.values()]
    p2_speeds = [v["stats"].get("speed", 70.0) for v in p2_revealed.values()]
    p2_mean_bst = float(np.mean(p2_bsts)) if p2_bsts else 450.0
    p2_max_bst = float(np.max(p2_bsts)) if p2_bsts else 450.0
    p2_mean_speed = float(np.mean(p2_speeds)) if p2_speeds else 70.0
    p2_max_speed = float(np.max(p2_speeds)) if p2_speeds else 70.0

    p1_bsts = [sum(_get_pokemon_stat(p, s) for s in STATS) for p in p1_team]
    p1_speeds = [_get_pokemon_stat(p, "speed") for p in p1_team]
    p1_mean_bst = float(np.mean(p1_bsts)) if p1_bsts else 450.0
    p1_mean_speed = float(np.mean(p1_speeds)) if p1_speeds else 70.0

    bst_diff_vs_known = p1_mean_bst - p2_mean_bst
    speed_diff_vs_known = p1_mean_speed - p2_mean_speed

    p1_se_count = 0
    p2_se_count = 0
    p1_wall_count = 0
    p2_wall_count = 0
    p2_threatened = set()
    p1_threatened = set()

    for p in p1_team:
        p_sp = _get_species_name(p)
        p_t = _get_pokemon_types(p)
        for q_sp, q_v in p2_revealed.items():
            q_t = q_v["types"]
            m1 = compute_type_multiplier(p_t, q_t, chart)
            m2 = compute_type_multiplier(q_t, p_t, chart)
            if m1 >= 2.0:
                p1_se_count += 1
                p2_threatened.add(q_sp)
            elif m1 <= 0.5:
                p2_wall_count += 1

            if m2 >= 2.0:
                p2_se_count += 1
                p1_threatened.add(p_sp)
            elif m2 <= 0.5 or m2 == 0.0:
                p1_wall_count += 1

    net_se_adv = p1_se_count - p2_se_count
    p2_threatened_ratio = float(len(p2_threatened) / max(1, n_p2_rev))
    p1_threatened_ratio = float(len(p1_threatened) / max(1, len(p1_team)))
    threatened_net_ratio = float(p2_threatened_ratio - p1_threatened_ratio)
    wall_adv = float(p1_wall_count - p2_wall_count)

    surv_p1 = [p for p in p1_team if _get_species_name(p) not in p1_fainted]
    surv_p2 = [q_v for q_sp, q_v in p2_revealed.items() if not q_v["fainted"]]
    surv_p1_se = 0
    surv_p2_se = 0
    for p in surv_p1:
        p_t = _get_pokemon_types(p)
        for q_v in surv_p2:
            q_t = q_v["types"]
            if compute_type_multiplier(p_t, q_t, chart) >= 2.0:
                surv_p1_se += 1
            if compute_type_multiplier(q_t, p_t, chart) >= 2.0:
                surv_p2_se += 1
    surv_net_se_adv = float(surv_p1_se - surv_p2_se)
    surv_p2_threatened_ratio = float(surv_p1_se > 0) if surv_p2 else 1.0

    p1_surv_bst = sum(sum(_get_pokemon_stat(p, s) for s in STATS) for p in p1_team if _get_species_name(p) not in p1_fainted)
    p2_surv_bst = sum(sum(v["stats"].values()) for v in p2_revealed.values() if not v["fainted"]) + max(0, 6 - n_p2_rev) * 500.0
    surv_bst_diff = float(p1_surv_bst - p2_surv_bst)

    p1_surv_spe = sum(_get_pokemon_stat(p, "speed") for p in p1_team if _get_species_name(p) not in p1_fainted)
    p2_surv_spe = sum(v["stats"].get("speed", 70.0) for v in p2_revealed.values() if not v["fainted"]) + max(0, 6 - n_p2_rev) * 70.0
    surv_spe_diff = float(p1_surv_spe - p2_surv_spe)

    p1_eff_bst = sum(
        sum(_get_pokemon_stat(p, s) for s in STATS) * (p1_hp_map.get(_get_species_name(p), 100.0) / 100.0)
        for p in p1_team if _get_species_name(p) not in p1_fainted
    )
    p2_eff_bst = sum(
        sum(v["stats"].values()) * (v["hp"] / 100.0)
        for v in p2_revealed.values() if not v["fainted"]
    ) + max(0, 6 - n_p2_rev) * 500.0
    eff_bst_diff = float(p1_eff_bst - p2_eff_bst)

    p1_eff_spe = sum(
        _get_pokemon_stat(p, "speed") * (p1_hp_map.get(_get_species_name(p), 100.0) / 100.0)
        for p in p1_team if _get_species_name(p) not in p1_fainted
    )
    p2_eff_spe = sum(
        v["stats"].get("speed", 70.0) * (v["hp"] / 100.0)
        for v in p2_revealed.values() if not v["fainted"]
    ) + max(0, 6 - n_p2_rev) * 70.0
    eff_spe_diff = float(p1_eff_spe - p2_eff_spe)

    stage_mults = {
        6: 4.0, 5: 3.5, 4: 3.0, 3: 2.5, 2: 2.0, 1: 1.5, 0: 1.0,
        -1: 2/3, -2: 2/4, -3: 2/5, -4: 2/6, -5: 2/7, -6: 2/8
    }
    last_st1 = timeline[-1].get("p1_pokemon_state", timeline[-1].get("p1_state", {})) or {} if timeline else {}
    last_st2 = timeline[-1].get("p2_pokemon_state", timeline[-1].get("p2_state", {})) or {} if timeline else {}

    if end_p1_sp and end_p2_sp:
        dex1 = get_internal_species_info(end_p1_sp, species_dex)
        dex2 = get_internal_species_info(end_p2_sp, species_dex)
        t1_act = dex1.get("types", ["normal"])
        t2_act = dex2.get("types", ["normal"])
        m_active12 = compute_type_multiplier(t1_act, t2_act, chart)
        m_active21 = compute_type_multiplier(t2_act, t1_act, chart)
        end_active_adv = float(m_active12 - m_active21)
        end_active_speed_diff = float(dex1.get("stats", {}).get("speed", 70.0) - dex2.get("stats", {}).get("speed", 70.0))
        end_active_outspeeds = float(dex1.get("stats", {}).get("speed", 70.0) > dex2.get("stats", {}).get("speed", 70.0))
        end_active_is_p1_se = float(m_active12 >= 2.0)
        end_active_is_p2_se = float(m_active21 >= 2.0)

        b1_spe = (last_st1.get("boosts", {}) or {}).get("spe", 0)
        b2_spe = (last_st2.get("boosts", {}) or {}).get("spe", 0)
        par1 = 0.25 if str(last_st1.get("status", "")).lower() == "par" else 1.0
        par2 = 0.25 if str(last_st2.get("status", "")).lower() == "par" else 1.0

        eff1 = min(999.0, dex1.get("stats", {}).get("speed", 70.0) * stage_mults.get(b1_spe, 1.0) * par1)
        eff2 = min(999.0, dex2.get("stats", {}).get("speed", 70.0) * stage_mults.get(b2_spe, 1.0) * par2)
        turn30_eff_spe_diff = float(eff1 - eff2)

        threat1 = 1.0 if (eff1 > eff2 and m_active12 >= 2.0) else 0.0
        threat2 = 1.0 if (eff2 > eff1 and m_active21 >= 2.0) else 0.0
        speed_kill_threat_diff = float(threat1 - threat2)

        hp_act1 = _extract_hp(last_st1)
        hp_act2 = _extract_hp(last_st2)
        fin1 = 1.0 if (eff1 > eff2 and hp_act2 <= 35.0) else 0.0
        fin2 = 1.0 if (eff2 > eff1 and hp_act1 <= 35.0) else 0.0
        finisher_threat_diff = float(fin1 - fin2)

        surv_p1_bench = [p for p in p1_team if _get_species_name(p) not in p1_fainted and _get_species_name(p) != end_p1_sp]
        surv_p2_bench = [sp for sp, v in p2_revealed.items() if not v["fainted"] and sp != end_p2_sp]
        safe_p1 = sum(1 for p in surv_p1_bench if compute_type_multiplier(t2_act, _get_pokemon_types(p), chart) <= 0.5)
        safe_p2 = sum(1 for sp in surv_p2_bench if compute_type_multiplier(t1_act, p2_revealed[sp]["types"], chart) <= 0.5)
        safe_bench_switch_diff = float(safe_p1 - safe_p2)
    else:
        end_active_adv = 0.0
        end_active_speed_diff = 0.0
        end_active_outspeeds = 0.0
        end_active_is_p1_se = 0.0
        end_active_is_p2_se = 0.0
        turn30_eff_spe_diff = 0.0
        speed_kill_threat_diff = 0.0
        finisher_threat_diff = 0.0
        safe_bench_switch_diff = 0.0

    p1_spikes = sum(1 for t in timeline if str((t.get("p1_move_details") or {}).get("name", "")).lower() == "spikes")
    p2_spikes = sum(1 for t in timeline if str((t.get("p2_move_details") or {}).get("name", "")).lower() == "spikes")
    p1_grounded = sum(1 for p in p1_team if "flying" not in [str(tp).lower() for tp in _get_pokemon_types(p)])
    p2_grounded = sum(1 for v in p2_revealed.values() if "flying" not in [str(tp).lower() for tp in v.get("types", [])])
    grounded_spikes_adv = float((p2_grounded * p1_spikes - p1_grounded * p2_spikes) * 12.5)

    return {
        "p2_revealed_count": float(n_p2_rev),
        "p2_revealed_fainted_count": float(n_p2_fnt),
        "p2_revealed_surviving_count": float(n_p2_surv),
        "p2_revealed_faint_rate": p2_faint_rate,
        "p2_unrevealed_estimate": float(max(0, 6 - n_p2_rev)),
        "p2_known_mean_bst": p2_mean_bst,
        "p2_known_max_bst": p2_max_bst,
        "p2_known_mean_speed": p2_mean_speed,
        "p2_known_max_speed": p2_max_speed,
        "bst_diff_vs_known_p2": bst_diff_vs_known,
        "speed_diff_vs_known_p2": speed_diff_vs_known,
        "p1_se_on_p2_known_count": float(p1_se_count),
        "p2_known_se_on_p1_count": float(p2_se_count),
        "p1_vs_p2_known_net_se_adv": float(net_se_adv),
        "p2_known_threatened_ratio": p2_threatened_ratio,
        "p1_threatened_by_known_p2_ratio": p1_threatened_ratio,
        "threatened_net_ratio_vs_known": threatened_net_ratio,
        "p1_wall_advantage_vs_known_p2": wall_adv,
        "surv_p1_vs_p2_known_net_se_adv": surv_net_se_adv,
        "surv_p2_known_threatened_ratio": surv_p2_threatened_ratio,
        "surv_bst_diff": surv_bst_diff,
        "surv_speed_pool_diff": surv_spe_diff,
        "effective_bst_pool_diff": eff_bst_diff,
        "effective_speed_pool_diff": eff_spe_diff,
        "grounded_spikes_chip_exposure": grounded_spikes_adv,
        "end_active_type_advantage": end_active_adv,
        "end_active_speed_diff": end_active_speed_diff,
        "end_active_outspeeds": end_active_outspeeds,
        "end_active_is_p1_se": end_active_is_p1_se,
        "end_active_is_p2_se": end_active_is_p2_se,
        "speed_kill_threat_diff": speed_kill_threat_diff,
        "finisher_threat_diff": finisher_threat_diff,
        "turn30_effective_speed_diff": turn30_eff_spe_diff,
        "safe_bench_switch_diff": safe_bench_switch_diff,
    }

def extract_single_battle_features(
    battle: Dict[str, Any],
    type_chart: Optional[Dict[str, Any]] = None,
    historical_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    chart = type_chart if type_chart is not None else TYPE_CHART

    feats: Dict[str, Any] = {
        "battle_id": battle.get("battle_id"),
    }

    p1_team = battle.get("p1_team_details", [])
    feats["p1_team_size"] = len(p1_team)

    p1_bst_list = []
    p1_phys_bulk_list = []
    p1_spec_bulk_list = []
    p1_leftovers_count = 0
    p1_all_types = []

    for stat in STATS:
        vals = [_get_pokemon_stat(p, stat) for p in p1_team]
        feats[f"p1_mean_{stat}"] = float(np.mean(vals)) if vals else 0.0
        feats[f"p1_std_{stat}"] = float(np.std(vals)) if vals else 0.0
        feats[f"p1_max_{stat}"] = float(np.max(vals)) if vals else 0.0
        feats[f"p1_min_{stat}"] = float(np.min(vals)) if vals else 0.0

    for p in p1_team:
        hp = _get_pokemon_stat(p, "hp")
        atk = _get_pokemon_stat(p, "attack")
        dfn = _get_pokemon_stat(p, "defense")
        spa = _get_pokemon_stat(p, "special-attack")
        spd = _get_pokemon_stat(p, "special-defense")
        spe = _get_pokemon_stat(p, "speed")

        p1_bst_list.append(hp + atk + dfn + spa + spd + spe)
        p1_phys_bulk_list.append((hp * dfn) / 100.0)
        p1_spec_bulk_list.append((hp * spd) / 100.0)

        item = str(p.get("item", "")).lower()
        if "leftovers" in item:
            p1_leftovers_count += 1

        p1_all_types.extend(_get_pokemon_types(p))

    feats["p1_mean_bst"] = float(np.mean(p1_bst_list)) if p1_bst_list else 0.0
    feats["p1_std_bst"] = float(np.std(p1_bst_list)) if p1_bst_list else 0.0
    feats["p1_max_bst"] = float(np.max(p1_bst_list)) if p1_bst_list else 0.0
    feats["p1_mean_phys_bulk"] = float(np.mean(p1_phys_bulk_list)) if p1_phys_bulk_list else 0.0
    feats["p1_mean_spec_bulk"] = float(np.mean(p1_spec_bulk_list)) if p1_spec_bulk_list else 0.0
    feats["p1_leftovers_count"] = p1_leftovers_count
    feats["p1_unique_types_count"] = len(set(p1_all_types))

    mean_atk = feats.get("p1_mean_attack", 1.0)
    mean_spa = feats.get("p1_mean_special-attack", 1.0)
    feats["p1_phys_spec_ratio"] = float(mean_atk / (mean_spa + 1e-5))

    feats.update(compute_structural_type_synergy(p1_team, chart))

    feats.update(extract_empirical_synergy_features(battle, historical_stats))

    p2_lead = battle.get("p2_lead_details", {})
    p2_lead_types = _get_pokemon_types(p2_lead)
    p2_lead_bst = sum(_get_pokemon_stat(p2_lead, s) for s in STATS)
    p2_lead_speed = _get_pokemon_stat(p2_lead, "speed")

    feats["p2_lead_bst"] = p2_lead_bst
    for stat in STATS:
        feats[f"p2_lead_{stat}"] = _get_pokemon_stat(p2_lead, stat)

    p1_lead = p1_team[0] if len(p1_team) > 0 else {}
    p1_lead_types = _get_pokemon_types(p1_lead)
    p1_lead_bst = sum(_get_pokemon_stat(p1_lead, s) for s in STATS)
    p1_lead_speed = _get_pokemon_stat(p1_lead, "speed")

    feats["p1_lead_bst"] = p1_lead_bst
    for stat in STATS:
        feats[f"p1_lead_{stat}"] = _get_pokemon_stat(p1_lead, stat)

    feats["lead_bst_diff"] = p1_lead_bst - p2_lead_bst
    feats["lead_speed_diff"] = p1_lead_speed - p2_lead_speed
    feats["lead_p1_to_p2_mult"] = compute_type_multiplier(p1_lead_types, p2_lead_types, chart)
    feats["lead_p2_to_p1_mult"] = compute_type_multiplier(p2_lead_types, p1_lead_types, chart)
    feats["lead_type_advantage"] = feats["lead_p1_to_p2_mult"] - feats["lead_p2_to_p1_mult"]

    p1_se_vs_p2_lead = 0
    p1_weak_vs_p2_lead = 0
    p1_outspeeds_p2_lead = 0
    for p in p1_team:
        p_t = _get_pokemon_types(p)
        p_spe = _get_pokemon_stat(p, "speed")
        if compute_type_multiplier(p_t, p2_lead_types, chart) >= 2.0:
            p1_se_vs_p2_lead += 1
        if compute_type_multiplier(p2_lead_types, p_t, chart) >= 2.0:
            p1_weak_vs_p2_lead += 1
        if p_spe > p2_lead_speed:
            p1_outspeeds_p2_lead += 1

    feats["p1_team_se_vs_p2_lead_count"] = p1_se_vs_p2_lead
    feats["p1_team_weak_vs_p2_lead_count"] = p1_weak_vs_p2_lead
    feats["p1_team_outspeeds_p2_lead_count"] = p1_outspeeds_p2_lead

    timeline = battle.get("battle_timeline", [])
    feats.update(extract_timeline_dynamics(timeline))

    feats.update(
        compute_battle_length_and_offensive_advantage(
            p1_team=p1_team,
            timeline=timeline,
            extracted_timeline_feats=feats,
        )
    )

    sp_dex = historical_stats.get("species_dex") if historical_stats else None
    feats.update(
        trace_opponent_pokemon_and_features(
            battle=battle,
            type_chart=chart,
            species_dex=sp_dex,
        )
    )

    return feats

def extract_features(
    records: List[Dict[str, Any]],
    type_chart: Optional[Dict[str, Any]] = None,
    historical_stats: Optional[Dict[str, Any]] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:

    chart = type_chart if type_chart is not None else TYPE_CHART
    is_labeled = bool(records and "player_won" in records[0])

    if is_labeled and historical_stats is None and len(records) >= n_splits:
        rows: List[Optional[Dict[str, Any]]] = [None] * len(records)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        indices = np.arange(len(records))

        for train_idx, val_idx in kf.split(indices):
            fold_train = [records[i] for i in train_idx]
            fold_stats = fit_historical_stats(fold_train)
            for i in val_idx:
                rows[i] = extract_single_battle_features(
                    records[i], type_chart=chart, historical_stats=fold_stats
                )
        df = pd.DataFrame(rows)
    else:
        rows_list = [
            extract_single_battle_features(b, type_chart=chart, historical_stats=historical_stats)
            for b in records
        ]
        df = pd.DataFrame(rows_list)

    if is_labeled:
        df["player_won"] = [int(b["player_won"]) for b in records]

    return df

ZERO_VARIANCE_CONSTANTS: List[str] = ["p1_team_size", "turns_observed", "median_battle_length"]

PRIORITY_DOMAIN_SIGNALS: List[str] = [
    "true_hazard_chip_damage_diff",
    "active_spikes_at_turn30_diff",
    "active_tempo_advantage",
    "active_substitute_advantage",
    "hp_per_surv_diff",
    "active_stat_boost_advantage",
    "late_game_hp_momentum",
    "active_incapacitation_advantage",
    "grounded_spikes_chip_exposure",
    "effective_bst_pool_diff",
    "effective_speed_pool_diff",
    "speed_kill_threat_diff",
    "finisher_threat_diff",
    "turn30_effective_speed_diff",
    "safe_bench_switch_diff",
]

def prune_features(
    df: pd.DataFrame,
    target_col: str = "player_won",
    min_correlation: float = 0.018,
    drop_constants: bool = True,
    selected_cols: Optional[List[str]] = None,
    force_keep_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:

    ignore_cols = {"battle_id", target_col}
    all_feature_cols = [c for c in df.columns if c not in ignore_cols]

    if selected_cols is not None:
        valid_cols = [c for c in selected_cols if c in df.columns]
        output_cols = [c for c in df.columns if c in ignore_cols] + valid_cols
        return df[output_cols].copy(), valid_cols

    if force_keep_cols is None:
        force_keep_cols = [c for c in PRIORITY_DOMAIN_SIGNALS if c in df.columns]

    X = df[all_feature_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)

    stds = np.nanstd(X, axis=0)
    keep_indices = [
        i for i, col in enumerate(all_feature_cols)
        if not (drop_constants and (stds[i] < 1e-8 or col in ZERO_VARIANCE_CONSTANTS))
    ]

    if target_col in df:
        y = df[target_col].values.astype(float)
        filtered_indices = []
        for i in keep_indices:
            col_name = all_feature_cols[i]
            if col_name in force_keep_cols:
                filtered_indices.append(i)
                continue
            feat_vals = X[:, i]
            feat_std = stds[i]
            if feat_std < 1e-8:
                continue
            r = abs(np.corrcoef(feat_vals, y)[0, 1])
            if np.isnan(r) or r < min_correlation:
                continue
            filtered_indices.append(i)
        keep_indices = filtered_indices

    retained_cols = [all_feature_cols[i] for i in keep_indices]
    output_cols = [c for c in df.columns if c in ignore_cols] + retained_cols
    return df[output_cols].copy(), retained_cols
