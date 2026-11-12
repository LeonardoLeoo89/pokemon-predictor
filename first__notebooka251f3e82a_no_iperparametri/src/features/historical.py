"""Smoothed outcome estimates for species, teams and types."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from itertools import combinations
from .schema import STATS
from .parsing import _get_species_name, _get_pokemon_types, _get_pokemon_stat


def fit_historical_stats(
    train_records: List[Dict[str, Any]],
    smoothing_m: float = 20.0,
    pair_smoothing_m: float = 15.0,
) -> Dict[str, Any]:
    """Fit smoothed species, team-combination and type win rates from labelled battles."""
    wins_list = [int(b["player_won"]) for b in train_records if "player_won" in b]
    global_prior = float(np.mean(wins_list)) if wins_list else 0.5

    p1_species_stats: Dict[str, List[int]] = {}
    p2_species_stats: Dict[str, List[int]] = {}
    p1_k_tuple_stats: Dict[int, Dict[Tuple[str, ...], List[int]]] = {
        k: {} for k in range(1, 7)
    }
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
            p2_state = (
                turn_data.get("p2_pokemon_state", turn_data.get("p2_state", {})) or {}
            )
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

    p1_sp_wr = {
        sp: _smooth(w, c, smoothing_m) for sp, (w, c) in p1_species_stats.items()
    }

    p2_sp_wr = {
        sp: 1.0 - _smooth(w, c, smoothing_m) for sp, (w, c) in p2_species_stats.items()
    }

    k_smoothing_m = {
        1: smoothing_m,
        2: pair_smoothing_m,
        3: 10.0,
        4: 8.0,
        5: 5.0,
        6: 3.0,
    }
    p1_k_wr: Dict[int, Dict[Tuple[str, ...], float]] = {}
    p1_k_synergy: Dict[int, Dict[Tuple[str, ...], float]] = {}

    for k in range(1, 7):
        m_k = k_smoothing_m.get(k, 5.0)
        p1_k_wr[k] = {
            combo: _smooth(w, c, m_k) for combo, (w, c) in p1_k_tuple_stats[k].items()
        }
        if k >= 2:
            p1_k_synergy[k] = {}
            for combo, combo_wr in p1_k_wr[k].items():
                expected_wr = float(
                    np.mean([p1_sp_wr.get(sp, global_prior) for sp in combo])
                )
                p1_k_synergy[k][combo] = float(combo_wr - expected_wr)

    p1_pair_wr = p1_k_wr[2]
    p1_pair_synergy = p1_k_synergy[2]

    p1_tp_wr = {tp: _smooth(w, c, smoothing_m) for tp, (w, c) in p1_type_stats.items()}
    p1_tp_pair_wr = {}
    p1_tp_pair_synergy = {}
    for (tp_a, tp_b), (w, c) in p1_type_pair_stats.items():
        t_wr = _smooth(w, c, pair_smoothing_m)
        p1_tp_pair_wr[(tp_a, tp_b)] = t_wr
        expected_t_wr = 0.5 * (
            p1_tp_wr.get(tp_a, global_prior) + p1_tp_wr.get(tp_b, global_prior)
        )
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
    """Look up outcome and synergy estimates for a battle."""
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

    p1_wrs = (
        [p1_sp_map.get(sp, prior) for sp in p1_species_ordered]
        if p1_species_ordered
        else [prior]
    )
    p1_lead_sp = p1_species_ordered[0] if p1_species_ordered else ""
    p1_lead_wr = p1_sp_map.get(p1_lead_sp, prior)

    p2_lead_sp = _get_species_name(battle.get("p2_lead_details", {}))
    p2_lead_wr = p2_sp_map.get(p2_lead_sp, 1.0 - prior)

    p2_seen_species = set()
    if p2_lead_sp:
        p2_seen_species.add(p2_lead_sp)
    for turn_data in battle.get("battle_timeline", []):
        p2_state = (
            turn_data.get("p2_pokemon_state", turn_data.get("p2_state", {})) or {}
        )
        sp_turn = _get_species_name(p2_state)
        if sp_turn:
            p2_seen_species.add(sp_turn)

    p2_revealed_wrs = (
        [p2_sp_map.get(sp, 1.0 - prior) for sp in p2_seen_species]
        if p2_seen_species
        else [1.0 - prior]
    )

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

    type_wrs = (
        [p1_tp_map.get(tp, prior) for tp in p1_types_unique]
        if p1_types_unique
        else [prior]
    )
    type_pair_wrs = []
    type_pair_lifts = []
    for tp_a, tp_b in combinations(p1_types_unique, 2):
        type_pair_wrs.append(p1_tp_pair_wr_map.get((tp_a, tp_b), prior))
        type_pair_lifts.append(p1_tp_pair_syn_map.get((tp_a, tp_b), 0.0))

    lead_matchup_wr = (
        lead_matchup_map.get((p1_lead_sp, p2_lead_sp), prior)
        if (p1_lead_sp and p2_lead_sp)
        else prior
    )

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
    feats["p1_type_pair_mean_wr"] = (
        float(np.mean(type_pair_wrs)) if type_pair_wrs else prior
    )
    feats["p1_type_pair_synergy_lift_mean"] = (
        float(np.mean(type_pair_lifts)) if type_pair_lifts else 0.0
    )

    return feats
