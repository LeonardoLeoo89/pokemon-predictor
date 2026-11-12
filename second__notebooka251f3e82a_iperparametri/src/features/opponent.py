"""Track revealed opponents and calculate matchup features."""

from typing import Any, Dict, Optional
import numpy as np
from .schema import STATS
from .parsing import (
    _get_species_name,
    _get_pokemon_types,
    _get_pokemon_stat,
    _extract_hp,
)
from .team import compute_type_multiplier

# Shared across battles; clearing this changes metadata fallback.
INTERNAL_SPECIES_CACHE: Dict[str, Dict[str, Any]] = {}


def register_species_from_battle(battle: Dict[str, Any]) -> None:
    """Cache types and base stats from recorded team and lead details."""
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
    """Look up recorded species metadata, with neutral fallback values."""
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


def _track_rosters(battle, species_dex):
    """Track revealed species, HP and fainting through the observed turns."""
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

    return p2_revealed, p1_fainted, p1_hp_map, (end_p1_sp, end_p2_sp)


def _roster_type_features(p1_team, p2_revealed, p1_fainted):
    """Compare type coverage across revealed and surviving rosters."""
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
            m1 = compute_type_multiplier(p_t, q_t)
            m2 = compute_type_multiplier(q_t, p_t)
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
    wall_adv = float(p1_wall_count - p2_wall_count)

    surv_p1 = [p for p in p1_team if _get_species_name(p) not in p1_fainted]
    surv_p2 = [q_v for q_sp, q_v in p2_revealed.items() if not q_v["fainted"]]
    surv_p1_se = 0
    surv_p2_se = 0
    for p in surv_p1:
        p_t = _get_pokemon_types(p)
        for q_v in surv_p2:
            q_t = q_v["types"]
            if compute_type_multiplier(p_t, q_t) >= 2.0:
                surv_p1_se += 1
            if compute_type_multiplier(q_t, p_t) >= 2.0:
                surv_p2_se += 1
    surv_net_se_adv = float(surv_p1_se - surv_p2_se)
    surv_p2_threatened_ratio = float(surv_p1_se > 0) if surv_p2 else 1.0

    return {
        "p1_se_on_p2_known_count": float(p1_se_count),
        "p2_known_se_on_p1_count": float(p2_se_count),
        "p1_vs_p2_known_net_se_adv": float(net_se_adv),
        "p1_wall_advantage_vs_known_p2": wall_adv,
        "surv_p1_vs_p2_known_net_se_adv": surv_net_se_adv,
        "surv_p2_known_threatened_ratio": surv_p2_threatened_ratio,
    }


def _survivor_stat_features(p1_team, p2_revealed, p1_fainted, p1_hp_map):
    """Compare surviving stat pools and their HP-weighted counterparts."""
    n_p2_rev = len(p2_revealed)
    p1_surv_bst = sum(
        sum(_get_pokemon_stat(p, s) for s in STATS)
        for p in p1_team
        if _get_species_name(p) not in p1_fainted
    )
    p2_surv_bst = (
        sum(sum(v["stats"].values()) for v in p2_revealed.values() if not v["fainted"])
        + max(0, 6 - n_p2_rev) * 500.0
    )
    surv_bst_diff = float(p1_surv_bst - p2_surv_bst)

    p1_surv_spe = sum(
        _get_pokemon_stat(p, "speed")
        for p in p1_team
        if _get_species_name(p) not in p1_fainted
    )
    p2_surv_spe = (
        sum(
            v["stats"].get("speed", 70.0)
            for v in p2_revealed.values()
            if not v["fainted"]
        )
        + max(0, 6 - n_p2_rev) * 70.0
    )
    surv_spe_diff = float(p1_surv_spe - p2_surv_spe)

    p1_eff_bst = sum(
        sum(_get_pokemon_stat(p, s) for s in STATS)
        * (p1_hp_map.get(_get_species_name(p), 100.0) / 100.0)
        for p in p1_team
        if _get_species_name(p) not in p1_fainted
    )
    p2_eff_bst = (
        sum(
            sum(v["stats"].values()) * (v["hp"] / 100.0)
            for v in p2_revealed.values()
            if not v["fainted"]
        )
        + max(0, 6 - n_p2_rev) * 500.0
    )
    eff_bst_diff = float(p1_eff_bst - p2_eff_bst)

    p1_eff_spe = sum(
        _get_pokemon_stat(p, "speed")
        * (p1_hp_map.get(_get_species_name(p), 100.0) / 100.0)
        for p in p1_team
        if _get_species_name(p) not in p1_fainted
    )
    p2_eff_spe = (
        sum(
            v["stats"].get("speed", 70.0) * (v["hp"] / 100.0)
            for v in p2_revealed.values()
            if not v["fainted"]
        )
        + max(0, 6 - n_p2_rev) * 70.0
    )
    eff_spe_diff = float(p1_eff_spe - p2_eff_spe)

    return {
        "surv_bst_diff": surv_bst_diff,
        "surv_speed_pool_diff": surv_spe_diff,
        "effective_bst_pool_diff": eff_bst_diff,
        "effective_speed_pool_diff": eff_spe_diff,
    }


def _active_matchup_features(
    battle, p2_revealed, p1_fainted, active_species, species_dex
):
    """Compare active types, effective speeds, threats and bench options."""
    p1_team = battle.get("p1_team_details", [])
    timeline = battle.get("battle_timeline", [])
    end_p1_sp, end_p2_sp = active_species
    stage_mults = {
        stage: (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)
        for stage in range(-6, 7)
    }

    last_st1 = (
        timeline[-1].get("p1_pokemon_state", timeline[-1].get("p1_state", {})) or {}
        if timeline
        else {}
    )
    last_st2 = (
        timeline[-1].get("p2_pokemon_state", timeline[-1].get("p2_state", {})) or {}
        if timeline
        else {}
    )

    if end_p1_sp and end_p2_sp:
        dex1 = get_internal_species_info(end_p1_sp, species_dex)
        dex2 = get_internal_species_info(end_p2_sp, species_dex)
        t1_act = dex1.get("types", ["normal"])
        t2_act = dex2.get("types", ["normal"])
        m_active12 = compute_type_multiplier(t1_act, t2_act)
        m_active21 = compute_type_multiplier(t2_act, t1_act)
        end_active_adv = float(m_active12 - m_active21)
        end_active_is_p1_se = float(m_active12 >= 2.0)
        end_active_is_p2_se = float(m_active21 >= 2.0)

        b1_spe = (last_st1.get("boosts", {}) or {}).get("spe", 0)
        b2_spe = (last_st2.get("boosts", {}) or {}).get("spe", 0)
        par1 = 0.25 if str(last_st1.get("status", "")).lower() == "par" else 1.0
        par2 = 0.25 if str(last_st2.get("status", "")).lower() == "par" else 1.0

        eff1 = min(
            999.0,
            dex1.get("stats", {}).get("speed", 70.0)
            * stage_mults.get(b1_spe, 1.0)
            * par1,
        )
        eff2 = min(
            999.0,
            dex2.get("stats", {}).get("speed", 70.0)
            * stage_mults.get(b2_spe, 1.0)
            * par2,
        )
        turn30_eff_spe_diff = float(eff1 - eff2)

        threat1 = 1.0 if (eff1 > eff2 and m_active12 >= 2.0) else 0.0
        threat2 = 1.0 if (eff2 > eff1 and m_active21 >= 2.0) else 0.0
        speed_kill_threat_diff = float(threat1 - threat2)

        hp_act1 = _extract_hp(last_st1)
        hp_act2 = _extract_hp(last_st2)
        fin1 = 1.0 if (eff1 > eff2 and hp_act2 <= 35.0) else 0.0
        fin2 = 1.0 if (eff2 > eff1 and hp_act1 <= 35.0) else 0.0
        finisher_threat_diff = float(fin1 - fin2)

        surv_p1_bench = [
            p
            for p in p1_team
            if _get_species_name(p) not in p1_fainted
            and _get_species_name(p) != end_p1_sp
        ]
        surv_p2_bench = [
            sp for sp, v in p2_revealed.items() if not v["fainted"] and sp != end_p2_sp
        ]
        safe_p1 = sum(
            1
            for p in surv_p1_bench
            if compute_type_multiplier(t2_act, _get_pokemon_types(p)) <= 0.5
        )
        safe_p2 = sum(
            1
            for sp in surv_p2_bench
            if compute_type_multiplier(t1_act, p2_revealed[sp]["types"]) <= 0.5
        )
        safe_bench_switch_diff = float(safe_p1 - safe_p2)
    else:
        end_active_adv = 0.0
        end_active_is_p1_se = 0.0
        end_active_is_p2_se = 0.0
        turn30_eff_spe_diff = 0.0
        speed_kill_threat_diff = 0.0
        finisher_threat_diff = 0.0
        safe_bench_switch_diff = 0.0

    return {
        "end_active_type_advantage": end_active_adv,
        "end_active_is_p1_se": end_active_is_p1_se,
        "end_active_is_p2_se": end_active_is_p2_se,
        "speed_kill_threat_diff": speed_kill_threat_diff,
        "finisher_threat_diff": finisher_threat_diff,
        "turn30_effective_speed_diff": turn30_eff_spe_diff,
        "safe_bench_switch_diff": safe_bench_switch_diff,
    }


def _spikes_exposure(p1_team, p2_revealed, timeline):
    """Compare recorded Spikes exposure of grounded roster members."""
    p1_spikes = sum(
        1
        for t in timeline
        if str((t.get("p1_move_details") or {}).get("name", "")).lower() == "spikes"
    )
    p2_spikes = sum(
        1
        for t in timeline
        if str((t.get("p2_move_details") or {}).get("name", "")).lower() == "spikes"
    )
    p1_grounded = sum(
        1
        for p in p1_team
        if "flying" not in [str(tp).lower() for tp in _get_pokemon_types(p)]
    )
    p2_grounded = sum(
        1
        for v in p2_revealed.values()
        if "flying" not in [str(tp).lower() for tp in v.get("types", [])]
    )
    grounded_spikes_adv = float(
        (p2_grounded * p1_spikes - p1_grounded * p2_spikes) * 12.5
    )

    return grounded_spikes_adv


def trace_opponent_pokemon_and_features(
    battle: Dict[str, Any],
    species_dex: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Calculate matchup features for the revealed opponent roster."""
    register_species_from_battle(battle)
    p1_team = battle.get("p1_team_details", [])
    timeline = battle.get("battle_timeline", [])
    revealed, fainted, hp, active = _track_rosters(battle, species_dex)
    count = len(revealed)
    fainted_count = sum(1 for pokemon in revealed.values() if pokemon["fainted"])
    speeds = [pokemon["stats"].get("speed", 70.0) for pokemon in revealed.values()]
    types = _roster_type_features(p1_team, revealed, fainted)
    stats = _survivor_stat_features(p1_team, revealed, fainted, hp)
    active_features = _active_matchup_features(
        battle, revealed, fainted, active, species_dex
    )
    spikes = _spikes_exposure(p1_team, revealed, timeline)
    return {
        "p2_revealed_count": float(count),
        "p2_revealed_fainted_count": float(fainted_count),
        "p2_revealed_surviving_count": float(max(0, count - fainted_count)),
        "p2_unrevealed_estimate": float(max(0, 6 - count)),
        "p2_known_max_speed": float(np.max(speeds)) if speeds else 70.0,
        **types,
        **stats,
        "grounded_spikes_chip_exposure": spikes,
        **active_features,
    }
