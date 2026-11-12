"""HP, status, damage and move observations across turns."""

from typing import Any, Dict, List, Optional
from .schema import STATUS_WEIGHTS, STATUS_CONDITIONS
from .parsing import _extract_hp, _get_species_name


def extract_timeline_dynamics(battle_or_timeline: Any) -> Dict[str, Any]:
    """Summarize HP, status, damage and move observations across turns."""
    if isinstance(battle_or_timeline, dict):
        timeline = battle_or_timeline.get("battle_timeline", [])
    elif isinstance(battle_or_timeline, list):
        timeline = battle_or_timeline
    else:
        timeline = []

    feats: Dict[str, Any] = {}

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
        p1_state = (
            turn_data.get("p1_pokemon_state", turn_data.get("p1_state", {})) or {}
        )
        p2_state = (
            turn_data.get("p2_pokemon_state", turn_data.get("p2_state", {})) or {}
        )

        p1_sp = _get_species_name(p1_state) or f"p1_slot_{len(p1_revealed_roster)}"
        p2_sp = _get_species_name(p2_state) or f"p2_slot_{len(p2_revealed_roster)}"

        action_p1_switch = str(turn_data.get("p1_action", "")).lower() == "switch"
        action_p2_switch = str(turn_data.get("p2_action", "")).lower() == "switch"
        name_p1_switch = last_p1_sp is not None and p1_sp != last_p1_sp
        name_p2_switch = last_p2_sp is not None and p2_sp != last_p2_sp

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
            prev_p1 = (
                prev_turn.get("p1_pokemon_state", prev_turn.get("p1_state", {})) or {}
            )
            prev_p2 = (
                prev_turn.get("p2_pokemon_state", prev_turn.get("p2_state", {})) or {}
            )
            prev_sp1 = _get_species_name(prev_p1)
            prev_sp2 = _get_species_name(prev_p2)
            prev_hp1 = _extract_hp(prev_p1)
            prev_hp2 = _extract_hp(prev_p2)
            if prev_sp1 == p1_sp and prev_hp1 > p1_hp:
                p2_damage_dealt += prev_hp1 - p1_hp
            if prev_sp2 == p2_sp and prev_hp2 > p2_hp:
                p1_damage_dealt += prev_hp2 - p2_hp

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

    feats["fainted_momentum_shift"] = (
        feats["fainted_diff_late"] - feats["fainted_diff_early"]
    )

    feats["p1_remaining_estimated"] = 6 - p1_fainted_final
    feats["p2_remaining_estimated"] = 6 - p2_fainted_final
    feats["remaining_diff"] = (
        feats["p1_remaining_estimated"] - feats["p2_remaining_estimated"]
    )

    p1_unrev = max(0, 6 - len(p1_revealed_roster))
    p2_unrev = max(0, 6 - len(p2_revealed_roster))
    p1_total_hp = (
        sum(v["hp"] for v in p1_revealed_roster.values() if not v["fainted"])
        + p1_unrev * 100.0
    )
    p2_total_hp = (
        sum(v["hp"] for v in p2_revealed_roster.values() if not v["fainted"])
        + p2_unrev * 100.0
    )

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

    p1_sev = sum(
        STATUS_WEIGHTS.get(v["status"], 0.0)
        for v in p1_revealed_roster.values()
        if not v["fainted"]
    )
    p2_sev = sum(
        STATUS_WEIGHTS.get(v["status"], 0.0)
        for v in p2_revealed_roster.values()
        if not v["fainted"]
    )

    feats["status_severity_diff"] = float(p2_sev - p1_sev)

    feats["p1_damage_dealt"] = float(p1_damage_dealt)
    feats["p2_damage_dealt"] = float(p2_damage_dealt)
    feats["damage_dealt_diff"] = float(p1_damage_dealt - p2_damage_dealt)
    feats["p1_spikes_set"] = p1_spikes
    feats["p2_spikes_set"] = p2_spikes
    feats["spikes_diff"] = p1_spikes - p2_spikes
    feats["p1_spikes_chip_damage"] = float(p1_spikes * p2_switches * 12.5)
    feats["p2_spikes_chip_damage"] = float(p2_spikes * p1_switches * 12.5)
    feats["spikes_chip_damage_diff"] = (
        feats["p1_spikes_chip_damage"] - feats["p2_spikes_chip_damage"]
    )
    feats["p1_rests_used"] = p1_rests

    end_p1 = (
        timeline[-1].get("p1_pokemon_state", timeline[-1].get("p1_state", {}))
        if timeline
        else {}
    )
    end_p2 = (
        timeline[-1].get("p2_pokemon_state", timeline[-1].get("p2_state", {}))
        if timeline
        else {}
    )
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
        feats["late_game_hp_momentum"] = float(
            (p1_active_hp - hp1_20) - (p2_active_hp - hp2_20)
        )
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
    feats["active_spikes_at_turn30_diff"] = float(p1_spikes_live) - float(
        p2_spikes_live
    )
    feats["explosions_diff"] = float(p1_booms - p2_booms)

    eff1 = [str(e).lower() for e in (end_p1.get("effects", []) or [])]
    eff2 = [str(e).lower() for e in (end_p2.get("effects", []) or [])]
    sub1 = 1.0 if "substitute" in eff1 else 0.0
    sub2 = 1.0 if "substitute" in eff2 else 0.0
    feats["active_substitute_advantage"] = float(sub1 - sub2)

    hp_act_diff = p1_active_hp - p2_active_hp

    feats["active_tempo_advantage"] = float(
        (hp_act_diff / 100.0) + (b1_net - b2_net) * 0.5
    )

    p1_surv_count = max(1, 6 - p1_fainted_final)
    p2_surv_count = max(1, 6 - p2_fainted_final)
    feats["hp_per_surv_diff"] = float(
        (p1_total_hp / p1_surv_count) - (p2_total_hp / p2_surv_count)
    )

    if len(p1_hp_history) >= 2:
        feats["p1_hp_slope"] = float(
            (p1_hp_history[-1] - p1_hp_history[0]) / len(p1_hp_history)
        )
    else:
        feats["p1_hp_slope"] = 0.0

    return feats


def compute_battle_length_and_offensive_advantage(
    p1_team: List[Dict[str, Any]],
    timeline: List[Dict[str, Any]],
    extracted_timeline_feats: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Estimate battle length from HP depletion and locate the first faint."""

    if extracted_timeline_feats:
        p1_hp_pool = float(extracted_timeline_feats.get("p1_total_hp_pool", 600.0))
        p2_hp_pool = float(extracted_timeline_feats.get("p2_total_hp_pool", 600.0))
    else:
        p1_hp_pool = 600.0
        p2_hp_pool = 600.0

    p1_lost = max(1.0, 600.0 - p1_hp_pool)
    p2_lost = max(1.0, 600.0 - p2_hp_pool)
    faster_hp_rate = max(p1_lost, p2_lost) / 30.0
    b_len_hp = float(min(80, 600.0 / faster_hp_rate))

    first_faint_turn = 30.0
    for t_idx, t in enumerate(timeline):
        st1 = t.get("p1_pokemon_state", t.get("p1_state", {})) or {}
        st2 = t.get("p2_pokemon_state", t.get("p2_state", {})) or {}
        if (
            st1.get("status") == "fnt"
            or _extract_hp(st1) <= 0.0
            or st2.get("status") == "fnt"
            or _extract_hp(st2) <= 0.0
        ):
            first_faint_turn = float(t.get("turn", t_idx + 1))
            break

    return {
        "battle_length_est": b_len_hp,
        "first_faint_turn": first_faint_turn,
    }
