"""Assemble battle rows and historical cross-fitted feature frames."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from ..reproducibility import DEFAULT_SEED
from .schema import STATIC_FEATURES
from .parsing import _get_pokemon_stat
from .team import _team_stat_features, compute_structural_type_synergy
from .historical import fit_historical_stats, extract_empirical_synergy_features
from .timeline import (
    extract_timeline_dynamics,
    compute_battle_length_and_offensive_advantage,
)
from .opponent import trace_opponent_pokemon_and_features


def extract_single_battle_features(
    battle: Dict[str, Any],
    historical_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the retained feature values for one battle."""
    feats: Dict[str, Any] = {
        "battle_id": battle.get("battle_id"),
    }

    p1_team = battle.get("p1_team_details", [])
    feats.update(_team_stat_features(p1_team))

    feats.update(compute_structural_type_synergy(p1_team))

    feats.update(extract_empirical_synergy_features(battle, historical_stats))

    p1_lead = p1_team[0] if len(p1_team) > 0 else {}
    feats["p1_lead_special-defense"] = _get_pokemon_stat(p1_lead, "special-defense")

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
            species_dex=sp_dex,
        )
    )

    return {k: feats[k] for k in ["battle_id"] + STATIC_FEATURES if k in feats}


def _historical_oof_rows(records, n_splits, random_state):
    """Encode each row with historical estimates fitted on the other folds."""
    rows: List[Optional[Dict[str, Any]]] = [None] * len(records)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    indices = np.arange(len(records))

    for train_idx, val_idx in kf.split(indices):
        fold_train = [records[i] for i in train_idx]
        fold_stats = fit_historical_stats(fold_train)
        for i in val_idx:
            rows[i] = extract_single_battle_features(
                records[i], historical_stats=fold_stats
            )
    return rows


def extract_features(
    records: List[Dict[str, Any]],
    historical_stats: Optional[Dict[str, Any]] = None,
    n_splits: int = 5,
    random_state: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Build the ordered feature frame, including the target when available."""
    is_labeled = bool(records and "player_won" in records[0])

    if is_labeled and historical_stats is None and len(records) >= n_splits:
        rows = _historical_oof_rows(records, n_splits, random_state)
        df = pd.DataFrame(rows)
    else:
        rows_list = [
            extract_single_battle_features(b, historical_stats=historical_stats)
            for b in records
        ]
        df = pd.DataFrame(rows_list)

    if is_labeled:
        df["player_won"] = [int(b["player_won"]) for b in records]

    valid_cols = (
        ["battle_id"]
        + (["player_won"] if is_labeled else [])
        + [c for c in STATIC_FEATURES if c in df.columns]
    )
    return df[valid_cols]
