from .data_loader import load_jsonl, parse_metadata_dataframe, load_type_chart
from .features import (
    TYPE_CHART,
    compute_type_multiplier,
    compute_structural_type_synergy,
    extract_timeline_dynamics,
    compute_battle_length_and_offensive_advantage,
    trace_opponent_pokemon_and_features,
    extract_single_battle_features,
    extract_features,
    fit_historical_stats,
    ZERO_VARIANCE_CONSTANTS,
    PRIORITY_DOMAIN_SIGNALS,
)

__all__ = [
    "load_jsonl",
    "parse_metadata_dataframe",
    "load_type_chart",
    "TYPE_CHART",
    "compute_type_multiplier",
    "compute_structural_type_synergy",
    "extract_timeline_dynamics",
    "compute_battle_length_and_offensive_advantage",
    "trace_opponent_pokemon_and_features",
    "extract_single_battle_features",
    "extract_features",
    "fit_historical_stats",
    "ZERO_VARIANCE_CONSTANTS",
    "PRIORITY_DOMAIN_SIGNALS",
    "PokemonBattlePredictionPipeline",
]


def __getattr__(name):
    if name == "PokemonBattlePredictionPipeline":
        from .pipeline import PokemonBattlePredictionPipeline

        return PokemonBattlePredictionPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
