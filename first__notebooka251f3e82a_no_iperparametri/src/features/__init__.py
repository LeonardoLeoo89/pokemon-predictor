"""Feature extraction API."""

from .schema import (
    STATUS_WEIGHTS,
    STATIC_FEATURES,
    TYPE_CHART,
    ZERO_VARIANCE_CONSTANTS,
    PRIORITY_DOMAIN_SIGNALS,
    STATS,
    STATUS_CONDITIONS,
)
from .team import (
    compute_type_multiplier,
    compute_structural_type_synergy,
)
from .historical import fit_historical_stats, extract_empirical_synergy_features
from .timeline import (
    extract_timeline_dynamics,
    compute_battle_length_and_offensive_advantage,
)
from .opponent import (
    register_species_from_battle,
    get_internal_species_info,
    trace_opponent_pokemon_and_features,
)
from .extraction import (
    extract_single_battle_features,
    extract_features,
)
from .opponent import INTERNAL_SPECIES_CACHE

__all__ = [
    "STATUS_WEIGHTS",
    "STATIC_FEATURES",
    "TYPE_CHART",
    "ZERO_VARIANCE_CONSTANTS",
    "PRIORITY_DOMAIN_SIGNALS",
    "STATS",
    "STATUS_CONDITIONS",
    "compute_type_multiplier",
    "compute_structural_type_synergy",
    "fit_historical_stats",
    "extract_empirical_synergy_features",
    "extract_timeline_dynamics",
    "compute_battle_length_and_offensive_advantage",
    "register_species_from_battle",
    "get_internal_species_info",
    "trace_opponent_pokemon_and_features",
    "extract_single_battle_features",
    "extract_features",
    "INTERNAL_SPECIES_CACHE",
]
