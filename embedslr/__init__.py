from importlib import metadata as _m
from .embeddings import get_embeddings, list_models
from .similarity import rank_by_cosine
from .bibliometrics import full_report, indicators
from .colab_app import run as colab_run
from .multi_embedding import (
    multi_model_analysis,
    select_top_publications,
    calculate_model_metrics,
    rank_models_multi_criteria,
    group_by_model_consensus,
    analyze_hierarchical_groups,
    create_radar_chart,
    create_comparison_charts,
)

try:
    __version__ = _m.version(__name__)
except _m.PackageNotFoundError:
    __version__ = "0.6.0"

__all__ = [
    # Original functions
    "get_embeddings", "list_models", "rank_by_cosine",
    "full_report", "indicators", "colab_run",
    # Multi-embedding functions
    "multi_model_analysis",
    "select_top_publications",
    "calculate_model_metrics",
    "rank_models_multi_criteria",
    "group_by_model_consensus",
    "analyze_hierarchical_groups",
    "create_radar_chart",
    "create_comparison_charts",
]
