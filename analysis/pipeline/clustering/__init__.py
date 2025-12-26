"""Clustering Pipeline.

Re-exports main functions for backward compatibility.
"""

from .main import run_clustering_pipeline
from .features import (
    apply_subcluster_weights,
    extract_embedding_features,
    load_temporal_metadata,
    prepare_features,
)
from .algorithms import run_hac, run_birch, run_spectral, run_kmeans, run_dbscan
from .analysis import analyze_cluster
from .subclustering import (
    run_subcluster_pipeline,
    find_optimal_subclusters,
    auto_tune_subcluster_weights,
)
from .constants import EMBEDDING_DIM_NAMES, FEATURE_WEIGHT_INDICES

__all__ = [
    # Main pipeline
    "run_clustering_pipeline",
    # Features
    "apply_subcluster_weights",
    "extract_embedding_features",
    "load_temporal_metadata",
    "prepare_features",
    # Algorithms
    "run_hac",
    "run_birch",
    "run_spectral",
    "run_kmeans",
    "run_dbscan",
    # Analysis
    "analyze_cluster",
    # Sub-clustering
    "run_subcluster_pipeline",
    "find_optimal_subclusters",
    "auto_tune_subcluster_weights",
    # Constants
    "EMBEDDING_DIM_NAMES",
    "FEATURE_WEIGHT_INDICES",
]
