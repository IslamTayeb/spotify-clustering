#!/usr/bin/env python3
"""Centralized configuration for music analysis pipeline.

This module provides a single source of truth for all configuration settings
used across the CLI (run_analysis.py) and Streamlit dashboard
(interactive_interpretability.py).
"""

from typing import Dict, Any

# =============================================================================
# PCA CONFIGURATION
# =============================================================================

# PCA Components for 75% cumulative variance (per mode)
# Determined empirically via tools/find_optimal_pca.py
PCA_COMPONENTS_MAP: Dict[str, int] = {
    "audio": 118,      # 75.01% variance explained
    "lyrics": 162,     # 75.02% variance explained
    "combined": 142,   # 75.04% variance (audio) + 75.04% variance (lyrics)
}


# =============================================================================
# CLUSTERING CONFIGURATION
# =============================================================================

# Default clustering parameters for Hierarchical Agglomerative Clustering (HAC)
DEFAULT_CLUSTERING_PARAMS: Dict[str, Any] = {
    "algorithm": "hac",
    "n_clusters_hac": 5,
    "linkage_method": "ward",
}

# Default UMAP parameters for dimensionality reduction to 3D
DEFAULT_UMAP_PARAMS: Dict[str, Any] = {
    "n_neighbors": 20,
    "min_dist": 0.2,
    "n_components": 3,
}


# =============================================================================
# INTERPRETABLE FEATURE SCALES
# =============================================================================

# Theme scale for interpretable lyric features
# Maps lyric themes to ordinal values (introspective → energetic/positive)
THEME_SCALE: Dict[str, float] = {
    "party": 1.0,
    "flex": 0.9,
    "love": 0.8,
    "social": 0.7,
    "spirituality": 0.6,
    "introspection": 0.5,
    "street": 0.4,
    "heartbreak": 0.3,
    "struggle": 0.2,
    "other": 0.1,
    "none": 0.0,
}

# Language scale for interpretable lyric features
# Maps detected language to ordinal encoding
LANGUAGE_SCALE: Dict[str, float] = {
    "english": 1.0,
    "spanish": 0.86,
    "french": 0.71,
    "arabic": 0.57,
    "korean": 0.43,
    "japanese": 0.29,
    "unknown": 0.14,
    "none": 0.0,
}


# =============================================================================
# CACHE PATHS
# =============================================================================

# Feature cache file paths
CACHE_PATHS: Dict[str, str] = {
    "audio": "cache/audio_features.pkl",
    "lyrics_bge": "cache/lyric_features.pkl",          # Legacy default (bge-m3)
    "lyrics_e5": "cache/lyric_features_e5.pkl",        # Higher quality (E5)
    "mert": "cache/mert_embeddings_24khz_30s_cls.pkl",
    "lyric_interpretable": "cache/lyric_interpretable_features.pkl",
}


def get_lyric_cache_path(backend: str) -> str:
    """Get cache path for lyric embedding backend.

    Args:
        backend: Lyric embedding backend ("bge-m3", "e5", or custom)

    Returns:
        Path to cache file for specified backend
    """
    if backend == "bge-m3":
        return CACHE_PATHS["lyrics_bge"]
    elif backend == "e5":
        return CACHE_PATHS["lyrics_e5"]
    else:
        # Custom backend - generate cache path
        return f"cache/lyric_features_{backend}.pkl"
