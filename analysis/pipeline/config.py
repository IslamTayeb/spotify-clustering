#!/usr/bin/env python3
"""Centralized configuration for music analysis pipeline.

This module provides a single source of truth for all configuration settings
used across the CLI (analysis/run_analysis.py) and Streamlit dashboard
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
    "clustering_algorithm": "hac",
    "n_clusters_hac": 5,
    "linkage_method": "ward",
}

# Default UMAP parameters for dimensionality reduction to 3D
DEFAULT_UMAP_PARAMS: Dict[str, Any] = {
    "umap_n_neighbors": 20,
    "umap_min_dist": 0.2,
    "umap_n_components": 3,
}


# =============================================================================
# INTERPRETABLE FEATURE SCALES
# =============================================================================

# Theme scale for interpretable lyric features
# Maps lyric themes to ordinal values (energetic/positive → heavy/dark)
# 10 themes evenly spaced across [0, 1] (spacing = 1/9 ≈ 0.111)
THEME_SCALE: Dict[str, float] = {
    "party": 1.0,
    "flex": 0.889,
    "love": 0.778,
    "social": 0.667,
    "spirituality": 0.556,
    "introspection": 0.444,
    "street": 0.333,
    "heartbreak": 0.222,
    "struggle": 0.111,
    "other": 0.0,
    "none": 0.5,  # Centered for instrumental tracks (equidistant from all categories)
}

# Language scale for interpretable lyric features
# Maps detected language to ordinal encoding
# Language family buckets - grouped by musical tradition similarity
# 8 families evenly spaced across [0, 1] (spacing = 1/7 ≈ 0.143)
LANGUAGE_SCALE: Dict[str, float] = {
    # English (standalone - dominant Western pop/hip-hop baseline)
    "english": 1.0,
    # Romance (Latin pop, reggaeton, shared production)
    "spanish": 0.857,
    "portuguese": 0.857,
    "french": 0.857,
    # Germanic (European pop traditions)
    "german": 0.714,
    "swedish": 0.714,
    "norwegian": 0.714,
    # Slavic (Eastern European traditions)
    "russian": 0.571,
    "ukrainian": 0.571,
    "serbian": 0.571,
    "czech": 0.571,
    # Middle Eastern (maqam scales, shared instrumentation)
    "arabic": 0.429,
    "hebrew": 0.429,
    "turkish": 0.429,
    # South Asian
    "punjabi": 0.286,
    # East Asian (K-pop/J-pop/C-pop aesthetics)
    "korean": 0.143,
    "japanese": 0.143,
    "chinese": 0.143,
    "vietnamese": 0.143,
    # African
    "luganda": 0.0,
    # Other/Unknown - centered for instrumental tracks (equidistant from all categories)
    "multilingual": 0.5,
    "unknown": 0.5,
    "none": 0.5,
}


# =============================================================================
# CACHE PATHS
# =============================================================================

# Feature cache file paths
CACHE_PATHS: Dict[str, str] = {
    "audio": "analysis/cache/audio_features.pkl",
    "lyrics_bge": "analysis/cache/lyric_features.pkl",          # Legacy default (bge-m3)
    "lyrics_e5": "analysis/cache/lyric_features_e5.pkl",        # Higher quality (E5)
    "mert": "analysis/cache/mert_embeddings_24khz_30s_cls.pkl",
    "lyric_interpretable": "analysis/cache/lyric_interpretable_features.pkl",
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
        return f"analysis/cache/lyric_features_{backend}.pkl"
