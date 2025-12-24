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
# Language family buckets - grouped by musical tradition similarity
LANGUAGE_SCALE: Dict[str, float] = {
    # English (standalone - dominant Western pop/hip-hop baseline)
    "english": 1.0,
    # Romance (Latin pop, reggaeton, shared production)
    "spanish": 0.85,
    "portuguese": 0.85,
    "french": 0.85,
    # Germanic (European pop traditions)
    "german": 0.70,
    "swedish": 0.70,
    "norwegian": 0.70,
    # Slavic (Eastern European traditions)
    "russian": 0.55,
    "ukrainian": 0.55,
    "serbian": 0.55,
    "czech": 0.55,
    # Middle Eastern (maqam scales, shared instrumentation)
    "arabic": 0.40,
    "hebrew": 0.40,
    "turkish": 0.40,
    # South Asian
    "punjabi": 0.30,
    # East Asian (K-pop/J-pop/C-pop aesthetics)
    "korean": 0.20,
    "japanese": 0.20,
    "chinese": 0.20,
    "vietnamese": 0.20,
    # African
    "luganda": 0.10,
    # Other/Unknown
    "multilingual": 0.0,
    "unknown": 0.0,
    "none": 0.0,
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
