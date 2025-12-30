#!/usr/bin/env python3
"""Feature caching utilities for music analysis pipeline.

This module provides smart caching helpers to eliminate complex branching logic
in the main pipeline. All cache loading/saving is centralized here.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_from_cache(cache_path: str, feature_type: str) -> Optional[List[Dict[str, Any]]]:
    """Load features from cache file if it exists.

    Args:
        cache_path: Path to cache file
        feature_type: Human-readable type (for logging): "audio", "lyric (e5)", etc.

    Returns:
        List of feature dicts if cache exists, None otherwise
    """
    path = Path(cache_path)
    if not path.exists():
        logger.info(f"{feature_type} cache not found: {cache_path}")
        return None

    logger.info(f"Loading {feature_type} features from cache: {cache_path}")
    try:
        with open(path, "rb") as f:
            features = pickle.load(f)
        logger.info(f"✓ Loaded {len(features)} {feature_type} features from cache")
        return features
    except Exception as e:
        logger.error(f"Failed to load {feature_type} cache from {cache_path}: {e}")
        return None


def save_to_cache(features: List[Dict[str, Any]], cache_path: str, feature_type: str):
    """Save features to cache file.

    Args:
        features: List of feature dicts to save
        cache_path: Path to cache file
        feature_type: Human-readable type (for logging): "audio", "lyric (e5)", etc.
    """
    logger.info(f"Saving {len(features)} {feature_type} features to cache: {cache_path}")
    try:
        # Ensure parent directory exists
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

        with open(cache_path, "wb") as f:
            pickle.dump(features, f)
        logger.info(f"✓ Saved {feature_type} cache to {cache_path}")
    except Exception as e:
        logger.error(f"Failed to save {feature_type} cache to {cache_path}: {e}")
        raise


def upgrade_audio_cache_if_needed(audio_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Upgrade audio cache with genre_fusion if needed.

    The genre_fusion feature was added later and uses an entropy-based calculation.
    This function checks if cached audio features need the upgrade and applies it
    if necessary.

    Args:
        audio_features: Audio features loaded from cache

    Returns:
        audio_features with genre_fusion added/updated if needed
    """
    from analysis.pipeline.genre_fusion import add_genre_fusion_to_features

    # Check if any of the first 10 tracks have the entropy-based version
    has_entropy_version = any(
        f.get("genre_fusion_version") == "entropy"
        for f in audio_features[:10]
    )

    if not has_entropy_version:
        logger.info("Upgrading genre_fusion to entropy-based version...")
        audio_features = add_genre_fusion_to_features(audio_features)

        # Save updated cache
        save_to_cache(audio_features, "analysis/cache/audio_features.pkl", "audio")
        logger.info("✓ Updated audio cache with genre_fusion")

    return audio_features


def load_audio_features(
    fresh: bool = False,
    re_classify: bool = False
) -> List[Dict[str, Any]]:
    """Load audio features with smart caching and upgrade logic.

    Args:
        fresh: If True, force re-extraction (ignore cache)
        re_classify: If True, re-run classifiers on cached features

    Returns:
        List of audio feature dicts
    """
    from analysis.pipeline.audio_analysis import extract_audio_features, update_cached_features

    cache_path = "analysis/cache/audio_features.pkl"

    # Re-classify existing cache
    if re_classify and Path(cache_path).exists():
        logger.info("Updating cached audio features with new classifiers...")
        return update_cached_features()

    # Load from cache (if not forcing fresh extraction)
    if not fresh:
        cached = load_from_cache(cache_path, "audio")
        if cached:
            return upgrade_audio_cache_if_needed(cached)

    # Extract fresh features
    logger.info("Extracting audio features from MP3 files...")
    return extract_audio_features()


def load_lyric_features(
    backend: str = "e5",
    fresh: bool = False
) -> List[Dict[str, Any]]:
    """Load lyric features with backend-specific smart caching.

    Args:
        backend: Lyric embedding backend ("bge-m3", "e5", or custom)
        fresh: If True, force re-extraction (ignore cache)

    Returns:
        List of lyric feature dicts
    """
    from analysis.pipeline.lyric_analysis import extract_lyric_features
    from analysis.pipeline.config import get_lyric_cache_path

    cache_path = get_lyric_cache_path(backend)

    # Load from cache (if not forcing fresh extraction)
    if not fresh:
        cached = load_from_cache(cache_path, f"lyric ({backend})")
        if cached:
            return cached

    # Extract fresh features
    logger.info(f"Extracting lyric features using {backend}...")
    return extract_lyric_features(backend=backend, cache_path=cache_path)


def load_interpretable_lyric_features(
    fresh: bool = False
) -> List[Dict[str, Any]]:
    """Load GPT-extracted interpretable lyric features.

    These are the semantic features extracted by GPT (lyric_valence, lyric_arousal,
    lyric_theme, lyric_language, etc.) stored separately from embedding-based features.

    Args:
        fresh: If True, force re-extraction (ignore cache)

    Returns:
        List of interpretable lyric feature dicts

    Raises:
        FileNotFoundError: If cache doesn't exist (run --extract-interpretable-lyrics first)
    """
    from analysis.pipeline.config import CACHE_PATHS

    cache_path = CACHE_PATHS["lyric_interpretable"]

    # Load from cache
    if not fresh:
        cached = load_from_cache(cache_path, "interpretable lyric (GPT)")
        if cached:
            return cached

    # Cache doesn't exist - user needs to extract first
    raise FileNotFoundError(
        f"Interpretable lyric features not found at {cache_path}.\n"
        "Run 'python analysis/run_analysis.py --extract-interpretable-lyrics' first to extract GPT features."
    )


def load_mert_embeddings(
    cache_path: str = "analysis/cache/mert_embeddings_24khz_30s_cls.pkl",
    fresh: bool = False
) -> List[Dict[str, Any]]:
    """Load MERT audio embeddings with smart caching.

    Args:
        cache_path: Path to MERT cache file
        fresh: If True, force re-extraction (ignore cache)

    Returns:
        List of MERT embedding dicts
    """
    from analysis.pipeline.mert_embedding import extract_mert_embeddings

    # Load from cache (if not forcing fresh extraction)
    if not fresh:
        cached = load_from_cache(cache_path, "MERT")
        if cached:
            return cached

    # Extract fresh MERT embeddings
    logger.info("Extracting MERT embeddings from MP3 files...")
    return extract_mert_embeddings(cache_path=cache_path, use_cache=False)
