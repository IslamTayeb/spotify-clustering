#!/usr/bin/env python3
"""
Genre Fusion Feature Extraction (Entropy-based)

Computes a continuous 0-1 "genre_fusion" score for each song where:
- 0.0 = Pure genre (song clearly belongs to one genre)
- 1.0 = High genre fusion (song crosses many genres, AI uncertain)

This measures "how categorizable" a song is - a meta-feature about the song's
relationship to genre traditions:
- Low values = Artist working WITHIN a genre tradition (pure genre)
- High values = Artist CROSSING genre boundaries (fusion)

Uses entropy of the 400-dimensional genre probability vector from Essentia's
discogs400 classifier.
"""

import logging
import json
import numpy as np
from typing import Dict, List
from pathlib import Path
from scipy.stats import entropy

logger = logging.getLogger(__name__)


def load_genre_labels(model_name: str = "discogs400") -> List[str]:
    """Load genre labels for a specific model."""
    if model_name == "discogs400":
        filepath = (
            Path(__file__).parent.parent / "labels" / "genre_discogs400_labels.json"
        )
        if filepath.exists():
            with open(filepath, "r") as f:
                return json.load(f)
    return []


def compute_genre_entropy(genre_probs: np.ndarray) -> float:
    """
    Compute normalized entropy of genre probability distribution.

    Args:
        genre_probs: 400-dimensional probability vector from genre classifier

    Returns:
        Normalized entropy in [0, 1] range
    """
    if genre_probs is None or len(genre_probs) == 0:
        return 0.5  # Default for missing data

    probs = np.array(genre_probs)

    # Normalize to ensure it sums to 1
    probs = probs / (probs.sum() + 1e-10)

    # Filter out very small probabilities (noise)
    probs = probs[probs > 0.001]

    if len(probs) == 0:
        return 0.5

    # Compute entropy
    ent = entropy(probs)

    return float(ent)


def migrate_genre_purity_to_fusion(audio_features: List[Dict]) -> List[Dict]:
    """
    Migrate old 'genre_purity' field name to new 'genre_fusion' field name.

    This handles backward compatibility with caches that used the old naming.
    The values are semantically identical (0=pure, 1=fusion), just renamed
    for clarity since high "purity" meaning "fusion" was confusing.

    Args:
        audio_features: List of audio feature dictionaries

    Returns:
        Same list with 'genre_purity' renamed to 'genre_fusion'
    """
    migrated_count = 0
    for track in audio_features:
        # Migrate genre_purity -> genre_fusion if old key exists
        if "genre_purity" in track and "genre_fusion" not in track:
            track["genre_fusion"] = track.pop("genre_purity")
            migrated_count += 1
        # Migrate genre_purity_version -> genre_fusion_version
        if "genre_purity_version" in track and "genre_fusion_version" not in track:
            track["genre_fusion_version"] = track.pop("genre_purity_version")

    if migrated_count > 0:
        logger.info(f"Migrated {migrated_count} tracks from genre_purity to genre_fusion")

    return audio_features


def add_genre_fusion_to_features(audio_features: List[Dict]) -> List[Dict]:
    """
    Main entry point: add genre_fusion field to all audio features.

    The genre_fusion measures how much a song crosses genre boundaries:
    - 0.0 = Pure (clearly one genre, e.g., pure Trap)
    - 1.0 = Fusion (crosses many genres, e.g., experimental crossover)

    Args:
        audio_features: List of audio feature dictionaries with 'genre_probs'

    Returns:
        Same list with 'genre_fusion' field added to each track
    """
    # First, migrate any old genre_purity fields to genre_fusion
    audio_features = migrate_genre_purity_to_fusion(audio_features)

    logger.info("Computing genre fusion scores (entropy-based)...")

    # First pass: collect all entropies to normalize
    raw_entropies = []
    for track in audio_features:
        genre_probs = track.get("genre_probs")
        if genre_probs is not None:
            ent = compute_genre_entropy(genre_probs)
            raw_entropies.append(ent)
        else:
            raw_entropies.append(None)

    # Get valid entropies for normalization
    valid_entropies = [e for e in raw_entropies if e is not None]

    if not valid_entropies:
        logger.warning("No genre probabilities found - setting all to 0.5")
        for track in audio_features:
            track["genre_fusion"] = 0.5
        return audio_features

    # Compute min/max for normalization
    min_ent = min(valid_entropies)
    max_ent = max(valid_entropies)
    range_ent = max_ent - min_ent if max_ent > min_ent else 1.0

    # Second pass: normalize and assign
    fusion_values = []
    for i, track in enumerate(audio_features):
        if raw_entropies[i] is not None:
            # Normalize to [0, 1]
            normalized = (raw_entropies[i] - min_ent) / range_ent
            normalized = max(0.0, min(1.0, normalized))
            track["genre_fusion"] = float(normalized)
            track["genre_fusion_version"] = "entropy"  # Mark as entropy-based
            fusion_values.append(normalized)
        else:
            track["genre_fusion"] = 0.5  # Default
            track["genre_fusion_version"] = "entropy"

    # Log statistics
    if fusion_values:
        logger.info(f"Genre fusion computed for {len(fusion_values)} songs")
        logger.info(f"  - Mean: {np.mean(fusion_values):.3f}")
        logger.info(f"  - Min: {np.min(fusion_values):.3f}")
        logger.info(f"  - Max: {np.max(fusion_values):.3f}")
        logger.info(f"  - Std: {np.std(fusion_values):.3f}")
        logger.info("  - Interpretation: 0=Pure genre, 1=Genre fusion")

    return audio_features
