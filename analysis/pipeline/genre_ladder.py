#!/usr/bin/env python3
"""
Genre Ladder Feature Extraction (Entropy-based)

Computes a continuous 0-1 "genre_ladder" score for each song where:
- 0.0 = High genre purity (song clearly belongs to one genre)
- 1.0 = High genre fusion (song crosses many genres, AI uncertain)

This measures "how categorizable" a song is - a meta-feature about the song's
relationship to genre traditions:
- Low values = Artist working WITHIN a genre tradition
- High values = Artist CROSSING genre boundaries

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
            Path(__file__).parent.parent / "data" / "genre_discogs400_labels.json"
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


def add_genre_ladder_to_features(audio_features: List[Dict]) -> List[Dict]:
    """
    Main entry point: add genre_ladder field to all audio features.

    The genre_ladder measures genre purity/fusion:
    - 0.0 = Pure (clearly one genre, e.g., pure Trap)
    - 1.0 = Fusion (crosses many genres, e.g., experimental crossover)

    Args:
        audio_features: List of audio feature dictionaries with 'genre_probs'

    Returns:
        Same list with 'genre_ladder' field added to each track
    """
    logger.info("Computing genre ladder scores (entropy-based)...")

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
            track["genre_ladder"] = 0.5
        return audio_features

    # Compute min/max for normalization
    min_ent = min(valid_entropies)
    max_ent = max(valid_entropies)
    range_ent = max_ent - min_ent if max_ent > min_ent else 1.0

    # Second pass: normalize and assign
    ladder_values = []
    for i, track in enumerate(audio_features):
        if raw_entropies[i] is not None:
            # Normalize to [0, 1]
            normalized = (raw_entropies[i] - min_ent) / range_ent
            normalized = max(0.0, min(1.0, normalized))
            track["genre_ladder"] = float(normalized)
            track["genre_ladder_version"] = "entropy"  # Mark as entropy-based
            ladder_values.append(normalized)
        else:
            track["genre_ladder"] = 0.5  # Default
            track["genre_ladder_version"] = "entropy"

    # Log statistics
    if ladder_values:
        logger.info(f"Genre ladder computed for {len(ladder_values)} songs")
        logger.info(f"  - Mean: {np.mean(ladder_values):.3f}")
        logger.info(f"  - Min: {np.min(ladder_values):.3f}")
        logger.info(f"  - Max: {np.max(ladder_values):.3f}")
        logger.info(f"  - Std: {np.std(ladder_values):.3f}")
        logger.info("  - Interpretation: 0=Pure genre, 1=Genre fusion")

    return audio_features
