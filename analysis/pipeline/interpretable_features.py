#!/usr/bin/env python3
"""Interpretable feature construction for music analysis.

This module provides the single source of truth for constructing 30-dimensional
interpretable feature vectors from audio and lyric features. Used by both the
CLI (analysis/run_analysis.py) and Streamlit dashboard (interactive_interpretability.py).

Vector Structure (30 dimensions):
- Audio features (14 dims): BPM, danceability, instrumentalness, valence, arousal,
                           engagement, approachability, moods (5), voice gender, genre ladder
- Key features (3 dims): Circular encoding (sin/cos) of pitch + major/minor scale (weighted 0.33)
- Lyric features (10 dims): valence, arousal, moods (4), explicit, narrative, vocabulary, repetition
- Theme (1 dim): Semantic scale from introspective → energetic/positive
- Language (1 dim): Ordinal encoding
- Popularity (1 dim): Spotify popularity score normalized to [0, 1]

CRITICAL: All lyric-related features are weighted by (1 - instrumentalness) to ensure
that instrumental songs are NOT clustered based on non-existent lyric content.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from analysis.pipeline.config import THEME_SCALE, LANGUAGE_SCALE

logger = logging.getLogger(__name__)


def build_interpretable_features(
    audio_features: List[Dict[str, Any]],
    lyric_features: List[Dict[str, Any]],
    popularity_data: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """Construct 30-dimensional interpretable feature vectors.

    Args:
        audio_features: List of audio feature dicts (must have 'track_id' key)
        lyric_features: List of lyric feature dicts (must have 'track_id' key)
        popularity_data: Optional dict mapping track_id to popularity (0-100)

    Returns:
        audio_features with new 'embedding' key containing 30-dim np.array

    Note:
        Modifies audio_features in-place by adding 'embedding' key.
        Lyric features weighted by (1 - instrumentalness) to prevent
        instrumental songs from being clustered by default lyric values.
        Popularity normalized from Spotify's 0-100 scale to [0,1].
    """
    logger.info("Building interpretable feature vectors (30 dimensions)")

    # Create lyric lookup by track_id
    lyric_by_id = {f["track_id"]: f for f in lyric_features}

    # Compute global normalization ranges for BPM, valence, arousal, popularity
    bpms = [float(t.get("bpm", 0) or 0) for t in audio_features]
    valences = [float(t.get("valence", 0) or 0) for t in audio_features]
    arousals = [float(t.get("arousal", 0) or 0) for t in audio_features]

    def get_range(values, default_min, default_max):
        """Get min/max range from valid (non-zero) values."""
        valid = [v for v in values if v > 0]
        if not valid:
            return default_min, default_max
        return min(valid), max(valid)

    min_bpm, max_bpm = get_range(bpms, 50, 200)
    min_val, max_val = get_range(valences, 1, 9)
    min_ar, max_ar = get_range(arousals, 1, 9)

    # Normalize popularity (0-100 from Spotify) to [0, 1]
    if popularity_data:
        popularities = [popularity_data.get(t["track_id"], 0) for t in audio_features]
        min_pop, max_pop = get_range(popularities, 0, 100)
        logger.info(f"Normalization ranges - BPM: [{min_bpm:.1f}, {max_bpm:.1f}], "
                    f"Valence: [{min_val:.1f}, {max_val:.1f}], "
                    f"Arousal: [{min_ar:.1f}, {max_ar:.1f}], "
                    f"Popularity: [{min_pop:.1f}, {max_pop:.1f}]")
    else:
        min_pop, max_pop = 0, 100
        logger.info(f"Normalization ranges - BPM: [{min_bpm:.1f}, {max_bpm:.1f}], "
                    f"Valence: [{min_val:.1f}, {max_val:.1f}], "
                    f"Arousal: [{min_ar:.1f}, {max_ar:.1f}]")

    # Track songs without lyric features (for logging only, NOT skipped)
    songs_without_lyrics = []

    # Process ALL tracks - no skipping
    tracks_to_process = []
    for track in audio_features:
        track_id = track["track_id"]
        track_name = track.get("track_name", "unknown")
        artist = track.get("artist", "unknown")
        instrumentalness = track.get("instrumentalness", 0.5)
        lyric = lyric_by_id.get(track_id, {})

        has_lyric_features = bool(lyric)  # Empty dict = no lyric features

        if not has_lyric_features:
            # Log but DON'T skip - will use 0 for lyric features
            songs_without_lyrics.append({
                "track_id": track_id,
                "track_name": track_name,
                "artist": artist,
                "instrumentalness": instrumentalness,
            })

        tracks_to_process.append((track, lyric))

    # Save songs without lyrics to JSON for reference (not skipped, just noted)
    if songs_without_lyrics:
        noted_path = Path("analysis/outputs/songs_without_lyric_features.json")
        noted_path.parent.mkdir(parents=True, exist_ok=True)
        with open(noted_path, "w") as f:
            json.dump(songs_without_lyrics, f, indent=2)
        logger.info(
            f"Note: {len(songs_without_lyrics)} songs have no lyric features (will use 0s). "
            f"Saved to {noted_path}"
        )

    logger.info(f"Processing ALL {len(tracks_to_process)} tracks")

    # Process valid tracks
    for track, lyric in tracks_to_process:
        track_id = track["track_id"]
        track_name = track.get("track_name", "unknown")
        instrumentalness = track.get("instrumentalness", 0.5)
        is_instrumental = instrumentalness >= 0.5

        # Helper functions - raise errors if data is missing
        def require_float(k):
            """Get float from track, raise error if missing."""
            v = track.get(k)
            if v is None:
                raise ValueError(f"Missing required audio feature '{k}' for track '{track_name}' ({track_id})")
            try:
                return float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid audio feature '{k}' for track '{track_name}' ({track_id}): {v}") from e

        def get_lyric_float(k, default=0.0):
            """Get float from lyric, return default if missing (for any song)."""
            v = lyric.get(k)
            if v is None:
                return default  # Use default for any missing lyric feature
            try:
                return float(v)
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid lyric feature '{k}' for track '{track_name}' ({track_id}): {v}, using default {default}")
                return default

        # =====================================================================
        # NORMALIZE AUDIO FEATURES
        # =====================================================================

        # Normalize BPM to [0, 1]
        raw_bpm = require_float("bpm")
        norm_bpm = (
            (raw_bpm - min_bpm) / (max_bpm - min_bpm)
            if (max_bpm > min_bpm)
            else 0.5
        )
        norm_bpm = max(0.0, min(1.0, norm_bpm))

        # Normalize Valence to [0, 1]
        raw_val = require_float("valence")
        norm_val = (
            (raw_val - min_val) / (max_val - min_val)
            if (max_val > min_val)
            else 0.5
        )
        norm_val = max(0.0, min(1.0, norm_val))

        # Normalize Arousal to [0, 1]
        raw_ar = require_float("arousal")
        norm_ar = (
            (raw_ar - min_ar) / (max_ar - min_ar)
            if (max_ar > min_ar)
            else 0.5
        )
        norm_ar = max(0.0, min(1.0, norm_ar))

        # =====================================================================
        # AUDIO FEATURES (14 dimensions)
        # =====================================================================
        audio_scalars = [
            norm_bpm,                               # 0: BPM (normalized)
            require_float("danceability"),          # 1: Danceability
            require_float("instrumentalness"),      # 2: Instrumentalness
            norm_val,                               # 3: Valence (normalized)
            norm_ar,                                # 4: Arousal (normalized)
            require_float("engagement_score"),      # 5: Engagement
            require_float("approachability_score"), # 6: Approachability
            require_float("mood_happy"),            # 7: Mood - Happy
            require_float("mood_sad"),              # 8: Mood - Sad
            require_float("mood_aggressive"),       # 9: Mood - Aggressive
            require_float("mood_relaxed"),          # 10: Mood - Relaxed
            require_float("mood_party"),            # 11: Mood - Party
            require_float("voice_gender_male"),     # 12: Voice Gender (0=female, 1=male)
            require_float("genre_ladder"),          # 13: Genre Ladder (0=pure, 1=fusion)
        ]

        # =====================================================================
        # KEY FEATURES (3 dimensions) - Circular encoding
        # =====================================================================
        key_vec = [0.0, 0.0, 0.0]
        key_str = track.get("key", "")
        if isinstance(key_str, str) and key_str:
            k = key_str.lower().strip()
            scale_val = 1.0 if "major" in k else 0.0

            # Pitch class mapping (C=0, C#/Db=1, ..., B=11)
            pitch_map = {
                "c": 0, "c#": 1, "db": 1,
                "d": 2, "d#": 3, "eb": 3,
                "e": 4,
                "f": 5, "f#": 6, "gb": 6,
                "g": 7, "g#": 8, "ab": 8,
                "a": 9, "a#": 10, "bb": 10,
                "b": 11,
            }

            parts = k.split()
            if parts and parts[0] in pitch_map:
                p = pitch_map[parts[0]]
                KEY_WEIGHT = 0.33  # Weight so 3 dims ≈ 1 equivalent dimension

                # Circular encoding: sin/cos to capture octave equivalence
                sin_val = (0.5 * np.sin(2 * np.pi * p / 12) + 0.5) * KEY_WEIGHT
                cos_val = (0.5 * np.cos(2 * np.pi * p / 12) + 0.5) * KEY_WEIGHT
                scale_val = scale_val * KEY_WEIGHT
                key_vec = [sin_val, cos_val, scale_val]

        # =====================================================================
        # LYRIC FEATURES (10 dimensions)
        # CRITICAL: Weighted by (1 - instrumentalness) to prevent instrumental
        #           songs from being clustered by default lyric values
        # =====================================================================
        instrumentalness_val = require_float("instrumentalness")
        lyric_weight = 1.0 - instrumentalness_val  # Key weighting factor!

        lyric_scalars = [
            get_lyric_float("lyric_valence", 0.5) * lyric_weight,             # 17: Lyric Valence
            get_lyric_float("lyric_arousal", 0.5) * lyric_weight,             # 18: Lyric Arousal
            get_lyric_float("lyric_mood_happy", 0.0) * lyric_weight,          # 19: Lyric Mood - Happy
            get_lyric_float("lyric_mood_sad", 0.0) * lyric_weight,            # 20: Lyric Mood - Sad
            get_lyric_float("lyric_mood_aggressive", 0.0) * lyric_weight,     # 21: Lyric Mood - Aggressive
            get_lyric_float("lyric_mood_relaxed", 0.0) * lyric_weight,        # 22: Lyric Mood - Relaxed
            get_lyric_float("lyric_explicit", 0.0) * lyric_weight,            # 23: Explicit content
            get_lyric_float("lyric_narrative", 0.0) * lyric_weight,           # 24: Narrative style
            get_lyric_float("lyric_vocabulary_richness", 0.0) * lyric_weight, # 25: Vocabulary richness
            get_lyric_float("lyric_repetition", 0.0) * lyric_weight,          # 26: Repetition score
        ]

        # =====================================================================
        # THEME (1 dimension) - Semantic scale, weighted by lyric_weight
        # =====================================================================
        theme = lyric.get("lyric_theme")
        if theme is None:
            theme = "none"  # Default for missing lyric features
        if not isinstance(theme, str):
            logger.warning(f"Invalid lyric_theme for track '{track_name}' ({track_id}): {theme}, using 'none'")
            theme = "none"
        theme = theme.lower().strip()
        if theme not in THEME_SCALE:
            logger.warning(f"Unknown theme '{theme}' for track '{track_name}' ({track_id}), using 'other'")
            theme = "other"
        theme_val = THEME_SCALE[theme] * lyric_weight  # 27: Theme

        # =====================================================================
        # LANGUAGE (1 dimension) - Ordinal encoding, weighted by lyric_weight
        # =====================================================================
        lang = lyric.get("lyric_language")
        if lang is None:
            lang = "none"  # Default for missing lyric features
        if not isinstance(lang, str):
            logger.warning(f"Invalid lyric_language for track '{track_name}' ({track_id}): {lang}, using 'none'")
            lang = "none"
        lang = lang.lower().strip()
        if lang not in LANGUAGE_SCALE:
            logger.warning(f"Unknown language '{lang}' for track '{track_name}' ({track_id}), using 'unknown'")
            lang = "unknown"
        lang_val = LANGUAGE_SCALE[lang] * lyric_weight  # 28: Language

        # =====================================================================
        # POPULARITY (1 dimension) - Normalized Spotify popularity score
        # =====================================================================
        if popularity_data:
            raw_pop = popularity_data.get(track_id, 50)  # Default to 50 if missing
        else:
            raw_pop = 50  # Default popularity
        norm_pop = (
            (raw_pop - min_pop) / (max_pop - min_pop)
            if (max_pop > min_pop)
            else 0.5
        )
        norm_pop = max(0.0, min(1.0, norm_pop))

        # =====================================================================
        # COMBINE ALL FEATURES (30 dimensions total)
        # =====================================================================
        embedding = np.array(
            audio_scalars + key_vec + lyric_scalars + [theme_val, lang_val, norm_pop],
            dtype=np.float32,
        )

        # Add embedding to track dict
        track["embedding"] = embedding

    # Return ALL processed tracks (no skipping)
    processed_tracks = [track for track, _ in tracks_to_process]
    logger.info(f"Built interpretable features for {len(processed_tracks)} tracks (ALL included)")
    return processed_tracks


def apply_feature_weights(
    audio_features: List[Dict[str, Any]],
    weights: Dict[str, float]
) -> List[Dict[str, Any]]:
    """Apply user-defined weights to feature vector dimensions.

    This allows dashboard users to emphasize or de-emphasize specific feature
    groups during clustering (e.g., weight moods more heavily than genre).

    Args:
        audio_features: List with 'embedding' key containing 30-dim vectors
        weights: Dict with keys:
            - 'core_audio': Weight for BPM, danceability, etc. (indices 0-6)
            - 'mood': Weight for audio moods (indices 7-11)
            - 'genre': Weight for voice gender + genre ladder (indices 12-13)
            - 'key': Weight for key features (indices 14-16)
            - 'lyric_emotion': Weight for lyric emotions (indices 17-22)
            - 'lyric_content': Weight for content features (indices 23-26)
            - 'theme': Weight for theme (index 27)
            - 'language': Weight for language (index 28)
            - 'popularity': Weight for popularity (index 29)

    Returns:
        audio_features with modified 'embedding' (in-place modification)

    Note:
        Modifies audio_features in-place.
    """
    logger.info(f"Applying feature weights: {weights}")

    # Index ranges for each feature group
    index_ranges = {
        'core_audio': (0, 7),      # BPM through approachability
        'mood': (7, 12),           # 5 mood dimensions
        'genre': (12, 14),         # Voice gender + genre ladder
        'key': (14, 17),           # 3 key dimensions
        'lyric_emotion': (17, 23), # Lyric valence/arousal + 4 moods
        'lyric_content': (23, 27), # Explicit, narrative, vocabulary, repetition
        'theme': (27, 28),         # Theme dimension
        'language': (28, 29),      # Language dimension
        'popularity': (29, 30),    # Popularity dimension
    }

    for track in audio_features:
        if "embedding" not in track:
            logger.warning(f"Track {track.get('track_id', 'unknown')} missing embedding, skipping")
            continue

        emb = track["embedding"].copy()

        # Apply weights to each feature group
        for group_name, (start_idx, end_idx) in index_ranges.items():
            weight = weights.get(group_name, 1.0)
            emb[start_idx:end_idx] *= weight

        track["embedding"] = emb

    logger.info(f"Applied weights to {len(audio_features)} track embeddings")
    return audio_features
