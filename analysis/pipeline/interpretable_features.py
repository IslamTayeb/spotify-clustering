#!/usr/bin/env python3
"""Interpretable feature construction for music analysis.

This module provides the single source of truth for constructing 32-dimensional
interpretable feature vectors from audio and lyric features. Used by both the
CLI (analysis/run_analysis.py) and Streamlit dashboard (interactive_interpretability.py).

Vector Structure (32 dimensions):
- Audio features (16 dims): BPM, danceability, instrumentalness, valence, arousal,
                           engagement, approachability, moods (5), voice gender, genre ladder,
                           acoustic/electronic (NEW), timbre brightness (NEW)
- Key features (3 dims): Circular encoding (sin/cos) of pitch + major/minor scale (weighted 0.33)
- Lyric features (10 dims): valence, arousal, moods (4), explicit, narrative, vocabulary, repetition
- Theme (1 dim): Semantic scale from introspective → energetic/positive
- Language (1 dim): Ordinal encoding
- Popularity (1 dim): Spotify popularity score normalized to [0, 1]

CRITICAL: Lyric-related features use different weighting strategies based on semantics:
- Bipolar scales (valence, arousal): interpolate toward 0.5 (neutral)
- Presence/absence (moods, explicit, etc.): scale toward 0 (absent)
- Categorical (theme, language): hard threshold at instrumentalness > 0.5 → centered "none"
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
    metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    popularity_data: Optional[Dict[str, float]] = None  # Backward compatibility
) -> List[Dict[str, Any]]:
    """Construct 33-dimensional interpretable feature vectors.

    Args:
        audio_features: List of audio feature dicts (must have 'track_id' key)
        lyric_features: List of lyric feature dicts (must have 'track_id' key)
        metadata: Optional dict mapping track_id to {'popularity': float, 'release_year': float}
        popularity_data: DEPRECATED - use metadata instead (kept for backward compatibility)

    Returns:
        audio_features with new 'embedding' key containing 33-dim np.array

    Note:
        Modifies audio_features in-place by adding 'embedding' key.
        Lyric features weighted by (1 - instrumentalness) to prevent
        instrumental songs from being clustered by default lyric values.
        Popularity normalized from Spotify's 0-100 scale to [0,1].
        Release year encoded as decade buckets (0.0=1950s, 1.0=2020s).
    """
    logger.info("Building interpretable feature vectors (33 dimensions)")

    # Backward compatibility: convert old popularity_data to metadata format
    if metadata is None and popularity_data is not None:
        metadata = {tid: {'popularity': pop, 'release_year': 0.5}
                   for tid, pop in popularity_data.items()}

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
    if metadata:
        popularities = [metadata.get(t["track_id"], {}).get('popularity', 50) for t in audio_features]
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
        # AUDIO FEATURES (16 dimensions)
        # =====================================================================
        # NEW: Acoustic vs Electronic (0=electronic, 1=acoustic)
        mood_acoustic = require_float("mood_acoustic")
        mood_electronic = require_float("mood_electronic")
        electronic_acoustic = (mood_acoustic - mood_electronic + 1) / 2  # Rescale [-1,1] to [0,1]

        # NEW: Timbre brightness (0=dark, 1=bright)
        timbre_brightness = require_float("timbre_bright")

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
            0.5 if is_instrumental else require_float("voice_gender_male"),  # 12: Voice Gender (0=female, 0.5=instrumental, 1=male)
            require_float("genre_fusion"),          # 13: Genre Fusion (0=pure, 1=fusion)
            electronic_acoustic,                    # 14: Acoustic/Electronic (0=electronic, 1=acoustic)
            timbre_brightness,                      # 15: Timbre Brightness (0=dark, 1=bright)
        ]

        # =====================================================================
        # KEY FEATURES (3 dimensions, indices 16-18) - Circular encoding
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
        # LYRIC FEATURES (10 dimensions, indices 19-28)
        # Three weighting strategies based on semantic type:
        # - Bipolar (valence, arousal): interpolate toward 0.5 (neutral)
        # - Presence/absence: scale toward 0 (absent)
        # - Categorical (theme, language): hard threshold (handled below)
        # =====================================================================
        instrumentalness_val = require_float("instrumentalness")
        lyric_weight = 1.0 - instrumentalness_val  # Key weighting factor!

        # BIPOLAR SCALES: interpolate toward 0.5 (neutral)
        # Rationale: These are negative↔positive scales. An instrumental track
        # isn't "lyrically negative"—it's lyrically absent, which should be neutral.
        lyric_valence_raw = get_lyric_float("lyric_valence", 0.5)
        lyric_arousal_raw = get_lyric_float("lyric_arousal", 0.5)
        lyric_valence = 0.5 + (lyric_valence_raw - 0.5) * lyric_weight
        lyric_arousal = 0.5 + (lyric_arousal_raw - 0.5) * lyric_weight

        # PRESENCE/ABSENCE: scale toward 0 (absent)
        # Rationale: These are "happy vs non_happy" classifiers. An instrumental
        # track is definitively non_happy, non_sad, non_explicit, etc.
        lyric_scalars = [
            lyric_valence,                                                    # 19: Lyric Valence
            lyric_arousal,                                                    # 20: Lyric Arousal
            get_lyric_float("lyric_mood_happy", 0.0) * lyric_weight,          # 21: Lyric Mood - Happy
            get_lyric_float("lyric_mood_sad", 0.0) * lyric_weight,            # 22: Lyric Mood - Sad
            get_lyric_float("lyric_mood_aggressive", 0.0) * lyric_weight,     # 23: Lyric Mood - Aggressive
            get_lyric_float("lyric_mood_relaxed", 0.0) * lyric_weight,        # 24: Lyric Mood - Relaxed
            get_lyric_float("lyric_explicit", 0.0) * lyric_weight,            # 25: Explicit content
            get_lyric_float("lyric_narrative", 0.0) * lyric_weight,           # 26: Narrative style
            get_lyric_float("lyric_vocabulary_richness", 0.0) * lyric_weight, # 27: Vocabulary richness
            get_lyric_float("lyric_repetition", 0.0) * lyric_weight,          # 28: Repetition score
        ]

        # =====================================================================
        # THEME (1 dimension, index 29) - Categorical with hard threshold
        # Rationale: Themes are categorical, not continuous. A track at
        # instrumentalness=0.9 doesn't have "10% Japanese"—it either has
        # meaningful lyrics with a theme or it doesn't.
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
        # CATEGORICAL: hard threshold with centered "none"
        if instrumentalness_val > 0.5:
            theme_val = 0.5  # Centered "none" for instrumental tracks
        else:
            theme_val = THEME_SCALE[theme]  # 29: Theme

        # =====================================================================
        # LANGUAGE (1 dimension, index 30) - Categorical with hard threshold
        # Same rationale as theme: categorical, not continuous.
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
        # CATEGORICAL: hard threshold with centered "none"
        if instrumentalness_val > 0.5:
            lang_val = 0.5  # Centered "none" for instrumental tracks
        else:
            lang_val = LANGUAGE_SCALE[lang]  # 30: Language

        # =====================================================================
        # POPULARITY (1 dimension, index 31) - Normalized Spotify popularity score
        # =====================================================================
        if metadata:
            raw_pop = metadata.get(track_id, {}).get('popularity', 50)  # Default to 50 if missing
        else:
            raw_pop = 50  # Default popularity
        norm_pop = (
            (raw_pop - min_pop) / (max_pop - min_pop)
            if (max_pop > min_pop)
            else 0.5
        )
        norm_pop = max(0.0, min(1.0, norm_pop))

        # =====================================================================
        # RELEASE YEAR (1 dimension, index 32) - Decade bucket encoding
        # =====================================================================
        if metadata:
            release_year = metadata.get(track_id, {}).get('release_year', 0.5)
        else:
            release_year = 0.5  # Default: centered "unknown"

        # =====================================================================
        # COMBINE ALL FEATURES (33 dimensions total)
        # =====================================================================
        embedding = np.array(
            audio_scalars + key_vec + lyric_scalars + [theme_val, lang_val, norm_pop, release_year],
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
        audio_features: List with 'embedding' key containing 33-dim vectors
        weights: Dict with keys:
            - 'core_audio': Weight for BPM, danceability, etc. (indices 0-6)
            - 'mood': Weight for audio moods (indices 7-11)
            - 'genre': Weight for voice gender, genre ladder, acoustic/electronic, timbre (indices 12-15)
            - 'key': Weight for key features (indices 16-18)
            - 'lyric_emotion': Weight for lyric emotions (indices 19-24)
            - 'lyric_content': Weight for content features (indices 25-28)
            - 'theme': Weight for theme (index 29)
            - 'language': Weight for language (index 30)
            - 'metadata': Weight for metadata (popularity + release year) (indices 31-32)

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
        'genre': (12, 16),         # Voice gender + genre ladder + acoustic/electronic + timbre
        'key': (16, 19),           # 3 key dimensions
        'lyric_emotion': (19, 25), # Lyric valence/arousal + 4 moods
        'lyric_content': (25, 29), # Explicit, narrative, vocabulary, repetition
        'theme': (29, 30),         # Theme dimension
        'language': (30, 31),      # Language dimension
        'metadata': (31, 33),      # Popularity + Release Year
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
