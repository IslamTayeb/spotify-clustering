import sys
from pathlib import Path

# Add project root to sys.path to allow imports from export/
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import umap
import plotly.graph_objects as go
from sklearn.cluster import (
    AgglomerativeClustering,
    SpectralClustering,
    Birch,
    KMeans,
    DBSCAN,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from collections import Counter

# Try to import export tools
try:
    from export.create_playlists import (
        create_spotify_client,
        create_playlist,
        add_tracks_to_playlist,
    )

    HAS_EXPORT_TOOLS = True
except ImportError:
    HAS_EXPORT_TOOLS = False
    print("Warning: Could not import export tools. Spotify export will be disabled.")

st.set_page_config(layout="wide", page_title="Music Clustering Tuner")


@st.cache_data
def load_data():
    # Robust path resolution (handles running from root or analysis/ dir)
    cache_dirs = [Path("cache"), Path("../cache")]
    cache_dir = next((d for d in cache_dirs if d.exists()), None)

    if not cache_dir:
        st.error("Cache directory not found! Expected 'cache/' or '../cache/'")
        return [], [], []

    # Load standard features
    with open(cache_dir / "audio_features.pkl", "rb") as f:
        audio_features = pickle.load(f)
    with open(cache_dir / "lyric_features.pkl", "rb") as f:
        lyric_features = pickle.load(f)

    # Load MERT features if available
    mert_features = []
    mert_path = cache_dir / "mert_embeddings_24khz_30s_cls.pkl"
    if mert_path.exists():
        with open(mert_path, "rb") as f:
            mert_features = pickle.load(f)

    # Align features by track_id
    audio_by_id = {f["track_id"]: f for f in audio_features}
    lyric_by_id = {f["track_id"]: f for f in lyric_features}
    mert_by_id = {f["track_id"]: f for f in mert_features} if mert_features else {}

    # Find common IDs (Essentia + Lyrics)
    # Note: We don't strictly require MERT to exist for all tracks to support backward compatibility
    common_ids = set(audio_by_id.keys()) & set(lyric_by_id.keys())

    sorted_ids = sorted(common_ids)
    aligned_audio = [audio_by_id[tid] for tid in sorted_ids]
    aligned_lyrics = [lyric_by_id[tid] for tid in sorted_ids]

    # Align MERT (fill with None if missing)
    aligned_mert = [mert_by_id.get(tid) for tid in sorted_ids]

    return aligned_audio, aligned_lyrics, aligned_mert


def prepare_features_for_mode(
    audio_features,
    lyric_features,
    mode,
    n_pca_components,
    skip_pca=False,
    interpretable_mode=False,
):
    """Prepare PCA-reduced features for clustering.

    Args:
        interpretable_mode: If True and mode="combined", use only audio embeddings
            (which already contain lyric interpretable features). Does NOT use
            the cached lyric embeddings.
    """
    # For combined mode, filter out vocal songs without lyrics
    # Note: Instrumental songs (instrumentalness >= 0.5) are kept even without lyrics
    #       Non-instrumental songs (vocal) without lyrics are filtered out
    if mode == "combined":
        # Filter tracks where instrumentalness < 0.5 (vocal) but has_lyrics is False
        # Keep instrumental songs (instrumentalness >= 0.5) even without lyrics
        valid_mask = [
            not (
                audio.get("instrumentalness", 0.5) < 0.5
                and not lyric.get("has_lyrics", False)
            )
            for audio, lyric in zip(audio_features, lyric_features)
        ]
        audio_features = [f for f, valid in zip(audio_features, valid_mask) if valid]
        lyric_features = [f for f, valid in zip(lyric_features, valid_mask) if valid]

        filtered_count = sum(1 for v in valid_mask if not v)
        if filtered_count > 0:
            st.sidebar.info(
                f"ℹ️ Filtered out {filtered_count} vocal songs without lyrics in combined mode"
            )

    audio_emb = np.vstack([f["embedding"] for f in audio_features])

    # For interpretable mode with combined, we don't use lyric embeddings
    # The audio embedding already contains lyric interpretable features
    if not interpretable_mode:
        lyric_emb = np.vstack([f["embedding"] for f in lyric_features])

    # If skipping PCA, just standardize and return
    if skip_pca:
        if mode == "audio":
            features_reduced = StandardScaler().fit_transform(audio_emb)
            valid_indices = list(range(len(audio_features)))
        elif mode == "lyrics":
            has_lyrics = np.array([f["has_lyrics"] for f in lyric_features])
            valid_indices = np.where(has_lyrics)[0].tolist()
            lyric_emb = np.vstack([f["embedding"] for f in lyric_features])
            features_reduced = StandardScaler().fit_transform(lyric_emb[has_lyrics])
        else:  # combined
            if interpretable_mode:
                # Interpretable mode: audio embedding already has lyric features
                # Just use audio embeddings (no lyric embeddings needed)
                features_reduced = StandardScaler().fit_transform(audio_emb)
            else:
                audio_norm = StandardScaler().fit_transform(audio_emb)
                lyric_norm = StandardScaler().fit_transform(lyric_emb)
                features_reduced = np.hstack([audio_norm, lyric_norm])
            valid_indices = list(range(len(audio_features)))

        return features_reduced, valid_indices, 1.0  # 100% variance explained

    if mode == "audio":
        # Standardize then PCA
        audio_norm = StandardScaler().fit_transform(audio_emb)
        n_components = min(audio_norm.shape[0], audio_norm.shape[1], n_pca_components)

        pca = PCA(n_components=n_components, random_state=42)
        features_reduced = pca.fit_transform(audio_norm)
        valid_indices = list(range(len(audio_features)))
        explained_var = np.sum(pca.explained_variance_ratio_)

    elif mode == "lyrics":
        has_lyrics = np.array([f["has_lyrics"] for f in lyric_features])
        valid_indices = np.where(has_lyrics)[0].tolist()

        # Standardize then PCA
        lyric_emb = np.vstack([f["embedding"] for f in lyric_features])
        lyric_norm = StandardScaler().fit_transform(lyric_emb[has_lyrics])
        n_components = min(lyric_norm.shape[0], lyric_norm.shape[1], n_pca_components)

        pca = PCA(n_components=n_components, random_state=42)
        features_reduced = pca.fit_transform(lyric_norm)
        explained_var = np.sum(pca.explained_variance_ratio_)

    else:  # combined
        if interpretable_mode:
            # Interpretable mode: audio embedding already has lyric features
            audio_norm = StandardScaler().fit_transform(audio_emb)
            n_components = min(
                audio_norm.shape[0], audio_norm.shape[1], n_pca_components
            )
            pca = PCA(n_components=n_components, random_state=42)
            features_reduced = pca.fit_transform(audio_norm)
            explained_var = np.sum(pca.explained_variance_ratio_)
        else:
            # Standardize first
            audio_norm = StandardScaler().fit_transform(audio_emb)
            lyric_norm = StandardScaler().fit_transform(lyric_emb)

            # PCA Reduction to balance modalities
            n_components_audio = min(
                audio_norm.shape[0], audio_norm.shape[1], n_pca_components
            )
            n_components_lyric = min(
                lyric_norm.shape[0], lyric_norm.shape[1], n_pca_components
            )

            pca_audio = PCA(n_components=n_components_audio, random_state=42)
            audio_reduced = pca_audio.fit_transform(audio_norm)

            pca_lyric = PCA(n_components=n_components_lyric, random_state=42)
            lyric_reduced = pca_lyric.fit_transform(lyric_norm)

            explained_var = (
                np.sum(pca_audio.explained_variance_ratio_)
                + np.sum(pca_lyric.explained_variance_ratio_)
            ) / 2

            # Combine the balanced, reduced vectors
            features_reduced = np.hstack([audio_reduced, lyric_reduced])
        valid_indices = list(range(len(audio_features)))

    return features_reduced, valid_indices, explained_var


def main():
    st.title("🎵 Music Clustering Tuner")
    st.markdown("""
    **Pipeline Architecture:**
    1. **Features → PCA** (dimensionality reduction, tunable below)
    2. **PCA Features → HAC/Birch/Spectral** (clustering, determines colors)
    3. **PCA Features → UMAP** (visualization only, determines 3D positions)
    """)

    # Load Data
    with st.spinner("Loading cached features..."):
        audio_features, lyric_features, mert_features = load_data()

    st.sidebar.header("⚙️ Feature Preparation")

    # Backend Selection
    has_mert = any(x is not None for x in mert_features)
    backend_options = ["Essentia (Default)"]
    if has_mert:
        backend_options.append("MERT (Transformer)")
    backend_options.append("Interpretable Features (Audio)")

    backend = st.sidebar.selectbox(
        "Audio Embedding Backend",
        backend_options,
        index=1 if has_mert else 0,
        help="Choose backend. 'Interpretable' uses BPM, Key, Moods, etc. 'MERT' uses transformers.",
    )

    # Apply backend override
    if "MERT" in backend:
        # Check integrity
        if len(mert_features) != len(audio_features):
            st.warning(
                f"MERT features count ({len(mert_features)}) mismatch with Essentia ({len(audio_features)})"
            )

        # Override embeddings in the audio_features list (in-memory only)
        # We assume mert_features aligns with audio_features from load_data
        valid_mert_count = 0
        for i, mert_item in enumerate(mert_features):
            if mert_item is not None and i < len(audio_features):
                audio_features[i]["embedding"] = mert_item["embedding"]
                valid_mert_count += 1

        st.sidebar.success(f"Using MERT embeddings for {valid_mert_count} tracks")

    elif "Interpretable" in backend:
        st.sidebar.info("✨ Using interpretable features: Audio + Lyric (combined)")

        # Create lyric lookup by track_id
        lyric_by_id = {f["track_id"]: f for f in lyric_features}

        # Calculate dynamic global min/max for normalization
        bpms = [float(t.get("bpm", 0) or 0) for t in audio_features]
        valences = [float(t.get("valence", 0) or 0) for t in audio_features]
        arousals = [float(t.get("arousal", 0) or 0) for t in audio_features]

        # Helper for safe min/max
        def get_range(values, default_min, default_max):
            valid = [v for v in values if v > 0]
            if not valid:
                return default_min, default_max
            return min(valid), max(valid)

        min_bpm, max_bpm = get_range(bpms, 50, 200)
        min_val, max_val = get_range(valences, 1, 9)
        min_ar, max_ar = get_range(arousals, 1, 9)

        # Theme & Language: 1 dimension each, natural 0-1 scale
        # Theme ordered by emotional energy/positivity (high=positive/energetic, low=negative/introspective)
        THEME_SCALE = {
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
        # Language: ordinal encoding, natural 0-1 scale
        LANGUAGE_SCALE = {
            "english": 1.0,
            "spanish": 0.86,
            "french": 0.71,
            "arabic": 0.57,
            "korean": 0.43,
            "japanese": 0.29,
            "unknown": 0.14,
            "none": 0.0,
        }

        count = 0
        for track in audio_features:
            lyric = lyric_by_id.get(track["track_id"], {})

            # Helper for float extraction
            def get_float(d, k, default=0.0):
                v = d.get(k)
                if v is None:
                    return default
                try:
                    return float(v)
                except Exception:
                    return default

            # Normalize BPM
            raw_bpm = get_float(track, "bpm", 120)
            norm_bpm = (
                (raw_bpm - min_bpm) / (max_bpm - min_bpm)
                if (max_bpm > min_bpm)
                else 0.5
            )
            norm_bpm = max(0.0, min(1.0, norm_bpm))

            # Normalize Audio Valence
            raw_val = get_float(track, "valence", 4.5)
            norm_val = (
                (raw_val - min_val) / (max_val - min_val)
                if (max_val > min_val)
                else 0.5
            )
            norm_val = max(0.0, min(1.0, norm_val))

            # Normalize Audio Arousal
            raw_ar = get_float(track, "arousal", 4.5)
            norm_ar = (
                (raw_ar - min_ar) / (max_ar - min_ar) if (max_ar > min_ar) else 0.5
            )
            norm_ar = max(0.0, min(1.0, norm_ar))

            # ═══════════════════════════════════════════════════════════════
            # AUDIO FEATURES (17 dimensions)
            # ═══════════════════════════════════════════════════════════════
            audio_scalars = [
                norm_bpm,  # 0: BPM (norm)
                get_float(track, "danceability", 0.5),  # 1: Danceability
                get_float(track, "instrumentalness", 0.0),  # 2: Instrumentalness
                norm_val,  # 3: Valence (norm)
                norm_ar,  # 4: Arousal (norm)
                get_float(track, "engagement_score", 0.5),  # 5: Engagement
                get_float(track, "approachability_score", 0.5),  # 6: Approachability
                get_float(track, "mood_happy", 0.0),  # 7: Mood Happy
                get_float(track, "mood_sad", 0.0),  # 8: Mood Sad
                get_float(track, "mood_aggressive", 0.0),  # 9: Mood Aggressive
                get_float(track, "mood_relaxed", 0.0),  # 10: Mood Relaxed
                get_float(track, "mood_party", 0.0),  # 11: Mood Party
                get_float(track, "voice_gender_male", 0.5),  # 12: Voice Gender Male
                get_float(track, "genre_ladder", 0.5),  # 13: Genre Ladder
            ]

            # Key Features (3 dimensions, weighted)
            key_vec = [0.0, 0.0, 0.0]
            key_str = track.get("key", "")
            if isinstance(key_str, str) and key_str:
                k = key_str.lower().strip()
                scale_val = 1.0 if "major" in k else 0.0
                pitch_map = {
                    "c": 0,
                    "c#": 1,
                    "db": 1,
                    "d": 2,
                    "d#": 3,
                    "eb": 3,
                    "e": 4,
                    "f": 5,
                    "f#": 6,
                    "gb": 6,
                    "g": 7,
                    "g#": 8,
                    "ab": 8,
                    "a": 9,
                    "a#": 10,
                    "bb": 10,
                    "b": 11,
                }
                parts = k.split()
                if parts and parts[0] in pitch_map:
                    p = pitch_map[parts[0]]
                    KEY_WEIGHT = 0.33  # 3 dims × 0.33 ≈ 1 equivalent dimension
                    sin_val = (0.5 * np.sin(2 * np.pi * p / 12) + 0.5) * KEY_WEIGHT
                    cos_val = (0.5 * np.cos(2 * np.pi * p / 12) + 0.5) * KEY_WEIGHT
                    scale_val = scale_val * KEY_WEIGHT
                    key_vec = [sin_val, cos_val, scale_val]

            # ═══════════════════════════════════════════════════════════════
            # LYRIC FEATURES (8 continuous + 11 theme + 8 language = 27 dims)
            # Weight lyric features by (1 - instrumentalness) so that
            # highly instrumental songs don't get clustered by lyrics
            # ═══════════════════════════════════════════════════════════════
            instrumentalness_val = get_float(track, "instrumentalness", 0.0)
            lyric_weight = (
                1.0 - instrumentalness_val
            )  # 0 if fully instrumental, 1 if fully vocal

            lyric_scalars = [
                get_float(lyric, "lyric_valence", 0.5)
                * lyric_weight,  # 0: Lyric Valence
                get_float(lyric, "lyric_arousal", 0.5)
                * lyric_weight,  # 1: Lyric Arousal
                get_float(lyric, "lyric_mood_happy", 0.0)
                * lyric_weight,  # 2: Lyric Happy
                get_float(lyric, "lyric_mood_sad", 0.0) * lyric_weight,  # 3: Lyric Sad
                get_float(lyric, "lyric_mood_aggressive", 0.0)
                * lyric_weight,  # 4: Lyric Aggressive
                get_float(lyric, "lyric_mood_relaxed", 0.0)
                * lyric_weight,  # 5: Lyric Relaxed
                get_float(lyric, "lyric_explicit", 0.0) * lyric_weight,  # 6: Explicit
                get_float(lyric, "lyric_narrative", 0.0) * lyric_weight,  # 7: Narrative
                get_float(lyric, "lyric_vocabulary_richness", 0.0)
                * lyric_weight,  # 8: Vocab Richness
                get_float(lyric, "lyric_repetition", 0.0)
                * lyric_weight,  # 9: Repetition
            ]

            # Theme (1 dim) - semantic scale, weighted by lyric_weight
            theme = lyric.get("lyric_theme", "other")
            if not isinstance(theme, str):
                theme = "other"
            theme = theme.lower().strip()
            theme_val = THEME_SCALE.get(theme, 0.1) * lyric_weight  # default to "other"

            # Language (1 dim) - ordinal scale, weighted by lyric_weight
            lang = lyric.get("lyric_language", "unknown")
            if not isinstance(lang, str):
                lang = "unknown"
            lang = lang.lower().strip()
            lang_val = (
                LANGUAGE_SCALE.get(lang, 0.14) * lyric_weight
            )  # default to "unknown"

            # Combine all features (29 dims total)
            full_vector = (
                audio_scalars + key_vec + lyric_scalars + [theme_val, lang_val]
            )
            track["embedding"] = np.array(full_vector, dtype=np.float32)
            count += 1

        st.sidebar.success(
            f"Constructed interpretable vectors for {count} tracks (Audio + Lyrics)"
        )

        # Debug: Show sample vector breakdown
        with st.sidebar.expander("🔍 Debug: Sample Vector", expanded=False):
            sample_track = audio_features[0]
            sample_lyric = lyric_by_id.get(sample_track["track_id"], {})
            emb = sample_track["embedding"]
            sample_instrumentalness = sample_track.get("instrumentalness", 0.0)
            sample_lyric_weight = 1.0 - sample_instrumentalness

            st.write(f"**{sample_track['track_name']}** by {sample_track['artist']}")
            st.write(f"Vector length: {len(emb)}")
            st.write(
                f"**Instrumentalness:** {sample_instrumentalness:.2f} → **Lyric Weight:** {sample_lyric_weight:.2f}"
            )

            # Audio features (0-16)
            st.markdown("**Audio Features:**")
            audio_names = [
                "BPM",
                "Danceability",
                "Instrumentalness",
                "Valence",
                "Arousal",
                "Engagement",
                "Approachability",
                "Mood Happy",
                "Mood Sad",
                "Mood Aggressive",
                "Mood Relaxed",
                "Mood Party",
                "Voice Gender Male",
                "Genre Ladder",
                "Key Sin",
                "Key Cos",
                "Key Scale",
            ]
            for i, name in enumerate(audio_names):
                if i < len(emb):
                    st.write(f"`{name}`: {emb[i]:.4f}")

            # Lyric features (17-26)
            st.markdown("**Lyric Features:**")
            lyric_names = [
                "Lyric Valence",
                "Lyric Arousal",
                "Lyric Happy",
                "Lyric Sad",
                "Lyric Aggressive",
                "Lyric Relaxed",
                "Explicit",
                "Narrative",
                "Vocab Richness",
                "Repetition",
            ]
            for i, name in enumerate(lyric_names):
                idx = 17 + i
                if idx < len(emb):
                    st.write(f"`{name}`: {emb[idx]:.4f}")

            # Theme/Language (1 dim each, indices 27-28)
            st.markdown("**Categorical (1 dim each):**")
            theme_raw = sample_lyric.get("lyric_theme", "N/A")
            lang_raw = sample_lyric.get("lyric_language", "N/A")
            st.write(
                f"Theme: {theme_raw} → {THEME_SCALE.get(str(theme_raw).lower().strip(), 0.1):.2f}"
            )
            st.write(
                f"Language: {lang_raw} → {LANGUAGE_SCALE.get(str(lang_raw).lower().strip(), 0.14):.2f}"
            )
            if len(emb) >= 29:
                st.write(f"`Theme val`: {emb[27]:.4f}")
                st.write(f"`Language val`: {emb[28]:.4f}")

        # Add weight controls for feature groups
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎛️ Audio Feature Weights")

        core_weight = st.sidebar.slider(
            "🎵 Core Audio (BPM, Danceability, etc.)",
            0.0,
            2.0,
            1.0,
            0.1,
            help="BPM, danceability, instrumentalness, valence, arousal, engagement, approachability",
        )
        mood_weight = st.sidebar.slider(
            "😊 Audio Moods",
            0.0,
            2.0,
            1.0,
            0.1,
            help="happy, sad, aggressive, relaxed, party",
        )
        genre_weight = st.sidebar.slider(
            "🎸 Genre Ladder",
            0.0,
            2.0,
            1.0,
            0.1,
            help="0=acoustic, 1=electronic",
        )
        key_weight = st.sidebar.slider(
            "🎹 Key",
            0.0,
            2.0,
            1.0,
            0.1,
            help="Musical key (sin, cos, scale)",
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("📝 Lyric Feature Weights")

        lyric_emotion_weight = st.sidebar.slider(
            "💭 Lyric Emotions (Valence, Arousal, Moods)",
            0.0,
            2.0,
            1.0,
            0.1,
            help="Lyric valence, arousal, happy, sad, aggressive, relaxed",
        )
        lyric_content_weight = st.sidebar.slider(
            "🔞 Content (Explicit, Narrative, Vocab)",
            0.0,
            2.0,
            1.0,
            0.1,
            help="Explicit content, narrative style, vocabulary richness, repetition",
        )
        theme_weight = st.sidebar.slider(
            "🏷️ Theme",
            0.0,
            2.0,
            1.0,
            0.1,
            help="Theme categories: love, heartbreak, party, flex, street, etc.",
        )
        language_weight = st.sidebar.slider(
            "🌍 Language",
            0.0,
            2.0,
            1.0,
            0.1,
            help="Language of lyrics",
        )

        # Apply weights to embeddings
        # Vector structure (29 dims total):
        #   0-6:   Core audio (bpm, danceability, instrumentalness, valence, arousal, engagement, approachability)
        #   7-11:  Audio moods (happy, sad, aggressive, relaxed, party)
        #   12-13: Voice gender + genre ladder
        #   14-16: Key (sin, cos, scale)
        #   17-22: Lyric emotions (valence, arousal, happy, sad, aggressive, relaxed)
        #   23-26: Lyric content (explicit, narrative, vocabulary_richness, repetition)
        #   27:    Theme (1 dim)
        #   28:    Language (1 dim)
        for track in audio_features:
            emb = track["embedding"].copy()

            # Audio Core (indices 0-6): 7 features
            emb[0:7] = emb[0:7] * core_weight

            # Audio Moods (indices 7-11): 5 features
            emb[7:12] = emb[7:12] * mood_weight

            # Voice Gender + Genre Ladder (indices 12-13)
            emb[12:14] = emb[12:14] * core_weight

            # Key features (indices 14-16): 3 features
            emb[14:17] = emb[14:17] * key_weight

            # Lyric emotions (indices 17-22): valence, arousal, happy, sad, aggressive, relaxed
            emb[17:23] = emb[17:23] * lyric_emotion_weight

            # Lyric content (indices 23-26): explicit, narrative, vocab, repetition
            emb[23:27] = emb[23:27] * lyric_content_weight

            # Theme (index 27): 1 dim
            emb[27] = emb[27] * theme_weight

            # Language (index 28): 1 dim
            emb[28] = emb[28] * language_weight

            track["embedding"] = emb

    # Mode Selection
    mode = st.sidebar.selectbox("Feature Mode", ["combined", "audio", "lyrics"])

    # PCA Control
    skip_pca = st.sidebar.checkbox(
        "Skip PCA (Use Raw Features)",
        value=False,
        help="Pass high-dimensional embeddings directly to clustering.",
    )

    # PCA Parameters - use mode-specific defaults for 75% variance
    pca_defaults = {"audio": 118, "lyrics": 162, "combined": 142}
    default_pca = pca_defaults.get(mode, 140)

    if not skip_pca:
        n_pca_components = st.sidebar.slider(
            "PCA Components",
            5,
            200,
            default_pca,
            step=5,
            help=f"Number of PCA components for dimensionality reduction before clustering. Default ({default_pca}) achieves ~75% variance for {mode} mode.",
        )
    else:
        n_pca_components = default_pca  # Placeholder, ignored

    # Prepare features
    is_interpretable = "Interpretable" in backend
    with st.spinner("Preparing features..."):
        pca_features, valid_indices, explained_var = prepare_features_for_mode(
            audio_features,
            lyric_features,
            mode,
            n_pca_components,
            skip_pca,
            interpretable_mode=is_interpretable,
        )

    st.sidebar.info(
        f"**PCA Features Shape:** {pca_features.shape}\n\n"
        f"**Explained Variance:** {explained_var:.2%}"
    )

    st.sidebar.header("🎯 Clustering Algorithm")

    # Clustering Algorithm Selection
    clustering_algorithm = st.sidebar.selectbox(
        "Algorithm",
        [
            "HAC (Hierarchical Agglomerative)",
            "Birch",
            "Spectral Clustering",
            "K-Means",
            "DBSCAN",
        ],
        help="Choose the clustering algorithm to use on PCA-reduced features.",
    )

    # Algorithm-specific parameters
    if clustering_algorithm == "HAC (Hierarchical Agglomerative)":
        st.sidebar.subheader("HAC Parameters")
        n_clusters_hac = st.sidebar.slider(
            "Number of Clusters",
            2,
            50,
            20,
            help="Fixed number of clusters to create.",
        )
        linkage_method = st.sidebar.selectbox(
            "Linkage Method",
            ["ward", "complete", "average", "single"],
            index=0,
            help="'ward' minimizes variance within clusters. 'complete' uses max distance between clusters. 'average' uses average distance. 'single' uses min distance.",
        )

        # Run HAC
        with st.spinner("Running Hierarchical Agglomerative Clustering..."):
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters_hac, linkage=linkage_method
            )
            labels = clusterer.fit_predict(pca_features)

    elif clustering_algorithm == "Birch":
        st.sidebar.subheader("Birch Parameters")
        st.sidebar.info(
            "⚡ Fast hierarchical clustering, similar to HAC but more efficient"
        )
        n_clusters_birch = st.sidebar.slider(
            "Number of Clusters",
            2,
            50,
            20,
            help="Final number of clusters to create.",
        )
        threshold = st.sidebar.slider(
            "Threshold",
            0.1,
            2.0,
            0.5,
            step=0.1,
            help="Radius of subcluster. Lower = more granular subclusters, might be slower.",
        )
        branching_factor = st.sidebar.slider(
            "Branching Factor",
            10,
            100,
            50,
            step=10,
            help="Max subclusters per node. Higher = more memory but better clustering.",
        )

        # Run Birch
        with st.spinner("Running Birch Clustering..."):
            clusterer = Birch(
                n_clusters=n_clusters_birch,
                threshold=threshold,
                branching_factor=branching_factor,
            )
            labels = clusterer.fit_predict(pca_features)

    elif clustering_algorithm == "Spectral Clustering":
        st.sidebar.subheader("Spectral Clustering Parameters")
        st.sidebar.info("🌐 Graph-based clustering, great for non-convex shapes")
        n_clusters_spectral = st.sidebar.slider(
            "Number of Clusters",
            2,
            50,
            20,
            help="Number of clusters to find.",
        )
        affinity = st.sidebar.selectbox(
            "Affinity",
            ["nearest_neighbors", "rbf"],
            index=0,
            help="'nearest_neighbors' uses k-NN graph. 'rbf' uses RBF kernel (slower).",
        )
        n_neighbors = st.sidebar.slider(
            "N Neighbors",
            5,
            50,
            15,
            help="Number of neighbors for k-NN graph (only used if affinity=nearest_neighbors).",
        )
        assign_labels = st.sidebar.selectbox(
            "Label Assignment",
            ["kmeans", "discretize"],
            index=0,
            help="'kmeans' is faster and usually better. 'discretize' is an alternative method.",
        )

        # Run Spectral Clustering
        with st.spinner("Running Spectral Clustering..."):
            clusterer = SpectralClustering(
                n_clusters=n_clusters_spectral,
                affinity=affinity,
                n_neighbors=n_neighbors if affinity == "nearest_neighbors" else 10,
                assign_labels=assign_labels,
                random_state=42,
            )
            labels = clusterer.fit_predict(pca_features)

    elif clustering_algorithm == "K-Means":
        st.sidebar.subheader("K-Means Parameters")
        n_clusters_kmeans = st.sidebar.slider(
            "Number of Clusters",
            2,
            50,
            20,
            help="Number of clusters to create.",
        )
        init_method = st.sidebar.selectbox(
            "Initialization Method",
            ["k-means++", "random"],
            index=0,
            help="'k-means++' selects initial cluster centroids using sampling based on an empirical probability distribution of the points' contribution to the overall inertia.",
        )

        # Run K-Means
        with st.spinner("Running K-Means Clustering..."):
            clusterer = KMeans(
                n_clusters=n_clusters_kmeans,
                init=init_method,
                n_init=10,
                random_state=42,
            )
            labels = clusterer.fit_predict(pca_features)

    elif clustering_algorithm == "DBSCAN":
        st.sidebar.subheader("DBSCAN Parameters")
        st.sidebar.info("Density-based clustering. Finds outliers automatically (-1).")
        eps = st.sidebar.slider(
            "Epsilon (eps)",
            0.1,
            5.0,
            0.5,
            step=0.1,
            help="The maximum distance between two samples for one to be considered as in the neighborhood of the other.",
        )
        min_samples = st.sidebar.slider(
            "Min Samples",
            2,
            20,
            5,
            help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.",
        )

        # Run DBSCAN
        with st.spinner("Running DBSCAN Clustering..."):
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(pca_features)

    # Calculate metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = (labels == -1).sum()
    pct_outliers = (n_outliers / len(labels)) * 100

    # Silhouette score
    if len(set(labels)) > 1:
        sil_score = silhouette_score(pca_features, labels)
    else:
        sil_score = 0.0

    # Metrics Display
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clusters Found", n_clusters)
    col2.metric("Outliers", f"{n_outliers} ({pct_outliers:.1f}%)")
    col3.metric("Total Songs", len(labels))
    col4.metric("Silhouette Score", f"{sil_score:.3f}")

    st.sidebar.header("📊 UMAP Visualization")
    st.sidebar.markdown(
        "*These parameters only affect the plot layout, NOT clustering!*"
    )

    n_neighbors_viz = st.sidebar.slider(
        "n_neighbors (Viz)",
        5,
        100,
        20,
        help="Controls balance between local and global structure for the plot. Lower = more local detail, higher = more global overview.",
    )
    min_dist_viz = st.sidebar.slider(
        "min_dist (Viz)",
        0.0,
        1.0,
        0.2,
        step=0.01,
        help="How tightly points are packed in the visualization. Lower = tighter packing, clearer separation between visual clusters.",
    )

    # Run UMAP for visualization ONLY
    with st.spinner("Running UMAP for visualization..."):
        reducer_viz = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors_viz,
            min_dist=min_dist_viz,
            metric="cosine",
            random_state=42,
        )
        umap_coords = reducer_viz.fit_transform(pca_features)

    # Create DataFrame for Plotting
    plot_data = []
    for i, idx in enumerate(valid_indices):
        track = audio_features[idx]
        lyric_track = lyric_features[idx] if idx < len(lyric_features) else {}

        row_data = {
            "x": umap_coords[i, 0],
            "y": umap_coords[i, 1],
            "z": umap_coords[i, 2],
            "label": labels[i],
            "track_id": track["track_id"],
            "track_name": track["track_name"],
            "artist": track["artist"],
            "genre": track["top_3_genres"][0][0]
            if track["top_3_genres"]
            else "unknown",
            "mood_happy": track.get("mood_happy", 0.5),
            "mood_sad": track.get("mood_sad", 0.5),
            "mood_aggressive": track.get("mood_aggressive", 0.5),
            "mood_relaxed": track.get("mood_relaxed", 0.5),
            "mood_party": track.get("mood_party", 0.5),
            "valence": track.get("valence", 0.5),
            "arousal": track.get("arousal", 0.5),
            "engagement_score": track.get("engagement_score", 0.0),
            "approachability_score": track.get("approachability_score", 0.0),
            "danceability": track.get("danceability", 0.5),
            "instrumentalness": track.get("instrumentalness", 0.5),
            "bpm": track["bpm"],
            "key": track["key"],
            "genre_ladder": track.get("genre_ladder", 0.5),
            # Lyric Features (Tier 1: Parallel emotional dimensions)
            "lyric_valence": lyric_track.get("lyric_valence", 0.5),
            "lyric_arousal": lyric_track.get("lyric_arousal", 0.5),
            "lyric_mood_happy": lyric_track.get("lyric_mood_happy", 0),
            "lyric_mood_sad": lyric_track.get("lyric_mood_sad", 0),
            "lyric_mood_aggressive": lyric_track.get("lyric_mood_aggressive", 0),
            "lyric_mood_relaxed": lyric_track.get("lyric_mood_relaxed", 0),
            # Lyric Features (Tier 3: Lyric-unique)
            "lyric_explicit": lyric_track.get("lyric_explicit", 0),
            "lyric_narrative": lyric_track.get("lyric_narrative", 0),
            "lyric_theme": lyric_track.get("lyric_theme", "other"),
            "lyric_language": lyric_track.get("lyric_language", "unknown"),
            "lyric_vocabulary_richness": lyric_track.get(
                "lyric_vocabulary_richness", 0
            ),
            "lyric_repetition": lyric_track.get("lyric_repetition", 0),
        }

        plot_data.append(row_data)

    df = pd.DataFrame(plot_data)

    # --- EXPORT SECTION ---
    if HAS_EXPORT_TOOLS:
        st.sidebar.markdown("---")
        st.sidebar.header("📤 Export to Spotify")

        playlist_prefix = st.sidebar.text_input(
            "Playlist Prefix", value="Tuner Export", help="Prefix for created playlists"
        )
        is_private = st.sidebar.checkbox("Private Playlists", value=False)

        if st.sidebar.button("Create Spotify Playlists"):
            with st.spinner("Connecting to Spotify..."):
                try:
                    sp = create_spotify_client()
                    user_id = sp.current_user()["id"]
                    st.sidebar.success(f"Connected as {user_id}")

                    # Create playlists for each cluster
                    unique_labels = sorted(df["label"].unique())
                    progress_bar = st.sidebar.progress(0)
                    status_text = st.sidebar.empty()

                    for i, label in enumerate(unique_labels):
                        if label == -1:  # Skip outliers
                            continue

                        # Get tracks for this cluster
                        cluster_tracks = df[df["label"] == label]
                        track_uris = [
                            f"spotify:track:{tid}" for tid in cluster_tracks["track_id"]
                        ]

                        # Generate description
                        # 1. Top Genre
                        genre_counts = Counter(cluster_tracks["genre"])
                        top_genre = genre_counts.most_common(1)[0][0]

                        # 2. Dominant Mood (averaging mood scores)
                        mood_cols = [
                            "mood_happy",
                            "mood_sad",
                            "mood_aggressive",
                            "mood_relaxed",
                            "mood_party",
                        ]
                        avg_moods = cluster_tracks[mood_cols].mean()
                        dominant_mood_col = avg_moods.idxmax()
                        dominant_mood = dominant_mood_col.replace(
                            "mood_", ""
                        ).capitalize()

                        playlist_name = (
                            f"{playlist_prefix} - Cluster {label} ({top_genre})"
                        )
                        description = f"Auto-generated: {len(cluster_tracks)} tracks. Genre: {top_genre}, Mood: {dominant_mood}. Mode: {mode}"

                        status_text.write(f"Creating: {playlist_name}...")

                        # Create and populate
                        playlist = create_playlist(
                            sp,
                            user_id,
                            playlist_name,
                            description,
                            public=not is_private,
                        )
                        add_tracks_to_playlist(sp, playlist["id"], track_uris)

                        progress_bar.progress((i + 1) / len(unique_labels))

                    status_text.write("✅ Export Complete!")
                    st.sidebar.balloons()

                except Exception as e:
                    st.sidebar.error(f"Export failed: {str(e)}")
                    st.error(f"Full error: {str(e)}")

    # Visualization
    fig = go.Figure()

    # Add clusters
    unique_labels = sorted(df["label"].unique())
    for label in unique_labels:
        cluster_points = df[df["label"] == label]
        if cluster_points.empty:
            continue

        if label == -1:
            name = f"Outliers ({len(cluster_points)})"
            color_val = "lightgrey"
            size = 3
            opacity = 0.3
        else:
            name = f"Cluster {label} ({len(cluster_points)})"
            color_val = label
            size = 4
            opacity = 0.8

        # Build hover text
        def build_hover_text(r):
            text = (
                f"<b>{r['track_name']}</b><br>"
                f"Artist: {r['artist']}<br>"
                f"Cluster: {r['label']}<br>"
                f"Genre: {r['genre']}<br>"
                f"BPM: {r['bpm']:.0f} | Key: {r['key']}<br>"
                f"Danceability: {r['danceability']:.2f}<br>"
                f"Instrumentalness: {r['instrumentalness']:.2f}<br>"
                f"Valence: {r['valence']:.2f} | Arousal: {r['arousal']:.2f}<br>"
                f"Engagement: {r['engagement_score']:.2f} | Approachability: {r['approachability_score']:.2f}<br>"
                f"Moods:<br>"
                f"- Happy: {r['mood_happy']:.2f}<br>"
                f"- Sad: {r['mood_sad']:.2f}<br>"
                f"- Aggressive: {r['mood_aggressive']:.2f}<br>"
                f"- Relaxed: {r['mood_relaxed']:.2f}<br>"
                f"- Party: {r['mood_party']:.2f}<br>"
                f"Genre Ladder: {r['genre_ladder']:.2f} ({'Acoustic' if r['genre_ladder'] < 0.4 else 'Electronic' if r['genre_ladder'] > 0.6 else 'Hybrid'})<br>"
            )
            return text

        fig.add_trace(
            go.Scatter3d(
                x=cluster_points["x"],
                y=cluster_points["y"],
                z=cluster_points["z"],
                mode="markers",
                name=name,
                marker=dict(
                    size=size,
                    color=color_val,
                    colorscale="Viridis",
                    opacity=opacity,
                ),
                text=cluster_points.apply(build_hover_text, axis=1),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        height=800,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        title=f"{clustering_algorithm} Clustering ({mode} mode) - UMAP Visualization",
    )

    st.caption("💡 Click on a point to play the song in Spotify!")

    # 1. State Management
    if "last_chart_selection" not in st.session_state:
        st.session_state.last_chart_selection = None
    if "last_df_selection" not in st.session_state:
        st.session_state.last_df_selection = None
    if "current_track" not in st.session_state:
        st.session_state.current_track = None

    # 2. Render Chart
    chart_selection = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="main_chart",
    )

    # 3. Cluster Inspector (List View)
    st.markdown("---")
    st.subheader("📋 Cluster Inspector")

    col_filter, col_spacer = st.columns([1, 2])
    with col_filter:
        unique_labels_list = sorted(df["label"].unique())
        selected_cluster_view = st.selectbox(
            "Filter List by Cluster",
            ["All"]
            + [
                f"Cluster {lbl}" if lbl != -1 else "Outliers"
                for lbl in unique_labels_list
            ],
        )

    if selected_cluster_view != "All":
        if selected_cluster_view == "Outliers":
            view_label = -1
        else:
            view_label = int(selected_cluster_view.split(" ")[1])
        view_df = df[df["label"] == view_label]
    else:
        view_df = df

    # Prepare DataFrame for display
    cols_to_show = [
        "track_name",
        "artist",
        "label",
        "genre",
        "bpm",
        "key",
        "mood_happy",
        "mood_sad",
        "mood_party",
        "danceability",
    ]
    # Ensure columns exist
    display_df = view_df[[c for c in cols_to_show if c in view_df.columns]].copy()

    st.caption("👇 Click on a row to play the song!")
    df_selection = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="cluster_list",
        column_config={
            "label": "Cluster",
            "track_name": "Track",
            "artist": "Artist",
            "genre": "Genre",
            "bpm": st.column_config.NumberColumn("BPM", format="%d"),
            "mood_happy": st.column_config.ProgressColumn(
                "Happy", min_value=0, max_value=1
            ),
            "mood_sad": st.column_config.ProgressColumn(
                "Sad", min_value=0, max_value=1
            ),
            "mood_party": st.column_config.ProgressColumn(
                "Party", min_value=0, max_value=1
            ),
            "danceability": st.column_config.ProgressColumn(
                "Danceability", min_value=0, max_value=1
            ),
        },
    )

    # 4. Logic to determine playing track
    new_track_found = False

    # Check Chart Selection Change
    if chart_selection != st.session_state.last_chart_selection:
        st.session_state.last_chart_selection = chart_selection
        if chart_selection and chart_selection["selection"]["points"]:
            point = chart_selection["selection"]["points"][0]
            curve_idx = point["curve_number"]
            point_idx = point["point_index"]

            # Map back to data
            if curve_idx < len(unique_labels):
                label = unique_labels[curve_idx]
                cluster_subset = df[df["label"] == label]
                if point_idx < len(cluster_subset):
                    st.session_state.current_track = cluster_subset.iloc[point_idx]
                    new_track_found = True

    # Check DataFrame Selection Change (Overrides chart if happened later/simultaneously in this rerun logic)
    if df_selection != st.session_state.last_df_selection:
        st.session_state.last_df_selection = df_selection
        if df_selection and df_selection["selection"]["rows"]:
            row_idx = df_selection["selection"]["rows"][0]
            if row_idx < len(display_df):
                # map back to original df via index if possible, but display_df is a copy.
                # simpler: display_df has the data we need directly, but we need 'track_id'.
                # We didn't include 'track_id' in cols_to_show.
                # Let's get it from the index or include it hidden?
                # display_df comes from view_df. iloc on view_df should work if sorted same way.
                # Yes, we just sliced columns.
                st.session_state.current_track = view_df.iloc[row_idx]
                new_track_found = True

    # 5. Render Player
    if st.session_state.current_track is not None:
        track = st.session_state.current_track
        try:
            st.sidebar.markdown("---")
            st.sidebar.header("🎧 Now Playing")
            st.sidebar.markdown(f"**{track['track_name']}**")
            st.sidebar.caption(f"{track['artist']}")

            spotify_embed = f"""
            <iframe style="border-radius:12px"
            src="https://open.spotify.com/embed/track/{track["track_id"]}?utm_source=generator"
            width="100%" height="152" frameBorder="0"
            allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
            loading="lazy"></iframe>
            """
            st.sidebar.markdown(spotify_embed, unsafe_allow_html=True)
        except Exception as e:
            st.sidebar.error(f"Error loading track: {e}")


if __name__ == "__main__":
    main()
