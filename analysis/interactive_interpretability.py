"""
Interactive Cluster Interpretability Dashboard

Streamlit app for exploring cluster interpretability with:
- Dynamic Clustering: Tune parameters and re-cluster on the fly
- EDA Explorer: Overall statistics and extremes
- Feature Importance: What makes clusters unique
- Cluster Comparison: Statistical comparison of clusters
- Lyric Themes: Keyword extraction, sentiment, complexity
- Overview: Global similarity and summary
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import umap

# scikit-learn imports for dynamic clustering
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, Birch, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our interpretability modules
from analysis.interpretability.feature_importance import (
    compute_feature_importance,
    get_top_features,
    get_feature_interpretation,
)
from analysis.interpretability.cluster_comparison import (
    compare_two_clusters,
    compute_cluster_similarity_matrix,
    find_most_different_pairs,
)
from analysis.interpretability.lyric_themes import (
    load_lyrics_for_cluster,
    extract_tfidf_keywords,
    analyze_sentiment,
    compute_lyric_complexity,
    extract_common_phrases,
    compare_cluster_keywords,
)

# Page configuration
st.set_page_config(
    page_title="Cluster Interpretability Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1DB954;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1DB954;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_raw_data():
    """Load raw feature data from cache for dynamic clustering."""
    # Robust path resolution
    cache_dirs = [Path("cache"), Path("../cache")]
    cache_dir = next((d for d in cache_dirs if d.exists()), None)

    if not cache_dir:
        st.error("Cache directory not found! Expected 'cache/' or '../cache/'")
        return [], [], []

    # Load standard features
    try:
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

        common_ids = set(audio_by_id.keys()) & set(lyric_by_id.keys())
        sorted_ids = sorted(common_ids)

        aligned_audio = [audio_by_id[tid] for tid in sorted_ids]
        aligned_lyrics = [lyric_by_id[tid] for tid in sorted_ids]
        aligned_mert = [mert_by_id.get(tid) for tid in sorted_ids]

        return aligned_audio, aligned_lyrics, aligned_mert
    except Exception as e:
        st.error(f"Error loading raw data: {e}")
        return [], [], []


def prepare_features_dynamic(
    audio_features,
    lyric_features,
    mode,
    n_pca_components,
    skip_pca=False,
    interpretable_mode=False,
):
    """Prepare features for clustering (logic ported from interactive_tuner.py).

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

    audio_emb = np.vstack([f["embedding"] for f in audio_features])

    # For interpretable mode with combined, we don't use lyric embeddings
    # The audio embedding already contains lyric interpretable features
    if not interpretable_mode:
        lyric_emb = np.vstack([f["embedding"] for f in lyric_features])

    valid_indices = list(range(len(audio_features)))

    # If skipping PCA, just standardize
    if skip_pca:
        if mode == "audio":
            features_reduced = StandardScaler().fit_transform(audio_emb)
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
        return features_reduced, valid_indices

    # PCA
    if mode == "audio":
        audio_norm = StandardScaler().fit_transform(audio_emb)
        n_components = min(audio_norm.shape[0], audio_norm.shape[1], n_pca_components)
        pca = PCA(n_components=n_components, random_state=42)
        features_reduced = pca.fit_transform(audio_norm)

    elif mode == "lyrics":
        has_lyrics = np.array([f["has_lyrics"] for f in lyric_features])
        valid_indices = np.where(has_lyrics)[0].tolist()
        lyric_emb = np.vstack([f["embedding"] for f in lyric_features])
        lyric_norm = StandardScaler().fit_transform(lyric_emb[has_lyrics])
        n_components = min(lyric_norm.shape[0], lyric_norm.shape[1], n_pca_components)
        pca = PCA(n_components=n_components, random_state=42)
        features_reduced = pca.fit_transform(lyric_norm)

    else:  # combined
        if interpretable_mode:
            # Interpretable mode: audio embedding already has lyric features
            audio_norm = StandardScaler().fit_transform(audio_emb)
            n_components = min(
                audio_norm.shape[0], audio_norm.shape[1], n_pca_components
            )
            pca = PCA(n_components=n_components, random_state=42)
            features_reduced = pca.fit_transform(audio_norm)
        else:
            audio_norm = StandardScaler().fit_transform(audio_emb)
            lyric_norm = StandardScaler().fit_transform(lyric_emb)

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

            features_reduced = np.hstack([audio_reduced, lyric_reduced])

    return features_reduced, valid_indices


def create_dataframe_from_clustering(
    audio_features, valid_indices, labels, umap_coords=None, lyric_features=None
):
    """Create a DataFrame compatible with the dashboard from dynamic clustering results."""
    data = []

    # Map genre labels
    try:
        labels_path = Path(__file__).parent / "data" / "genre_discogs400_labels.json"
        if labels_path.exists():
            with open(labels_path, "r") as f:
                genre_labels = json.load(f)
        else:
            genre_labels = []
    except:
        genre_labels = []

    def get_genre_label(g_idx):
        if not genre_labels:
            return str(g_idx)
        try:
            s = str(g_idx)
            if s.startswith("genre_"):
                s = s.replace("genre_", "")
            idx = int(float(s))
            if 0 <= idx < len(genre_labels):
                return genre_labels[idx]
            return str(g_idx)
        except:
            return str(g_idx)

    # Create lyric lookup by track_id if lyric_features provided
    lyric_by_id = {}
    if lyric_features:
        lyric_by_id = {f["track_id"]: f for f in lyric_features}

    for i, idx in enumerate(valid_indices):
        track = audio_features[idx]
        lyric_track = lyric_by_id.get(track["track_id"], {})

        # Determine top genre
        genre = "Unknown"
        if track.get("top_3_genres"):
            genre = get_genre_label(track["top_3_genres"][0][0])

        row = {
            "track_id": track["track_id"],
            "track_name": track["track_name"],
            "artist": track["artist"],
            "filename": track.get("filename", ""),
            "cluster": labels[i],
            "top_genre": genre,
            # Core Features
            "bpm": track.get("bpm", 0),
            "danceability": track.get("danceability", 0),
            "instrumentalness": track.get("instrumentalness", 0),
            "valence": track.get("valence", 0),
            "arousal": track.get("arousal", 0),
            # Moods
            "mood_happy": track.get("mood_happy", 0),
            "mood_sad": track.get("mood_sad", 0),
            "mood_aggressive": track.get("mood_aggressive", 0),
            "mood_relaxed": track.get("mood_relaxed", 0),
            "mood_party": track.get("mood_party", 0),
            "mood_acoustic": track.get("mood_acoustic", 0),
            "mood_electronic": track.get("mood_electronic", 0),
            # Additional
            "engagement_score": track.get("engagement_score", 0),
            "approachability_score": track.get("approachability_score", 0),
            "voice_gender_male": track.get("voice_gender_male", 0),
            "voice_gender_female": track.get("voice_gender_female", 0),
            "genre_ladder": track.get("genre_ladder", 0.5),
            # Lyric Features (Tier 1: Parallel emotional dimensions)
            # Try track first (if merged), then lyric_track
            "lyric_valence": track.get("lyric_valence")
            if "lyric_valence" in track
            else lyric_track.get("lyric_valence", 0.5),
            "lyric_arousal": track.get("lyric_arousal")
            if "lyric_arousal" in track
            else lyric_track.get("lyric_arousal", 0.5),
            "lyric_mood_happy": track.get("lyric_mood_happy")
            if "lyric_mood_happy" in track
            else lyric_track.get("lyric_mood_happy", 0),
            "lyric_mood_sad": track.get("lyric_mood_sad")
            if "lyric_mood_sad" in track
            else lyric_track.get("lyric_mood_sad", 0),
            "lyric_mood_aggressive": track.get("lyric_mood_aggressive")
            if "lyric_mood_aggressive" in track
            else lyric_track.get("lyric_mood_aggressive", 0),
            "lyric_mood_relaxed": track.get("lyric_mood_relaxed")
            if "lyric_mood_relaxed" in track
            else lyric_track.get("lyric_mood_relaxed", 0),
            # Lyric Features (Tier 3: Lyric-unique)
            "lyric_explicit": track.get("lyric_explicit")
            if "lyric_explicit" in track
            else lyric_track.get("lyric_explicit", 0),
            "lyric_narrative": track.get("lyric_narrative")
            if "lyric_narrative" in track
            else lyric_track.get("lyric_narrative", 0),
            "lyric_theme": track.get("lyric_theme")
            if "lyric_theme" in track
            else lyric_track.get("lyric_theme", "other"),
            "lyric_language": track.get("lyric_language")
            if "lyric_language" in track
            else lyric_track.get("lyric_language", "unknown"),
            "lyric_vocabulary_richness": track.get("lyric_vocabulary_richness")
            if "lyric_vocabulary_richness" in track
            else lyric_track.get("lyric_vocabulary_richness", 0),
            "lyric_repetition": track.get("lyric_repetition")
            if "lyric_repetition" in track
            else lyric_track.get("lyric_repetition", 0),
        }

        # Add UMAP coords if available
        if umap_coords is not None:
            row["umap_x"] = umap_coords[i, 0]
            row["umap_y"] = umap_coords[i, 1]
            row["umap_z"] = umap_coords[i, 2]

        data.append(row)

    return pd.DataFrame(data)


def main():
    # Header
    st.markdown(
        '<div class="main-header">🎵 Music Cluster Interpretability Dashboard</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        data_source = st.radio(
            "Data Source", ["Static File (Pre-computed)", "Dynamic Tuning (Live)"]
        )

        df = None
        mode = "combined"  # default

        if data_source == "Static File (Pre-computed)":
            # Automatically find available data files
            output_dir = Path("analysis/outputs")
            available_files = list(output_dir.glob("*.pkl"))
            file_options = [str(f) for f in available_files]

            default_file = "analysis/outputs/analysis_data.pkl"
            if default_file not in file_options and file_options:
                default_index = 0
            elif default_file in file_options:
                default_index = file_options.index(default_file)
            else:
                default_index = 0

            data_file = st.selectbox(
                "Select Analysis Data",
                options=file_options,
                index=default_index,
                help="Choose the analysis results file to explore",
            )

            # Helper to load static file
            @st.cache_data
            def load_analysis_data_static(fp):
                with open(fp, "rb") as f:
                    return pickle.load(f)

            if data_file:
                try:
                    all_data = load_analysis_data_static(data_file)
                    st.success("✓ Data loaded successfully")

                    if "metadata" in all_data:
                        meta = all_data["metadata"]
                        st.info(f"backend: {meta.get('audio_backend', 'unknown')}")

                    mode = st.selectbox(
                        "Clustering mode", ["combined", "audio", "lyrics"]
                    )

                    # Extract dataframe
                    df = all_data[mode]["dataframe"].copy()

                    # Map genre indices if needed (logic similar to get_dataframe)
                    # ... (omitted for brevity, relying on user to use correct file or implementation above)
                    # Using simplified mapping here for safety if needed

                except Exception as e:
                    st.error(f"Error loading data: {e}")

        else:  # Dynamic Tuning
            st.info("⚡ Live Tuning Mode")

            with st.spinner("Loading raw features..."):
                audio_features, lyric_features, mert_features = load_raw_data()

            if not audio_features:
                st.stop()

            # --- Dynamic Settings ---

            # Backend Selection
            has_mert = any(x is not None for x in mert_features)
            backend_options = ["Essentia (Default)"]
            if has_mert:
                backend_options.append("MERT (Transformer)")
            backend_options.append("Interpretable Features (Audio)")

            backend = st.selectbox(
                "Embedding Backend", backend_options, index=1 if has_mert else 0
            )

            # Apply backend
            if "MERT" in backend:
                for i, mert_item in enumerate(mert_features):
                    if mert_item is not None and i < len(audio_features):
                        audio_features[i]["embedding"] = mert_item["embedding"]
                st.success(f"Using MERT for {len(audio_features)} tracks")

            elif "Interpretable" in backend:
                st.info("✨ Using interpretable features: Audio + Lyric (combined)")
                st.caption(
                    "Note: Lyric embeddings are NOT used. Only interpretable lyric features (valence, moods, themes, etc.)"
                )

                # Create lyric lookup by track_id
                lyric_by_id = {f["track_id"]: f for f in lyric_features}

                # Filter out vocal songs without lyrics for combined mode
                # Vocal = instrumentalness < 0.5
                # Keep instrumental songs even without lyrics
                filtered_out_tracks = []
                keep_indices = []
                for i, (audio, lyric) in enumerate(zip(audio_features, lyric_features)):
                    is_vocal = audio.get("instrumentalness", 0.5) < 0.5
                    has_lyrics = lyric.get("has_lyrics", False)
                    if is_vocal and not has_lyrics:
                        filtered_out_tracks.append(
                            {
                                "track_name": audio.get("track_name", "Unknown"),
                                "artist": audio.get("artist", "Unknown"),
                                "instrumentalness": audio.get("instrumentalness", 0),
                            }
                        )
                    else:
                        keep_indices.append(i)

                if filtered_out_tracks:
                    with st.expander(
                        f"⚠️ Filtered out {len(filtered_out_tracks)} vocal songs without lyrics",
                        expanded=False,
                    ):
                        st.write(
                            "These are non-instrumental songs that don't have lyrics data:"
                        )
                        filtered_df = pd.DataFrame(filtered_out_tracks)
                        st.dataframe(
                            filtered_df, use_container_width=True, hide_index=True
                        )
                        st.caption(
                            "These songs are excluded because they're vocal (instrumentalness < 0.5) but have no lyrics."
                        )

                # Apply the filter
                audio_features = [audio_features[i] for i in keep_indices]
                lyric_features = [lyric_features[i] for i in keep_indices]
                # Update lyric lookup after filtering
                lyric_by_id = {f["track_id"]: f for f in lyric_features}

                st.success(
                    f"Using {len(audio_features)} tracks (filtered {len(filtered_out_tracks)} vocal songs without lyrics)"
                )

                # Dynamic global min/max for normalization
                bpms = [float(t.get("bpm", 0) or 0) for t in audio_features]
                valences = [float(t.get("valence", 0) or 0) for t in audio_features]
                arousals = [float(t.get("arousal", 0) or 0) for t in audio_features]

                def get_range(values, default_min, default_max):
                    valid = [v for v in values if v > 0]
                    if not valid:
                        return default_min, default_max
                    return min(valid), max(valid)

                min_bpm, max_bpm = get_range(bpms, 50, 200)
                min_val, max_val = get_range(valences, 1, 9)
                min_ar, max_ar = get_range(arousals, 1, 9)

                # Theme & Language: 1 dimension each (semantic ordering)
                # "none" is isolated at 0.0 with a gap to distinguish instrumentals/no-lyrics
                # Theme ordered by emotional energy/positivity (high=positive/energetic, low=negative/introspective)
                THEME_SCALE = {
                    "party": 1.0,  # highest energy
                    "flex": 0.92,  # confident, boastful
                    "love": 0.85,  # positive emotion
                    "social": 0.75,  # community focused
                    "spirituality": 0.68,  # contemplative but uplifting
                    "introspection": 0.6,  # neutral, internal
                    "street": 0.5,  # raw, realistic
                    "heartbreak": 0.4,  # sad
                    "struggle": 0.3,  # difficult
                    "other": 0.2,  # has lyrics, unknown theme
                    "none": 0.0,  # no lyrics/theme (gap of 0.2)
                }
                # Language: ordinal encoding with gap for "none"
                LANGUAGE_SCALE = {
                    "english": 1.0,
                    "spanish": 0.85,
                    "french": 0.7,
                    "arabic": 0.6,
                    "korean": 0.5,
                    "japanese": 0.4,
                    "unknown": 0.25,  # has lyrics, unknown language
                    "none": 0.0,  # no lyrics (gap of 0.25)
                }

                for track in audio_features:
                    lyric = lyric_by_id.get(track["track_id"], {})

                    def get_float(d, k, default=0.0):
                        v = d.get(k)
                        try:
                            return float(v) if v is not None else default
                        except Exception:
                            return default

                    # Normalize BPM/Valence/Arousal
                    raw_bpm = get_float(track, "bpm", 120)
                    norm_bpm = max(
                        0.0,
                        min(
                            1.0,
                            (raw_bpm - min_bpm) / (max_bpm - min_bpm)
                            if max_bpm > min_bpm
                            else 0.5,
                        ),
                    )
                    raw_val = get_float(track, "valence", 4.5)
                    norm_val = max(
                        0.0,
                        min(
                            1.0,
                            (raw_val - min_val) / (max_val - min_val)
                            if max_val > min_val
                            else 0.5,
                        ),
                    )
                    raw_ar = get_float(track, "arousal", 4.5)
                    norm_ar = max(
                        0.0,
                        min(
                            1.0,
                            (raw_ar - min_ar) / (max_ar - min_ar)
                            if max_ar > min_ar
                            else 0.5,
                        ),
                    )

                    # Audio scalars (14 dims)
                    audio_scalars = [
                        norm_bpm,
                        get_float(track, "danceability", 0.5),
                        get_float(track, "instrumentalness", 0.0),
                        norm_val,
                        norm_ar,
                        get_float(track, "engagement_score", 0.5),
                        get_float(track, "approachability_score", 0.5),
                        get_float(track, "mood_happy", 0.0),
                        get_float(track, "mood_sad", 0.0),
                        get_float(track, "mood_aggressive", 0.0),
                        get_float(track, "mood_relaxed", 0.0),
                        get_float(track, "mood_party", 0.0),
                        get_float(track, "voice_gender_male", 0.5),
                        get_float(track, "genre_ladder", 0.5),
                    ]

                    # Key features (3 dims)
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
                            KEY_WEIGHT = 0.5
                            key_vec = [
                                (0.5 * np.sin(2 * np.pi * p / 12) + 0.5) * KEY_WEIGHT,
                                (0.5 * np.cos(2 * np.pi * p / 12) + 0.5) * KEY_WEIGHT,
                                scale_val * KEY_WEIGHT,
                            ]

                    # Lyric scalars (10 dims)
                    lyric_scalars = [
                        get_float(lyric, "lyric_valence", 0.5),
                        get_float(lyric, "lyric_arousal", 0.5),
                        get_float(lyric, "lyric_mood_happy", 0.0),
                        get_float(lyric, "lyric_mood_sad", 0.0),
                        get_float(lyric, "lyric_mood_aggressive", 0.0),
                        get_float(lyric, "lyric_mood_relaxed", 0.0),
                        get_float(lyric, "lyric_explicit", 0.0),
                        get_float(lyric, "lyric_narrative", 0.0),
                        get_float(lyric, "lyric_vocabulary_richness", 0.0),
                        get_float(lyric, "lyric_repetition", 0.0),
                    ]

                    # Theme (1 dim) - semantic scale
                    theme = lyric.get("lyric_theme", "other")
                    theme = theme.lower().strip() if isinstance(theme, str) else "other"
                    theme_val = THEME_SCALE.get(theme, 0.2)  # default to "other"

                    # Language (1 dim) - ordinal scale
                    lang = lyric.get("lyric_language", "unknown")
                    lang = lang.lower().strip() if isinstance(lang, str) else "unknown"
                    lang_val = LANGUAGE_SCALE.get(lang, 0.25)  # default to "unknown"

                    track["embedding"] = np.array(
                        audio_scalars + key_vec + lyric_scalars + [theme_val, lang_val],
                        dtype=np.float32,
                    )

                # Debug expander
                with st.expander("🔍 Debug: Sample Vector", expanded=False):
                    sample_track = audio_features[0]
                    sample_lyric = lyric_by_id.get(sample_track["track_id"], {})
                    emb = sample_track["embedding"]
                    st.write(
                        f"**{sample_track['track_name']}** by {sample_track['artist']}"
                    )
                    st.write(
                        f"Vector length: {len(emb)} (Audio:14 + Key:3 + Lyric:10 + Theme:1 + Lang:1 = 29)"
                    )
                    theme_raw = sample_lyric.get("lyric_theme", "N/A")
                    lang_raw = sample_lyric.get("lyric_language", "N/A")
                    st.write(
                        f"Theme: {theme_raw} → {THEME_SCALE.get(str(theme_raw).lower().strip(), 0.2):.2f}, "
                        f"Language: {lang_raw} → {LANGUAGE_SCALE.get(str(lang_raw).lower().strip(), 0.25):.2f}"
                    )

                # Weight controls
                st.markdown("---")
                st.subheader("🎛️ Audio Feature Weights")
                core_weight = st.slider("🎵 Core Audio", 0.0, 2.0, 1.0, 0.1)
                mood_weight = st.slider("😊 Audio Moods", 0.0, 2.0, 1.0, 0.1)
                genre_weight = st.slider("🎸 Genre Ladder", 0.0, 2.0, 1.0, 0.1)
                key_weight = st.slider("🎹 Key", 0.0, 2.0, 1.0, 0.1)

                st.markdown("---")
                st.subheader("📝 Lyric Feature Weights")
                lyric_emotion_weight = st.slider(
                    "💭 Lyric Emotions", 0.0, 2.0, 1.0, 0.1
                )
                lyric_content_weight = st.slider("🔞 Content", 0.0, 2.0, 1.0, 0.1)
                theme_weight = st.slider("🏷️ Theme", 0.0, 2.0, 1.0, 0.1)
                language_weight = st.slider("🌍 Language", 0.0, 2.0, 1.0, 0.1)

                # Apply weights
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
                    emb[0:7] = emb[0:7] * core_weight  # Core audio
                    emb[7:12] = emb[7:12] * mood_weight  # Audio moods
                    emb[12:14] = emb[12:14] * core_weight  # Voice gender + genre ladder
                    emb[14:17] = emb[14:17] * key_weight  # Key
                    emb[17:23] = emb[17:23] * lyric_emotion_weight  # Lyric emotions
                    emb[23:27] = emb[23:27] * lyric_content_weight  # Lyric content
                    emb[27] = emb[27] * theme_weight  # Theme (1 dim)
                    emb[28] = emb[28] * language_weight  # Language (1 dim)
                    track["embedding"] = emb

            mode = st.selectbox("Feature Mode", ["combined", "audio", "lyrics"])

            # Clustering Algo
            algo = st.selectbox("Algorithm", ["K-Means", "HAC", "Spectral", "Birch"])
            n_clusters = st.slider("Clusters", 2, 50, 20)

            # Run Clustering Button
            if st.button("Run Clustering"):
                with st.spinner("Processing..."):
                    # 1. Prepare Features
                    is_interpretable = "Interpretable" in backend
                    pca_feats, valid_idx = prepare_features_dynamic(
                        audio_features,
                        lyric_features,
                        mode,
                        n_pca_components=100,
                        skip_pca=is_interpretable,
                        interpretable_mode=is_interpretable,
                    )

                    # 2. Cluster
                    if algo == "K-Means":
                        model = KMeans(
                            n_clusters=n_clusters, n_init=10, random_state=42
                        )
                        labels = model.fit_predict(pca_feats)
                    elif algo == "HAC":
                        model = AgglomerativeClustering(n_clusters=n_clusters)
                        labels = model.fit_predict(pca_feats)
                    elif algo == "Birch":
                        model = Birch(n_clusters=n_clusters)
                        labels = model.fit_predict(pca_feats)
                    elif algo == "Spectral":
                        model = SpectralClustering(
                            n_clusters=n_clusters, random_state=42
                        )
                        labels = model.fit_predict(pca_feats)

                    # 3. UMAP for Viz
                    reducer = umap.UMAP(n_components=3, random_state=42)
                    coords = reducer.fit_transform(pca_feats)

                    # 4. Create DF
                    df = create_dataframe_from_clustering(
                        audio_features, valid_idx, labels, coords, lyric_features
                    )

                    # Save to session state to persist across reruns
                    st.session_state["dynamic_df"] = df
                    st.success("Clustering Complete!")

            # Retrieve from session state if available
            if "dynamic_df" in st.session_state:
                df = st.session_state["dynamic_df"]

        # --- Shared Processing ---
        if df is not None:
            # Display stats
            st.markdown("---")
            st.metric("Total Songs", len(df))
            st.metric("Clusters", df["cluster"].nunique())
            st.caption(f"Analyzing {mode} mode")

    if df is None:
        st.info("👈 Select a data source or run dynamic clustering to begin.")
        st.stop()

    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 EDA Explorer",
            "🎯 Feature Importance",
            "⚖️ Cluster Comparison",
            "📝 Lyric Themes",
            "🔍 Overview",
        ]
    )

    with tab1:
        render_eda_explorer(df)

    with tab2:
        render_feature_importance(df)

    with tab3:
        render_cluster_comparison(df)

    with tab4:
        render_lyric_themes(df)

    with tab5:
        render_overview(df)


def render_eda_explorer(df: pd.DataFrame):
    """Render EDA Explorer view with comprehensive statistics."""
    st.header("📊 Exploratory Data Analysis")

    # Overall Statistics
    with st.expander("📈 Overall Statistics", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Songs", len(df))

        with col2:
            st.metric("Number of Clusters", df["cluster"].nunique())

        with col3:
            if "has_lyrics" in df.columns:
                # Approximate check if has_lyrics column exists or infer from filename match
                st.metric("Songs", len(df))
            else:
                st.metric("Songs", len(df))

        with col4:
            if "added_at" in df.columns:
                try:
                    min_date = pd.to_datetime(df["added_at"]).min()
                    max_date = pd.to_datetime(df["added_at"]).max()
                    date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
                    st.metric("Date Range", date_range)
                except:
                    st.metric("Date Range", "N/A")
            else:
                st.metric("Date Range", "N/A")

    # Genre Analysis
    with st.expander("🎸 Genre Analysis", expanded=False):
        st.subheader("Top 20 Most Common Genres")

        if "top_genre" in df.columns:
            genre_counts = df["top_genre"].value_counts().head(20)

            fig = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation="h",
                labels={"x": "Number of Songs", "y": "Genre"},
                title="Top 20 Genres in Your Library",
            )
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Genre distribution across clusters
            st.subheader("Genre Distribution Across Clusters")
            # Get top 10 genres
            top_genres = df["top_genre"].value_counts().head(10).index

            genre_cluster_data = []
            for cluster_id in sorted(df["cluster"].unique()):
                cluster_df = df[df["cluster"] == cluster_id]
                for genre in top_genres:
                    count = (cluster_df["top_genre"] == genre).sum()
                    genre_cluster_data.append(
                        {
                            "Cluster": f"Cluster {cluster_id}",
                            "Genre": genre,
                            "Count": count,
                        }
                    )

            genre_cluster_df = pd.DataFrame(genre_cluster_data)

            fig = px.bar(
                genre_cluster_df,
                x="Cluster",
                y="Count",
                color="Genre",
                title="Top 10 Genres by Cluster",
                barmode="stack",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Genre information not available in this dataset")

    # Audio Extremes
    with st.expander("🔊 Audio Extremes", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Most/Least Approachable
            if "approachability_score" in df.columns:
                st.subheader("Most Approachable Songs")
                top_approachable = df.nlargest(10, "approachability_score")[
                    ["track_name", "artist", "approachability_score"]
                ]
                st.dataframe(
                    top_approachable, use_container_width=True, hide_index=True
                )

                st.subheader("Least Approachable Songs (Most Niche)")
                bottom_approachable = df.nsmallest(10, "approachability_score")[
                    ["track_name", "artist", "approachability_score"]
                ]
                st.dataframe(
                    bottom_approachable, use_container_width=True, hide_index=True
                )

        with col2:
            # Most/Least Engaging
            if "engagement_score" in df.columns:
                st.subheader("Most Engaging Songs")
                top_engaging = df.nlargest(10, "engagement_score")[
                    ["track_name", "artist", "engagement_score"]
                ]
                st.dataframe(top_engaging, use_container_width=True, hide_index=True)

                st.subheader("Least Engaging Songs")
                bottom_engaging = df.nsmallest(10, "engagement_score")[
                    ["track_name", "artist", "engagement_score"]
                ]
                st.dataframe(bottom_engaging, use_container_width=True, hide_index=True)

        # BPM Extremes
        if "bpm" in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Fastest Songs (Highest BPM)")
                fastest = df.nlargest(10, "bpm")[["track_name", "artist", "bpm"]]
                st.dataframe(fastest, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Slowest Songs (Lowest BPM)")
                slowest = df.nsmallest(10, "bpm")[["track_name", "artist", "bpm"]]
                st.dataframe(slowest, use_container_width=True, hide_index=True)

        # Danceability Extremes
        if "danceability" in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Most Danceable Songs")
                danceable = df.nlargest(10, "danceability")[
                    ["track_name", "artist", "danceability"]
                ]
                st.dataframe(danceable, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Least Danceable Songs")
                not_danceable = df.nsmallest(10, "danceability")[
                    ["track_name", "artist", "danceability"]
                ]
                st.dataframe(not_danceable, use_container_width=True, hide_index=True)

    # Mood Analysis
    with st.expander("😊 Mood Analysis", expanded=False):
        mood_cols = [
            "mood_happy",
            "mood_sad",
            "mood_aggressive",
            "mood_relaxed",
            "mood_party",
        ]

        if all(col in df.columns for col in mood_cols):
            # 5D Radar plot for library mood profile
            st.subheader("Overall Library Mood Profile")

            avg_moods = {
                "Happy": df["mood_happy"].mean() * 100,
                "Sad": df["mood_sad"].mean() * 100,
                "Aggressive": df["mood_aggressive"].mean() * 100,
                "Relaxed": df["mood_relaxed"].mean() * 100,
                "Party": df["mood_party"].mean() * 100,
            }

            fig = go.Figure(
                data=go.Scatterpolar(
                    r=list(avg_moods.values()),
                    theta=list(avg_moods.keys()),
                    fill="toself",
                    name="Library Average",
                )
            )

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title="Average Mood Distribution (%)",
            )

            st.plotly_chart(fig, use_container_width=True)

            # Valence vs Arousal scatter plot (emotional quadrants)
            if "valence" in df.columns and "arousal" in df.columns:
                st.subheader("Emotional Quadrants (Valence vs Arousal)")

                fig = px.scatter(
                    df,
                    x="valence",
                    y="arousal",
                    color="cluster",
                    hover_data=["track_name", "artist"],
                    labels={
                        "valence": "Valence (Pleasant)",
                        "arousal": "Arousal (Energy)",
                    },
                    title="Songs by Emotional Content",
                    color_continuous_scale="Viridis",
                )

                # Add quadrant lines
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)

                # Add quadrant labels
                fig.add_annotation(
                    x=0.75, y=0.75, text="Happy/Energetic", showarrow=False, opacity=0.5
                )
                fig.add_annotation(
                    x=0.25, y=0.75, text="Angry/Tense", showarrow=False, opacity=0.5
                )
                fig.add_annotation(
                    x=0.25, y=0.25, text="Sad/Depressed", showarrow=False, opacity=0.5
                )
                fig.add_annotation(
                    x=0.75, y=0.25, text="Calm/Peaceful", showarrow=False, opacity=0.5
                )

                st.plotly_chart(fig, use_container_width=True)

            # Mood extremes
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Happiest Songs")
                happiest = df.nlargest(10, "mood_happy")[
                    ["track_name", "artist", "mood_happy"]
                ]
                st.dataframe(happiest, use_container_width=True, hide_index=True)

                st.subheader("Most Aggressive Songs")
                aggressive = df.nlargest(10, "mood_aggressive")[
                    ["track_name", "artist", "mood_aggressive"]
                ]
                st.dataframe(aggressive, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Saddest Songs")
                saddest = df.nlargest(10, "mood_sad")[
                    ["track_name", "artist", "mood_sad"]
                ]
                st.dataframe(saddest, use_container_width=True, hide_index=True)

                st.subheader("Most Relaxed Songs")
                relaxed = df.nlargest(10, "mood_relaxed")[
                    ["track_name", "artist", "mood_relaxed"]
                ]
                st.dataframe(relaxed, use_container_width=True, hide_index=True)

        else:
            st.warning("Mood information not available in this dataset")

    # Vocal Analysis
    with st.expander("🎤 Vocal Analysis", expanded=False):
        if "instrumentalness" in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Most Instrumental Songs")
                instrumental = df.nlargest(10, "instrumentalness")[
                    ["track_name", "artist", "instrumentalness"]
                ]
                st.dataframe(instrumental, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Most Vocal Songs")
                vocal = df.nsmallest(10, "instrumentalness")[
                    ["track_name", "artist", "instrumentalness"]
                ]
                st.dataframe(vocal, use_container_width=True, hide_index=True)

        # Voice gender distribution
        if "voice_gender_male" in df.columns and "voice_gender_female" in df.columns:
            st.subheader("Voice Gender Distribution")

            # Classify songs as predominantly male, female, or mixed
            df_vocal = df.copy()
            df_vocal["dominant_gender"] = "Mixed"
            df_vocal.loc[df_vocal["voice_gender_male"] > 0.6, "dominant_gender"] = (
                "Male"
            )
            df_vocal.loc[df_vocal["voice_gender_female"] > 0.6, "dominant_gender"] = (
                "Female"
            )

            gender_counts = df_vocal["dominant_gender"].value_counts()

            fig = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title="Voice Gender Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Acoustic vs Electronic
        if "mood_acoustic" in df.columns and "mood_electronic" in df.columns:
            st.subheader("Acoustic vs Electronic Distribution")

            # Classify songs
            df_production = df.copy()
            df_production["production_style"] = "Mixed"
            df_production.loc[
                df_production["mood_acoustic"] > 0.6, "production_style"
            ] = "Acoustic"
            df_production.loc[
                df_production["mood_electronic"] > 0.6, "production_style"
            ] = "Electronic"

            production_counts = df_production["production_style"].value_counts()

            fig = px.pie(
                values=production_counts.values,
                names=production_counts.index,
                title="Production Style Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Genre Ladder Analysis
    with st.expander("🎸 Genre Ladder Analysis", expanded=False):
        if "genre_ladder" in df.columns:
            st.subheader("Acoustic ↔ Electronic Distribution (Genre-Based)")
            st.caption(
                "Genre ladder captures stylistic intent (0=acoustic/traditional, 1=electronic/synthetic), complementing mood_acoustic/mood_electronic which analyze actual audio production."
            )

            fig = px.histogram(
                df,
                x="genre_ladder",
                nbins=50,
                title="Genre Ladder Distribution (0=Acoustic, 1=Electronic)",
                labels={
                    "genre_ladder": "Genre Ladder Score",
                    "count": "Number of Songs",
                },
            )
            fig.add_vline(
                x=0.5, line_dash="dash", line_color="gray", annotation_text="Hybrid"
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Most Acoustic Songs (by Genre)**")
                acoustic = df.nsmallest(10, "genre_ladder")[
                    ["track_name", "artist", "top_genre", "genre_ladder"]
                ]
                st.dataframe(acoustic, use_container_width=True, hide_index=True)

            with col2:
                st.write("**Most Electronic Songs (by Genre)**")
                electronic = df.nlargest(10, "genre_ladder")[
                    ["track_name", "artist", "top_genre", "genre_ladder"]
                ]
                st.dataframe(electronic, use_container_width=True, hide_index=True)

            # Compare with mood_acoustic/mood_electronic if available
            if "mood_acoustic" in df.columns and "mood_electronic" in df.columns:
                st.markdown("---")
                st.subheader("Genre Ladder vs Audio Production Analysis")
                st.caption(
                    "Compare genre-based classification (genre_ladder) with audio signal analysis (mood_acoustic/mood_electronic)"
                )

                # Scatter plot
                fig = px.scatter(
                    df,
                    x="genre_ladder",
                    y="mood_electronic",
                    color="cluster",
                    hover_data=["track_name", "artist", "top_genre"],
                    labels={
                        "genre_ladder": "Genre Ladder (0=Acoustic, 1=Electronic)",
                        "mood_electronic": "Audio Electronic Score (from Essentia)",
                    },
                    title="Genre Taxonomy vs Audio Production",
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Correlation
                correlation = df["genre_ladder"].corr(df["mood_electronic"])
                st.metric(
                    "Correlation",
                    f"{correlation:.3f}",
                    help="How well genre taxonomy aligns with audio production analysis",
                )
        else:
            st.warning("Genre ladder information not available in this dataset")

    # 3D Cluster Visualization
    with st.expander("🗺️ Interactive 3D Cluster Map", expanded=False):
        st.subheader("3D UMAP Visualization of Clusters")
        st.write(
            "Explore your music clusters in 3D space. Points are colored by cluster assignment."
        )

        # Check if we have UMAP coordinates already
        if "umap_x" in df.columns and "umap_y" in df.columns and "umap_z" in df.columns:
            # Create 3D scatter plot
            fig = go.Figure()

            unique_clusters = sorted(df["cluster"].unique())
            colors = px.colors.qualitative.Plotly

            for i, cluster_id in enumerate(unique_clusters):
                cluster_df = df[df["cluster"] == cluster_id]

                # Build hover text
                hover_texts = []
                for _, row in cluster_df.iterrows():
                    text = (
                        f"<b>{row['track_name']}</b><br>"
                        f"Artist: {row['artist']}<br>"
                        f"Cluster: {row['cluster']}<br>"
                    )

                    if "top_genre" in row:
                        text += f"Genre: {row['top_genre']}<br>"
                    if "bpm" in row:
                        text += f"BPM: {row['bpm']:.0f}<br>"
                    if "danceability" in row:
                        text += f"Danceability: {row['danceability']:.2f}<br>"
                    if "valence" in row and "arousal" in row:
                        text += f"Valence: {row['valence']:.2f} | Arousal: {row['arousal']:.2f}<br>"
                    if "mood_happy" in row:
                        text += f"Moods: Happy {row['mood_happy']:.2f}, Sad {row.get('mood_sad', 0):.2f}<br>"

                    hover_texts.append(text)

                fig.add_trace(
                    go.Scatter3d(
                        x=cluster_df["umap_x"],
                        y=cluster_df["umap_y"],
                        z=cluster_df["umap_z"],
                        mode="markers",
                        name=f"Cluster {cluster_id} ({len(cluster_df)})",
                        marker=dict(size=4, color=colors[i % len(colors)], opacity=0.8),
                        text=hover_texts,
                        hovertemplate="%{text}<extra></extra>",
                    )
                )

            fig.update_layout(
                height=700,
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                ),
                title="3D Cluster Visualization (UMAP)",
                showlegend=True,
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning(
                "UMAP coordinates not found in data. Cannot display cluster map."
            )

    # Data Preview
    with st.expander("🔍 Data Preview & Export", expanded=False):
        st.subheader("Full Dataset Preview")

        # Create a copy for display and convert problematic columns
        display_df = df.copy()

        # Convert object columns that might have mixed types to strings
        for col in display_df.columns:
            if display_df[col].dtype == "object":
                display_df[col] = display_df[col].astype(str)

        st.dataframe(display_df, use_container_width=True, height=400)

        # Export button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="cluster_data.csv",
            mime="text/csv",
        )


@st.cache_data
def compute_all_cluster_importance(df: pd.DataFrame):
    """Compute feature importance for all clusters (cached)."""
    importance_data = {}
    cluster_ids = sorted(df["cluster"].unique())

    for cluster_id in cluster_ids:
        result = get_top_features(df, cluster_id, n=20)
        importance_data[cluster_id] = result

    return importance_data


def render_feature_importance(df: pd.DataFrame):
    """Render Feature Importance view."""
    st.header("🎯 Feature Importance Analysis")

    st.write(
        "Identify which features make each cluster distinctive using Cohen's d effect sizes."
    )

    # Cluster selection
    cluster_ids = sorted(df["cluster"].unique())
    selected_cluster = st.selectbox(
        "Select Cluster to Analyze",
        options=cluster_ids,
        format_func=lambda x: f"Cluster {x}",
    )

    with st.spinner("Computing feature importance..."):
        # Get feature importance for selected cluster
        cluster_info = get_top_features(df, selected_cluster, n=20)

        # Display cluster info metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Cluster Size", cluster_info["cluster_size"])

        with col2:
            st.metric(
                "Percentage of Library", f"{cluster_info['cluster_percentage']:.1f}%"
            )

        with col3:
            # Get top feature
            if len(cluster_info["top_features"]) > 0:
                top_feature = cluster_info["top_features"].iloc[0]
                st.metric(
                    f"Top Feature: {top_feature['feature']}",
                    f"Effect size: {top_feature['effect_size']:.2f}",
                )

        st.markdown("---")

        # Top 3 distinctive features with interpretations
        st.subheader(
            f"🌟 Top 3 Most Distinctive Features for Cluster {selected_cluster}"
        )

        if len(cluster_info["top_features"]) >= 3:
            for i, row in cluster_info["top_features"].head(3).iterrows():
                feature = row["feature"]
                effect_size = row["effect_size"]
                cluster_mean = row["cluster_mean"]
                global_mean = row["global_mean"]

                interpretation = get_feature_interpretation(effect_size)

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**{i}. {feature}**")
                    st.write(
                        f"Cluster mean: {cluster_mean:.3f} | Global mean: {global_mean:.3f}"
                    )
                    st.write(f"Effect size: **{effect_size:.2f}** ({interpretation})")

                with col2:
                    # Simple visual indicator
                    if abs(effect_size) >= 0.8:
                        st.success("🔥 Large effect")
                    elif abs(effect_size) >= 0.5:
                        st.info("⚡ Medium effect")
                    else:
                        st.warning("✨ Small effect")

        # Full feature importance table
        st.markdown("---")
        st.subheader("📊 Full Feature Importance Ranking")

        # Prepare dataframe for display
        display_df = cluster_info["all_features"].copy()
        display_df["effect_size"] = display_df["effect_size"].round(3)
        display_df["cluster_mean"] = display_df["cluster_mean"].round(3)
        display_df["global_mean"] = display_df["global_mean"].round(3)

        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            column_config={
                "feature": "Feature",
                "effect_size": st.column_config.NumberColumn(
                    "Effect Size (Cohen's d)",
                    help="How many standard deviations this cluster differs from average",
                    format="%.3f",
                ),
                "cluster_mean": "Cluster Mean",
                "global_mean": "Global Mean",
                "importance_rank": "Rank",
            },
            hide_index=True,
        )

        # Download button
        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download Cluster {selected_cluster} Importance Data",
            data=csv,
            file_name=f"cluster_{selected_cluster}_importance.csv",
            mime="text/csv",
        )

    # Feature importance heatmap for all clusters
    st.markdown("---")
    st.subheader("🔥 Feature Importance Heatmap (All Clusters)")

    with st.spinner("Computing importance for all clusters..."):
        all_importance = compute_all_cluster_importance(df)

        # Build heatmap data
        top_n_features = 15  # Top features to show
        feature_names = set()

        # Collect top features from all clusters
        for cluster_id, data in all_importance.items():
            top_features = data["top_features"].head(top_n_features)
            feature_names.update(top_features["feature"].tolist())

        feature_names = sorted(list(feature_names))[:20]  # Limit to 20 features

        # Build matrix
        heatmap_data = []
        for feature in feature_names:
            row = [feature]
            for cluster_id in cluster_ids:
                importance_df = all_importance[cluster_id]["all_features"]
                feature_row = importance_df[importance_df["feature"] == feature]

                if len(feature_row) > 0:
                    effect_size = feature_row.iloc[0]["effect_size"]
                    row.append(effect_size)
                else:
                    row.append(0.0)

            heatmap_data.append(row)

        # Create heatmap
        heatmap_df = pd.DataFrame(
            heatmap_data,
            columns=["Feature"] + [f"Cluster {cid}" for cid in cluster_ids],
        )

        fig = px.imshow(
            heatmap_df.set_index("Feature").T,
            labels=dict(x="Feature", y="Cluster", color="Effect Size"),
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            aspect="auto",
            title="Feature Effect Sizes Across All Clusters",
        )

        fig.update_xaxes(side="bottom")
        fig.update_layout(height=400 + len(cluster_ids) * 50)

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "💡 Red = higher than average, Blue = lower than average, White = near average"
        )

    # Distribution violin plots for top features
    st.markdown("---")
    st.subheader("📊 Feature Distribution Comparison")

    # Let user select a feature to visualize
    all_features = cluster_info["all_features"]["feature"].tolist()
    selected_feature = st.selectbox(
        "Select feature to visualize distribution", options=all_features, index=0
    )

    if selected_feature in df.columns:
        # Create violin plot
        fig = go.Figure()

        for cluster_id in cluster_ids:
            cluster_values = df[df["cluster"] == cluster_id][selected_feature].dropna()

            fig.add_trace(
                go.Violin(
                    y=cluster_values,
                    name=f"Cluster {cluster_id}",
                    box_visible=True,
                    meanline_visible=True,
                )
            )

        fig.update_layout(
            title=f"Distribution of '{selected_feature}' Across Clusters",
            yaxis_title=selected_feature,
            xaxis_title="Cluster",
            showlegend=True,
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add statistical summary
        st.subheader("Statistical Summary")

        summary_data = []
        for cluster_id in cluster_ids:
            cluster_values = df[df["cluster"] == cluster_id][selected_feature].dropna()

            summary_data.append(
                {
                    "Cluster": f"Cluster {cluster_id}",
                    "Mean": cluster_values.mean(),
                    "Median": cluster_values.median(),
                    "Std Dev": cluster_values.std(),
                    "Min": cluster_values.min(),
                    "Max": cluster_values.max(),
                    "Count": len(cluster_values),
                }
            )

        summary_df = pd.DataFrame(summary_data)

        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    else:
        st.warning(f"Feature '{selected_feature}' not found in dataframe")


def render_cluster_comparison(df: pd.DataFrame):
    """Render Cluster Comparison view."""
    st.header("⚖️ Statistical Cluster Comparison")

    st.write("Compare multiple clusters using statistical tests and visualizations.")

    # Cluster selection - allow multiple
    cluster_ids = sorted(df["cluster"].unique())

    # Multi-select for clusters
    selected_clusters = st.multiselect(
        "Select Clusters to Compare (select 2 or more)",
        options=cluster_ids,
        default=cluster_ids[: min(2, len(cluster_ids))],
        format_func=lambda x: f"Cluster {x}",
        help="Select 2 or more clusters to compare. The radar plot will show all selected clusters.",
    )

    if len(selected_clusters) < 2:
        st.warning("⚠️ Please select at least 2 clusters to compare")
        return

    # Show number of clusters being compared
    st.info(
        f"📊 Comparing {len(selected_clusters)} clusters: {', '.join([f'Cluster {c}' for c in selected_clusters])}"
    )

    # Basic cluster information
    st.markdown("---")
    st.subheader("📊 Cluster Overview")

    # Create overview table for all selected clusters
    overview_data = []
    for cluster_id in selected_clusters:
        cluster_df = df[df["cluster"] == cluster_id]
        row = {
            "Cluster": f"Cluster {cluster_id}",
            "Size": len(cluster_df),
            "Percentage": f"{len(cluster_df) / len(df) * 100:.1f}%",
        }

        if "bpm" in df.columns:
            row["Avg BPM"] = f"{cluster_df['bpm'].mean():.1f}"
        if "danceability" in df.columns:
            row["Avg Danceability"] = f"{cluster_df['danceability'].mean():.2f}"
        if "mood_happy" in df.columns:
            row["Avg Happiness"] = f"{cluster_df['mood_happy'].mean():.2f}"
        if "valence" in df.columns:
            row["Avg Valence"] = f"{cluster_df['valence'].mean():.2f}"

        overview_data.append(row)

    overview_df = pd.DataFrame(overview_data)
    st.dataframe(overview_df, use_container_width=True, hide_index=True)

    # Pairwise Statistical Comparisons
    st.markdown("---")
    st.subheader("📈 Pairwise Statistical Comparisons")

    if len(selected_clusters) == 2:
        # For 2 clusters, show detailed comparison
        with st.spinner("Running statistical tests..."):
            comparison_df = compare_two_clusters(
                df, selected_clusters[0], selected_clusters[1]
            )

            if len(comparison_df) > 0:
                # Show only significant differences by default
                show_all = st.checkbox(
                    "Show all features (including non-significant)", value=False
                )

                if not show_all:
                    display_df = comparison_df[comparison_df["significant"]].copy()
                    st.write(
                        f"**Showing {len(display_df)} significant differences (p < 0.05)**"
                    )
                else:
                    display_df = comparison_df.copy()
                    st.write(f"**Showing all {len(display_df)} features**")

                if len(display_df) > 0:
                    # Format for display
                    display_df["cluster_a_mean"] = display_df["cluster_a_mean"].round(3)
                    display_df["cluster_b_mean"] = display_df["cluster_b_mean"].round(3)
                    display_df["difference"] = display_df["difference"].round(3)
                    display_df["effect_size"] = display_df["effect_size"].round(3)
                    display_df["t_statistic"] = display_df["t_statistic"].round(3)
                    display_df["p_value"] = display_df["p_value"].apply(
                        lambda x: f"{x:.4f}"
                    )

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=400,
                        column_config={
                            "feature": "Feature",
                            "cluster_a_mean": f"Cluster {selected_clusters[0]} Mean",
                            "cluster_b_mean": f"Cluster {selected_clusters[1]} Mean",
                            "difference": "Difference",
                            "effect_size": st.column_config.NumberColumn(
                                "Effect Size",
                                help="Cohen's d - measures practical significance",
                                format="%.3f",
                            ),
                            "t_statistic": "t-statistic",
                            "p_value": "p-value",
                            "significant": st.column_config.CheckboxColumn(
                                "Significant?"
                            ),
                        },
                        hide_index=True,
                    )

                    # Download button
                    csv = comparison_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"📥 Download Comparison Data",
                        data=csv,
                        file_name=f"cluster_{selected_clusters[0]}_vs_{selected_clusters[1]}_comparison.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No significant differences found between these clusters")
            else:
                st.warning("Unable to compare these clusters")
    else:
        # For 3+ clusters, show summary of all pairwise comparisons
        st.write(
            f"**Showing summary of all pairwise comparisons for {len(selected_clusters)} clusters**"
        )

        with st.spinner("Computing all pairwise comparisons..."):
            # Compute all pairs
            comparison_summaries = []

            for i, cluster_a in enumerate(selected_clusters):
                for cluster_b in selected_clusters[i + 1 :]:
                    comparison_df = compare_two_clusters(df, cluster_a, cluster_b)

                    if len(comparison_df) > 0:
                        # Count significant differences
                        n_significant = comparison_df["significant"].sum()
                        avg_effect_size = comparison_df["effect_size"].abs().mean()

                        comparison_summaries.append(
                            {
                                "Cluster A": f"Cluster {cluster_a}",
                                "Cluster B": f"Cluster {cluster_b}",
                                "Significant Differences": n_significant,
                                "Avg Effect Size": f"{avg_effect_size:.3f}",
                                "Most Different Feature": comparison_df.iloc[0][
                                    "feature"
                                ],
                            }
                        )

            summary_df = pd.DataFrame(comparison_summaries)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.info(
                "💡 Select exactly 2 clusters to see detailed statistical comparison"
            )

    # Radar plot comparison - now supports multiple clusters!
    st.markdown("---")
    st.subheader("🎯 Multi-Dimensional Comparison")

    # Select key features for radar plot
    radar_features = [
        "bpm",
        "danceability",
        "valence",
        "arousal",
        "mood_happy",
        "mood_sad",
        "mood_aggressive",
        "mood_relaxed",
    ]
    radar_features = [f for f in radar_features if f in df.columns]

    if len(radar_features) >= 3:
        # Normalize features to 0-1 scale for fair comparison
        normalized_df = df[radar_features].copy()
        for col in radar_features:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val > min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (
                    max_val - min_val
                )

        # Add cluster column back
        normalized_df["cluster"] = df["cluster"].values

        # Create radar plot with all selected clusters
        fig = go.Figure()

        colors = px.colors.qualitative.Plotly

        for i, cluster_id in enumerate(selected_clusters):
            cluster_means = normalized_df[normalized_df["cluster"] == cluster_id][
                radar_features
            ].mean()

            fig.add_trace(
                go.Scatterpolar(
                    r=cluster_means.values,
                    theta=radar_features,
                    fill="toself",
                    name=f"Cluster {cluster_id}",
                    line_color=colors[i % len(colors)],
                    opacity=0.6,
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title=f"Multi-Cluster Comparison: {len(selected_clusters)} Clusters (Normalized Features)",
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "💡 All features normalized to 0-1 scale for fair comparison. Each cluster shown as a different color."
        )

    # Genre comparison
    st.markdown("---")
    st.subheader("🎸 Genre Comparison")

    if "top_genre" in df.columns:
        # Show top genres for each selected cluster
        num_cols = min(len(selected_clusters), 3)  # Max 3 columns
        cols = st.columns(num_cols)

        for i, cluster_id in enumerate(selected_clusters):
            cluster_df = df[df["cluster"] == cluster_id]
            col_idx = i % num_cols

            with cols[col_idx]:
                st.write(f"**Cluster {cluster_id} - Top 10 Genres**")
                cluster_genres = cluster_df["top_genre"].value_counts().head(10)

                fig = px.bar(
                    x=cluster_genres.values,
                    y=cluster_genres.index,
                    orientation="h",
                    labels={"x": "Count", "y": "Genre"},
                    color_discrete_sequence=[
                        px.colors.qualitative.Plotly[
                            i % len(px.colors.qualitative.Plotly)
                        ]
                    ],
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Genre overlap analysis for all selected clusters
        st.markdown("---")
        st.write(f"**Genre Overlap Analysis ({len(selected_clusters)} clusters)**")

        genre_sets = {
            cluster_id: set(df[df["cluster"] == cluster_id]["top_genre"].unique())
            for cluster_id in selected_clusters
        }

        # Find shared genres across all clusters
        shared_genres = (
            set.intersection(*genre_sets.values()) if len(genre_sets) > 0 else set()
        )

        overlap_data = []
        for cluster_id in selected_clusters:
            # Genres unique to this cluster
            other_clusters = [c for c in selected_clusters if c != cluster_id]
            other_genres = (
                set.union(*[genre_sets[c] for c in other_clusters])
                if other_clusters
                else set()
            )
            unique_genres = genre_sets[cluster_id] - other_genres

            overlap_data.append(
                {
                    "Cluster": f"Cluster {cluster_id}",
                    "Total Genres": len(genre_sets[cluster_id]),
                    "Unique Genres": len(unique_genres),
                    "Shared with All": len(shared_genres),
                }
            )

        overlap_df = pd.DataFrame(overlap_data)
        st.dataframe(overlap_df, use_container_width=True, hide_index=True)

        if len(shared_genres) > 0:
            st.info(
                f"🎵 {len(shared_genres)} genres appear in all selected clusters: {', '.join(list(shared_genres)[:10])}{('...' if len(shared_genres) > 10 else '')}"
            )

    # Sample songs from each cluster
    st.markdown("---")
    st.subheader("🎵 Sample Songs from Each Cluster")

    num_cols = min(len(selected_clusters), 3)  # Max 3 columns
    cols = st.columns(num_cols)

    for i, cluster_id in enumerate(selected_clusters):
        cluster_df = df[df["cluster"] == cluster_id]
        col_idx = i % num_cols

        with cols[col_idx]:
            st.write(f"**Cluster {cluster_id} - Random Sample**")
            sample_df = cluster_df.sample(min(10, len(cluster_df)))[
                ["track_name", "artist"]
            ]
            st.dataframe(sample_df, use_container_width=True, hide_index=True)


@st.cache_data
def load_all_lyrics(df: pd.DataFrame, lyrics_dir: str = "lyrics/temp/"):
    """Load all lyrics for the dataset (cached)."""
    all_lyrics = []

    for _, row in df.iterrows():
        filename = row.get("filename", "")
        if not filename:
            continue

        lyric_filename = filename.replace(".mp3", ".txt")
        lyric_file = Path(lyrics_dir) / lyric_filename

        if lyric_file.exists():
            try:
                with open(lyric_file, "r", encoding="utf-8") as f:
                    lyrics_text = f.read().strip()
                    if lyrics_text:
                        all_lyrics.append(lyrics_text)
            except:
                pass

    return all_lyrics


def render_lyric_themes(df: pd.DataFrame):
    """Render Lyric Themes view."""
    st.header("📝 Lyric Theme Analysis")

    st.write("Explore lyric themes, sentiment, and complexity across clusters.")

    # Lyrics directory configuration
    lyrics_dir = st.text_input(
        "Lyrics directory",
        value="lyrics/temp/",
        help="Directory containing lyric .txt files",
    )

    if not Path(lyrics_dir).exists():
        st.error(f"Lyrics directory not found: {lyrics_dir}")
        st.info("💡 Update the path above to point to your lyrics directory")
        return

    # Cluster selection
    cluster_ids = sorted(df["cluster"].unique())
    selected_cluster = st.selectbox(
        "Select Cluster to Analyze",
        options=cluster_ids,
        format_func=lambda x: f"Cluster {x}",
        key="lyric_cluster",
    )

    with st.spinner("Loading lyrics..."):
        # Load lyrics for selected cluster
        cluster_lyrics_data = load_lyrics_for_cluster(df, selected_cluster, lyrics_dir)

        # Load all lyrics for TF-IDF comparison
        all_lyrics = load_all_lyrics(df, lyrics_dir)

        if not cluster_lyrics_data:
            st.warning(f"No lyrics found for Cluster {selected_cluster}")
            st.info(
                "Make sure lyrics are stored as .txt files matching the MP3 filenames"
            )
            return

        cluster_lyrics = [lyrics for _, lyrics in cluster_lyrics_data]

        # Basic metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            coverage_pct = (
                len(cluster_lyrics_data)
                / len(df[df["cluster"] == selected_cluster])
                * 100
            )
            st.metric("Songs with Lyrics", f"{len(cluster_lyrics_data)}")
            st.caption(f"{coverage_pct:.1f}% coverage")

        with col2:
            if cluster_lyrics:
                avg_word_count = sum(
                    len(lyrics.split()) for lyrics in cluster_lyrics
                ) / len(cluster_lyrics)
                st.metric("Avg Word Count", f"{avg_word_count:.0f}")

        with col3:
            if cluster_lyrics:
                total_words = sum(len(lyrics.split()) for lyrics in cluster_lyrics)
                unique_words = len(set(" ".join(cluster_lyrics).lower().split()))
                st.metric("Unique Words", unique_words)

    # Keyword Analysis
    st.markdown("---")
    st.subheader("🔑 Keyword Analysis (TF-IDF)")

    with st.spinner("Extracting keywords..."):
        if all_lyrics and cluster_lyrics:
            keywords_data = extract_tfidf_keywords(
                all_lyrics, cluster_lyrics, top_n=30, ngram_range=(1, 3)
            )

            if keywords_data["unigrams"]:
                tab1, tab2, tab3 = st.tabs(["Unigrams", "Bigrams", "Trigrams"])

                with tab1:
                    st.write("**Top 30 Single Words**")
                    unigrams_df = pd.DataFrame(
                        keywords_data["unigrams"], columns=["Word", "TF-IDF Score"]
                    )
                    unigrams_df["TF-IDF Score"] = unigrams_df["TF-IDF Score"].round(4)

                    # Bar chart
                    fig = px.bar(
                        unigrams_df.head(20),
                        x="TF-IDF Score",
                        y="Word",
                        orientation="h",
                        title="Top 20 Keywords",
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(unigrams_df, use_container_width=True, hide_index=True)

                with tab2:
                    if keywords_data["bigrams"]:
                        st.write("**Top 30 Two-Word Phrases**")
                        bigrams_df = pd.DataFrame(
                            keywords_data["bigrams"], columns=["Phrase", "TF-IDF Score"]
                        )
                        bigrams_df["TF-IDF Score"] = bigrams_df["TF-IDF Score"].round(4)
                        st.dataframe(
                            bigrams_df, use_container_width=True, hide_index=True
                        )
                    else:
                        st.info("No significant bigrams found")

                with tab3:
                    if keywords_data["trigrams"]:
                        st.write("**Top 30 Three-Word Phrases**")
                        trigrams_df = pd.DataFrame(
                            keywords_data["trigrams"],
                            columns=["Phrase", "TF-IDF Score"],
                        )
                        trigrams_df["TF-IDF Score"] = trigrams_df["TF-IDF Score"].round(
                            4
                        )
                        st.dataframe(
                            trigrams_df, use_container_width=True, hide_index=True
                        )
                    else:
                        st.info("No significant trigrams found")

                # Word Cloud
                st.markdown("---")
                st.subheader("☁️ Word Cloud")

                try:
                    from wordcloud import WordCloud
                    import matplotlib.pyplot as plt

                    # Create word cloud from keywords
                    word_freq = {
                        word: score for word, score in keywords_data["unigrams"][:50]
                    }

                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color="white",
                        colormap="viridis",
                        relative_scaling=0.5,
                        min_font_size=10,
                    ).generate_from_frequencies(word_freq)

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close()

                except ImportError:
                    st.warning(
                        "WordCloud library not available. Install with: pip install wordcloud"
                    )

            else:
                st.warning(
                    "No keywords extracted. Try adjusting the lyrics directory path."
                )

    # Sentiment Analysis
    st.markdown("---")
    st.subheader("😊 Sentiment Analysis")

    with st.spinner("Analyzing sentiment..."):
        if cluster_lyrics:
            sentiments = []

            for lyrics in cluster_lyrics:
                sentiment = analyze_sentiment(lyrics)
                sentiments.append(sentiment)

            # Average sentiment
            avg_compound = np.mean([s["compound_score"] for s in sentiments])
            avg_positive = np.mean([s["positive"] for s in sentiments])
            avg_negative = np.mean([s["negative"] for s in sentiments])
            avg_neutral = np.mean([s["neutral"] for s in sentiments])

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Compound Score",
                    f"{avg_compound:.3f}",
                    help="Overall sentiment: -1 (very negative) to +1 (very positive)",
                )

            with col2:
                st.metric("Positive", f"{avg_positive:.3f}")

            with col3:
                st.metric("Negative", f"{avg_negative:.3f}")

            with col4:
                st.metric("Neutral", f"{avg_neutral:.3f}")

            # Sentiment distribution
            fig = go.Figure()

            fig.add_trace(
                go.Histogram(
                    x=[s["compound_score"] for s in sentiments],
                    nbinsx=20,
                    name="Sentiment Distribution",
                )
            )

            fig.update_layout(
                title="Distribution of Sentiment Scores",
                xaxis_title="Compound Sentiment Score",
                yaxis_title="Number of Songs",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Most positive/negative songs
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Most Positive Songs**")
                # Get top 5 most positive
                sorted_sentiments = sorted(
                    enumerate(sentiments),
                    key=lambda x: x[1]["compound_score"],
                    reverse=True,
                )
                positive_songs = []
                for idx, sentiment in sorted_sentiments[:5]:
                    track_name, _ = cluster_lyrics_data[idx]
                    positive_songs.append(
                        {
                            "Song": track_name,
                            "Sentiment": f"{sentiment['compound_score']:.3f}",
                        }
                    )
                st.dataframe(
                    pd.DataFrame(positive_songs),
                    use_container_width=True,
                    hide_index=True,
                )

            with col2:
                st.write("**Most Negative Songs**")
                negative_songs = []
                for idx, sentiment in sorted_sentiments[-5:][::-1]:
                    track_name, _ = cluster_lyrics_data[idx]
                    negative_songs.append(
                        {
                            "Song": track_name,
                            "Sentiment": f"{sentiment['compound_score']:.3f}",
                        }
                    )
                st.dataframe(
                    pd.DataFrame(negative_songs),
                    use_container_width=True,
                    hide_index=True,
                )

    # Lyric Complexity
    st.markdown("---")
    st.subheader("📚 Lyric Complexity")

    with st.spinner("Computing complexity metrics..."):
        if cluster_lyrics:
            complexities = []

            for lyrics in cluster_lyrics:
                complexity = compute_lyric_complexity(lyrics)
                complexities.append(complexity)

            # Average complexity metrics
            avg_richness = np.mean([c["vocabulary_richness"] for c in complexities])
            avg_word_length = np.mean([c["avg_word_length"] for c in complexities])
            avg_flesch = np.mean(
                [
                    c["flesch_reading_ease"]
                    for c in complexities
                    if c["flesch_reading_ease"] > 0
                ]
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Avg Vocabulary Richness",
                    f"{avg_richness:.3f}",
                    help="Unique words / Total words (higher = more diverse vocabulary)",
                )

            with col2:
                st.metric("Avg Word Length", f"{avg_word_length:.1f} chars")

            with col3:
                if avg_flesch > 0:
                    st.metric(
                        "Flesch Reading Ease",
                        f"{avg_flesch:.1f}",
                        help="0-100 scale, higher = easier to read",
                    )

            # Complexity extremes
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Most Complex Lyrics**")
                sorted_complexity = sorted(
                    enumerate(complexities),
                    key=lambda x: x[1]["vocabulary_richness"],
                    reverse=True,
                )
                complex_songs = []
                for idx, complexity in sorted_complexity[:5]:
                    track_name, _ = cluster_lyrics_data[idx]
                    complex_songs.append(
                        {
                            "Song": track_name,
                            "Richness": f"{complexity['vocabulary_richness']:.3f}",
                            "Words": complexity["word_count"],
                        }
                    )
                st.dataframe(
                    pd.DataFrame(complex_songs),
                    use_container_width=True,
                    hide_index=True,
                )

            with col2:
                st.write("**Simplest Lyrics**")
                simple_songs = []
                for idx, complexity in sorted_complexity[-5:][::-1]:
                    track_name, _ = cluster_lyrics_data[idx]
                    simple_songs.append(
                        {
                            "Song": track_name,
                            "Richness": f"{complexity['vocabulary_richness']:.3f}",
                            "Words": complexity["word_count"],
                        }
                    )
                st.dataframe(
                    pd.DataFrame(simple_songs),
                    use_container_width=True,
                    hide_index=True,
                )

    # Common Phrases
    st.markdown("---")
    st.subheader("🔁 Common Repeated Phrases")

    with st.spinner("Finding common phrases..."):
        if cluster_lyrics:
            common_phrases = extract_common_phrases(cluster_lyrics, top_n=20)

            if common_phrases:
                phrases_df = pd.DataFrame(common_phrases, columns=["Phrase", "Count"])
                st.write("**Top 20 Repeated Phrases (appear more than once)**")
                st.dataframe(phrases_df, use_container_width=True, hide_index=True)
            else:
                st.info("No commonly repeated phrases found")


def render_overview(df: pd.DataFrame):
    """Render Overview view."""
    st.header("🔍 Global Overview")

    st.write("High-level summary of all clusters and their relationships.")

    # Summary Statistics
    st.subheader("📊 Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Songs", len(df))

    with col2:
        st.metric("Number of Clusters", df["cluster"].nunique())

    with col3:
        if "has_lyrics" in df.columns:
            lyric_pct = (df["has_lyrics"].sum() / len(df) * 100) if len(df) > 0 else 0
            st.metric("Songs with Lyrics", f"{lyric_pct:.1f}%")
        else:
            st.metric("Songs with Lyrics", "N/A")

    with col4:
        # Calculate average silhouette score if available
        st.metric("Clustering Mode", "Combined")

    # Cluster sizes
    st.markdown("---")
    st.subheader("📏 Cluster Size Distribution")

    cluster_sizes = df["cluster"].value_counts().sort_index()

    fig = px.bar(
        x=cluster_sizes.index,
        y=cluster_sizes.values,
        labels={"x": "Cluster", "y": "Number of Songs"},
        title="Songs per Cluster",
    )
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)

    # Show cluster size table
    cluster_info = []
    for cluster_id in sorted(df["cluster"].unique()):
        size = len(df[df["cluster"] == cluster_id])
        percentage = size / len(df) * 100

        cluster_info.append(
            {
                "Cluster": f"Cluster {cluster_id}",
                "Size": size,
                "Percentage": f"{percentage:.1f}%",
            }
        )

    st.dataframe(pd.DataFrame(cluster_info), use_container_width=True, hide_index=True)

    # Cluster similarity matrix
    st.markdown("---")
    st.subheader("🔗 Cluster Similarity Matrix")

    st.write("Lower values = more similar clusters (based on average effect sizes)")

    with st.spinner("Computing cluster similarities..."):
        similarity_matrix = compute_cluster_similarity_matrix(df)

        fig = px.imshow(
            similarity_matrix,
            labels=dict(x="Cluster", y="Cluster", color="Dissimilarity"),
            x=similarity_matrix.columns,
            y=similarity_matrix.index,
            color_continuous_scale="YlOrRd",
            title="Cluster Dissimilarity Matrix (Lower = More Similar)",
            aspect="auto",
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Most and least similar cluster pairs
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Most Similar Cluster Pairs**")

        # Get all pairs with their similarity scores
        pairs = []
        cluster_ids = sorted(df["cluster"].unique())

        for i, cluster_a in enumerate(cluster_ids):
            for cluster_b in cluster_ids[i + 1 :]:
                dissimilarity = similarity_matrix.loc[cluster_a, cluster_b]
                pairs.append(
                    {
                        "Pair": f"Cluster {cluster_a} & {cluster_b}",
                        "Dissimilarity": f"{dissimilarity:.3f}",
                    }
                )

        pairs_df = pd.DataFrame(pairs)
        pairs_df = pairs_df.sort_values("Dissimilarity")
        st.dataframe(pairs_df.head(5), use_container_width=True, hide_index=True)

    with col2:
        st.write("**Most Different Cluster Pairs**")
        st.dataframe(
            pairs_df.tail(5).iloc[::-1], use_container_width=True, hide_index=True
        )

    # Key Feature Summary
    st.markdown("---")
    st.subheader("🎯 Key Feature Summary Across All Clusters")

    # Select a few key features to summarize
    key_features = [
        "bpm",
        "danceability",
        "valence",
        "arousal",
        "mood_happy",
        "mood_sad",
    ]
    key_features = [f for f in key_features if f in df.columns]

    if key_features:
        summary_data = []

        for feature in key_features:
            feature_data = {"Feature": feature}

            for cluster_id in sorted(df["cluster"].unique()):
                cluster_df = df[df["cluster"] == cluster_id]
                mean_val = cluster_df[feature].mean()
                feature_data[f"Cluster {cluster_id}"] = f"{mean_val:.3f}"

            summary_data.append(feature_data)

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Genre distribution across clusters
    if "top_genre" in df.columns:
        st.markdown("---")
        st.subheader("🎸 Top Genre Distribution Across Clusters")

        # Get top 5 overall genres
        top_genres = df["top_genre"].value_counts().head(5).index

        genre_cluster_data = []
        for genre in top_genres:
            row = {"Genre": genre}
            for cluster_id in sorted(df["cluster"].unique()):
                cluster_df = df[df["cluster"] == cluster_id]
                count = (cluster_df["top_genre"] == genre).sum()
                percentage = (
                    (count / len(cluster_df) * 100) if len(cluster_df) > 0 else 0
                )
                row[f"Cluster {cluster_id}"] = f"{percentage:.1f}%"

            genre_cluster_data.append(row)

        genre_dist_df = pd.DataFrame(genre_cluster_data)
        st.dataframe(genre_dist_df, use_container_width=True, hide_index=True)

    # Mood radar comparison
    mood_cols = [
        "mood_happy",
        "mood_sad",
        "mood_aggressive",
        "mood_relaxed",
        "mood_party",
    ]

    if all(col in df.columns for col in mood_cols):
        st.markdown("---")
        st.subheader("😊 Mood Profiles by Cluster")

        fig = go.Figure()

        cluster_ids = sorted(df["cluster"].unique())
        colors = px.colors.qualitative.Plotly

        for i, cluster_id in enumerate(cluster_ids):
            cluster_df = df[df["cluster"] == cluster_id]

            mood_means = [
                cluster_df["mood_happy"].mean() * 100,
                cluster_df["mood_sad"].mean() * 100,
                cluster_df["mood_aggressive"].mean() * 100,
                cluster_df["mood_relaxed"].mean() * 100,
                cluster_df["mood_party"].mean() * 100,
            ]

            fig.add_trace(
                go.Scatterpolar(
                    r=mood_means,
                    theta=["Happy", "Sad", "Aggressive", "Relaxed", "Party"],
                    fill="toself",
                    name=f"Cluster {cluster_id}",
                    line_color=colors[i % len(colors)],
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Mood Profiles by Cluster",
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

    # Export all overview data
    st.markdown("---")
    st.subheader("📥 Export Data")

    col1, col2 = st.columns(2)

    with col1:
        # Export similarity matrix
        csv = similarity_matrix.to_csv().encode("utf-8")
        st.download_button(
            label="Download Similarity Matrix",
            data=csv,
            file_name="cluster_similarity_matrix.csv",
            mime="text/csv",
        )

    with col2:
        # Export cluster summary
        summary_csv = pd.DataFrame(cluster_info).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cluster Summary",
            data=summary_csv,
            file_name="cluster_summary.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
