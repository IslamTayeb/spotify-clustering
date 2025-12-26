"""Feature selection and weighting widgets.

This module provides Streamlit widgets for selecting backends and adjusting
feature weights for interpretable mode.
"""

import streamlit as st
from typing import Dict, Tuple


def render_backend_selector(has_mert: bool = False) -> str:
    """Render backend selection widget.

    Args:
        has_mert: Whether MERT embeddings are available

    Returns:
        Selected backend name
    """
    backend_options = ["Essentia (Default)"]
    if has_mert:
        backend_options.append("MERT (Transformer)")
    backend_options.append("Interpretable Features (Audio)")

    default_index = 1 if has_mert else 0

    return st.sidebar.selectbox(
        "Audio Embedding Backend",
        backend_options,
        index=default_index,
        help="Choose backend. 'Interpretable' uses BPM, Key, Moods, etc. 'MERT' uses transformers.",
    )


def render_audio_feature_weights() -> Dict[str, float]:
    """Render audio feature weight sliders.

    Returns:
        Dictionary of audio feature weights
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Audio Feature Weights")

    weights = {}

    weights["core_audio"] = st.sidebar.slider(
        "🎵 Core Audio (BPM, Danceability, etc.)",
        0.0,
        2.0,
        1.0,
        0.1,
        help="BPM, danceability, instrumentalness, valence, arousal, engagement, approachability",
    )

    weights["mood"] = st.sidebar.slider(
        "😊 Audio Moods",
        0.0,
        2.0,
        1.0,
        0.1,
        help="happy, sad, aggressive, relaxed, party",
    )

    weights["genre"] = st.sidebar.slider(
        "🎸 Genre Ladder",
        0.0,
        2.0,
        1.0,
        0.1,
        help="0=pure genre, 1=genre fusion (entropy-based)",
    )

    weights["key"] = st.sidebar.slider(
        "🎹 Key",
        0.0,
        2.0,
        1.0,
        0.1,
        help="Musical key (sin, cos, scale)",
    )

    return weights


def render_lyric_feature_weights() -> Dict[str, float]:
    """Render lyric feature weight sliders.

    Returns:
        Dictionary of lyric feature weights
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("📝 Lyric Feature Weights")

    weights = {}

    weights["lyric_emotion"] = st.sidebar.slider(
        "💭 Lyric Emotions (Valence, Arousal, Moods)",
        0.0,
        2.0,
        1.0,
        0.1,
        help="Lyric valence, arousal, happy, sad, aggressive, relaxed",
    )

    weights["lyric_content"] = st.sidebar.slider(
        "🔞 Content (Explicit, Narrative, Vocab)",
        0.0,
        2.0,
        1.0,
        0.1,
        help="Explicit content, narrative style, vocabulary richness, repetition",
    )

    weights["theme"] = st.sidebar.slider(
        "🏷️ Theme",
        0.0,
        2.0,
        1.0,
        0.1,
        help="Theme categories: love, heartbreak, party, flex, street, etc.",
    )

    weights["language"] = st.sidebar.slider(
        "🌍 Language",
        0.0,
        2.0,
        1.0,
        0.1,
        help="Language of lyrics",
    )

    weights["metadata"] = st.sidebar.slider(
        "📊 Metadata (Popularity + Release Year)",
        0.0,
        2.0,
        1.0,
        0.1,
        help="Spotify popularity (mainstream vs. niche) + Release year (vintage vs. modern)",
    )

    return weights


def render_pca_controls(mode: str, interpretable_mode: bool = False) -> Dict[str, any]:
    """Render PCA control widgets.

    Args:
        mode: Feature mode (audio/lyrics/combined)
        interpretable_mode: If True, PCA is skipped (interpretable vector is already low-dim)

    Returns:
        Dictionary with PCA configuration
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Feature Preparation")

    # For interpretable mode, always skip PCA (already low-dimensional and interpretable)
    if interpretable_mode:
        st.sidebar.info("ℹ️ PCA skipped for interpretable features")
        return {
            "skip_pca": True,
            "n_pca_components": None,  # Not used
        }

    # Skip PCA checkbox (only shown for non-interpretable backends)
    skip_pca = st.sidebar.checkbox(
        "Skip PCA (Use Raw Features)",
        value=False,
        help="Pass high-dimensional embeddings directly to clustering.",
    )

    # PCA component slider (disabled if skip_pca)
    pca_defaults = {"audio": 118, "lyrics": 162, "combined": 142}
    default_pca = pca_defaults.get(mode, 140)

    if not skip_pca:
        n_pca_components = st.sidebar.slider(
            "PCA Components",
            5,
            200,
            default_pca,
            step=5,
            help=f"Number of PCA components for dimensionality reduction. Default ({default_pca}) achieves ~75% variance for {mode} mode.",
        )
    else:
        n_pca_components = default_pca  # Placeholder, will be ignored

    return {
        "skip_pca": skip_pca,
        "n_pca_components": n_pca_components,
    }


def render_subcluster_feature_weights() -> Dict[str, float]:
    """Render feature weight sliders for sub-clustering.

    These weights are applied independently of main clustering weights,
    allowing different emphasis when drilling into a specific cluster.

    Returns:
        Dictionary of feature weights for sub-clustering
    """
    with st.sidebar.expander("🎛️ Sub-Cluster Feature Weights", expanded=False):
        st.caption("Adjust feature importance for sub-clustering")

        weights = {}

        weights["core_audio"] = st.slider(
            "🎵 Core Audio",
            0.0,
            2.0,
            1.0,
            0.1,
            help="BPM, danceability, instrumentalness, valence, arousal, engagement, approachability",
            key="sc_weight_core_audio",
        )

        weights["mood"] = st.slider(
            "😊 Audio Moods",
            0.0,
            2.0,
            1.0,
            0.1,
            help="happy, sad, aggressive, relaxed, party",
            key="sc_weight_mood",
        )

        weights["genre"] = st.slider(
            "🎸 Genre/Timbre",
            0.0,
            2.0,
            1.0,
            0.1,
            help="Voice gender, genre ladder, acoustic/electronic, timbre",
            key="sc_weight_genre",
        )

        weights["key"] = st.slider(
            "🎹 Key",
            0.0,
            2.0,
            1.0,
            0.1,
            help="Musical key (sin, cos, scale)",
            key="sc_weight_key",
        )

        weights["lyric_emotion"] = st.slider(
            "💭 Lyric Emotions",
            0.0,
            2.0,
            1.0,
            0.1,
            help="Lyric valence, arousal, moods",
            key="sc_weight_lyric_emotion",
        )

        weights["lyric_content"] = st.slider(
            "🔞 Lyric Content",
            0.0,
            2.0,
            1.0,
            0.1,
            help="Explicit, narrative, vocabulary, repetition",
            key="sc_weight_lyric_content",
        )

        weights["theme"] = st.slider(
            "🏷️ Theme",
            0.0,
            2.0,
            1.0,
            0.1,
            help="Theme categories",
            key="sc_weight_theme",
        )

        weights["language"] = st.slider(
            "🌍 Language",
            0.0,
            2.0,
            1.0,
            0.1,
            help="Language of lyrics",
            key="sc_weight_language",
        )

        weights["metadata"] = st.slider(
            "📊 Metadata",
            0.0,
            2.0,
            1.0,
            0.1,
            help="Popularity + Release Year",
            key="sc_weight_metadata",
        )

        # Quick presets
        st.markdown("---")
        st.caption("Quick Presets")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Audio Focus", key="sc_preset_audio", use_container_width=True):
                st.session_state["sc_weight_core_audio"] = 1.5
                st.session_state["sc_weight_mood"] = 1.5
                st.session_state["sc_weight_genre"] = 1.0
                st.session_state["sc_weight_lyric_emotion"] = 0.3
                st.session_state["sc_weight_lyric_content"] = 0.3
                st.rerun()
        with col2:
            if st.button("Lyric Focus", key="sc_preset_lyric", use_container_width=True):
                st.session_state["sc_weight_core_audio"] = 0.5
                st.session_state["sc_weight_mood"] = 0.5
                st.session_state["sc_weight_lyric_emotion"] = 1.5
                st.session_state["sc_weight_lyric_content"] = 1.5
                st.session_state["sc_weight_theme"] = 1.5
                st.rerun()

    return weights


def render_umap_controls() -> Dict[str, any]:
    """Render UMAP visualization parameter controls.

    Returns:
        Dictionary with UMAP parameters
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 UMAP Visualization")
    st.sidebar.markdown("*These parameters only affect the plot layout, NOT clustering!*")

    n_neighbors = st.sidebar.slider(
        "n_neighbors (Viz)",
        5,
        100,
        20,
        help="Controls balance between local and global structure for the plot. Lower = more local detail, higher = more global overview.",
    )

    min_dist = st.sidebar.slider(
        "min_dist (Viz)",
        0.0,
        1.0,
        0.2,
        step=0.01,
        help="How tightly points are packed in the visualization. Lower = tighter packing, clearer separation between visual clusters.",
    )

    return {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
    }
