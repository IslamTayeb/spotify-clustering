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
        help="0=acoustic, 1=electronic",
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
        interpretable_mode: If True, PCA is skipped by default (30-dim vector)

    Returns:
        Dictionary with PCA configuration
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Feature Preparation")

    # For interpretable mode, always skip PCA (30-dim vector is already interpretable)
    if interpretable_mode:
        st.sidebar.info("ℹ️ PCA skipped for interpretable 30-dim vector")
        return {
            "skip_pca": True,
            "n_pca_components": 30,  # Placeholder, will be ignored
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
