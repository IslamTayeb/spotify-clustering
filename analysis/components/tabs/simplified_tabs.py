"""Simplified tab components that leverage existing code.

Rather than duplicating 1,500+ lines of tab logic, these components
import and delegate to modular tab implementations.

All tabs have been extracted to proper component modules.
"""

import pandas as pd


# Import all tabs from proper component modules
try:
    from analysis.components.tabs.eda_explorer import render_eda_explorer
    from analysis.components.tabs.feature_importance import render_feature_importance
    from analysis.components.tabs.cluster_comparison import render_cluster_comparison
    from analysis.components.tabs.lyric_themes import render_lyric_themes
    from analysis.components.tabs.overview import render_overview
    from analysis.components.tabs.audio_vs_lyrics import render_audio_vs_lyrics
    from analysis.components.tabs.feature_explainers import render_feature_explainers
    HAS_ALL_TABS = True
except ImportError as e:
    HAS_ALL_TABS = False
    IMPORT_ERROR = str(e)

# Fallback stub implementations if imports fail
if not HAS_ALL_TABS:
    import streamlit as st

    def render_eda_explorer(df: pd.DataFrame) -> None:
        st.header("📊 EDA Explorer")
        st.error(f"Tab import failed: {IMPORT_ERROR}")
        st.dataframe(df.head(10))

    def render_feature_importance(df: pd.DataFrame) -> None:
        st.header("🎯 Feature Importance")
        st.error(f"Tab import failed: {IMPORT_ERROR}")

    def render_cluster_comparison(df: pd.DataFrame) -> None:
        st.header("⚖️ Cluster Comparison")
        st.error(f"Tab import failed: {IMPORT_ERROR}")

    def render_lyric_themes(df: pd.DataFrame) -> None:
        st.header("📝 Lyric Themes")
        st.error(f"Tab import failed: {IMPORT_ERROR}")

    def render_overview(df: pd.DataFrame) -> None:
        st.header("🔍 Overview")
        st.error(f"Tab import failed: {IMPORT_ERROR}")

    def render_audio_vs_lyrics(df: pd.DataFrame) -> None:
        st.header("🔀 Audio vs Lyrics")
        st.error(f"Tab import failed: {IMPORT_ERROR}")

    def render_feature_explainers(df: pd.DataFrame = None) -> None:
        st.header("📚 Feature Explainers")
        st.error(f"Tab import failed: {IMPORT_ERROR}")


# Re-export for easy importing
__all__ = [
    "render_eda_explorer",
    "render_feature_importance",
    "render_cluster_comparison",
    "render_lyric_themes",
    "render_overview",
    "render_audio_vs_lyrics",
    "render_feature_explainers",
]
