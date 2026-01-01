"""EDA Explorer Tab - Main orchestrator.

Comprehensive exploratory data analysis with:
- Overall statistics
- Feature vector analysis
- Genre analysis
- Audio/mood/vocal extremes
- Genre ladder analysis
- Language analysis
- Temporal analysis (11 visualizations)
- 3D cluster visualization
- Data preview & export
"""

import streamlit as st
import pandas as pd

from .feature_stats import (
    render_feature_vector_analysis,
    render_feature_distributions_by_category,
    render_raw_features,
    render_overall_statistics,
)
from .genre_analysis import render_genre_analysis, render_genre_fusion_analysis
from .mood_vocal import (
    render_audio_extremes,
    render_mood_analysis,
    render_vocal_analysis,
    render_language_analysis,
)
from .temporal import render_temporal_analysis
from .cluster_viz import render_3d_cluster_visualization, render_data_preview_export
from analysis.components.export.chart_export import render_export_section, get_selected_chart_count


def render_eda_explorer(df: pd.DataFrame):
    """Render EDA Explorer view with comprehensive statistics."""
    st.header("📊 Exploratory Data Analysis")

    # Feature Analysis
    render_feature_vector_analysis(df)
    render_feature_distributions_by_category(df)
    render_raw_features(df)

    # Statistics
    render_overall_statistics(df)

    # Genre Analysis
    render_genre_analysis(df)

    # Audio/Mood/Vocal
    render_audio_extremes(df)
    render_mood_analysis(df)
    render_vocal_analysis(df)

    # Genre Fusion & Language
    render_genre_fusion_analysis(df)
    render_language_analysis(df)

    # Temporal
    render_temporal_analysis(df)

    # 3D Visualization
    render_3d_cluster_visualization(df)

    # Data Export
    render_data_preview_export(df)

    # Chart Export Section (for all selected charts across all sections)
    if get_selected_chart_count() > 0:
        render_export_section("export/visualizations", "eda")
