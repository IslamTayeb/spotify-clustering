"""Reusable components for music analysis Streamlit dashboard.

This package contains modular components organized by layer:
- data: Data loading, feature preparation, DataFrame construction
- clustering: Clustering algorithms, controls, metrics
- visualization: 3D plots, distributions, comparisons
- widgets: Reusable UI widgets (selectors, inspectors, filters)
- tabs: Main tab render functions
- export: Spotify playlist export functionality
"""

# Session state keys (constants for consistency)
SESSION_DYNAMIC_DF = "dynamic_df"
SESSION_AUDIO_FEATURES = "audio_features"
SESSION_LYRIC_FEATURES = "lyric_features"
SESSION_MERT_FEATURES = "mert_features"
SESSION_SELECTED_BACKEND = "selected_backend"
SESSION_FEATURE_WEIGHTS = "feature_weights"

__version__ = "2.0.0"
