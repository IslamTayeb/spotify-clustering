"""Music Analysis Pipeline

Core modules for audio feature extraction, lyric analysis, clustering, and visualization.

NOTE: Static visualization/report generation has been removed.
All analysis is now done through the interactive Streamlit dashboard:
  streamlit run analysis/interactive_interpretability.py
"""

from .audio_analysis import extract_audio_features, AudioFeatureExtractor
from .lyric_analysis import extract_lyric_features
from .clustering import run_clustering_pipeline

__all__ = [
    'extract_audio_features',
    'AudioFeatureExtractor',
    'extract_lyric_features',
    'run_clustering_pipeline',
]
