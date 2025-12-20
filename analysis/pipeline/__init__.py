"""Music Analysis Pipeline

Core modules for audio feature extraction, lyric analysis, clustering, and visualization.
"""

from .audio_analysis import extract_audio_features, AudioFeatureExtractor
from .lyric_analysis import extract_lyric_features
from .clustering import run_clustering_pipeline
from .visualization import create_interactive_map, generate_report

__all__ = [
    'extract_audio_features',
    'AudioFeatureExtractor',
    'extract_lyric_features',
    'run_clustering_pipeline',
    'create_interactive_map',
    'generate_report',
]
