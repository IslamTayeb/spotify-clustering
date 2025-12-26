"""Feature preparation for clustering pipeline."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .constants import FEATURE_WEIGHT_INDICES

logger = logging.getLogger(__name__)


def apply_subcluster_weights(features: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
    """Apply feature weights to embedding features for sub-clustering.

    Args:
        features: Array of shape (n_samples, n_features) - typically 33 embedding dims
        weights: Dictionary mapping feature group names to weight values (0.0-2.0)

    Returns:
        Weighted features array of the same shape
    """
    if weights is None:
        return features

    weighted = features.copy()
    n_dims = weighted.shape[1]

    for group, (start, end) in FEATURE_WEIGHT_INDICES.items():
        if group in weights and start < n_dims:
            actual_end = min(end, n_dims)
            weighted[:, start:actual_end] *= weights[group]

    return weighted


def extract_embedding_features(df: pd.DataFrame, parent_cluster: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract embedding features from DataFrame for a specific cluster.

    Args:
        df: DataFrame with emb_* columns
        parent_cluster: Cluster ID to filter

    Returns:
        Tuple of (features array, cluster mask)
    """
    emb_cols = [col for col in df.columns if col.startswith('emb_')]

    if not emb_cols:
        raise ValueError("No embedding columns (emb_*) found in DataFrame")

    cluster_mask = df['cluster'].values == parent_cluster
    subset_df = df[cluster_mask]

    features = subset_df[emb_cols].values.astype(np.float64)

    return features, cluster_mask


def load_temporal_metadata(saved_tracks_path: str = 'spotify/saved_tracks.json') -> Dict:
    """Load temporal metadata from saved_tracks.json and return as dict keyed by track_id."""
    if not Path(saved_tracks_path).exists():
        logger.warning(f"Temporal metadata file not found: {saved_tracks_path}")
        return {}

    with open(saved_tracks_path, 'r') as f:
        tracks = json.load(f)

    temporal_data = {}
    for track in tracks:
        track_id = track['track_id']
        temporal_data[track_id] = {
            'added_at': track['added_at'],
            'release_date': track.get('release_date'),
            'popularity': track.get('popularity', 0),
            'album_name': track.get('album_name', ''),
            'album_type': track.get('album_type', ''),
            'explicit': track.get('explicit', False)
        }

    logger.info(f"Loaded temporal metadata for {len(temporal_data)} tracks")
    return temporal_data


def prepare_features(
    audio_features: List[Dict],
    lyric_features: List[Dict],
    mode: str = 'combined',
    n_pca_components: int = 50,
    audio_embeddings_override: List[Dict] = None,
    lyric_embeddings_override: List[Dict] = None
) -> Tuple[np.ndarray, List[int]]:
    """Prepare features for clustering by standardizing and applying PCA.

    Args:
        audio_features: Audio features (Essentia, used for interpretation)
        lyric_features: Lyric features (BGE-M3 default)
        mode: 'audio', 'lyrics', or 'combined'
        n_pca_components: Number of PCA components
        audio_embeddings_override: Optional MERT embeddings for clustering
        lyric_embeddings_override: Optional E5 embeddings for clustering

    Returns:
        Tuple of (pca_features, valid_indices)
    """
    audio_source = audio_embeddings_override if audio_embeddings_override else audio_features

    # Extract audio embeddings
    audio_emb = np.vstack([f['embedding'] for f in audio_source])
    logger.info(f"Raw Audio embedding shape: {audio_emb.shape}")

    # For interpretable mode (n_pca_components=None)
    if n_pca_components is None:
        if mode == 'audio':
            features = audio_emb[:, 0:19]
            logger.info(f"Using interpretable AUDIO features ({features.shape[1]} dims)")
        elif mode == 'lyrics':
            features = audio_emb[:, 19:33]
            logger.info(f"Using interpretable LYRIC+META features ({features.shape[1]} dims)")
        else:  # combined
            features = audio_emb
            logger.info(f"Using interpretable COMBINED features ({features.shape[1]} dims)")

        features_norm = StandardScaler().fit_transform(features)
        return features_norm, list(range(len(audio_source)))

    # Audio mode with PCA
    if mode == 'audio':
        audio_norm = StandardScaler().fit_transform(audio_emb)
        n_components = min(audio_norm.shape[0], audio_norm.shape[1], n_pca_components)

        logger.info(f"Reducing Audio to {n_components} components via PCA...")
        pca = PCA(n_components=n_components, random_state=42)
        audio_reduced = pca.fit_transform(audio_norm)
        logger.info(f"Explained Variance - Audio: {np.sum(pca.explained_variance_ratio_):.2f}")

        return audio_reduced, list(range(len(audio_features)))

    # Extract lyric embeddings for lyrics/combined modes
    lyric_source = lyric_embeddings_override if lyric_embeddings_override else lyric_features
    lyric_emb = np.vstack([f['embedding'] for f in lyric_source])
    logger.info(f"Raw Lyric embedding shape: {lyric_emb.shape}")

    if mode == 'lyrics':
        has_lyrics = np.array([f.get('has_lyrics', True) for f in lyric_features])
        valid_indices = np.where(has_lyrics)[0].tolist()

        lyric_norm = StandardScaler().fit_transform(lyric_emb[has_lyrics])

        if n_pca_components is None:
            logger.info(f"Using raw lyric features without PCA ({lyric_norm.shape[1]} dims)")
            return lyric_norm, valid_indices

        n_components = min(lyric_norm.shape[0], lyric_norm.shape[1], n_pca_components)

        logger.info(f"Reducing Lyrics to {n_components} components via PCA...")
        pca = PCA(n_components=n_components, random_state=42)
        lyric_reduced = pca.fit_transform(lyric_norm)
        logger.info(f"Explained Variance - Lyrics: {np.sum(pca.explained_variance_ratio_):.2f}")

        return lyric_reduced, valid_indices

    else:  # combined
        audio_norm = StandardScaler().fit_transform(audio_emb)
        lyric_norm = StandardScaler().fit_transform(lyric_emb)

        if n_pca_components is None:
            logger.info(f"Using interpretable features without PCA ({audio_norm.shape[1]} dims)")
            return audio_norm, list(range(len(audio_features)))

        n_components_audio = min(audio_norm.shape[0], audio_norm.shape[1], n_pca_components)
        n_components_lyric = min(lyric_norm.shape[0], lyric_norm.shape[1], n_pca_components)

        logger.info(f"Reducing Audio to {n_components_audio} components via PCA...")
        pca_audio = PCA(n_components=n_components_audio, random_state=42)
        audio_reduced = pca_audio.fit_transform(audio_norm)

        logger.info(f"Reducing Lyrics to {n_components_lyric} components via PCA...")
        pca_lyric = PCA(n_components=n_components_lyric, random_state=42)
        lyric_reduced = pca_lyric.fit_transform(lyric_norm)

        logger.info(f"Explained Variance - Audio: {np.sum(pca_audio.explained_variance_ratio_):.2f}")
        logger.info(f"Explained Variance - Lyrics: {np.sum(pca_lyric.explained_variance_ratio_):.2f}")

        combined = np.hstack([audio_reduced, lyric_reduced])
        logger.info(f"Combined feature shape: {combined.shape}")

        return combined, list(range(len(audio_features)))
