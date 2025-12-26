"""Main clustering pipeline orchestrator."""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import umap
from sklearn.metrics import silhouette_score

from .constants import EMBEDDING_DIM_NAMES
from .features import prepare_features, load_temporal_metadata
from .algorithms import run_hac, run_birch, run_spectral
from .analysis import analyze_cluster

logger = logging.getLogger(__name__)


def run_clustering_pipeline(
    audio_features: List[Dict],
    lyric_features: List[Dict],
    mode: str = 'combined',
    audio_embeddings_override: List[Dict] = None,
    lyric_embeddings_override: List[Dict] = None,
    n_pca_components: int = 50,
    clustering_algorithm: str = 'hac',
    # HAC parameters
    n_clusters_hac: int = 20,
    linkage_method: str = 'ward',
    # Birch parameters
    n_clusters_birch: int = 20,
    birch_threshold: float = 0.5,
    birch_branching_factor: int = 50,
    # Spectral parameters
    n_clusters_spectral: int = 20,
    spectral_affinity: str = 'nearest_neighbors',
    spectral_n_neighbors: int = 15,
    spectral_assign_labels: str = 'kmeans',
    # UMAP parameters
    umap_n_neighbors: int = 20,
    umap_min_dist: float = 0.2,
    umap_n_components: int = 3
) -> Dict:
    """Main clustering pipeline.

    Separates:
    1. PCA-reduced features for CLUSTERING
    2. UMAP for VISUALIZATION only

    Args:
        audio_features: Audio features (Essentia)
        lyric_features: Lyric features (BGE-M3)
        mode: 'audio', 'lyrics', or 'combined'
        audio_embeddings_override: Optional MERT embeddings
        lyric_embeddings_override: Optional E5 embeddings
        n_pca_components: Number of PCA components
        clustering_algorithm: 'hac', 'birch', 'spectral'
        ... (algorithm-specific parameters)

    Returns:
        Dict with dataframe, coordinates, labels, stats, etc.
    """
    logger.info(f"Running clustering in {mode} mode")
    logger.info(f"Clustering algorithm: {clustering_algorithm}")
    logger.info(f"PCA components: {n_pca_components}")

    # Align features by track_id
    audio_by_id = {f['track_id']: f for f in audio_features}
    lyric_by_id = {f['track_id']: f for f in lyric_features}

    logger.info(f"Audio features: {len(audio_features)}, Lyric features: {len(lyric_features)}")

    # Mode-specific track selection
    selected_ids = _select_tracks(audio_by_id, lyric_by_id, mode)

    aligned_audio = [audio_by_id[tid] for tid in sorted(selected_ids)]
    aligned_lyrics = [lyric_by_id.get(tid, {'track_id': tid}) for tid in sorted(selected_ids)]

    # Align override embeddings
    aligned_audio_override = None
    aligned_lyric_override = None

    if audio_embeddings_override:
        audio_override_by_id = {f['track_id']: f for f in audio_embeddings_override}
        aligned_audio_override = [audio_override_by_id[tid] for tid in sorted(selected_ids) if tid in audio_override_by_id]
        logger.info(f"Using override embeddings for audio ({len(aligned_audio_override)} tracks)")

    if lyric_embeddings_override:
        lyric_override_by_id = {f['track_id']: f for f in lyric_embeddings_override}
        aligned_lyric_override = [lyric_override_by_id[tid] for tid in sorted(selected_ids) if tid in lyric_override_by_id]
        logger.info(f"Using E5 embeddings for lyrics ({len(aligned_lyric_override)} tracks)")

    # Step 1: Prepare PCA-reduced features
    pca_features, valid_indices = prepare_features(
        aligned_audio,
        aligned_lyrics,
        mode,
        n_pca_components,
        audio_embeddings_override=aligned_audio_override,
        lyric_embeddings_override=aligned_lyric_override
    )
    logger.info(f"PCA-reduced features for clustering: {pca_features.shape}")

    # Step 2: Run clustering
    if clustering_algorithm == 'hac':
        cluster_labels = run_hac(pca_features, n_clusters_hac, linkage_method)
    elif clustering_algorithm == 'birch':
        cluster_labels = run_birch(pca_features, n_clusters_birch, birch_threshold, birch_branching_factor)
    elif clustering_algorithm == 'spectral':
        cluster_labels = run_spectral(pca_features, n_clusters_spectral, spectral_affinity, spectral_n_neighbors, spectral_assign_labels)
    else:
        raise ValueError(f"Unknown clustering algorithm: {clustering_algorithm}")

    # Step 3: Run UMAP for visualization
    logger.info(f"Running UMAP for visualization...")
    reducer = umap.UMAP(
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric='cosine',
        random_state=42
    )
    umap_coords = reducer.fit_transform(pca_features)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_outliers = (cluster_labels == -1).sum()

    logger.info(f"Found {n_clusters} clusters and {n_outliers} outliers")

    # Build DataFrame
    df = _build_dataframe(
        aligned_audio, aligned_lyrics, valid_indices,
        cluster_labels, umap_coords, umap_n_components
    )

    # Load and merge temporal metadata
    logger.info("Loading temporal metadata...")
    temporal_metadata = load_temporal_metadata()

    if temporal_metadata:
        df['added_at'] = df['track_id'].map(lambda tid: temporal_metadata.get(tid, {}).get('added_at'))
        df['release_date'] = df['track_id'].map(lambda tid: temporal_metadata.get(tid, {}).get('release_date'))
        df['popularity'] = df['track_id'].map(lambda tid: temporal_metadata.get(tid, {}).get('popularity', 0))
        df['album_name'] = df['track_id'].map(lambda tid: temporal_metadata.get(tid, {}).get('album_name', ''))
        df['album_type'] = df['track_id'].map(lambda tid: temporal_metadata.get(tid, {}).get('album_type', ''))
        df['explicit'] = df['track_id'].map(lambda tid: temporal_metadata.get(tid, {}).get('explicit', False))

    # Add temporal features
    if 'added_at' in df.columns and 'release_date' in df.columns:
        df['added_at'] = pd.to_datetime(df['added_at'])
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

        if df['added_at'].dt.tz is not None and df['release_date'].dt.tz is None:
            df['release_date'] = df['release_date'].dt.tz_localize('UTC')
        elif df['added_at'].dt.tz is None and df['release_date'].dt.tz is not None:
            df['added_at'] = df['added_at'].dt.tz_localize('UTC')

        df['age_at_add_years'] = (df['added_at'] - df['release_date']).dt.days / 365.25
        df['added_year'] = df['added_at'].dt.year
        df['added_month'] = df['added_at'].dt.to_period('M').astype(str)

        logger.info("Temporal metadata merged successfully")

    # Analyze clusters
    cluster_stats = {}
    for cluster_id in sorted(df['cluster'].unique()):
        if cluster_id != -1:
            cluster_stats[cluster_id] = analyze_cluster(cluster_id, df, pca_features, valid_indices)

    outlier_songs = df[df['cluster'] == -1]['filename'].tolist()

    # Calculate silhouette score
    if len(set(cluster_labels)) > 1:
        sil_score = silhouette_score(pca_features, cluster_labels)
    else:
        sil_score = 0.0

    logger.info(f"Silhouette score (on PCA features): {sil_score:.3f}")

    return {
        'dataframe': df,
        'umap_coords': umap_coords,
        'cluster_labels': cluster_labels,
        'cluster_stats': cluster_stats,
        'outlier_songs': outlier_songs,
        'pca_features': pca_features,
        'valid_indices': valid_indices,
        'n_clusters': n_clusters,
        'n_outliers': n_outliers,
        'silhouette_score': float(sil_score),
        'mode': mode
    }


def _select_tracks(audio_by_id: Dict, lyric_by_id: Dict, mode: str) -> set:
    """Select tracks based on mode."""
    if mode == 'audio':
        selected_ids = set(audio_by_id.keys())
        logger.info(f"Audio mode: using all {len(selected_ids)} tracks")
        return selected_ids

    elif mode == 'combined':
        selected_ids = set()
        skipped_vocal_no_lyrics = 0
        for tid in audio_by_id.keys():
            audio = audio_by_id[tid]
            lyric = lyric_by_id.get(tid, {})

            is_instrumental = audio.get('instrumentalness', 0.5) >= 0.5
            lyric_lang = lyric.get('lyric_language', 'none')
            has_lyrics = lyric_lang not in ('none', None)

            if is_instrumental or has_lyrics:
                selected_ids.add(tid)
            else:
                skipped_vocal_no_lyrics += 1

        logger.info(f"Combined mode: {len(selected_ids)} tracks (skipped {skipped_vocal_no_lyrics} vocal without lyrics)")
        return selected_ids

    else:  # lyrics mode
        common_ids = set(audio_by_id.keys()) & set(lyric_by_id.keys())
        logger.info(f"Found {len(common_ids)} tracks with both audio and lyric features")

        filtered_ids = set()
        for tid in common_ids:
            audio = audio_by_id[tid]
            lyric = lyric_by_id[tid]

            lyric_lang = lyric.get('lyric_language', 'none')
            has_lyrics = lyric_lang not in ('none', None)
            is_instrumental = audio.get('instrumentalness', 0.5) >= 0.5

            if not has_lyrics or is_instrumental:
                filtered_ids.add(tid)

        if filtered_ids:
            logger.info(f"Filtering out {len(filtered_ids)} tracks in lyrics mode")
            common_ids = common_ids - filtered_ids

        logger.info(f"Remaining tracks for lyrics analysis: {len(common_ids)}")
        return common_ids


def _build_dataframe(
    aligned_audio: List[Dict],
    aligned_lyrics: List[Dict],
    valid_indices: List[int],
    cluster_labels: np.ndarray,
    umap_coords: np.ndarray,
    umap_n_components: int
) -> pd.DataFrame:
    """Build result DataFrame."""
    df_data = []
    for i, idx in enumerate(valid_indices):
        audio_f = aligned_audio[idx]
        lyric_f = aligned_lyrics[idx]

        top_genre = audio_f['top_3_genres'][0][0] if audio_f['top_3_genres'] else 'unknown'
        dominant_mood = max(
            [('happy', audio_f['mood_happy']),
             ('sad', audio_f['mood_sad']),
             ('aggressive', audio_f['mood_aggressive']),
             ('relaxed', audio_f['mood_relaxed']),
             ('party', audio_f['mood_party'])],
            key=lambda x: x[1]
        )[0]

        coords_dict = {
            'umap_x': float(umap_coords[i, 0]),
            'umap_y': float(umap_coords[i, 1]),
        }
        if umap_n_components >= 3:
            coords_dict['umap_z'] = float(umap_coords[i, 2])

        row = {
            'track_id': audio_f['track_id'],
            'track_name': audio_f['track_name'],
            'artist': audio_f['artist'],
            'filename': audio_f['filename'],
            'filepath': audio_f['filepath'],
            'cluster': int(cluster_labels[i]),
            **coords_dict,
            'top_genre': top_genre,
            'top_3_genres': audio_f['top_3_genres'],
            'genre_probs': audio_f['genre_probs'],
            'dominant_mood': dominant_mood,
            'mood_happy': audio_f['mood_happy'],
            'mood_sad': audio_f['mood_sad'],
            'mood_aggressive': audio_f['mood_aggressive'],
            'mood_relaxed': audio_f['mood_relaxed'],
            'mood_party': audio_f['mood_party'],
            'valence': audio_f.get('valence', 0.5),
            'arousal': audio_f.get('arousal', 0.5),
            'approachability_score': audio_f.get('approachability_score', 0.0),
            'engagement_score': audio_f.get('engagement_score', 0.0),
            'mtg_jamendo_probs': audio_f.get('mtg_jamendo_probs', []),
            'bpm': audio_f['bpm'],
            'genre_ladder': audio_f.get('genre_ladder', 0.5),
            'key': audio_f['key'],
            'danceability': audio_f['danceability'],
            'instrumentalness': audio_f['instrumentalness'],
            'is_vocal': audio_f['instrumentalness'] < 0.5,
            'language': lyric_f.get('language', lyric_f.get('lyric_language', 'unknown')),
            'has_lyrics': lyric_f.get('has_lyrics', lyric_f.get('lyric_language', 'none') != 'none'),
            'lyric_valence': lyric_f.get('lyric_valence', 0.5),
            'lyric_arousal': lyric_f.get('lyric_arousal', 0.5),
            'lyric_mood_happy': lyric_f.get('lyric_mood_happy', 0),
            'lyric_mood_sad': lyric_f.get('lyric_mood_sad', 0),
            'lyric_mood_aggressive': lyric_f.get('lyric_mood_aggressive', 0),
            'lyric_mood_relaxed': lyric_f.get('lyric_mood_relaxed', 0),
            'lyric_explicit': lyric_f.get('lyric_explicit', 0),
            'lyric_narrative': lyric_f.get('lyric_narrative', 0),
            'lyric_theme': lyric_f.get('lyric_theme', 'other'),
            'lyric_language': lyric_f.get('lyric_language', 'unknown'),
            'lyric_vocabulary_richness': lyric_f.get('lyric_vocabulary_richness', 0),
            'lyric_repetition': lyric_f.get('lyric_repetition', 0)
        }

        # Add embedding dimensions
        if 'embedding' in audio_f and audio_f['embedding'] is not None:
            emb = audio_f['embedding']
            if len(emb) == len(EMBEDDING_DIM_NAMES):
                for dim_idx, dim_name in enumerate(EMBEDDING_DIM_NAMES):
                    row[dim_name] = float(emb[dim_idx])

        df_data.append(row)

    return pd.DataFrame(df_data)
