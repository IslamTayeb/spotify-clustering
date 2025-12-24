#!/usr/bin/env python3
"""
Clustering Pipeline

Supports multiple embedding backends via override parameters:
- Audio: Essentia (default, for interpretation) or MERT (via override, for clustering)
- Lyrics: BGE-M3 (default) or E5 (via override)

Design: Essentia always runs for interpretation (genre/mood/BPM). MERT/E5 are optional
overrides passed in-memory, with separate caches (no schema changes to existing caches).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, Birch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_temporal_metadata(saved_tracks_path: str = 'spotify/saved_tracks.json') -> Dict:
    """Load temporal metadata from saved_tracks.json and return as dict keyed by track_id"""
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
    # NEW: Embedding overrides for MERT/E5 support
    audio_embeddings_override: List[Dict] = None,
    lyric_embeddings_override: List[Dict] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Prepare features for clustering by standardizing and applying PCA.
    This output is used for CLUSTERING (not visualization).

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
    # Use override embeddings if provided, else use default
    audio_source = audio_embeddings_override if audio_embeddings_override else audio_features
    lyric_source = lyric_embeddings_override if lyric_embeddings_override else lyric_features

    audio_emb = np.vstack([f['embedding'] for f in audio_source])
    lyric_emb = np.vstack([f['embedding'] for f in lyric_source])

    logger.info(f"Raw Audio embedding shape: {audio_emb.shape}")
    logger.info(f"Raw Lyric embedding shape: {lyric_emb.shape}")

    if mode == 'audio':
        # Standardize then PCA
        audio_norm = StandardScaler().fit_transform(audio_emb)
        n_components = min(audio_norm.shape[0], audio_norm.shape[1], n_pca_components)

        logger.info(f"Reducing Audio to {n_components} components via PCA...")
        pca = PCA(n_components=n_components, random_state=42)
        audio_reduced = pca.fit_transform(audio_norm)
        logger.info(f"Explained Variance - Audio: {np.sum(pca.explained_variance_ratio_):.2f}")

        return audio_reduced, list(range(len(audio_features)))

    elif mode == 'lyrics':
        has_lyrics = np.array([f['has_lyrics'] for f in lyric_features])
        valid_indices = np.where(has_lyrics)[0].tolist()

        # Standardize then PCA
        lyric_norm = StandardScaler().fit_transform(lyric_emb[has_lyrics])
        n_components = min(lyric_norm.shape[0], lyric_norm.shape[1], n_pca_components)

        logger.info(f"Reducing Lyrics to {n_components} components via PCA...")
        pca = PCA(n_components=n_components, random_state=42)
        lyric_reduced = pca.fit_transform(lyric_norm)
        logger.info(f"Explained Variance - Lyrics: {np.sum(pca.explained_variance_ratio_):.2f}")

        return lyric_reduced, valid_indices

    else:  # combined
        # Standardize first
        audio_norm = StandardScaler().fit_transform(audio_emb)
        lyric_norm = StandardScaler().fit_transform(lyric_emb)

        # PCA Reduction to balance modalities
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

        # Combine the balanced, reduced vectors
        combined = np.hstack([audio_reduced, lyric_reduced])
        logger.info(f"Combined feature shape: {combined.shape}")

        return combined, list(range(len(audio_features)))


def analyze_cluster(
    cluster_id: int,
    df: pd.DataFrame,
    pca_features: np.ndarray = None,
    valid_indices: List[int] = None,
    lyrics_dir: str = 'lyrics/data/',
    include_lyric_themes: bool = True
) -> Dict:
    """
    Analyze cluster statistics and select representative tracks.

    Args:
        cluster_id: Cluster ID to analyze
        df: Full DataFrame with all tracks
        pca_features: PCA-reduced features used for clustering (optional)
        valid_indices: Indices mapping df rows to pca_features (optional)

    Returns:
        Dictionary with cluster statistics
    """
    cluster_df = df[df['cluster'] == cluster_id]

    genre_matrix = np.vstack(cluster_df['genre_probs'].values)
    avg_genre_probs = genre_matrix.mean(axis=0)
    top_3_indices = np.argsort(avg_genre_probs)[-3:][::-1]

    genre_labels = cluster_df['top_3_genres'].iloc[0]
    if isinstance(genre_labels, list) and len(genre_labels) > 0:
        all_genres = {}
        for top_3 in cluster_df['top_3_genres'].values:
            for genre, prob in top_3:
                if genre not in all_genres:
                    all_genres[genre] = []
                all_genres[genre].append(prob)

        top_genres = sorted(
            [(g, np.mean(p)) for g, p in all_genres.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
    else:
        top_genres = [("unknown", 0.0)] * 3

    mood_cols = ['mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed', 'mood_party']
    mood_dist = {col: float(cluster_df[col].mean()) for col in mood_cols}

    major_count = cluster_df['key'].str.contains('major', case=False, na=False).sum()
    minor_count = cluster_df['key'].str.contains('minor', case=False, na=False).sum()
    total_key = major_count + minor_count
    key_dist = {
        'major': major_count / total_key if total_key > 0 else 0,
        'minor': minor_count / total_key if total_key > 0 else 0
    }

    # Representative track selection
    # CRITICAL: Use PCA features (clustering space) NOT UMAP (visualization only)
    if pca_features is not None and valid_indices is not None:
        # Get PCA features for this cluster
        cluster_mask = df['cluster'].values == cluster_id
        cluster_pca_features = pca_features[cluster_mask]

        # Compute centroid in PCA space (actual clustering space)
        centroid = cluster_pca_features.mean(axis=0)

        # Find 5 closest points to centroid in PCA space
        distances = np.linalg.norm(cluster_pca_features - centroid, axis=1)
        representative_indices = np.argsort(distances)[:5]
        representative_songs = cluster_df.iloc[representative_indices]['filename'].tolist()
    else:
        # Fallback to UMAP if PCA features not provided (backward compatibility)
        # WARNING: This uses visualization coordinates, not actual clustering features
        centroid = cluster_df[['umap_x', 'umap_y']].mean().values
        distances = np.linalg.norm(
            cluster_df[['umap_x', 'umap_y']].values - centroid,
            axis=1
        )
        representative_indices = np.argsort(distances)[:5]
        representative_songs = cluster_df.iloc[representative_indices]['filename'].tolist()

    language_dist = cluster_df['language'].value_counts().to_dict()

    # Lyric theme analysis (optional)
    lyric_themes = None
    if include_lyric_themes and Path(lyrics_dir).exists():
        try:
            from analysis.interpretability.lyric_themes import (
                load_lyrics_for_cluster,
                extract_tfidf_keywords,
                analyze_sentiment,
                compute_lyric_complexity
            )

            # Load lyrics for this cluster
            lyric_data = load_lyrics_for_cluster(df, cluster_id, lyrics_dir=lyrics_dir)

            if lyric_data and len(lyric_data) >= 3:  # Need minimum lyrics for TF-IDF
                cluster_lyrics = [text for _, text in lyric_data]

                # Load all lyrics from dataset for TF-IDF IDF calculation
                # (This is expensive, but TF-IDF needs global statistics)
                all_lyrics = []
                for _, row in df.iterrows():
                    if row.get('has_lyrics', False):
                        filename = row.get('filename', '')
                        if filename:
                            lyric_filename = filename.replace('.mp3', '.txt')
                            lyric_file = Path(lyrics_dir) / lyric_filename
                            if lyric_file.exists():
                                try:
                                    with open(lyric_file, 'r', encoding='utf-8') as f:
                                        all_lyrics.append(f.read())
                                except Exception:
                                    pass

                # Extract lyric themes
                if all_lyrics:
                    keywords = extract_tfidf_keywords(all_lyrics, cluster_lyrics, top_n=10)

                    # Sentiment analysis
                    sentiments = [analyze_sentiment(text) for text in cluster_lyrics]
                    avg_sentiment = np.mean([s['compound_score'] for s in sentiments])

                    if avg_sentiment > 0.05:
                        sentiment_label = 'positive'
                    elif avg_sentiment < -0.05:
                        sentiment_label = 'negative'
                    else:
                        sentiment_label = 'neutral'

                    # Complexity analysis
                    complexities = [compute_lyric_complexity(text) for text in cluster_lyrics]
                    avg_complexity = np.mean([c['vocabulary_richness'] for c in complexities])

                    lyric_themes = {
                        'top_keywords': keywords,
                        'avg_sentiment': float(avg_sentiment),
                        'sentiment_label': sentiment_label,
                        'avg_complexity': float(avg_complexity),
                        'n_lyrics': len(lyric_data)
                    }
        except Exception as e:
            logger.warning(f"Could not extract lyric themes for cluster {cluster_id}: {e}")

    return {
        'n_songs': len(cluster_df),
        'percentage': len(cluster_df) / len(df) * 100,
        'top_3_genres': top_genres,
        'median_bpm': float(cluster_df['bpm'].median()),
        'mood_distribution': mood_dist,
        'language_distribution': language_dist,
        'key_distribution': key_dist,
        'avg_danceability': float(cluster_df['danceability'].mean()),
        'representative_songs': representative_songs,
        'lyric_themes': lyric_themes  # NEW
    }


def run_clustering_pipeline(
    audio_features: List[Dict],
    lyric_features: List[Dict],
    mode: str = 'combined',
    # NEW: Embedding override parameters for MERT/E5
    audio_embeddings_override: List[Dict] = None,
    lyric_embeddings_override: List[Dict] = None,
    # Existing parameters
    n_pca_components: int = 50,              # PCA dimensionality for clustering
    clustering_algorithm: str = 'hac',       # 'hac', 'birch', 'spectral'
    # HAC parameters
    n_clusters_hac: int = 20,
    linkage_method: str = 'ward',            # 'ward', 'complete', 'average', 'single'
    # Birch parameters
    n_clusters_birch: int = 20,
    birch_threshold: float = 0.5,
    birch_branching_factor: int = 50,
    # Spectral Clustering parameters
    n_clusters_spectral: int = 20,
    spectral_affinity: str = 'nearest_neighbors',  # 'nearest_neighbors' or 'rbf'
    spectral_n_neighbors: int = 15,
    spectral_assign_labels: str = 'kmeans',  # 'kmeans' or 'discretize'
    # UMAP visualization parameters
    umap_n_neighbors: int = 20,
    umap_min_dist: float = 0.2,
    umap_n_components: int = 3               # 2D or 3D visualization
) -> Dict:
    """
    Clustering pipeline that separates:
    1. PCA-reduced features for CLUSTERING
    2. UMAP for VISUALIZATION only

    Design pattern for MERT/E5 support:
    - audio_features: ALWAYS Essentia (for interpretation: genre/mood/BPM)
    - audio_embeddings_override: Optional MERT embeddings (for clustering)
    - lyric_features: Default BGE-M3
    - lyric_embeddings_override: Optional E5 embeddings (for clustering)

    Supported algorithms:
    - 'hac': Hierarchical Agglomerative Clustering (default, recommended)
    - 'birch': Fast hierarchical clustering
    - 'spectral': Graph-based clustering for non-convex shapes
    """
    logger.info(f"Running clustering in {mode} mode")
    logger.info(f"Clustering algorithm: {clustering_algorithm}")
    logger.info(f"PCA components: {n_pca_components}")

    # Align features by track_id
    audio_by_id = {f['track_id']: f for f in audio_features}
    lyric_by_id = {f['track_id']: f for f in lyric_features}

    common_ids = set(audio_by_id.keys()) & set(lyric_by_id.keys())
    logger.info(f"Found {len(common_ids)} tracks with both audio and lyric features")
    logger.info(f"Audio features: {len(audio_features)}, Lyric features: {len(lyric_features)}")

    # For combined mode, filter out vocal songs without lyrics
    # Note: Instrumental songs (instrumentalness >= 0.5) are kept even without lyrics
    #       Non-instrumental songs (vocal) without lyrics are filtered out
    if mode == 'combined':
        filtered_ids = set()
        for tid in common_ids:
            audio = audio_by_id[tid]
            lyric = lyric_by_id[tid]
            # Exclude if song is vocal (instrumentalness < 0.5) but has no lyrics
            # Keep instrumental songs (instrumentalness >= 0.5) even without lyrics
            if audio.get('instrumentalness', 0.5) < 0.5 and not lyric.get('has_lyrics', False):
                filtered_ids.add(tid)

        if filtered_ids:
            logger.info(f"Filtering out {len(filtered_ids)} vocal songs without lyrics in combined mode")
            common_ids = common_ids - filtered_ids
            logger.info(f"Remaining tracks for combined analysis: {len(common_ids)}")

    aligned_audio = [audio_by_id[tid] for tid in sorted(common_ids)]
    aligned_lyrics = [lyric_by_id[tid] for tid in sorted(common_ids)]

    # Align override embeddings if provided
    aligned_audio_override = None
    aligned_lyric_override = None

    if audio_embeddings_override:
        audio_override_by_id = {f['track_id']: f for f in audio_embeddings_override}
        aligned_audio_override = [audio_override_by_id[tid] for tid in sorted(common_ids) if tid in audio_override_by_id]
        logger.info(f"Using MERT embeddings for audio clustering ({len(aligned_audio_override)} tracks)")

    if lyric_embeddings_override:
        lyric_override_by_id = {f['track_id']: f for f in lyric_embeddings_override}
        aligned_lyric_override = [lyric_override_by_id[tid] for tid in sorted(common_ids) if tid in lyric_override_by_id]
        logger.info(f"Using E5 embeddings for lyric clustering ({len(aligned_lyric_override)} tracks)")

    # Step 1: Prepare PCA-reduced features for CLUSTERING
    pca_features, valid_indices = prepare_features(
        aligned_audio,
        aligned_lyrics,
        mode,
        n_pca_components,
        audio_embeddings_override=aligned_audio_override,
        lyric_embeddings_override=aligned_lyric_override
    )
    logger.info(f"PCA-reduced features for clustering: {pca_features.shape}")

    # Step 2: Run CLUSTERING on PCA features
    if clustering_algorithm == 'hac':
        logger.info(f"Running Hierarchical Agglomerative Clustering (n_clusters={n_clusters_hac}, "
                   f"linkage={linkage_method})...")
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters_hac,
            linkage=linkage_method
        )
        cluster_labels = clusterer.fit_predict(pca_features)

    elif clustering_algorithm == 'birch':
        logger.info(f"Running Birch clustering (n_clusters={n_clusters_birch}, "
                   f"threshold={birch_threshold}, branching_factor={birch_branching_factor})...")
        clusterer = Birch(
            n_clusters=n_clusters_birch,
            threshold=birch_threshold,
            branching_factor=birch_branching_factor
        )
        cluster_labels = clusterer.fit_predict(pca_features)

    elif clustering_algorithm == 'spectral':
        logger.info(f"Running Spectral Clustering (n_clusters={n_clusters_spectral}, "
                   f"affinity={spectral_affinity}, n_neighbors={spectral_n_neighbors})...")
        clusterer = SpectralClustering(
            n_clusters=n_clusters_spectral,
            affinity=spectral_affinity,
            n_neighbors=spectral_n_neighbors if spectral_affinity == 'nearest_neighbors' else 10,
            assign_labels=spectral_assign_labels,
            random_state=42
        )
        cluster_labels = clusterer.fit_predict(pca_features)

    else:
        raise ValueError(f"Unknown clustering algorithm: {clustering_algorithm}")

    # Step 3: Run UMAP for VISUALIZATION ONLY (not used for clustering)
    logger.info(f"Running UMAP for visualization (n_neighbors={umap_n_neighbors}, "
               f"min_dist={umap_min_dist}, n_components={umap_n_components})...")
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

        # Build coordinate dict (supports 2D or 3D)
        coords_dict = {
            'umap_x': float(umap_coords[i, 0]),
            'umap_y': float(umap_coords[i, 1]),
        }
        if umap_n_components >= 3:
            coords_dict['umap_z'] = float(umap_coords[i, 2])

        df_data.append({
            'track_id': audio_f['track_id'],
            'track_name': audio_f['track_name'],
            'artist': audio_f['artist'],
            'filename': audio_f['filename'],
            'filepath': audio_f['filepath'],
            'cluster': int(cluster_labels[i]),
            **coords_dict,  # Unpack coordinates (2D or 3D)
            'top_genre': top_genre,
            'top_3_genres': audio_f['top_3_genres'],
            'genre_probs': audio_f['genre_probs'],
            'dominant_mood': dominant_mood,
            'mood_happy': audio_f['mood_happy'],
            'mood_sad': audio_f['mood_sad'],
            'mood_aggressive': audio_f['mood_aggressive'],
            'mood_relaxed': audio_f['mood_relaxed'],
            'mood_party': audio_f['mood_party'],
            # Missing features added below
            'valence': audio_f.get('valence', 0.5), # Default if missing
            'arousal': audio_f.get('arousal', 0.5),
            'approachability_score': audio_f.get('approachability_score', 0.0),
            'engagement_score': audio_f.get('engagement_score', 0.0),
            'mtg_jamendo_probs': audio_f.get('mtg_jamendo_probs', []),

            'bpm': audio_f['bpm'],
            'key': audio_f['key'],
            'danceability': audio_f['danceability'],
            'instrumentalness': audio_f['instrumentalness'],
            'is_vocal': audio_f['instrumentalness'] < 0.5,
            'language': lyric_f['language'],
            'word_count': lyric_f['word_count'],
            'has_lyrics': lyric_f['has_lyrics'],
            # Lyric Features (Tier 1: Parallel emotional dimensions)
            'lyric_valence': lyric_f.get('lyric_valence', 0.5),
            'lyric_arousal': lyric_f.get('lyric_arousal', 0.5),
            'lyric_mood_happy': lyric_f.get('lyric_mood_happy', 0),
            'lyric_mood_sad': lyric_f.get('lyric_mood_sad', 0),
            'lyric_mood_aggressive': lyric_f.get('lyric_mood_aggressive', 0),
            'lyric_mood_relaxed': lyric_f.get('lyric_mood_relaxed', 0),
            # Lyric Features (Tier 3: Lyric-unique)
            'lyric_explicit': lyric_f.get('lyric_explicit', 0),
            'lyric_narrative': lyric_f.get('lyric_narrative', 0),
            'lyric_theme': lyric_f.get('lyric_theme', 'other'),
            'lyric_language': lyric_f.get('lyric_language', 'unknown'),
            'lyric_vocabulary_richness': lyric_f.get('lyric_vocabulary_richness', 0),
            'lyric_repetition': lyric_f.get('lyric_repetition', 0)
        })

    df = pd.DataFrame(df_data)

    # Load and merge temporal metadata
    logger.info("Loading temporal metadata...")
    temporal_metadata = load_temporal_metadata()

    if temporal_metadata:
        # Add temporal columns
        df['added_at'] = df['track_id'].map(lambda tid: temporal_metadata.get(tid, {}).get('added_at'))
        df['release_date'] = df['track_id'].map(lambda tid: temporal_metadata.get(tid, {}).get('release_date'))
        df['popularity'] = df['track_id'].map(lambda tid: temporal_metadata.get(tid, {}).get('popularity', 0))
        df['album_name'] = df['track_id'].map(lambda tid: temporal_metadata.get(tid, {}).get('album_name', ''))
        df['album_type'] = df['track_id'].map(lambda tid: temporal_metadata.get(tid, {}).get('album_type', ''))
        df['explicit'] = df['track_id'].map(lambda tid: temporal_metadata.get(tid, {}).get('explicit', False))

    # Add temporal features if available
    if 'added_at' in df.columns and 'release_date' in df.columns:
        df['added_at'] = pd.to_datetime(df['added_at'])
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

        # Ensure compatible timezones
        if df['added_at'].dt.tz is not None and df['release_date'].dt.tz is None:
             df['release_date'] = df['release_date'].dt.tz_localize('UTC')
        elif df['added_at'].dt.tz is None and df['release_date'].dt.tz is not None:
             df['added_at'] = df['added_at'].dt.tz_localize('UTC')

        df['age_at_add_years'] = (df['added_at'] - df['release_date']).dt.days / 365.25

        # NOTE: Temporal features removed from clustering as requested
        # Time period groupings
        df['added_year'] = df['added_at'].dt.year
        df['added_month'] = df['added_at'].dt.to_period('M').astype(str)

        logger.info("Temporal metadata merged successfully")

    # Analyze each cluster (passing PCA features for representative track selection)
    cluster_stats = {}
    for cluster_id in sorted(df['cluster'].unique()):
        if cluster_id != -1:
            cluster_stats[cluster_id] = analyze_cluster(
                cluster_id, df, pca_features, valid_indices
            )

    outlier_songs = df[df['cluster'] == -1]['filename'].tolist()

    # Calculate silhouette score on PCA features (not UMAP visualization)
    # HAC, Birch, and Spectral don't produce outliers
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
        'pca_features': pca_features,  # NEW: Include for potential future use
        'valid_indices': valid_indices,  # NEW: Mapping from df to pca_features
        'n_clusters': n_clusters,
        'n_outliers': n_outliers,
        'silhouette_score': float(sil_score),
        'mode': mode
    }


if __name__ == '__main__':
    import pickle

    with open('cache/audio_features.pkl', 'rb') as f:
        audio_features = pickle.load(f)
    with open('cache/lyric_features.pkl', 'rb') as f:
        lyric_features = pickle.load(f)

    results = run_clustering_pipeline(audio_features, lyric_features)
    print(f"\nClustering complete: {results['n_clusters']} clusters, {results['n_outliers']} outliers")
