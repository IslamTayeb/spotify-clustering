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

# Names for all 30 embedding dimensions (used for clustering)
# These match the structure in interpretable_features.py
EMBEDDING_DIM_NAMES = [
    # Audio features (14 dims: indices 0-13)
    "emb_bpm",                    # 0: BPM (normalized to [0,1])
    "emb_danceability",           # 1: Danceability
    "emb_instrumentalness",       # 2: Instrumentalness
    "emb_valence",                # 3: Valence (normalized to [0,1])
    "emb_arousal",                # 4: Arousal (normalized to [0,1])
    "emb_engagement",             # 5: Engagement score
    "emb_approachability",        # 6: Approachability score
    "emb_mood_happy",             # 7: Mood - Happy
    "emb_mood_sad",               # 8: Mood - Sad
    "emb_mood_aggressive",        # 9: Mood - Aggressive
    "emb_mood_relaxed",           # 10: Mood - Relaxed
    "emb_mood_party",             # 11: Mood - Party
    "emb_voice_gender",           # 12: Voice Gender (0=female, 1=male)
    "emb_genre_ladder",           # 13: Genre Ladder (0=acoustic, 1=electronic)
    # Key features (3 dims: indices 14-16)
    "emb_key_sin",                # 14: Key pitch (sin component)
    "emb_key_cos",                # 15: Key pitch (cos component)
    "emb_key_scale",              # 16: Key scale (0=minor, 0.33=major)
    # Lyric features (10 dims: indices 17-26) - weighted by (1-instrumentalness)
    "emb_lyric_valence",          # 17: Lyric valence
    "emb_lyric_arousal",          # 18: Lyric arousal
    "emb_lyric_mood_happy",       # 19: Lyric mood - Happy
    "emb_lyric_mood_sad",         # 20: Lyric mood - Sad
    "emb_lyric_mood_aggressive",  # 21: Lyric mood - Aggressive
    "emb_lyric_mood_relaxed",     # 22: Lyric mood - Relaxed
    "emb_lyric_explicit",         # 23: Explicit content
    "emb_lyric_narrative",        # 24: Narrative style
    "emb_lyric_vocabulary",       # 25: Vocabulary richness
    "emb_lyric_repetition",       # 26: Repetition score
    # Theme, Language, Popularity (3 dims: indices 27-29)
    "emb_theme",                  # 27: Theme (ordinal scale)
    "emb_language",               # 28: Language (ordinal scale)
    "emb_popularity",             # 29: Popularity (normalized to [0,1])
]


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

    # Extract audio embeddings (always needed)
    audio_emb = np.vstack([f['embedding'] for f in audio_source])
    logger.info(f"Raw Audio embedding shape: {audio_emb.shape}")

    # For interpretable mode (n_pca_components=None), slice the 30-dim vector by mode:
    # - audio: dims 0-16 (14 audio + 3 key = 17 dims)
    # - lyrics: dims 17-28 (10 lyric + 1 theme + 1 language = 12 dims)
    # - combined: all 30 dims
    if n_pca_components is None:
        if mode == 'audio':
            # Audio features only: BPM, danceability, instrumentalness, valence, arousal,
            # engagement, approachability, moods (5), voice gender, genre ladder, key (3)
            features = audio_emb[:, 0:17]
            logger.info(f"Using interpretable AUDIO features ({features.shape[1]} dims)")
        elif mode == 'lyrics':
            # Lyric features only: lyric valence, arousal, moods (4), explicit, narrative,
            # vocabulary, repetition, theme, language
            features = audio_emb[:, 17:29]
            logger.info(f"Using interpretable LYRIC features ({features.shape[1]} dims)")
        else:  # combined
            # All 30 dims
            features = audio_emb
            logger.info(f"Using interpretable COMBINED features ({features.shape[1]} dims)")

        features_norm = StandardScaler().fit_transform(features)
        return features_norm, list(range(len(audio_source)))

    # Audio mode: only use audio embeddings (no lyric embeddings needed)
    if mode == 'audio':
        # Standardize
        audio_norm = StandardScaler().fit_transform(audio_emb)

        n_components = min(audio_norm.shape[0], audio_norm.shape[1], n_pca_components)

        logger.info(f"Reducing Audio to {n_components} components via PCA...")
        pca = PCA(n_components=n_components, random_state=42)
        audio_reduced = pca.fit_transform(audio_norm)
        logger.info(f"Explained Variance - Audio: {np.sum(pca.explained_variance_ratio_):.2f}")

        return audio_reduced, list(range(len(audio_features)))

    # For lyrics and combined modes, extract lyric embeddings
    lyric_source = lyric_embeddings_override if lyric_embeddings_override else lyric_features
    lyric_emb = np.vstack([f['embedding'] for f in lyric_source])
    logger.info(f"Raw Lyric embedding shape: {lyric_emb.shape}")

    if mode == 'lyrics':
        has_lyrics = np.array([f.get('has_lyrics', True) for f in lyric_features])
        valid_indices = np.where(has_lyrics)[0].tolist()

        # Standardize
        lyric_norm = StandardScaler().fit_transform(lyric_emb[has_lyrics])

        # Skip PCA if n_pca_components is None (e.g., interpretable mode)
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
        # Standardize first
        audio_norm = StandardScaler().fit_transform(audio_emb)
        lyric_norm = StandardScaler().fit_transform(lyric_emb)

        # Skip PCA if n_pca_components is None (e.g., interpretable mode)
        if n_pca_components is None:
            # For interpretable mode, audio_emb already contains combined features
            # No need to concatenate with lyrics - just use audio directly
            logger.info(f"Using interpretable features without PCA ({audio_norm.shape[1]} dims)")
            return audio_norm, list(range(len(audio_features)))

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

    logger.info(f"Audio features: {len(audio_features)}, Lyric features: {len(lyric_features)}")

    # Mode-specific track selection
    # - audio: ALL tracks (use audio dims 0-16 of interpretable vector)
    # - lyrics: only tracks with actual lyrics (use lyric dims 17-28)
    # - combined: ALL tracks (missing lyrics get 0s, weighted by 1-instrumentalness)

    if mode == 'audio':
        # Audio mode: use all audio features
        selected_ids = set(audio_by_id.keys())
        logger.info(f"Audio mode: using all {len(selected_ids)} tracks")
    elif mode == 'combined':
        # Combined mode: instrumental + vocal with lyrics
        # Skip vocal songs (instrumentalness < 0.5) that don't have lyrics
        selected_ids = set()
        skipped_vocal_no_lyrics = 0
        for tid in audio_by_id.keys():
            audio = audio_by_id[tid]
            lyric = lyric_by_id.get(tid, {})

            is_instrumental = audio.get('instrumentalness', 0.5) >= 0.5
            lyric_lang = lyric.get('lyric_language', 'none')
            has_lyrics = lyric_lang not in ('none', None)

            # Include if: instrumental OR (vocal AND has lyrics)
            if is_instrumental or has_lyrics:
                selected_ids.add(tid)
            else:
                skipped_vocal_no_lyrics += 1

        logger.info(f"Combined mode: {len(selected_ids)} tracks (skipped {skipped_vocal_no_lyrics} vocal without lyrics)")
    else:  # lyrics mode
        # Lyrics mode: only tracks with actual lyric features (non-zero lyric dims)
        # Exclude instrumental songs and songs without lyric data
        common_ids = set(audio_by_id.keys()) & set(lyric_by_id.keys())
        logger.info(f"Found {len(common_ids)} tracks with both audio and lyric features")

        filtered_ids = set()
        for tid in common_ids:
            audio = audio_by_id[tid]
            lyric = lyric_by_id[tid]

            # Check if song has lyrics:
            # - GPT interpretable format: lyric_language != 'none'
            lyric_lang = lyric.get('lyric_language', 'none')
            has_lyrics = lyric_lang not in ('none', None)

            is_instrumental = audio.get('instrumentalness', 0.5) >= 0.5

            # Lyrics mode: only include tracks with actual lyrics, exclude instrumental
            if not has_lyrics or is_instrumental:
                filtered_ids.add(tid)

        if filtered_ids:
            logger.info(f"Filtering out {len(filtered_ids)} tracks in lyrics mode (instrumental or no lyrics)")
            common_ids = common_ids - filtered_ids

        selected_ids = common_ids
        logger.info(f"Remaining tracks for lyrics analysis: {len(selected_ids)}")

    aligned_audio = [audio_by_id[tid] for tid in sorted(selected_ids)]
    # For audio mode, create empty lyric placeholders (not used but needed for alignment)
    aligned_lyrics = [lyric_by_id.get(tid, {'track_id': tid}) for tid in sorted(selected_ids)]

    # Align override embeddings if provided
    aligned_audio_override = None
    aligned_lyric_override = None

    if audio_embeddings_override:
        audio_override_by_id = {f['track_id']: f for f in audio_embeddings_override}
        aligned_audio_override = [audio_override_by_id[tid] for tid in sorted(selected_ids) if tid in audio_override_by_id]
        logger.info(f"Using override embeddings for audio clustering ({len(aligned_audio_override)} tracks)")

    if lyric_embeddings_override:
        lyric_override_by_id = {f['track_id']: f for f in lyric_embeddings_override}
        aligned_lyric_override = [lyric_override_by_id[tid] for tid in sorted(selected_ids) if tid in lyric_override_by_id]
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
            'genre_ladder': audio_f.get('genre_ladder', 0.5),
            'key': audio_f['key'],
            'danceability': audio_f['danceability'],
            'instrumentalness': audio_f['instrumentalness'],
            'is_vocal': audio_f['instrumentalness'] < 0.5,
            # Support both old-style (language, has_lyrics) and GPT interpretable (lyric_language)
            'language': lyric_f.get('language', lyric_f.get('lyric_language', 'unknown')),
            'has_lyrics': lyric_f.get('has_lyrics', lyric_f.get('lyric_language', 'none') != 'none'),
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

        # Add all 30 embedding dimensions if available (for interpretable mode)
        # This allows inspection of exactly what values are used for clustering
        if 'embedding' in audio_f and audio_f['embedding'] is not None:
            emb = audio_f['embedding']
            if len(emb) == 30:
                for dim_idx, dim_name in enumerate(EMBEDDING_DIM_NAMES):
                    df_data[-1][dim_name] = float(emb[dim_idx])

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


def run_subcluster_pipeline(
    df: pd.DataFrame,
    pca_features: np.ndarray,
    parent_cluster: int,
    n_subclusters: int = 2,
    algorithm: str = 'hac',
    linkage: str = 'ward',
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
) -> Dict:
    """
    Re-cluster songs within a single parent cluster.

    This function takes an existing clustering result and creates sub-clusters
    within a specific cluster, allowing for hierarchical exploration of music taste.

    Args:
        df: Full DataFrame with 'cluster' column from main clustering
        pca_features: Full PCA features array (aligned with df rows)
        parent_cluster: Which cluster ID to sub-cluster
        n_subclusters: Number of sub-clusters to create (2-10)
        algorithm: Clustering algorithm ('hac', 'birch', 'spectral')
        linkage: HAC linkage method ('ward', 'complete', 'average')
        umap_n_neighbors: UMAP n_neighbors parameter
        umap_min_dist: UMAP min_dist parameter

    Returns:
        Dict with:
        - 'subcluster_df': DataFrame for parent cluster with 'subcluster' column
        - 'subcluster_labels': Array of sub-cluster assignments (0, 1, 2, ...)
        - 'umap_coords': New 3D UMAP coordinates for sub-cluster visualization
        - 'parent_cluster': The parent cluster ID
        - 'n_subclusters': Actual number of sub-clusters created
        - 'silhouette_score': Quality metric for the sub-clustering
        - 'pca_features_subset': PCA features for the subset (for potential further sub-clustering)
    """
    logger.info(f"Sub-clustering Cluster {parent_cluster} into {n_subclusters} sub-clusters")
    logger.info(f"Algorithm: {algorithm}, Linkage: {linkage}")

    # Step 1: Filter to parent cluster
    cluster_mask = df['cluster'].values == parent_cluster
    subset_df = df[cluster_mask].copy()
    subset_pca = pca_features[cluster_mask]

    n_songs = len(subset_df)
    logger.info(f"Parent cluster {parent_cluster} contains {n_songs} songs")

    if n_songs < n_subclusters:
        logger.warning(f"Cannot create {n_subclusters} sub-clusters from {n_songs} songs")
        n_subclusters = max(2, n_songs // 2)
        logger.info(f"Adjusted to {n_subclusters} sub-clusters")

    if n_songs < 2:
        logger.error(f"Cluster {parent_cluster} has fewer than 2 songs, cannot sub-cluster")
        return {
            'subcluster_df': subset_df,
            'subcluster_labels': np.zeros(n_songs, dtype=int),
            'umap_coords': np.zeros((n_songs, 3)),
            'parent_cluster': parent_cluster,
            'n_subclusters': 1,
            'silhouette_score': 0.0,
            'pca_features_subset': subset_pca,
        }

    # Step 2: Run clustering on subset
    if algorithm == 'hac':
        logger.info(f"Running HAC (n_clusters={n_subclusters}, linkage={linkage})")
        clusterer = AgglomerativeClustering(
            n_clusters=n_subclusters,
            linkage=linkage
        )
    elif algorithm == 'birch':
        logger.info(f"Running Birch (n_clusters={n_subclusters})")
        clusterer = Birch(
            n_clusters=n_subclusters,
            threshold=0.5,
            branching_factor=50
        )
    elif algorithm == 'spectral':
        logger.info(f"Running Spectral Clustering (n_clusters={n_subclusters})")
        # Adjust n_neighbors if subset is small
        actual_n_neighbors = min(15, n_songs - 1)
        clusterer = SpectralClustering(
            n_clusters=n_subclusters,
            affinity='nearest_neighbors',
            n_neighbors=actual_n_neighbors,
            assign_labels='kmeans',
            random_state=42
        )
    else:
        logger.warning(f"Unknown algorithm '{algorithm}', defaulting to HAC")
        clusterer = AgglomerativeClustering(
            n_clusters=n_subclusters,
            linkage='ward'
        )

    subcluster_labels = clusterer.fit_predict(subset_pca)
    subset_df['subcluster'] = subcluster_labels

    actual_n_subclusters = len(set(subcluster_labels)) - (1 if -1 in subcluster_labels else 0)
    logger.info(f"Created {actual_n_subclusters} sub-clusters")

    # Step 3: Compute UMAP for visualization of subset
    logger.info(f"Computing UMAP for sub-cluster visualization...")
    # Adjust UMAP parameters for smaller datasets
    actual_n_neighbors = min(umap_n_neighbors, n_songs - 1)
    if actual_n_neighbors < 2:
        actual_n_neighbors = 2

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=actual_n_neighbors,
        min_dist=umap_min_dist,
        metric='cosine',
        random_state=42
    )
    umap_coords = reducer.fit_transform(subset_pca)

    # Step 4: Calculate silhouette score
    if actual_n_subclusters > 1 and n_songs > actual_n_subclusters:
        sil_score = silhouette_score(subset_pca, subcluster_labels)
    else:
        sil_score = 0.0

    logger.info(f"Sub-clustering complete. Silhouette score: {sil_score:.3f}")

    # Log sub-cluster sizes
    for sc_id in sorted(set(subcluster_labels)):
        sc_size = (subcluster_labels == sc_id).sum()
        logger.info(f"  Sub-cluster {sc_id}: {sc_size} songs")

    return {
        'subcluster_df': subset_df,
        'subcluster_labels': subcluster_labels,
        'umap_coords': umap_coords,
        'parent_cluster': parent_cluster,
        'n_subclusters': actual_n_subclusters,
        'silhouette_score': float(sil_score),
        'pca_features_subset': subset_pca,
    }


if __name__ == '__main__':
    import pickle

    with open('analysis/cache/audio_features.pkl', 'rb') as f:
        audio_features = pickle.load(f)
    with open('analysis/cache/lyric_features.pkl', 'rb') as f:
        lyric_features = pickle.load(f)

    results = run_clustering_pipeline(audio_features, lyric_features)
    print(f"\nClustering complete: {results['n_clusters']} clusters, {results['n_outliers']} outliers")
