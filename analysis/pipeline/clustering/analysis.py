"""Cluster analysis functions."""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_cluster(
    cluster_id: int,
    df: pd.DataFrame,
    pca_features: np.ndarray = None,
    valid_indices: List[int] = None,
    lyrics_dir: str = 'lyrics/data/',
    include_lyric_themes: bool = True
) -> Dict:
    """Analyze cluster statistics and select representative tracks.

    Args:
        cluster_id: Cluster ID to analyze
        df: Full DataFrame with all tracks
        pca_features: PCA-reduced features used for clustering (optional)
        valid_indices: Indices mapping df rows to pca_features (optional)
        lyrics_dir: Directory containing lyrics files
        include_lyric_themes: Whether to include lyric theme analysis

    Returns:
        Dictionary with cluster statistics
    """
    cluster_df = df[df['cluster'] == cluster_id]

    # Genre analysis
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

    # Mood distribution
    mood_cols = ['mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed', 'mood_party']
    mood_dist = {col: float(cluster_df[col].mean()) for col in mood_cols}

    # Key distribution
    major_count = cluster_df['key'].str.contains('major', case=False, na=False).sum()
    minor_count = cluster_df['key'].str.contains('minor', case=False, na=False).sum()
    total_key = major_count + minor_count
    key_dist = {
        'major': major_count / total_key if total_key > 0 else 0,
        'minor': minor_count / total_key if total_key > 0 else 0
    }

    # Representative track selection
    if pca_features is not None and valid_indices is not None:
        cluster_mask = df['cluster'].values == cluster_id
        cluster_pca_features = pca_features[cluster_mask]

        centroid = cluster_pca_features.mean(axis=0)
        distances = np.linalg.norm(cluster_pca_features - centroid, axis=1)
        representative_indices = np.argsort(distances)[:5]
        representative_songs = cluster_df.iloc[representative_indices]['filename'].tolist()
    else:
        # Fallback to UMAP coordinates
        centroid = cluster_df[['umap_x', 'umap_y']].mean().values
        distances = np.linalg.norm(cluster_df[['umap_x', 'umap_y']].values - centroid, axis=1)
        representative_indices = np.argsort(distances)[:5]
        representative_songs = cluster_df.iloc[representative_indices]['filename'].tolist()

    language_dist = cluster_df['language'].value_counts().to_dict()

    # Lyric theme analysis (optional)
    lyric_themes = None
    if include_lyric_themes and Path(lyrics_dir).exists():
        lyric_themes = _extract_lyric_themes(df, cluster_id, cluster_df, lyrics_dir)

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
        'lyric_themes': lyric_themes
    }


def _extract_lyric_themes(df: pd.DataFrame, cluster_id: int, cluster_df: pd.DataFrame, lyrics_dir: str) -> Dict:
    """Extract lyric themes for a cluster."""
    try:
        from analysis.interpretability.lyric_themes import (
            load_lyrics_for_cluster,
            extract_tfidf_keywords,
            analyze_sentiment,
            compute_lyric_complexity
        )

        lyric_data = load_lyrics_for_cluster(df, cluster_id, lyrics_dir=lyrics_dir)

        if lyric_data and len(lyric_data) >= 3:
            cluster_lyrics = [text for _, text in lyric_data]

            # Load all lyrics for TF-IDF
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

            if all_lyrics:
                keywords = extract_tfidf_keywords(all_lyrics, cluster_lyrics, top_n=10)

                sentiments = [analyze_sentiment(text) for text in cluster_lyrics]
                avg_sentiment = np.mean([s['compound_score'] for s in sentiments])

                if avg_sentiment > 0.05:
                    sentiment_label = 'positive'
                elif avg_sentiment < -0.05:
                    sentiment_label = 'negative'
                else:
                    sentiment_label = 'neutral'

                complexities = [compute_lyric_complexity(text) for text in cluster_lyrics]
                avg_complexity = np.mean([c['vocabulary_richness'] for c in complexities])

                return {
                    'top_keywords': keywords,
                    'avg_sentiment': float(avg_sentiment),
                    'sentiment_label': sentiment_label,
                    'avg_complexity': float(avg_complexity),
                    'n_lyrics': len(lyric_data)
                }
    except Exception as e:
        logger.warning(f"Could not extract lyric themes for cluster {cluster_id}: {e}")

    return None
