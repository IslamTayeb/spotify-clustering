"""Comprehensive Data Export Script for Spotify Clustering Analysis

This script exports ALL data and visualizations from the clustering analysis
into a data-first format optimized for sharing with Claude.

Exports:
- 45+ CSV/JSON files with raw data (primary)
- 12+ PNG images for key spatial visualizations (supplementary)

Usage:
    python analysis/export_all_data.py                    # Export everything
    python analysis/export_all_data.py --skip-images      # Data only (faster)
    python analysis/export_all_data.py --mode audio       # Specific mode
"""

import sys
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import argparse
import json
import os
import pickle
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer

# Import project modules
from analysis.components.data import loaders

print("Loading analysis modules...")


# ============================================================================
# INFRASTRUCTURE FUNCTIONS
# ============================================================================

def setup_output_directories(base_dir: str) -> None:
    """Create output directory structure"""
    dirs = [
        f"{base_dir}/data/core",
        f"{base_dir}/data/overview",
        f"{base_dir}/data/feature_importance",
        f"{base_dir}/data/cluster_comparison",
        f"{base_dir}/data/eda/temporal",
        f"{base_dir}/data/lyrics",
        f"{base_dir}/data/subclusters",
        f"{base_dir}/plots/spatial/subclusters",
        f"{base_dir}/plots/heatmaps/subclusters",
        f"{base_dir}/plots/patterns",
        f"{base_dir}/plots/wordclouds",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: str, description: str = None) -> None:
    """Save DataFrame to CSV with optional description"""
    df.to_csv(path, index=False)
    if description:
        print(f"  ✓ {description}: {os.path.basename(path)}")


def save_json(data: dict, path: str) -> None:
    """Save dictionary to JSON file"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def save_plotly_as_png(fig, path: str, width: int = 1200, height: int = 800) -> None:
    """Save Plotly figure as PNG (requires kaleido)"""
    try:
        import plotly.io as pio
        pio.write_image(fig, path, width=width, height=height)
    except ImportError:
        print(f"  ⚠️  Skipping {os.path.basename(path)}: kaleido not installed")
    except Exception as e:
        print(f"  ⚠️  Failed to save {os.path.basename(path)}: {e}")


def generate_metadata(args, source_metadata: dict, duration: float, file_counts: dict) -> dict:
    """Generate export metadata summary"""
    # Get date range from df if available
    date_range = "N/A"
    if 'added_at' in source_metadata:
        date_range = f"{source_metadata['added_at']['min']} to {source_metadata['added_at']['max']}"

    # Find sub-cluster files
    subclusters_exported = []
    subclusters_dir = Path(args.subclusters_dir)
    if subclusters_dir.exists():
        subclusters_exported = [f.stem for f in subclusters_dir.glob("*.pkl")]

    return {
        "export_timestamp": datetime.now().isoformat(),
        "source_file": args.data_file,
        "clustering_mode": args.mode,
        "audio_backend": source_metadata.get('audio_backend', 'unknown'),
        "total_songs": source_metadata.get('total_songs', 0),
        "num_clusters": source_metadata.get('num_clusters', 0),
        "num_embedding_dimensions": 33,
        "date_range": date_range,
        "subclusters_exported": subclusters_exported,
        "files_exported": {
            "data_files": file_counts.get('data', 0),
            "plot_images": file_counts.get('images', 0),
            "total_size_mb": 0  # Will be computed in main()
        },
        "feature_vector_info": {
            "audio_dims": "0-15 (BPM, moods, genre, production, etc.)",
            "key_dims": "16-18 (pitch sin/cos, major/minor)",
            "lyric_dims": "19-28 (valence, arousal, moods, explicit, narrative, vocab, repetition)",
            "meta_dims": "29-32 (theme, language, popularity, release_year)"
        },
        "export_duration_seconds": round(duration, 2)
    }


# ============================================================================
# CORE DATA EXPORT
# ============================================================================

def export_core_data(df: pd.DataFrame, pca_features, output_dir: str, skip_full_dataset: bool = False) -> int:
    """Export core clustering data"""
    file_count = 0

    # 1. Full dataset (exclude embedding vector columns to save space)
    # Skip entirely if --skip-full-dataset flag is used (saves ~10MB for LLM uploads)
    if not skip_full_dataset:
        # Only include interpretable features and metadata, not raw 33-dim vectors
        vector_cols = [col for col in df.columns if col.startswith('emb_')]
        df_export = df.drop(columns=vector_cols, errors='ignore')
        save_dataframe(df_export, f"{output_dir}/full_dataset.csv", "Full dataset")
        file_count += 1
    else:
        print("  ⚡ Skipping full_dataset.csv (--skip-full-dataset flag)")

    # 2. Embedding vectors - SKIPPED to reduce file size
    # These 33-dim vectors for all tracks take up ~500KB each
    # If needed, they can be reconstructed from the pickle file

    # 2b. Cluster + vectors consolidated - SKIPPED for same reason

    # 3. Cluster statistics
    cluster_stats = df.groupby('cluster').agg({
        'cluster': 'count'
    }).rename(columns={'cluster': 'size'})
    cluster_stats['percentage'] = (cluster_stats['size'] / len(df) * 100).round(2)
    cluster_stats = cluster_stats.reset_index()
    save_dataframe(cluster_stats, f"{output_dir}/cluster_stats.csv", "Cluster statistics")
    file_count += 1

    # 4. Cluster centroids (mean feature values)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude coordinate columns
    feature_cols = [c for c in numeric_cols if not c.startswith(('x', 'y', 'z', 'cluster'))]

    centroids = df.groupby('cluster')[feature_cols].mean()
    centroids = centroids.reset_index()
    save_dataframe(centroids, f"{output_dir}/cluster_centroids.csv", "Cluster centroids")
    file_count += 1

    # 5. Similarity matrix (pairwise cluster dissimilarity)
    centroid_values = centroids[feature_cols].values
    similarity_matrix = cdist(centroid_values, centroid_values, metric='euclidean')
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=[f"cluster_{i}" for i in centroids['cluster']],
        columns=[f"cluster_{i}" for i in centroids['cluster']]
    )
    save_dataframe(similarity_df, f"{output_dir}/similarity_matrix.csv", "Similarity matrix")
    file_count += 1

    return file_count


# ============================================================================
# OVERVIEW DATA EXPORT
# ============================================================================

def export_overview_data(df: pd.DataFrame, output_dir: str) -> int:
    """Export overview tab data"""
    file_count = 0

    # 1. Cluster sizes
    cluster_sizes = df.groupby('cluster').size().reset_index(name='size')
    cluster_sizes['percentage'] = (cluster_sizes['size'] / len(df) * 100).round(2)
    save_dataframe(cluster_sizes, f"{output_dir}/cluster_sizes.csv", "Cluster sizes")
    file_count += 1

    # 2. Similarity pairs (most/least similar clusters)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if not c.startswith(('x', 'y', 'z', 'cluster'))]
    centroids = df.groupby('cluster')[feature_cols].mean().values

    similarity_matrix = cdist(centroids, centroids, metric='euclidean')
    clusters = sorted(df['cluster'].unique())

    pairs = []
    for i, c1 in enumerate(clusters):
        for j, c2 in enumerate(clusters):
            if i < j:  # Only upper triangle
                pairs.append({
                    'cluster_a': c1,
                    'cluster_b': c2,
                    'dissimilarity': similarity_matrix[i, j]
                })

    pairs_df = pd.DataFrame(pairs).sort_values('dissimilarity')
    save_dataframe(pairs_df, f"{output_dir}/similarity_pairs.csv", "Similarity pairs")
    file_count += 1

    # 3. Key features summary
    key_features = ['bpm', 'danceability', 'valence', 'arousal', 'mood_happy', 'mood_sad']
    available_features = [f for f in key_features if f in df.columns]

    if available_features:
        key_summary = df.groupby('cluster')[available_features].mean().round(3)
        key_summary = key_summary.reset_index()
        save_dataframe(key_summary, f"{output_dir}/key_features_summary.csv", "Key features summary")
        file_count += 1

    # 4. Genre distribution
    if 'top_genre' in df.columns:
        genre_dist = []
        for cluster in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster]
            genre_counts = Counter(cluster_df['top_genre'].dropna())
            for genre, count in genre_counts.most_common(10):
                genre_dist.append({
                    'cluster': cluster,
                    'genre': genre,
                    'count': count,
                    'percentage': round(count / len(cluster_df) * 100, 2)
                })

        genre_df = pd.DataFrame(genre_dist)
        save_dataframe(genre_df, f"{output_dir}/genre_distribution.csv", "Genre distribution")
        file_count += 1

    # 5. Mood profiles
    mood_cols = ['mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed', 'mood_party']
    available_moods = [m for m in mood_cols if m in df.columns]

    if available_moods:
        mood_profiles = df.groupby('cluster')[available_moods].mean().round(3)
        mood_profiles = mood_profiles.reset_index()
        save_dataframe(mood_profiles, f"{output_dir}/mood_profiles.csv", "Mood profiles")
        file_count += 1

    return file_count


# ============================================================================
# FEATURE IMPORTANCE DATA EXPORT
# ============================================================================

def export_feature_importance_data(df: pd.DataFrame, output_dir: str) -> int:
    """Export feature importance analysis"""
    file_count = 0

    # Get numeric feature columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if not c.startswith(('x', 'y', 'z', 'cluster'))]

    # 1. Feature importance (Cohen's d for all features × all clusters)
    importance_records = []
    global_means = df[feature_cols].mean()
    global_stds = df[feature_cols].std()

    for cluster in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster]
        cluster_means = cluster_df[feature_cols].mean()

        for feature in feature_cols:
            cohens_d = (cluster_means[feature] - global_means[feature]) / global_stds[feature] if global_stds[feature] > 0 else 0
            importance_records.append({
                'cluster': cluster,
                'feature': feature,
                'cohens_d': round(cohens_d, 4),
                'cluster_mean': round(cluster_means[feature], 4),
                'global_mean': round(global_means[feature], 4),
                'abs_cohens_d': round(abs(cohens_d), 4)
            })

    importance_df = pd.DataFrame(importance_records)
    importance_df = importance_df.sort_values(['cluster', 'abs_cohens_d'], ascending=[True, False])
    save_dataframe(importance_df, f"{output_dir}/importance_all_clusters.csv", "Feature importance")
    file_count += 1

    # 2. Statistical summary per cluster
    summary_records = []
    for cluster in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster]
        for feature in feature_cols:
            summary_records.append({
                'cluster': cluster,
                'feature': feature,
                'mean': round(cluster_df[feature].mean(), 4),
                'median': round(cluster_df[feature].median(), 4),
                'std': round(cluster_df[feature].std(), 4),
                'min': round(cluster_df[feature].min(), 4),
                'max': round(cluster_df[feature].max(), 4),
                'count': len(cluster_df)
            })

    summary_df = pd.DataFrame(summary_records)
    save_dataframe(summary_df, f"{output_dir}/statistical_summary.csv", "Statistical summary")
    file_count += 1

    # 3. Top 3 features per cluster (JSON)
    top_features = {}
    for cluster in sorted(df['cluster'].unique()):
        cluster_importance = importance_df[importance_df['cluster'] == cluster].nlargest(3, 'abs_cohens_d')
        top_features[f"cluster_{cluster}"] = cluster_importance[['feature', 'cohens_d', 'cluster_mean', 'global_mean']].to_dict('records')

    save_json(top_features, f"{output_dir}/top_features_per_cluster.json")
    file_count += 1

    # 4. Top 5 defining features per cluster (CSV format for easier viewing)
    top_features_records = []
    for cluster in sorted(df['cluster'].unique()):
        cluster_importance = importance_df[importance_df['cluster'] == cluster].nlargest(5, 'abs_cohens_d')
        for rank, (_, row) in enumerate(cluster_importance.iterrows(), 1):
            top_features_records.append({
                'cluster': cluster,
                'rank': rank,
                'feature': row['feature'],
                'cohens_d': row['cohens_d'],
                'cluster_mean': row['cluster_mean'],
                'global_mean': row['global_mean'],
                'direction': 'higher' if row['cohens_d'] > 0 else 'lower'
            })

    if top_features_records:
        top_features_csv = pd.DataFrame(top_features_records)
        save_dataframe(top_features_csv, f"{output_dir}/top_defining_features.csv", "Top defining features")
        file_count += 1

    return file_count


# ============================================================================
# CLUSTER COMPARISON DATA EXPORT
# ============================================================================

def export_cluster_comparison_data(df: pd.DataFrame, output_dir: str) -> int:
    """Export cluster comparison data"""
    file_count = 0

    # Get numeric feature columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if not c.startswith(('x', 'y', 'z', 'cluster'))]

    # 1. Cluster overview
    overview_data = []
    key_metrics = ['bpm', 'danceability', 'mood_happy', 'valence']
    available_metrics = [m for m in key_metrics if m in df.columns]

    for cluster in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster]
        row = {
            'cluster': cluster,
            'size': len(cluster_df),
            'percentage': round(len(cluster_df) / len(df) * 100, 2)
        }
        for metric in available_metrics:
            row[f'avg_{metric}'] = round(cluster_df[metric].mean(), 3)
        overview_data.append(row)

    overview_df = pd.DataFrame(overview_data)
    save_dataframe(overview_df, f"{output_dir}/cluster_overview.csv", "Cluster overview")
    file_count += 1

    # 2. Pairwise statistical comparisons
    clusters = sorted(df['cluster'].unique())
    pairwise_comparisons = []

    for i, c1 in enumerate(clusters):
        for j, c2 in enumerate(clusters):
            if i < j:  # Only compare each pair once
                c1_df = df[df['cluster'] == c1]
                c2_df = df[df['cluster'] == c2]

                for feature in feature_cols:
                    try:
                        t_stat, p_value = stats.ttest_ind(
                            c1_df[feature].dropna(),
                            c2_df[feature].dropna()
                        )

                        # Cohen's d
                        pooled_std = np.sqrt((c1_df[feature].std()**2 + c2_df[feature].std()**2) / 2)
                        cohens_d = (c1_df[feature].mean() - c2_df[feature].mean()) / pooled_std if pooled_std > 0 else 0

                        pairwise_comparisons.append({
                            'cluster_a': c1,
                            'cluster_b': c2,
                            'feature': feature,
                            'cluster_a_mean': round(c1_df[feature].mean(), 4),
                            'cluster_b_mean': round(c2_df[feature].mean(), 4),
                            'difference': round(c1_df[feature].mean() - c2_df[feature].mean(), 4),
                            't_statistic': round(t_stat, 4),
                            'p_value': round(p_value, 6),
                            'cohens_d': round(cohens_d, 4),
                            'significant': p_value < 0.05
                        })
                    except:
                        pass  # Skip if comparison fails

    if pairwise_comparisons:
        pairwise_df = pd.DataFrame(pairwise_comparisons)
        pairwise_df = pairwise_df.sort_values(['cluster_a', 'cluster_b', 'p_value'])
        save_dataframe(pairwise_df, f"{output_dir}/pairwise_statistics.csv", "Pairwise statistics")
        file_count += 1

    # 3. Genre overlap analysis
    if 'top_genre' in df.columns:
        genre_overlap = []
        for cluster in clusters:
            cluster_genres = set(df[df['cluster'] == cluster]['top_genre'].dropna())
            all_genres = set(df['top_genre'].dropna())

            # Count shared genres with other clusters
            shared_count = 0
            for other_cluster in clusters:
                if other_cluster != cluster:
                    other_genres = set(df[df['cluster'] == other_cluster]['top_genre'].dropna())
                    shared_count += len(cluster_genres & other_genres)

            genre_overlap.append({
                'cluster': cluster,
                'total_genres': len(cluster_genres),
                'unique_genres': len(cluster_genres - set(df[df['cluster'] != cluster]['top_genre'].dropna())),
                'avg_shared_genres': round(shared_count / (len(clusters) - 1), 2) if len(clusters) > 1 else 0
            })

        overlap_df = pd.DataFrame(genre_overlap)
        save_dataframe(overlap_df, f"{output_dir}/genre_overlap.csv", "Genre overlap")
        file_count += 1

    # 4. Top genres per cluster
    if 'top_genre' in df.columns:
        top_genres_records = []
        for cluster in clusters:
            cluster_df = df[df['cluster'] == cluster]
            genre_counts = Counter(cluster_df['top_genre'].dropna())
            for rank, (genre, count) in enumerate(genre_counts.most_common(10), 1):
                top_genres_records.append({
                    'cluster': cluster,
                    'rank': rank,
                    'genre': genre,
                    'count': count,
                    'percentage': round(count / len(cluster_df) * 100, 2)
                })

        top_genres_df = pd.DataFrame(top_genres_records)
        save_dataframe(top_genres_df, f"{output_dir}/top_genres_per_cluster.csv", "Top genres per cluster")
        file_count += 1

    # 5. Top artists per cluster
    if 'artist' in df.columns:
        top_artists_records = []
        for cluster in clusters:
            cluster_df = df[df['cluster'] == cluster]
            artist_counts = Counter(cluster_df['artist'].dropna())
            for rank, (artist, count) in enumerate(artist_counts.most_common(10), 1):
                top_artists_records.append({
                    'cluster': cluster,
                    'rank': rank,
                    'artist': artist,
                    'count': count,
                    'percentage': round(count / len(cluster_df) * 100, 2)
                })

        top_artists_df = pd.DataFrame(top_artists_records)
        save_dataframe(top_artists_df, f"{output_dir}/top_artists_per_cluster.csv", "Top artists per cluster")
        file_count += 1

    # 6. Sample songs per cluster
    if 'track_name' in df.columns and 'artist' in df.columns:
        sample_songs = []
        for cluster in clusters:
            cluster_df = df[df['cluster'] == cluster]
            samples = cluster_df.sample(min(10, len(cluster_df)))[['track_name', 'artist']]
            samples['cluster'] = cluster
            sample_songs.append(samples)

        if sample_songs:
            samples_df = pd.concat(sample_songs, ignore_index=True)
            samples_df = samples_df[['cluster', 'track_name', 'artist']]
            save_dataframe(samples_df, f"{output_dir}/sample_songs_per_cluster.csv", "Sample songs per cluster")
            file_count += 1

    return file_count


# ============================================================================
# EDA DATA EXPORT
# ============================================================================

def export_eda_data(df: pd.DataFrame, output_dir: str) -> int:
    """Export exploratory data analysis data"""
    file_count = 0

    # Get numeric feature columns (excluding coordinates and cluster)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if not c.startswith(('x', 'y', 'z', 'cluster'))]

    # 1. Feature statistics
    feature_stats = []
    for feature in feature_cols:
        feature_stats.append({
            'feature': feature,
            'mean': round(df[feature].mean(), 4),
            'std': round(df[feature].std(), 4),
            'min': round(df[feature].min(), 4),
            'max': round(df[feature].max(), 4),
            'range': round(df[feature].max() - df[feature].min(), 4),
            'relative_importance': round(df[feature].std() / df[feature].mean(), 4) if df[feature].mean() != 0 else 0
        })

    stats_df = pd.DataFrame(feature_stats)
    save_dataframe(stats_df, f"{output_dir}/feature_stats.csv", "Feature statistics")
    file_count += 1

    # 2. Embedding metrics
    embedding_metrics = {
        'num_dimensions': len(feature_cols),
        'avg_vector_magnitude': float(np.linalg.norm(df[feature_cols].values, axis=1).mean()),
        'avg_feature_range': float(stats_df['range'].mean())
    }
    save_json(embedding_metrics, f"{output_dir}/embedding_metrics.json")
    file_count += 1

    # 3. Audio extremes
    audio_features = ['bpm', 'danceability', 'valence', 'arousal']
    available_audio = [f for f in audio_features if f in df.columns]

    audio_extremes = []
    for feature in available_audio:
        # Top 5
        top_5 = df.nlargest(5, feature)[['track_name', 'artist', feature] if 'track_name' in df.columns else [feature]]
        for idx, row in top_5.iterrows():
            audio_extremes.append({
                'feature': feature,
                'extreme': 'highest',
                'track_name': row.get('track_name', 'N/A'),
                'artist': row.get('artist', 'N/A'),
                'value': round(row[feature], 3)
            })

        # Bottom 5
        bottom_5 = df.nsmallest(5, feature)[['track_name', 'artist', feature] if 'track_name' in df.columns else [feature]]
        for idx, row in bottom_5.iterrows():
            audio_extremes.append({
                'feature': feature,
                'extreme': 'lowest',
                'track_name': row.get('track_name', 'N/A'),
                'artist': row.get('artist', 'N/A'),
                'value': round(row[feature], 3)
            })

    if audio_extremes:
        extremes_df = pd.DataFrame(audio_extremes)
        save_dataframe(extremes_df, f"{output_dir}/audio_extremes.csv", "Audio extremes")
        file_count += 1

    # 4. Mood extremes
    mood_features = ['mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed', 'mood_party']
    available_moods = [m for m in mood_features if m in df.columns]

    mood_extremes = []
    for mood in available_moods:
        top_5 = df.nlargest(5, mood)[['track_name', 'artist', mood] if 'track_name' in df.columns else [mood]]
        for idx, row in top_5.iterrows():
            mood_extremes.append({
                'mood': mood,
                'track_name': row.get('track_name', 'N/A'),
                'artist': row.get('artist', 'N/A'),
                'value': round(row[mood], 3)
            })

    if mood_extremes:
        mood_df = pd.DataFrame(mood_extremes)
        save_dataframe(mood_df, f"{output_dir}/mood_extremes.csv", "Mood extremes")
        file_count += 1

    # 5. Vocal extremes
    if 'instrumentalness' in df.columns:
        vocal_extremes = []

        # Most vocal
        most_vocal = df.nsmallest(5, 'instrumentalness')[['track_name', 'artist', 'instrumentalness'] if 'track_name' in df.columns else ['instrumentalness']]
        for idx, row in most_vocal.iterrows():
            vocal_extremes.append({
                'category': 'most_vocal',
                'track_name': row.get('track_name', 'N/A'),
                'artist': row.get('artist', 'N/A'),
                'instrumentalness': round(row['instrumentalness'], 3)
            })

        # Most instrumental
        most_instrumental = df.nlargest(5, 'instrumentalness')[['track_name', 'artist', 'instrumentalness'] if 'track_name' in df.columns else ['instrumentalness']]
        for idx, row in most_instrumental.iterrows():
            vocal_extremes.append({
                'category': 'most_instrumental',
                'track_name': row.get('track_name', 'N/A'),
                'artist': row.get('artist', 'N/A'),
                'instrumentalness': round(row['instrumentalness'], 3)
            })

        vocal_df = pd.DataFrame(vocal_extremes)
        save_dataframe(vocal_df, f"{output_dir}/vocal_extremes.csv", "Vocal extremes")
        file_count += 1

    # 6. Genre statistics
    if 'top_genre' in df.columns:
        genre_counts = Counter(df['top_genre'].dropna())
        genre_stats = pd.DataFrame([
            {'genre': genre, 'count': count, 'percentage': round(count / len(df) * 100, 2)}
            for genre, count in genre_counts.most_common()
        ])
        save_dataframe(genre_stats, f"{output_dir}/genre_stats.csv", "Genre statistics")
        file_count += 1

    # 7. Genre-cluster matrix
    if 'top_genre' in df.columns:
        # Get top 20 genres
        top_genres = [genre for genre, _ in Counter(df['top_genre'].dropna()).most_common(20)]

        genre_cluster_matrix = []
        for genre in top_genres:
            row = {'genre': genre}
            for cluster in sorted(df['cluster'].unique()):
                cluster_df = df[df['cluster'] == cluster]
                count = len(cluster_df[cluster_df['top_genre'] == genre])
                row[f'cluster_{cluster}'] = count
            genre_cluster_matrix.append(row)

        matrix_df = pd.DataFrame(genre_cluster_matrix)
        save_dataframe(matrix_df, f"{output_dir}/genre_cluster_matrix.csv", "Genre-cluster matrix")
        file_count += 1

    # 8. Genre purity (genre_ladder)
    if 'genre_ladder' in df.columns and 'track_name' in df.columns:
        purity_df = df[['track_name', 'artist', 'top_genre', 'genre_ladder']].copy()
        purity_df = purity_df.sort_values('genre_ladder')
        save_dataframe(purity_df, f"{output_dir}/genre_purity.csv", "Genre purity")
        file_count += 1

    # 9. Language statistics
    if 'language' in df.columns:
        lang_counts = Counter(df['language'].dropna())
        lang_stats = pd.DataFrame([
            {'language': lang, 'count': count, 'percentage': round(count / len(df) * 100, 2)}
            for lang, count in lang_counts.most_common()
        ])
        save_dataframe(lang_stats, f"{output_dir}/language_stats.csv", "Language statistics")
        file_count += 1

    # 10. Temporal data exports
    file_count += export_temporal_data(df, f"{output_dir}/temporal")

    return file_count


def export_temporal_data(df: pd.DataFrame, output_dir: str) -> int:
    """Export temporal analysis data"""
    file_count = 0

    # Check if temporal columns exist
    if 'added_at' not in df.columns:
        print("  ⚠️  No temporal data (added_at column missing)")
        return 0

    # Convert to datetime
    df['added_at_dt'] = pd.to_datetime(df['added_at'], errors='coerce')
    df_temporal = df.dropna(subset=['added_at_dt']).copy()

    if len(df_temporal) == 0:
        print("  ⚠️  No valid temporal data")
        return 0

    # 1. Temporal metrics
    date_range_days = (df_temporal['added_at_dt'].max() - df_temporal['added_at_dt'].min()).days

    # Calculate song age at addition if release_date available
    median_age = "N/A"
    if 'release_date' in df.columns:
        df_temporal['release_dt'] = pd.to_datetime(df_temporal['release_date'], errors='coerce')
        df_temporal['age_at_add'] = (df_temporal['added_at_dt'] - df_temporal['release_dt']).dt.days / 365.25
        median_age = round(df_temporal['age_at_add'].median(), 1) if not df_temporal['age_at_add'].isna().all() else "N/A"

    # Most active month
    df_temporal['month'] = df_temporal['added_at_dt'].dt.to_period('M')
    most_active_month = df_temporal['month'].value_counts().index[0].strftime('%Y-%m') if len(df_temporal) > 0 else "N/A"
    most_active_count = int(df_temporal['month'].value_counts().iloc[0]) if len(df_temporal) > 0 else 0

    temporal_metrics = {
        'date_range_days': date_range_days,
        'earliest_add': str(df_temporal['added_at_dt'].min().date()),
        'latest_add': str(df_temporal['added_at_dt'].max().date()),
        'median_song_age_at_add_years': median_age,
        'most_active_month': most_active_month,
        'most_active_month_count': most_active_count
    }
    save_json(temporal_metrics, f"{output_dir}/temporal_metrics.json")
    file_count += 1

    # 2. Library growth (cumulative additions)
    df_temporal_sorted = df_temporal.sort_values('added_at_dt')
    df_temporal_sorted['cumulative_count'] = range(1, len(df_temporal_sorted) + 1)
    growth_df = df_temporal_sorted[['added_at_dt', 'cumulative_count']].copy()
    growth_df.columns = ['date', 'cumulative_songs']
    save_dataframe(growth_df, f"{output_dir}/library_growth.csv", "Library growth")
    file_count += 1

    # 3. Monthly additions
    monthly_counts = df_temporal.groupby(df_temporal['added_at_dt'].dt.to_period('M')).size()
    monthly_df = pd.DataFrame({
        'month': monthly_counts.index.strftime('%Y-%m'),
        'songs_added': monthly_counts.values
    })
    save_dataframe(monthly_df, f"{output_dir}/monthly_additions.csv", "Monthly additions")
    file_count += 1

    # 4. Song age distribution (when added)
    if 'age_at_add' in df_temporal.columns:
        age_dist = df_temporal[['track_name', 'artist', 'age_at_add']].dropna()
        save_dataframe(age_dist, f"{output_dir}/song_age_distribution.csv", "Song age distribution")
        file_count += 1

    # 5. Release year distribution
    if 'release_date' in df.columns:
        df_temporal['release_year'] = pd.to_datetime(df_temporal['release_date'], errors='coerce').dt.year
        year_dist = df_temporal['release_year'].value_counts().sort_index()
        year_df = pd.DataFrame({
            'year': year_dist.index,
            'count': year_dist.values
        })
        save_dataframe(year_df, f"{output_dir}/release_year_distribution.csv", "Release year distribution")
        file_count += 1

        # 6. Decade distribution
        df_temporal['decade'] = (df_temporal['release_year'] // 10 * 10).astype('Int64')
        decade_dist = df_temporal['decade'].value_counts().sort_index()
        decade_df = pd.DataFrame({
            'decade': decade_dist.index,
            'count': decade_dist.values,
            'percentage': (decade_dist.values / len(df_temporal) * 100).round(2)
        })
        save_dataframe(decade_df, f"{output_dir}/decade_distribution.csv", "Decade distribution")
        file_count += 1

    # 7. Cluster evolution (4 time periods)
    quartiles = pd.qcut(df_temporal['added_at_dt'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    df_temporal['period'] = quartiles

    cluster_evolution = []
    for period in ['Q1', 'Q2', 'Q3', 'Q4']:
        period_df = df_temporal[df_temporal['period'] == period]
        total = len(period_df)
        for cluster in sorted(df['cluster'].unique()):
            count = len(period_df[period_df['cluster'] == cluster])
            cluster_evolution.append({
                'period': period,
                'cluster': cluster,
                'count': count,
                'percentage': round(count / total * 100, 2) if total > 0 else 0
            })

    evolution_df = pd.DataFrame(cluster_evolution)
    save_dataframe(evolution_df, f"{output_dir}/cluster_evolution.csv", "Cluster evolution")
    file_count += 1

    # 8. Cluster trends (rolling window)
    if len(df_temporal_sorted) >= 30:
        window_size = 30
        trends = []
        for i in range(len(df_temporal_sorted) - window_size + 1):
            window = df_temporal_sorted.iloc[i:i+window_size]
            window_date = window['added_at_dt'].iloc[-1]
            for cluster in sorted(df['cluster'].unique()):
                count = len(window[window['cluster'] == cluster])
                trends.append({
                    'date': window_date,
                    'cluster': cluster,
                    'percentage': round(count / window_size * 100, 2)
                })

        trends_df = pd.DataFrame(trends)
        save_dataframe(trends_df, f"{output_dir}/cluster_trends.csv", "Cluster trends")
        file_count += 1

    # 9. Mood trends (rolling window)
    mood_cols = ['mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed', 'mood_party']
    available_moods = [m for m in mood_cols if m in df_temporal.columns]

    if len(df_temporal_sorted) >= 30 and available_moods:
        window_size = 30
        mood_trends = []
        for i in range(len(df_temporal_sorted) - window_size + 1):
            window = df_temporal_sorted.iloc[i:i+window_size]
            window_date = window['added_at_dt'].iloc[-1]
            row = {'date': window_date}
            for mood in available_moods:
                row[mood] = round(window[mood].mean(), 4)
            mood_trends.append(row)

        mood_trends_df = pd.DataFrame(mood_trends)
        save_dataframe(mood_trends_df, f"{output_dir}/mood_trends.csv", "Mood trends")
        file_count += 1

    # 10. Genre trends (quarterly)
    if 'top_genre' in df_temporal.columns:
        df_temporal['quarter'] = df_temporal['added_at_dt'].dt.to_period('Q')

        # Get top 5 genre families
        top_genres = [g for g, _ in Counter(df_temporal['top_genre'].dropna()).most_common(5)]

        genre_trends = []
        for quarter in df_temporal['quarter'].unique():
            quarter_df = df_temporal[df_temporal['quarter'] == quarter]
            total = len(quarter_df)
            row = {'quarter': str(quarter)}
            for genre in top_genres:
                count = len(quarter_df[quarter_df['top_genre'] == genre])
                row[genre] = round(count / total * 100, 2) if total > 0 else 0
            genre_trends.append(row)

        genre_trends_df = pd.DataFrame(genre_trends)
        save_dataframe(genre_trends_df, f"{output_dir}/genre_trends.csv", "Genre trends")
        file_count += 1

    # 11. Temporal extremes
    extremes = []

    # Oldest/newest releases
    if 'release_year' in df_temporal.columns:
        oldest = df_temporal.nsmallest(5, 'release_year')[['track_name', 'artist', 'release_year']]
        for idx, row in oldest.iterrows():
            extremes.append({
                'category': 'oldest_release',
                'track_name': row['track_name'],
                'artist': row.get('artist', 'N/A'),
                'year': int(row['release_year'])
            })

        newest = df_temporal.nlargest(5, 'release_year')[['track_name', 'artist', 'release_year']]
        for idx, row in newest.iterrows():
            extremes.append({
                'category': 'newest_release',
                'track_name': row['track_name'],
                'artist': row.get('artist', 'N/A'),
                'year': int(row['release_year'])
            })

    # First/last added
    first_added = df_temporal.nsmallest(5, 'added_at_dt')[['track_name', 'artist', 'added_at_dt']]
    for idx, row in first_added.iterrows():
        extremes.append({
            'category': 'first_added',
            'track_name': row['track_name'],
            'artist': row.get('artist', 'N/A'),
            'date': str(row['added_at_dt'].date())
        })

    last_added = df_temporal.nlargest(5, 'added_at_dt')[['track_name', 'artist', 'added_at_dt']]
    for idx, row in last_added.iterrows():
        extremes.append({
            'category': 'last_added',
            'track_name': row['track_name'],
            'artist': row.get('artist', 'N/A'),
            'date': str(row['added_at_dt'].date())
        })

    if extremes:
        extremes_df = pd.DataFrame(extremes)
        save_dataframe(extremes_df, f"{output_dir}/temporal_extremes.csv", "Temporal extremes")
        file_count += 1

    return file_count


# ============================================================================
# LYRICS DATA EXPORT
# ============================================================================

def export_lyrics_data(df: pd.DataFrame, lyrics_dir: str, output_dir: str) -> int:
    """Export lyrics analysis data"""
    file_count = 0

    # Check if lyrics directory exists
    lyrics_path = Path(lyrics_dir)
    if not lyrics_path.exists():
        print(f"  ⚠️  Lyrics directory not found: {lyrics_dir}")
        return 0

    # Collect lyrics for all tracks
    lyrics_data = []
    for idx, row in df.iterrows():
        track_name = row.get('track_name', '')
        artist = row.get('artist', '')

        # Try to find lyrics file (normalize filename)
        normalized_name = f"{track_name}_{artist}".replace('/', '_').replace('\\', '_')[:200]
        lyric_files = list(lyrics_path.glob(f"{normalized_name}*.txt"))

        if lyric_files:
            try:
                with open(lyric_files[0], 'r', encoding='utf-8') as f:
                    lyrics = f.read()
                    lyrics_data.append({
                        'track_name': track_name,
                        'artist': artist,
                        'lyrics': lyrics,
                        'cluster': row.get('cluster', -1)
                    })
            except:
                pass

    if len(lyrics_data) == 0:
        print(f"  ⚠️  No lyrics found in {lyrics_dir}")
        return 0

    lyrics_df = pd.DataFrame(lyrics_data)
    print(f"  Found lyrics for {len(lyrics_df)} tracks")

    # 1. TF-IDF Analysis
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

        # Unigrams
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words=list(ENGLISH_STOP_WORDS),
            lowercase=True,
            ngram_range=(1, 1)
        )
        tfidf_matrix = vectorizer.fit_transform(lyrics_df['lyrics'])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1

        unigrams_df = pd.DataFrame({
            'word': feature_names,
            'tfidf_score': tfidf_scores
        }).sort_values('tfidf_score', ascending=False)
        save_dataframe(unigrams_df, f"{output_dir}/tfidf_unigrams.csv", "TF-IDF unigrams")
        file_count += 1

        # Bigrams
        vectorizer_bi = TfidfVectorizer(
            max_features=100,
            stop_words=list(ENGLISH_STOP_WORDS),
            lowercase=True,
            ngram_range=(2, 2)
        )
        tfidf_bi = vectorizer_bi.fit_transform(lyrics_df['lyrics'])
        bi_names = vectorizer_bi.get_feature_names_out()
        bi_scores = tfidf_bi.sum(axis=0).A1

        bigrams_df = pd.DataFrame({
            'phrase': bi_names,
            'tfidf_score': bi_scores
        }).sort_values('tfidf_score', ascending=False)
        save_dataframe(bigrams_df, f"{output_dir}/tfidf_bigrams.csv", "TF-IDF bigrams")
        file_count += 1

        # Trigrams
        vectorizer_tri = TfidfVectorizer(
            max_features=100,
            stop_words=list(ENGLISH_STOP_WORDS),
            lowercase=True,
            ngram_range=(3, 3)
        )
        tfidf_tri = vectorizer_tri.fit_transform(lyrics_df['lyrics'])
        tri_names = vectorizer_tri.get_feature_names_out()
        tri_scores = tfidf_tri.sum(axis=0).A1

        trigrams_df = pd.DataFrame({
            'phrase': tri_names,
            'tfidf_score': tri_scores
        }).sort_values('tfidf_score', ascending=False)
        save_dataframe(trigrams_df, f"{output_dir}/tfidf_trigrams.csv", "TF-IDF trigrams")
        file_count += 1

    except Exception as e:
        print(f"  ⚠️  TF-IDF analysis failed: {e}")

    # 2. Sentiment analysis (if columns exist)
    sentiment_cols = ['lyric_valence', 'lyric_arousal']
    available_sentiment = [c for c in sentiment_cols if c in df.columns]

    if available_sentiment:
        sentiment_tracks = df[['track_name', 'artist'] + available_sentiment].copy()
        save_dataframe(sentiment_tracks, f"{output_dir}/sentiment_all_tracks.csv", "Sentiment all tracks")
        file_count += 1

        # Sentiment metrics
        sentiment_metrics = {}
        for col in available_sentiment:
            sentiment_metrics[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std())
            }
        save_json(sentiment_metrics, f"{output_dir}/sentiment_metrics.json")
        file_count += 1

        # Sentiment extremes
        if 'lyric_valence' in df.columns:
            extremes = []

            # Most positive
            positive = df.nlargest(5, 'lyric_valence')[['track_name', 'artist', 'lyric_valence']]
            for idx, row in positive.iterrows():
                extremes.append({
                    'category': 'most_positive',
                    'track_name': row['track_name'],
                    'artist': row.get('artist', 'N/A'),
                    'valence': round(row['lyric_valence'], 3)
                })

            # Most negative
            negative = df.nsmallest(5, 'lyric_valence')[['track_name', 'artist', 'lyric_valence']]
            for idx, row in negative.iterrows():
                extremes.append({
                    'category': 'most_negative',
                    'track_name': row['track_name'],
                    'artist': row.get('artist', 'N/A'),
                    'valence': round(row['lyric_valence'], 3)
                })

            extremes_df = pd.DataFrame(extremes)
            save_dataframe(extremes_df, f"{output_dir}/sentiment_extremes.csv", "Sentiment extremes")
            file_count += 1

    # 3. Complexity metrics (if columns exist)
    complexity_cols = ['lyric_vocabulary', 'lyric_repetition']
    available_complexity = [c for c in complexity_cols if c in df.columns]

    if available_complexity:
        complexity_tracks = df[['track_name', 'artist'] + available_complexity].copy()
        save_dataframe(complexity_tracks, f"{output_dir}/complexity_all_tracks.csv", "Complexity all tracks")
        file_count += 1

        # Complexity metrics
        complexity_metrics = {}
        for col in available_complexity:
            complexity_metrics[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std())
            }
        save_json(complexity_metrics, f"{output_dir}/complexity_metrics.json")
        file_count += 1

        # Complexity extremes
        if 'lyric_vocabulary' in df.columns:
            extremes = []

            # Most complex
            complex_tracks = df.nlargest(5, 'lyric_vocabulary')[['track_name', 'artist', 'lyric_vocabulary']]
            for idx, row in complex_tracks.iterrows():
                extremes.append({
                    'category': 'most_complex',
                    'track_name': row['track_name'],
                    'artist': row.get('artist', 'N/A'),
                    'vocabulary': round(row['lyric_vocabulary'], 3)
                })

            # Simplest
            simple_tracks = df.nsmallest(5, 'lyric_vocabulary')[['track_name', 'artist', 'lyric_vocabulary']]
            for idx, row in simple_tracks.iterrows():
                extremes.append({
                    'category': 'simplest',
                    'track_name': row['track_name'],
                    'artist': row.get('artist', 'N/A'),
                    'vocabulary': round(row['lyric_vocabulary'], 3)
                })

            extremes_df = pd.DataFrame(extremes)
            save_dataframe(extremes_df, f"{output_dir}/complexity_extremes.csv", "Complexity extremes")
            file_count += 1

    # 4. Repeated phrases (simple approach: find common lines)
    if len(lyrics_df) > 0:
        all_lines = []
        for lyrics in lyrics_df['lyrics']:
            lines = [line.strip().lower() for line in lyrics.split('\n') if line.strip() and len(line.strip()) > 10]
            all_lines.extend(lines)

        line_counts = Counter(all_lines)
        # Only keep lines that appear more than once
        repeated = [(line, count) for line, count in line_counts.most_common(20) if count > 1]

        if repeated:
            repeated_df = pd.DataFrame(repeated, columns=['phrase', 'count'])
            save_dataframe(repeated_df, f"{output_dir}/repeated_phrases.csv", "Repeated phrases")
            file_count += 1

    return file_count


# ============================================================================
# SUB-CLUSTER DATA EXPORT
# ============================================================================

def export_all_subclusters(subclusters_dir: str, output_dir: str) -> int:
    """Export all saved sub-cluster analyses"""
    file_count = 0

    subclusters_path = Path(subclusters_dir)
    if not subclusters_path.exists():
        print(f"  ⚠️  Sub-clusters directory not found: {subclusters_dir}")
        return 0

    # Find all pickle files
    pkl_files = list(subclusters_path.glob("*.pkl"))

    if not pkl_files:
        print(f"  ⚠️  No sub-cluster files found in {subclusters_dir}")
        return 0

    print(f"  Found {len(pkl_files)} sub-cluster file(s)")

    for pkl_file in pkl_files:
        try:
            # Load sub-cluster data
            with open(pkl_file, 'rb') as f:
                subcluster_data = pickle.load(f)

            # Create output directory for this sub-cluster
            subdir_name = pkl_file.stem  # e.g., "parent_0_2025-12-28_k-means"
            subdir = f"{output_dir}/{subdir_name}"
            Path(subdir).mkdir(parents=True, exist_ok=True)

            print(f"  Exporting {subdir_name}...")

            # Extract data (key is 'subcluster_df' not 'parent_df')
            parent_df = subcluster_data.get('subcluster_df')
            if parent_df is None or len(parent_df) == 0:
                print(f"    ⚠️  No subcluster_df found, skipping")
                continue

            # 1. Subcluster assignments
            if 'subcluster' in parent_df.columns:
                assignments = parent_df[['track_name', 'artist', 'subcluster']].copy()
                save_dataframe(assignments, f"{subdir}/subcluster_assignments.csv", None)
                file_count += 1

                # 1b. Subcluster assignments with vectors - SKIPPED to reduce file size
                # These embedding vectors can take up 100-500KB per subcluster

            # 2. Subcluster statistics
            if 'subcluster' in parent_df.columns:
                stats_records = []
                for sc in sorted(parent_df['subcluster'].unique()):
                    sc_df = parent_df[parent_df['subcluster'] == sc]

                    # Get top genre
                    top_genre = "N/A"
                    if 'top_genre' in sc_df.columns:
                        genre_counts = Counter(sc_df['top_genre'].dropna())
                        top_genre = genre_counts.most_common(1)[0][0] if genre_counts else "N/A"

                    # Get dominant mood
                    dominant_mood = "N/A"
                    mood_cols = ['mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed', 'mood_party']
                    available_moods = [m for m in mood_cols if m in sc_df.columns]
                    if available_moods:
                        mood_avgs = {m: sc_df[m].mean() for m in available_moods}
                        dominant_mood = max(mood_avgs, key=mood_avgs.get)

                    stats_records.append({
                        'subcluster': sc,
                        'size': len(sc_df),
                        'top_genre': top_genre,
                        'avg_bpm': round(sc_df['bpm'].mean(), 1) if 'bpm' in sc_df.columns else 0,
                        'dominant_mood': dominant_mood,
                        'avg_danceability': round(sc_df['danceability'].mean(), 3) if 'danceability' in sc_df.columns else 0
                    })

                stats_df = pd.DataFrame(stats_records)
                save_dataframe(stats_df, f"{subdir}/subcluster_stats.csv", None)
                file_count += 1

            # 3. Subcluster centroids
            if 'subcluster' in parent_df.columns:
                numeric_cols = parent_df.select_dtypes(include=[np.number]).columns
                feature_cols = [c for c in numeric_cols if not c.startswith(('x', 'y', 'z', 'subcluster', 'cluster'))]

                centroids = parent_df.groupby('subcluster')[feature_cols].mean().reset_index()
                save_dataframe(centroids, f"{subdir}/subcluster_centroids.csv", None)
                file_count += 1

            # 4. Optimal k analysis (if available)
            optimal_k_results = subcluster_data.get('optimal_k_results')
            if optimal_k_results:
                save_dataframe(optimal_k_results, f"{subdir}/optimal_k_analysis.csv", None)
                file_count += 1

            # 5. Dissimilarity matrix
            if 'subcluster' in parent_df.columns:
                numeric_cols = parent_df.select_dtypes(include=[np.number]).columns
                feature_cols = [c for c in numeric_cols if not c.startswith(('x', 'y', 'z', 'subcluster', 'cluster'))]

                centroids = parent_df.groupby('subcluster')[feature_cols].mean().values
                dissimilarity = cdist(centroids, centroids, metric='euclidean')

                subclusters = sorted(parent_df['subcluster'].unique())
                dissim_df = pd.DataFrame(
                    dissimilarity,
                    index=[f"sc_{i}" for i in subclusters],
                    columns=[f"sc_{i}" for i in subclusters]
                )
                save_dataframe(dissim_df, f"{subdir}/dissimilarity_matrix.csv", None)
                file_count += 1

            # 6. Pairwise comparisons (basic)
            if 'subcluster' in parent_df.columns and len(parent_df['subcluster'].unique()) >= 2:
                subclusters = sorted(parent_df['subcluster'].unique())
                comparisons = []

                # Just do a few key features
                test_features = ['bpm', 'danceability', 'valence', 'mood_happy']
                available_features = [f for f in test_features if f in parent_df.columns]

                for i, sc1 in enumerate(subclusters):
                    for j, sc2 in enumerate(subclusters):
                        if i < j:
                            sc1_df = parent_df[parent_df['subcluster'] == sc1]
                            sc2_df = parent_df[parent_df['subcluster'] == sc2]

                            for feature in available_features:
                                try:
                                    t_stat, p_val = stats.ttest_ind(
                                        sc1_df[feature].dropna(),
                                        sc2_df[feature].dropna()
                                    )
                                    comparisons.append({
                                        'subcluster_a': sc1,
                                        'subcluster_b': sc2,
                                        'feature': feature,
                                        'mean_a': round(sc1_df[feature].mean(), 3),
                                        'mean_b': round(sc2_df[feature].mean(), 3),
                                        'p_value': round(p_val, 6),
                                        'significant': p_val < 0.05
                                    })
                                except:
                                    pass

                if comparisons:
                    comp_df = pd.DataFrame(comparisons)
                    save_dataframe(comp_df, f"{subdir}/pairwise_comparisons.csv", None)
                    file_count += 1

            # 7. Feature weights
            feature_weights = subcluster_data.get('feature_weights', {})
            if feature_weights:
                save_json(feature_weights, f"{subdir}/feature_weights.json")
                file_count += 1

            # 8. Top defining features per sub-cluster (Cohen's d)
            if 'subcluster' in parent_df.columns:
                numeric_cols = parent_df.select_dtypes(include=[np.number]).columns
                feature_cols = [c for c in numeric_cols if not c.startswith(('x', 'y', 'z', 'subcluster', 'cluster'))]

                # Calculate Cohen's d for each feature × subcluster
                global_means = parent_df[feature_cols].mean()
                global_stds = parent_df[feature_cols].std()

                top_features_records = []
                for sc in sorted(parent_df['subcluster'].unique()):
                    sc_df = parent_df[parent_df['subcluster'] == sc]
                    sc_means = sc_df[feature_cols].mean()

                    # Calculate Cohen's d for all features
                    feature_importance = []
                    for feature in feature_cols:
                        cohens_d = (sc_means[feature] - global_means[feature]) / global_stds[feature] if global_stds[feature] > 0 else 0
                        feature_importance.append({
                            'feature': feature,
                            'cohens_d': cohens_d,
                            'abs_cohens_d': abs(cohens_d),
                            'subcluster_mean': sc_means[feature],
                            'parent_mean': global_means[feature]
                        })

                    # Get top 5 by absolute Cohen's d
                    feature_importance.sort(key=lambda x: x['abs_cohens_d'], reverse=True)
                    for rank, feat_data in enumerate(feature_importance[:5], 1):
                        top_features_records.append({
                            'subcluster': sc,
                            'rank': rank,
                            'feature': feat_data['feature'],
                            'cohens_d': round(feat_data['cohens_d'], 4),
                            'subcluster_mean': round(feat_data['subcluster_mean'], 4),
                            'parent_mean': round(feat_data['parent_mean'], 4),
                            'direction': 'higher' if feat_data['cohens_d'] > 0 else 'lower'
                        })

                if top_features_records:
                    top_features_df = pd.DataFrame(top_features_records)
                    save_dataframe(top_features_df, f"{subdir}/top_defining_features.csv", None)
                    file_count += 1

            # 9. Tracks per subcluster
            if 'subcluster' in parent_df.columns and 'track_name' in parent_df.columns:
                tracks = parent_df[['subcluster', 'track_name', 'artist']].copy()
                save_dataframe(tracks, f"{subdir}/tracks_per_subcluster.csv", None)
                file_count += 1

        except Exception as e:
            print(f"    ⚠️  Failed to export {pkl_file.name}: {e}")

    return file_count


# ============================================================================
# VISUALIZATION IMAGE EXPORT (Optional)
# ============================================================================

def export_visualization_images(df: pd.DataFrame, output_dir: str, subclusters_dir: str) -> int:
    """Export key visualization images (PNG)"""
    file_count = 0

    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        print("  Generating visualization images...")

        # Note: Image generation is complex and would require:
        # 1. Importing and calling visualization functions from components
        # 2. Using plotly.io.write_image (requires kaleido)
        # 3. Handling 3D UMAP plots, heatmaps, radar charts, word clouds

        # This is a placeholder - full implementation would be extensive
        print("  ⚠️  Image generation not fully implemented yet")
        print("  ⚠️  To generate images, use --skip-images flag and export data only")

        return file_count

    except ImportError as e:
        print(f"  ⚠️  Cannot generate images: missing dependencies ({e})")
        return 0


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main export function - data-first approach"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Export all clustering data and visualizations for sharing with Claude',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analysis/export_all_data.py
  python analysis/export_all_data.py --skip-images
  python analysis/export_all_data.py --mode audio --output-dir ~/Desktop/export
        """
    )
    parser.add_argument('--data-file', default='analysis/outputs/analysis_data.pkl',
                        help='Path to analysis data pickle file')
    parser.add_argument('--mode', default='combined', choices=['combined', 'audio', 'lyrics'],
                        help='Clustering mode to export')
    parser.add_argument('--output-dir', default='analysis/outputs/raw',
                        help='Output directory for exported files')
    parser.add_argument('--subclusters-dir', default='analysis/outputs/subclusters',
                        help='Directory containing saved sub-cluster files')
    parser.add_argument('--lyrics-dir', default='lyrics/temp',
                        help='Directory containing lyrics files')
    parser.add_argument('--skip-images', action='store_true',
                        help='Skip PNG image generation (faster, data-only export)')
    parser.add_argument('--skip-full-dataset', action='store_true',
                        help='Skip full_dataset.csv export (saves ~10MB, use for LLM uploads)')
    parser.add_argument('--llm-bundle', action='store_true',
                        help='Create llm_bundle.txt with key files concatenated for easy LLM upload (~5K tokens)')
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 80)
    print("Spotify Clustering Data Export - Data-First Approach")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data file: {args.data_file}")
    print(f"  Mode: {args.mode}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Skip images: {args.skip_images}")
    print()

    # Load data
    print("Loading analysis data...")
    try:
        all_data = loaders.load_analysis_data(args.data_file)
        df = all_data[args.mode]['dataframe'].copy()
        pca_features = all_data[args.mode].get('pca_features')
        metadata = all_data.get('metadata', {})

        print(f"  ✓ Loaded {len(df)} tracks, {df['cluster'].nunique()} clusters")
        if pca_features is not None:
            print(f"  ✓ PCA features: {pca_features.shape}")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return 1

    # Setup output directories
    print("\nSetting up output directories...")
    setup_output_directories(args.output_dir)
    print(f"  ✓ Created directory structure in {args.output_dir}")

    file_count = {'data': 0, 'images': 0}

    # Export core data
    print("\n[1/8] Exporting core data...")
    file_count['data'] += export_core_data(df, pca_features, f"{args.output_dir}/data/core", skip_full_dataset=args.skip_full_dataset)

    print("\n[2/8] Exporting overview data...")
    file_count['data'] += export_overview_data(df, f"{args.output_dir}/data/overview")

    print("\n[3/8] Exporting feature importance data...")
    file_count['data'] += export_feature_importance_data(df, f"{args.output_dir}/data/feature_importance")

    print("\n[4/8] Exporting cluster comparison data...")
    file_count['data'] += export_cluster_comparison_data(df, f"{args.output_dir}/data/cluster_comparison")

    print("\n[5/8] Exporting EDA data...")
    file_count['data'] += export_eda_data(df, f"{args.output_dir}/data/eda")

    print("\n[6/8] Exporting lyrics data...")
    file_count['data'] += export_lyrics_data(df, args.lyrics_dir, f"{args.output_dir}/data/lyrics")

    print("\n[7/8] Exporting sub-cluster data...")
    file_count['data'] += export_all_subclusters(args.subclusters_dir, f"{args.output_dir}/data/subclusters")

    # Generate visualization images (optional)
    if not args.skip_images:
        print("\n[8/8] Generating visualization images...")
        file_count['images'] = export_visualization_images(
            df, f"{args.output_dir}/plots", args.subclusters_dir
        )
    else:
        print("\n[8/8] Skipping image generation (--skip-images)")

    # Generate metadata
    duration = time.time() - start_time

    # Update metadata with actual counts
    source_metadata = {
        'audio_backend': metadata.get('audio_backend', 'unknown'),
        'total_songs': len(df),
        'num_clusters': df['cluster'].nunique()
    }

    if 'added_at' in df.columns:
        source_metadata['added_at'] = {
            'min': str(df['added_at'].min()),
            'max': str(df['added_at'].max())
        }

    metadata_export = generate_metadata(args, source_metadata, duration, file_count)

    # Calculate total size
    total_size_bytes = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(args.output_dir)
        for f in files
    )
    total_size_mb = total_size_bytes / (1024 * 1024)
    metadata_export['files_exported']['total_size_mb'] = round(total_size_mb, 2)

    save_json(metadata_export, f"{args.output_dir}/metadata.json")

    # Summary
    print("\n" + "=" * 80)
    print("✅ Export Complete!")
    print("=" * 80)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Data files exported: {file_count['data']}")
    print(f"🖼️  Image files exported: {file_count['images']}")
    print(f"💾 Total size: {total_size_mb:.1f} MB")
    print(f"⏱️  Duration: {duration:.1f}s")
    print()
    print("📋 Next steps:")
    print("  1. Review exported files in analysis/outputs/raw/")
    print("  2. Check metadata.json for export summary")
    print("  3. Share the entire raw/ directory with Claude for analysis")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
