"""Sub-clustering pipeline and optimization functions."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import umap
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, Birch, KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from .constants import FEATURE_WEIGHT_INDICES
from .features import apply_subcluster_weights

logger = logging.getLogger(__name__)


def run_subcluster_pipeline(
    df: pd.DataFrame,
    pca_features: np.ndarray,
    parent_cluster: int,
    n_subclusters: int = 2,
    algorithm: str = 'hac',
    linkage: str = 'ward',
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    eps: float = 0.5,
    min_samples: int = 5,
    feature_weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """Re-cluster songs within a single parent cluster.

    Args:
        df: Full DataFrame with 'cluster' column from main clustering
        pca_features: Full PCA features array (aligned with df rows)
        parent_cluster: Which cluster ID to sub-cluster
        n_subclusters: Number of sub-clusters to create (2-10)
        algorithm: Clustering algorithm ('hac', 'birch', 'spectral', 'k-means', 'dbscan')
        linkage: HAC linkage method ('ward', 'complete', 'average')
        umap_n_neighbors: UMAP n_neighbors parameter
        umap_min_dist: UMAP min_dist parameter
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        feature_weights: Optional dict of feature group weights

    Returns:
        Dict with subcluster_df, labels, umap_coords, scores, etc.
    """
    logger.info(f"Sub-clustering Cluster {parent_cluster} into {n_subclusters} sub-clusters")
    logger.info(f"Algorithm: {algorithm}, Linkage: {linkage}")
    if feature_weights:
        logger.info(f"Feature weights: {feature_weights}")

    # Filter to parent cluster
    cluster_mask = df['cluster'].values == parent_cluster
    subset_df = df[cluster_mask].copy()

    # Use embedding features with weights if provided
    if feature_weights:
        emb_cols = [col for col in df.columns if col.startswith('emb_')]
        if emb_cols:
            subset_features = subset_df[emb_cols].values.astype(np.float64)
            subset_features = apply_subcluster_weights(subset_features, feature_weights)
            scaler = StandardScaler()
            subset_pca = scaler.fit_transform(subset_features)
            logger.info(f"Using weighted embedding features ({len(emb_cols)} dims)")
        else:
            logger.warning("No embedding columns found, falling back to pca_features")
            subset_pca = pca_features[cluster_mask]
    else:
        subset_pca = pca_features[cluster_mask]

    n_songs = len(subset_df)
    logger.info(f"Parent cluster {parent_cluster} contains {n_songs} songs")

    if n_songs < n_subclusters:
        logger.warning(f"Cannot create {n_subclusters} sub-clusters from {n_songs} songs")
        n_subclusters = max(2, n_songs // 2)
        logger.info(f"Adjusted to {n_subclusters} sub-clusters")

    if n_songs < 2:
        logger.error(f"Cluster {parent_cluster} has fewer than 2 songs")
        return {
            'subcluster_df': subset_df,
            'subcluster_labels': np.zeros(n_songs, dtype=int),
            'umap_coords': np.zeros((n_songs, 3)),
            'parent_cluster': parent_cluster,
            'n_subclusters': 1,
            'silhouette_score': 0.0,
            'pca_features_subset': subset_pca,
        }

    # Run clustering
    clusterer = _get_subcluster_clusterer(algorithm, n_subclusters, linkage, n_songs, eps, min_samples)
    subcluster_labels = clusterer.fit_predict(subset_pca)
    subset_df['subcluster'] = subcluster_labels

    actual_n_subclusters = len(set(subcluster_labels)) - (1 if -1 in subcluster_labels else 0)
    logger.info(f"Created {actual_n_subclusters} sub-clusters")

    # Compute UMAP for visualization
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

    # Calculate silhouette score
    if actual_n_subclusters > 1 and n_songs > actual_n_subclusters:
        sil_score = silhouette_score(subset_pca, subcluster_labels)
    else:
        sil_score = 0.0

    logger.info(f"Sub-clustering complete. Silhouette score: {sil_score:.3f}")

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
        'feature_weights': feature_weights,
    }


def _get_subcluster_clusterer(algorithm: str, n_clusters: int, linkage_method: str, n_songs: int, eps: float, min_samples: int):
    """Get clusterer for sub-clustering."""
    if algorithm == 'hac':
        return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    elif algorithm == 'birch':
        return Birch(n_clusters=n_clusters, threshold=0.5, branching_factor=50)
    elif algorithm == 'spectral':
        actual_n_neighbors = min(15, n_songs - 1)
        return SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors',
            n_neighbors=actual_n_neighbors,
            assign_labels='kmeans',
            random_state=42
        )
    elif algorithm == 'k-means':
        return KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    elif algorithm == 'dbscan':
        return DBSCAN(eps=eps, min_samples=min_samples)
    else:
        logger.warning(f"Unknown algorithm '{algorithm}', defaulting to HAC")
        return AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')


def find_optimal_subclusters(
    df: pd.DataFrame,
    pca_features: np.ndarray,
    parent_cluster: int,
    max_k: int = 10,
    algorithm: str = 'hac',
    linkage_method: str = 'ward',
    feature_weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """Find the optimal number of sub-clusters using multiple quality metrics.

    Args:
        df: Full DataFrame with 'cluster' column
        pca_features: Full PCA features array
        parent_cluster: Which cluster ID to analyze
        max_k: Maximum number of sub-clusters to try
        algorithm: Clustering algorithm
        linkage_method: HAC linkage method
        feature_weights: Optional dict of feature group weights

    Returns:
        Dict with k_values, scores, optimal_k, etc.
    """
    logger.info(f"Finding optimal sub-clusters for Cluster {parent_cluster}")
    logger.info(f"Algorithm: {algorithm}, Linkage: {linkage_method}, Max k: {max_k}")
    if feature_weights:
        logger.info(f"Feature weights: {feature_weights}")

    cluster_mask = df['cluster'].values == parent_cluster
    subset_df = df[cluster_mask]

    # Use embedding features with weights if provided
    if feature_weights:
        emb_cols = [col for col in df.columns if col.startswith('emb_')]
        if emb_cols:
            subset_features = subset_df[emb_cols].values.astype(np.float64)
            subset_features = apply_subcluster_weights(subset_features, feature_weights)
            scaler = StandardScaler()
            subset_pca = scaler.fit_transform(subset_features)
        else:
            subset_pca = pca_features[cluster_mask]
    else:
        subset_pca = pca_features[cluster_mask]

    n_songs = subset_pca.shape[0]
    logger.info(f"Cluster {parent_cluster} contains {n_songs} songs")

    actual_max_k = min(max_k, n_songs - 1, 10)
    if actual_max_k < 2:
        logger.warning(f"Cluster too small for sub-clustering (n={n_songs})")
        return {
            'k_values': [],
            'silhouette_scores': [],
            'calinski_harabasz_scores': [],
            'davies_bouldin_scores': [],
            'optimal_k': 1,
            'optimal_score': 0.0,
            'parent_cluster': parent_cluster,
            'cluster_size': n_songs,
        }

    k_values = list(range(2, actual_max_k + 1))
    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []

    if algorithm == 'hac':
        logger.info("Computing HAC linkage matrix...")
        Z = linkage(subset_pca, method=linkage_method)

        for k in k_values:
            labels = fcluster(Z, k, criterion='maxclust') - 1

            if len(set(labels)) > 1:
                sil = silhouette_score(subset_pca, labels)
                ch = calinski_harabasz_score(subset_pca, labels)
                db = davies_bouldin_score(subset_pca, labels)
            else:
                sil, ch, db = 0.0, 0.0, float('inf')

            silhouette_scores.append(float(sil))
            calinski_harabasz_scores.append(float(ch))
            davies_bouldin_scores.append(float(db))
            logger.info(f"  k={k}: silhouette={sil:.3f}, CH={ch:.1f}, DB={db:.3f}")
    else:
        for k in k_values:
            if algorithm == 'birch':
                clusterer = Birch(n_clusters=k)
            elif algorithm == 'spectral':
                clusterer = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
            elif algorithm == 'k-means':
                clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
            else:
                clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')

            labels = clusterer.fit_predict(subset_pca)

            if len(set(labels)) > 1:
                sil = silhouette_score(subset_pca, labels)
                ch = calinski_harabasz_score(subset_pca, labels)
                db = davies_bouldin_score(subset_pca, labels)
            else:
                sil, ch, db = 0.0, 0.0, float('inf')

            silhouette_scores.append(float(sil))
            calinski_harabasz_scores.append(float(ch))
            davies_bouldin_scores.append(float(db))
            logger.info(f"  k={k}: silhouette={sil:.3f}, CH={ch:.1f}, DB={db:.3f}")

    if silhouette_scores:
        optimal_idx = np.argmax(silhouette_scores)
        optimal_k = k_values[optimal_idx]
        optimal_score = silhouette_scores[optimal_idx]
    else:
        optimal_k = 2
        optimal_score = 0.0

    logger.info(f"Optimal k={optimal_k} with silhouette={optimal_score:.3f}")

    return {
        'k_values': k_values,
        'silhouette_scores': silhouette_scores,
        'calinski_harabasz_scores': calinski_harabasz_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'optimal_k': optimal_k,
        'optimal_score': optimal_score,
        'parent_cluster': parent_cluster,
        'cluster_size': n_songs,
        'feature_weights': feature_weights,
    }


def auto_tune_subcluster_weights(
    df: pd.DataFrame,
    pca_features: np.ndarray,
    parent_cluster: int,
    max_k: int = 10,
    algorithm: str = 'hac',
    linkage_method: str = 'ward',
) -> Dict:
    """Automatically find the best feature weights for sub-clustering.

    Args:
        df: Full DataFrame with 'cluster' column
        pca_features: Full PCA features array
        parent_cluster: Which cluster ID to analyze
        max_k: Maximum number of sub-clusters to try
        algorithm: Clustering algorithm
        linkage_method: HAC linkage method

    Returns:
        Dict with best_preset, best_weights, best_k, best_score, all_results
    """
    logger.info(f"Auto-tuning weights for Cluster {parent_cluster}")

    weight_presets = {
        'Balanced': {
            'core_audio': 1.0, 'mood': 1.0, 'genre': 1.0, 'key': 1.0,
            'lyric_emotion': 1.0, 'lyric_content': 1.0, 'theme': 1.0,
            'language': 1.0, 'metadata': 1.0
        },
        'Audio Focus': {
            'core_audio': 1.5, 'mood': 1.5, 'genre': 1.2, 'key': 0.8,
            'lyric_emotion': 0.3, 'lyric_content': 0.3, 'theme': 0.3,
            'language': 0.3, 'metadata': 0.5
        },
        'Lyric Focus': {
            'core_audio': 0.5, 'mood': 0.5, 'genre': 0.5, 'key': 0.3,
            'lyric_emotion': 1.5, 'lyric_content': 1.5, 'theme': 1.5,
            'language': 1.0, 'metadata': 0.5
        },
        'Mood & Emotion': {
            'core_audio': 0.8, 'mood': 2.0, 'genre': 0.5, 'key': 0.3,
            'lyric_emotion': 2.0, 'lyric_content': 0.5, 'theme': 1.0,
            'language': 0.3, 'metadata': 0.3
        },
        'Genre & Style': {
            'core_audio': 1.0, 'mood': 0.8, 'genre': 2.0, 'key': 0.5,
            'lyric_emotion': 0.5, 'lyric_content': 0.5, 'theme': 1.5,
            'language': 0.8, 'metadata': 0.5
        },
        'Energy & Tempo': {
            'core_audio': 2.0, 'mood': 1.2, 'genre': 0.5, 'key': 0.3,
            'lyric_emotion': 0.8, 'lyric_content': 0.3, 'theme': 0.5,
            'language': 0.3, 'metadata': 0.3
        },
        'Theme & Content': {
            'core_audio': 0.5, 'mood': 0.8, 'genre': 0.8, 'key': 0.3,
            'lyric_emotion': 1.0, 'lyric_content': 1.5, 'theme': 2.0,
            'language': 1.0, 'metadata': 0.5
        },
        'No Lyrics': {
            'core_audio': 1.5, 'mood': 1.5, 'genre': 1.5, 'key': 1.0,
            'lyric_emotion': 0.0, 'lyric_content': 0.0, 'theme': 0.0,
            'language': 0.0, 'metadata': 1.0
        },
        'Audio + Metadata': {
            'core_audio': 1.2, 'mood': 1.0, 'genre': 1.0, 'key': 0.5,
            'lyric_emotion': 0.5, 'lyric_content': 0.3, 'theme': 0.5,
            'language': 0.5, 'metadata': 2.0
        },
    }

    all_results = []
    best_score = -1
    best_preset = None
    best_weights = None
    best_k = 2

    for preset_name, weights in weight_presets.items():
        logger.info(f"Testing preset: {preset_name}")

        try:
            result = find_optimal_subclusters(
                df=df,
                pca_features=pca_features,
                parent_cluster=parent_cluster,
                max_k=max_k,
                algorithm=algorithm,
                linkage_method=linkage_method,
                feature_weights=weights,
            )

            preset_result = {
                'preset': preset_name,
                'weights': weights,
                'optimal_k': result['optimal_k'],
                'optimal_score': result['optimal_score'],
                'all_k_scores': list(zip(result['k_values'], result['silhouette_scores'])),
            }
            all_results.append(preset_result)

            if result['optimal_score'] > best_score:
                best_score = result['optimal_score']
                best_preset = preset_name
                best_weights = weights
                best_k = result['optimal_k']

            logger.info(f"  {preset_name}: k={result['optimal_k']}, score={result['optimal_score']:.3f}")

        except Exception as e:
            logger.warning(f"Failed to test preset {preset_name}: {e}")
            all_results.append({
                'preset': preset_name,
                'weights': weights,
                'optimal_k': None,
                'optimal_score': 0.0,
                'error': str(e),
            })

    logger.info(f"Best preset: {best_preset} with k={best_k}, score={best_score:.3f}")

    return {
        'best_preset': best_preset,
        'best_weights': best_weights,
        'best_k': best_k,
        'best_score': best_score,
        'all_results': all_results,
        'parent_cluster': parent_cluster,
    }
