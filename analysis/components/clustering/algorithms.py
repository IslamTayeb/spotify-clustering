"""Clustering algorithm implementations.

This module provides unified interface for running different clustering algorithms
on prepared features.
"""

import numpy as np
from sklearn.cluster import (
    AgglomerativeClustering,
    Birch,
    SpectralClustering,
    KMeans,
    DBSCAN,
)
from typing import Dict, Any


def run_hac(features: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Run Hierarchical Agglomerative Clustering.

    Args:
        features: Feature matrix (n_samples x n_features)
        params: Dictionary with keys:
            - n_clusters: Number of clusters
            - linkage: Linkage method ('ward', 'complete', 'average', 'single')

    Returns:
        Cluster labels (n_samples,)
    """
    clusterer = AgglomerativeClustering(
        n_clusters=params["n_clusters"],
        linkage=params.get("linkage", "ward"),
    )
    return clusterer.fit_predict(features)


def run_birch(features: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Run Birch clustering.

    Args:
        features: Feature matrix (n_samples x n_features)
        params: Dictionary with keys:
            - n_clusters: Number of clusters
            - threshold: Radius of subcluster
            - branching_factor: Max subclusters per node

    Returns:
        Cluster labels (n_samples,)
    """
    clusterer = Birch(
        n_clusters=params["n_clusters"],
        threshold=params.get("threshold", 0.5),
        branching_factor=params.get("branching_factor", 50),
    )
    return clusterer.fit_predict(features)


def run_spectral(features: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Run Spectral Clustering.

    Args:
        features: Feature matrix (n_samples x n_features)
        params: Dictionary with keys:
            - n_clusters: Number of clusters
            - affinity: 'nearest_neighbors' or 'rbf'
            - n_neighbors: Number of neighbors for k-NN graph (if affinity='nearest_neighbors')
            - assign_labels: 'kmeans' or 'discretize'

    Returns:
        Cluster labels (n_samples,)
    """
    affinity = params.get("affinity", "nearest_neighbors")
    n_neighbors = params.get("n_neighbors", 15) if affinity == "nearest_neighbors" else 10

    clusterer = SpectralClustering(
        n_clusters=params["n_clusters"],
        affinity=affinity,
        n_neighbors=n_neighbors,
        assign_labels=params.get("assign_labels", "kmeans"),
        random_state=42,
    )
    return clusterer.fit_predict(features)


def run_kmeans(features: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Run K-Means clustering.

    Args:
        features: Feature matrix (n_samples x n_features)
        params: Dictionary with keys:
            - n_clusters: Number of clusters
            - init: Initialization method ('k-means++' or 'random')

    Returns:
        Cluster labels (n_samples,)
    """
    clusterer = KMeans(
        n_clusters=params["n_clusters"],
        init=params.get("init", "k-means++"),
        n_init=10,
        random_state=42,
    )
    return clusterer.fit_predict(features)


def run_dbscan(features: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Run DBSCAN clustering (density-based).

    Args:
        features: Feature matrix (n_samples x n_features)
        params: Dictionary with keys:
            - eps: Maximum distance between two samples for neighborhood
            - min_samples: Min samples in neighborhood for core point

    Returns:
        Cluster labels (n_samples,), with -1 for outliers
    """
    clusterer = DBSCAN(
        eps=params.get("eps", 0.5),
        min_samples=params.get("min_samples", 5),
    )
    return clusterer.fit_predict(features)


# Unified dispatcher
CLUSTERING_ALGORITHMS = {
    "HAC (Hierarchical Agglomerative)": run_hac,
    "Birch": run_birch,
    "Spectral Clustering": run_spectral,
    "K-Means": run_kmeans,
    "DBSCAN": run_dbscan,
}


def run_clustering(
    algorithm_name: str,
    features: np.ndarray,
    params: Dict[str, Any]
) -> np.ndarray:
    """Unified clustering interface.

    Args:
        algorithm_name: Name of algorithm (key in CLUSTERING_ALGORITHMS)
        features: Feature matrix
        params: Algorithm-specific parameters

    Returns:
        Cluster labels

    Raises:
        ValueError: If algorithm_name not recognized
    """
    if algorithm_name not in CLUSTERING_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. "
            f"Available: {list(CLUSTERING_ALGORITHMS.keys())}"
        )

    return CLUSTERING_ALGORITHMS[algorithm_name](features, params)
