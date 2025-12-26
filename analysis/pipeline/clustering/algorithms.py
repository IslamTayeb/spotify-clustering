"""Clustering algorithm implementations."""

import logging
from typing import Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, Birch, KMeans, DBSCAN

logger = logging.getLogger(__name__)


def run_hac(
    features: np.ndarray,
    n_clusters: int = 20,
    linkage_method: str = 'ward'
) -> np.ndarray:
    """Run Hierarchical Agglomerative Clustering.

    Args:
        features: Feature matrix
        n_clusters: Number of clusters
        linkage_method: 'ward', 'complete', 'average', 'single'

    Returns:
        Cluster labels
    """
    logger.info(f"Running HAC (n_clusters={n_clusters}, linkage={linkage_method})...")
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    return clusterer.fit_predict(features)


def run_birch(
    features: np.ndarray,
    n_clusters: int = 20,
    threshold: float = 0.5,
    branching_factor: int = 50
) -> np.ndarray:
    """Run Birch clustering.

    Args:
        features: Feature matrix
        n_clusters: Number of clusters
        threshold: Birch threshold
        branching_factor: Birch branching factor

    Returns:
        Cluster labels
    """
    logger.info(f"Running Birch (n_clusters={n_clusters}, threshold={threshold})...")
    clusterer = Birch(
        n_clusters=n_clusters,
        threshold=threshold,
        branching_factor=branching_factor
    )
    return clusterer.fit_predict(features)


def run_spectral(
    features: np.ndarray,
    n_clusters: int = 20,
    affinity: str = 'nearest_neighbors',
    n_neighbors: int = 15,
    assign_labels: str = 'kmeans'
) -> np.ndarray:
    """Run Spectral Clustering.

    Args:
        features: Feature matrix
        n_clusters: Number of clusters
        affinity: 'nearest_neighbors' or 'rbf'
        n_neighbors: Number of neighbors for nearest_neighbors affinity
        assign_labels: 'kmeans' or 'discretize'

    Returns:
        Cluster labels
    """
    logger.info(f"Running Spectral (n_clusters={n_clusters}, affinity={affinity})...")
    clusterer = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        n_neighbors=n_neighbors if affinity == 'nearest_neighbors' else 10,
        assign_labels=assign_labels,
        random_state=42
    )
    return clusterer.fit_predict(features)


def run_kmeans(
    features: np.ndarray,
    n_clusters: int = 20,
    n_init: int = 10
) -> np.ndarray:
    """Run K-Means clustering.

    Args:
        features: Feature matrix
        n_clusters: Number of clusters
        n_init: Number of initializations

    Returns:
        Cluster labels
    """
    logger.info(f"Running K-Means (n_clusters={n_clusters})...")
    clusterer = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=n_init,
        random_state=42
    )
    return clusterer.fit_predict(features)


def run_dbscan(
    features: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5
) -> np.ndarray:
    """Run DBSCAN clustering.

    Args:
        features: Feature matrix
        eps: Maximum distance between samples
        min_samples: Minimum samples in a neighborhood

    Returns:
        Cluster labels (-1 for outliers)
    """
    logger.info(f"Running DBSCAN (eps={eps}, min_samples={min_samples})...")
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    return clusterer.fit_predict(features)


def get_clusterer(
    algorithm: str,
    n_clusters: int,
    **kwargs
) -> Tuple[object, bool]:
    """Get a clustering object by algorithm name.

    Args:
        algorithm: Algorithm name ('hac', 'birch', 'spectral', 'k-means', 'dbscan')
        n_clusters: Number of clusters (ignored for dbscan)
        **kwargs: Algorithm-specific parameters

    Returns:
        Tuple of (clusterer object, whether it produces outliers)
    """
    if algorithm == 'hac':
        linkage = kwargs.get('linkage', 'ward')
        return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage), False

    elif algorithm == 'birch':
        threshold = kwargs.get('threshold', 0.5)
        branching_factor = kwargs.get('branching_factor', 50)
        return Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor), False

    elif algorithm == 'spectral':
        affinity = kwargs.get('affinity', 'nearest_neighbors')
        n_neighbors = kwargs.get('n_neighbors', 15)
        assign_labels = kwargs.get('assign_labels', 'kmeans')
        return SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            n_neighbors=n_neighbors,
            assign_labels=assign_labels,
            random_state=42
        ), False

    elif algorithm == 'k-means':
        n_init = kwargs.get('n_init', 10)
        return KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init, random_state=42), False

    elif algorithm == 'dbscan':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        return DBSCAN(eps=eps, min_samples=min_samples), True

    else:
        logger.warning(f"Unknown algorithm '{algorithm}', defaulting to HAC")
        return AgglomerativeClustering(n_clusters=n_clusters, linkage='ward'), False
