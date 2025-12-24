"""Clustering algorithm parameter controls for Streamlit.

This module provides Streamlit widgets for configuring clustering parameters
for different algorithms.
"""

import streamlit as st
from typing import Dict, Any, Tuple


def render_algorithm_selector() -> str:
    """Render algorithm selection dropdown.

    Returns:
        Selected algorithm name
    """
    return st.sidebar.selectbox(
        "Algorithm",
        [
            "HAC (Hierarchical Agglomerative)",
            "Birch",
            "Spectral Clustering",
            "K-Means",
            "DBSCAN",
        ],
        help="Choose the clustering algorithm to use on PCA-reduced features.",
    )


def render_hac_controls() -> Dict[str, Any]:
    """Render HAC parameter controls.

    Returns:
        Dictionary of parameters for HAC
    """
    st.sidebar.subheader("HAC Parameters")

    n_clusters = st.sidebar.slider(
        "Number of Clusters",
        2,
        50,
        20,
        help="Fixed number of clusters to create.",
    )

    linkage = st.sidebar.selectbox(
        "Linkage Method",
        ["ward", "complete", "average", "single"],
        index=0,
        help="'ward' minimizes variance within clusters. 'complete' uses max distance between clusters. 'average' uses average distance. 'single' uses min distance.",
    )

    return {"n_clusters": n_clusters, "linkage": linkage}


def render_birch_controls() -> Dict[str, Any]:
    """Render Birch parameter controls.

    Returns:
        Dictionary of parameters for Birch
    """
    st.sidebar.subheader("Birch Parameters")
    st.sidebar.info("⚡ Fast hierarchical clustering, similar to HAC but more efficient")

    n_clusters = st.sidebar.slider(
        "Number of Clusters",
        2,
        50,
        20,
        help="Final number of clusters to create.",
    )

    threshold = st.sidebar.slider(
        "Threshold",
        0.1,
        2.0,
        0.5,
        step=0.1,
        help="Radius of subcluster. Lower = more granular subclusters, might be slower.",
    )

    branching_factor = st.sidebar.slider(
        "Branching Factor",
        10,
        100,
        50,
        step=10,
        help="Max subclusters per node. Higher = more memory but better clustering.",
    )

    return {
        "n_clusters": n_clusters,
        "threshold": threshold,
        "branching_factor": branching_factor,
    }


def render_spectral_controls() -> Dict[str, Any]:
    """Render Spectral Clustering parameter controls.

    Returns:
        Dictionary of parameters for Spectral Clustering
    """
    st.sidebar.subheader("Spectral Clustering Parameters")
    st.sidebar.info("🌐 Graph-based clustering, great for non-convex shapes")

    n_clusters = st.sidebar.slider(
        "Number of Clusters",
        2,
        50,
        20,
        help="Number of clusters to find.",
    )

    affinity = st.sidebar.selectbox(
        "Affinity",
        ["nearest_neighbors", "rbf"],
        index=0,
        help="'nearest_neighbors' uses k-NN graph. 'rbf' uses RBF kernel (slower).",
    )

    n_neighbors = st.sidebar.slider(
        "N Neighbors",
        5,
        50,
        15,
        help="Number of neighbors for k-NN graph (only used if affinity=nearest_neighbors).",
    )

    assign_labels = st.sidebar.selectbox(
        "Label Assignment",
        ["kmeans", "discretize"],
        index=0,
        help="'kmeans' is faster and usually better. 'discretize' is an alternative method.",
    )

    return {
        "n_clusters": n_clusters,
        "affinity": affinity,
        "n_neighbors": n_neighbors,
        "assign_labels": assign_labels,
    }


def render_kmeans_controls() -> Dict[str, Any]:
    """Render K-Means parameter controls.

    Returns:
        Dictionary of parameters for K-Means
    """
    st.sidebar.subheader("K-Means Parameters")

    n_clusters = st.sidebar.slider(
        "Number of Clusters",
        2,
        50,
        20,
        help="Number of clusters to create.",
    )

    init = st.sidebar.selectbox(
        "Initialization Method",
        ["k-means++", "random"],
        index=0,
        help="'k-means++' selects initial cluster centroids using sampling based on an empirical probability distribution of the points' contribution to the overall inertia.",
    )

    return {"n_clusters": n_clusters, "init": init}


def render_dbscan_controls() -> Dict[str, Any]:
    """Render DBSCAN parameter controls.

    Returns:
        Dictionary of parameters for DBSCAN
    """
    st.sidebar.subheader("DBSCAN Parameters")
    st.sidebar.info("🔍 Density-based clustering. Finds outliers automatically (-1).")

    eps = st.sidebar.slider(
        "Epsilon (eps)",
        0.1,
        5.0,
        0.5,
        step=0.1,
        help="The maximum distance between two samples for one to be considered as in the neighborhood of the other.",
    )

    min_samples = st.sidebar.slider(
        "Min Samples",
        2,
        20,
        5,
        help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.",
    )

    return {"eps": eps, "min_samples": min_samples}


# Unified control dispatcher
ALGORITHM_CONTROLS = {
    "HAC (Hierarchical Agglomerative)": render_hac_controls,
    "Birch": render_birch_controls,
    "Spectral Clustering": render_spectral_controls,
    "K-Means": render_kmeans_controls,
    "DBSCAN": render_dbscan_controls,
}


def render_clustering_controls() -> Tuple[str, Dict[str, Any]]:
    """Render complete clustering controls (algorithm selector + parameters).

    Returns:
        (algorithm_name, parameters)
    """
    algorithm_name = render_algorithm_selector()

    if algorithm_name not in ALGORITHM_CONTROLS:
        st.sidebar.error(f"Unknown algorithm: {algorithm_name}")
        return algorithm_name, {}

    params = ALGORITHM_CONTROLS[algorithm_name]()

    return algorithm_name, params
