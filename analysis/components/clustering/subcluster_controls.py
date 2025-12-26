"""Sub-clustering controls for Streamlit sidebar.

This module provides UI controls for drilling down into a specific cluster
and creating sub-clusters for more granular analysis.
"""

import streamlit as st
from typing import Tuple, Optional
import pandas as pd


def render_subcluster_controls(df: pd.DataFrame) -> Tuple[Optional[int], int, str, str, float, int]:
    """
    Render sub-clustering controls in the sidebar.

    Args:
        df: DataFrame with 'cluster' column from main clustering

    Returns:
        Tuple of (parent_cluster, n_subclusters, algorithm, linkage, eps, min_samples)
        Returns (None, 0, '', '', 0.5, 5) if no cluster is selected
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Sub-Clustering")
    st.sidebar.caption("Drill down into a cluster")

    # Get cluster options with song counts
    cluster_counts = df['cluster'].value_counts().sort_index()
    cluster_options = [None] + list(cluster_counts.index)

    parent_cluster = st.sidebar.selectbox(
        "Select cluster to sub-cluster",
        options=cluster_options,
        format_func=lambda x: "-- Select a cluster --" if x is None else f"Cluster {x} ({cluster_counts[x]} songs)",
        key="subcluster_parent_select",
    )

    if parent_cluster is None:
        st.sidebar.info("Select a cluster above to enable sub-clustering")
        return None, 0, '', '', 0.5, 5

    # Show selected cluster info
    cluster_size = cluster_counts[parent_cluster]
    st.sidebar.caption(f"Selected: Cluster {parent_cluster} with {cluster_size} songs")

    # Number of sub-clusters slider
    max_subclusters = min(10, cluster_size - 1) if cluster_size > 2 else 2
    n_subclusters = st.sidebar.slider(
        "Number of sub-clusters",
        min_value=2,
        max_value=max_subclusters,
        value=min(2, max_subclusters),
        help="How many sub-groups to create within this cluster",
        key="subcluster_n_clusters",
    )

    # Algorithm selector
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["HAC", "Birch", "Spectral", "K-Means", "DBSCAN"],
        index=0,
        help="HAC (Hierarchical) is recommended for interpretability",
        key="subcluster_algorithm",
    )

    # Linkage method (only for HAC)
    linkage = 'ward'
    if algorithm == "HAC":
        linkage = st.sidebar.selectbox(
            "Linkage Method",
            ["ward", "complete", "average"],
            index=0,
            help="'ward' minimizes variance, 'complete' uses max distance",
            key="subcluster_linkage",
        )

    # DBSCAN parameters (only for DBSCAN)
    eps = 0.5
    min_samples = 5
    if algorithm == "DBSCAN":
        st.sidebar.caption("⚠️ DBSCAN may produce outliers (label -1)")
        eps = st.sidebar.slider(
            "Epsilon (eps)",
            min_value=0.1,
            max_value=5.0,
            value=0.5,
            step=0.1,
            help="Maximum distance between samples in a neighborhood",
            key="subcluster_eps",
        )
        min_samples = st.sidebar.slider(
            "Min Samples",
            min_value=2,
            max_value=20,
            value=5,
            help="Minimum samples in a neighborhood for a core point",
            key="subcluster_min_samples",
        )

    return parent_cluster, n_subclusters, algorithm.lower(), linkage, eps, min_samples


def render_subcluster_button() -> bool:
    """
    Render the sub-clustering action button.

    Returns:
        True if button was clicked
    """
    return st.sidebar.button(
        "🔍 Run Sub-Clustering",
        type="primary",
        use_container_width=True,
        key="run_subcluster_btn",
    )


def render_clear_subcluster_button() -> bool:
    """
    Render button to clear sub-clustering results.

    Returns:
        True if button was clicked
    """
    return st.sidebar.button(
        "🗑️ Clear Sub-Clusters",
        use_container_width=True,
        key="clear_subcluster_btn",
    )
