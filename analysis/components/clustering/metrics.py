"""Clustering quality metrics display.

This module provides real-time metrics for evaluating clustering results.
"""

import streamlit as st
import numpy as np
from sklearn.metrics import silhouette_score
from typing import Optional


def compute_clustering_metrics(
    labels: np.ndarray,
    features: Optional[np.ndarray] = None
) -> dict:
    """Compute clustering quality metrics.

    Args:
        labels: Cluster labels (n_samples,)
        features: Feature matrix (n_samples x n_features), optional
            Required for silhouette score

    Returns:
        Dictionary with metrics:
            - n_clusters: Number of clusters found
            - n_outliers: Number of outliers (-1 label in DBSCAN)
            - pct_outliers: Percentage of outliers
            - total_songs: Total number of samples
            - silhouette_score: Silhouette score (if features provided)
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = (labels == -1).sum()
    pct_outliers = (n_outliers / len(labels)) * 100 if len(labels) > 0 else 0

    metrics = {
        "n_clusters": n_clusters,
        "n_outliers": int(n_outliers),
        "pct_outliers": pct_outliers,
        "total_songs": len(labels),
    }

    # Compute silhouette score if features provided
    if features is not None and len(set(labels)) > 1:
        # Filter out outliers for silhouette computation
        non_outlier_mask = labels != -1
        if non_outlier_mask.sum() > 0:
            try:
                sil_score = silhouette_score(
                    features[non_outlier_mask],
                    labels[non_outlier_mask]
                )
                metrics["silhouette_score"] = sil_score
            except ValueError:
                # Not enough samples or single cluster
                metrics["silhouette_score"] = 0.0
        else:
            metrics["silhouette_score"] = 0.0
    else:
        metrics["silhouette_score"] = None

    return metrics


def render_clustering_metrics(
    labels: np.ndarray,
    features: Optional[np.ndarray] = None
) -> None:
    """Display clustering quality metrics in Streamlit.

    Args:
        labels: Cluster labels
        features: Feature matrix for silhouette score (optional)
    """
    metrics = compute_clustering_metrics(labels, features)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Clusters Found", metrics["n_clusters"])

    with col2:
        outlier_text = f"{metrics['n_outliers']} ({metrics['pct_outliers']:.1f}%)"
        st.metric("Outliers", outlier_text)

    with col3:
        st.metric("Total Songs", metrics["total_songs"])

    with col4:
        if metrics["silhouette_score"] is not None:
            st.metric(
                "Silhouette Score",
                f"{metrics['silhouette_score']:.3f}",
                help="Measures cluster cohesion and separation. Range: -1 to 1. Higher is better.",
            )
        else:
            st.metric("Silhouette Score", "N/A")


def render_metrics_interpretation() -> None:
    """Render interpretation guide for clustering metrics."""
    with st.expander("ℹ️ Understanding Clustering Metrics"):
        st.markdown("""
        **Clusters Found**: Number of distinct clusters identified (excluding outliers).

        **Outliers**: Points that don't fit well into any cluster (DBSCAN specific).
        - Low outlier % (< 5%): Tight, well-defined clusters
        - High outlier % (> 20%): May need parameter tuning

        **Silhouette Score**: Measures how well-separated clusters are.
        - 0.7 - 1.0: Strong, well-separated clusters
        - 0.5 - 0.7: Reasonable structure
        - 0.25 - 0.5: Weak structure, clusters overlap
        - < 0.25: No meaningful structure or arbitrary clustering

        **Tips for Improvement**:
        - Low silhouette? Try different algorithm or adjust parameters
        - Too many outliers? Lower eps (DBSCAN) or try different algorithm
        - Too few clusters? Increase n_clusters or lower eps
        - Too many clusters? Decrease n_clusters or raise eps
        """)
