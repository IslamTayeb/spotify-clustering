"""UMAP 3D visualization utilities.

This module handles UMAP dimensionality reduction and 3D scatter plot creation.
"""

import umap
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from typing import Optional

from analysis.components.visualization.color_palette import (
    get_cluster_color,
    OUTLIER_COLOR,
)


def compute_umap_embedding(
    features: np.ndarray,
    n_neighbors: int = 20,
    min_dist: float = 0.2,
    random_state: int = 42
) -> np.ndarray:
    """Compute 3D UMAP embedding for visualization.

    Args:
        features: Feature matrix (n_samples x n_features)
        n_neighbors: Controls local vs global structure balance
        min_dist: How tightly points are packed
        random_state: Random seed for reproducibility

    Returns:
        UMAP coordinates (n_samples x 3)
    """
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def build_hover_text(row: pd.Series) -> str:
    """Build hover text for a single track.

    Args:
        row: DataFrame row with track metadata

    Returns:
        HTML-formatted hover text
    """
    text = (
        f"<b>{row['track_name']}</b><br>"
        f"Artist: {row['artist']}<br>"
        f"Cluster: {row['label']}<br>"
        f"Genre: {row.get('genre', 'Unknown')}<br>"
        f"BPM: {row.get('bpm', 0):.0f} | Key: {row.get('key', 'Unknown')}<br>"
        f"Danceability: {row.get('danceability', 0):.2f}<br>"
        f"Instrumentalness: {row.get('instrumentalness', 0):.2f}<br>"
        f"Valence: {row.get('valence', 0):.2f} | Arousal: {row.get('arousal', 0):.2f}<br>"
        f"Engagement: {row.get('engagement_score', 0):.2f} | Approachability: {row.get('approachability_score', 0):.2f}<br>"
        f"Moods:<br>"
        f"- Happy: {row.get('mood_happy', 0):.2f}<br>"
        f"- Sad: {row.get('mood_sad', 0):.2f}<br>"
        f"- Aggressive: {row.get('mood_aggressive', 0):.2f}<br>"
        f"- Relaxed: {row.get('mood_relaxed', 0):.2f}<br>"
        f"- Party: {row.get('mood_party', 0):.2f}<br>"
    )

    if 'genre_fusion' in row:
        genre_type = 'Pure' if row['genre_fusion'] < 0.4 else 'Fusion' if row['genre_fusion'] > 0.6 else 'Mixed'
        text += f"Genre Fusion: {row['genre_fusion']:.2f} ({genre_type})<br>"

    return text


def create_umap_3d_plot(
    df: pd.DataFrame,
    color_by: str = "label",
) -> go.Figure:
    """Create 3D UMAP scatter plot.

    Args:
        df: DataFrame with columns 'x', 'y', 'z', and 'label'
        color_by: Column name to color points by (default: 'label')

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Get unique labels
    unique_labels = sorted(df[color_by].unique())

    # Add trace for each cluster
    for label in unique_labels:
        cluster_points = df[df[color_by] == label]
        if cluster_points.empty:
            continue

        if label == -1:
            name = f"Outliers ({len(cluster_points)})"
            color_val = OUTLIER_COLOR  # Consistent light gray
            size = 3
            opacity = 0.3
        else:
            name = f"Cluster {label} ({len(cluster_points)})"
            color_val = get_cluster_color(label)  # Consistent color from palette
            size = 4
            opacity = 0.8

        # Build hover text
        hover_texts = cluster_points.apply(build_hover_text, axis=1)

        fig.add_trace(
            go.Scatter3d(
                x=cluster_points["x"],
                y=cluster_points["y"],
                z=cluster_points["z"],
                mode="markers",
                name=name,
                marker=dict(
                    size=size,
                    color=color_val,  # Direct hex color (no colorscale needed)
                    opacity=opacity,
                ),
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        height=800,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
    )

    return fig


def create_interactive_cluster_map(
    df: pd.DataFrame,
) -> go.Figure:
    """Create interactive 3D cluster map with selection support.

    Args:
        df: DataFrame with UMAP coordinates and cluster labels

    Returns:
        Plotly Figure with interactive selection
    """
    return create_umap_3d_plot(df)
