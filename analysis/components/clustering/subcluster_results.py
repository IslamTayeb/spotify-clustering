"""Sub-cluster results display with 3D visualization.

This module renders the results of sub-clustering, including:
- Metrics summary
- 3D UMAP visualization colored by sub-cluster
- Track tables grouped by sub-cluster
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict


def render_subcluster_results(subcluster_data: Dict) -> None:
    """
    Display sub-clustering results with 3D visualization.

    Args:
        subcluster_data: Dictionary returned by run_subcluster_pipeline()
    """
    parent_cluster = subcluster_data['parent_cluster']
    df = subcluster_data['subcluster_df']
    coords = subcluster_data['umap_coords']
    n_subclusters = subcluster_data['n_subclusters']
    sil_score = subcluster_data['silhouette_score']

    st.markdown("---")
    st.subheader(f"🔍 Sub-Clusters of Cluster {parent_cluster}")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Parent Cluster", parent_cluster)
    with col2:
        st.metric("Total Songs", len(df))
    with col3:
        st.metric("Sub-Clusters", n_subclusters)
    with col4:
        st.metric("Silhouette Score", f"{sil_score:.3f}")

    # 3D UMAP visualization
    st.markdown("### 3D Sub-Cluster Visualization")
    fig = _create_subcluster_3d_plot(df, coords, parent_cluster)
    st.plotly_chart(fig, use_container_width=True)

    # Sub-cluster statistics
    st.markdown("### Sub-Cluster Statistics")
    _render_subcluster_stats(df)

    # Track tables by sub-cluster
    st.markdown("### Tracks by Sub-Cluster")
    _render_subcluster_tracks(df)


def _create_subcluster_3d_plot(
    df: pd.DataFrame,
    coords: np.ndarray,
    parent_cluster: int
) -> go.Figure:
    """
    Create 3D scatter plot for sub-clusters.

    Args:
        df: DataFrame with 'subcluster' column
        coords: UMAP coordinates (n_samples x 3)
        parent_cluster: Parent cluster ID for title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, subcluster_id in enumerate(sorted(df['subcluster'].unique())):
        mask = df['subcluster'] == subcluster_id
        cluster_df = df[mask]
        cluster_coords = coords[mask.values]

        # Build hover text
        hover_texts = []
        for _, row in cluster_df.iterrows():
            text = (
                f"<b>{row['track_name']}</b><br>"
                f"Artist: {row['artist']}<br>"
                f"Sub-cluster: {subcluster_id}"
            )
            if 'top_genre' in row and pd.notna(row['top_genre']):
                text += f"<br>Genre: {row['top_genre']}"
            if 'bpm' in row and pd.notna(row['bpm']):
                text += f"<br>BPM: {row['bpm']:.0f}"
            if 'dominant_mood' in row and pd.notna(row['dominant_mood']):
                text += f"<br>Mood: {row['dominant_mood']}"
            hover_texts.append(text)

        fig.add_trace(go.Scatter3d(
            x=cluster_coords[:, 0],
            y=cluster_coords[:, 1],
            z=cluster_coords[:, 2],
            mode='markers',
            name=f"Sub-cluster {subcluster_id} ({len(cluster_df)})",
            marker=dict(
                size=6,
                color=colors[i % len(colors)],
                opacity=0.8,
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        height=600,
        title=f"Cluster {parent_cluster} Sub-Clusters (3D UMAP)",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )

    return fig


def _render_subcluster_stats(df: pd.DataFrame) -> None:
    """
    Render statistics table for each sub-cluster.

    Args:
        df: DataFrame with 'subcluster' column
    """
    stats_data = []

    for subcluster_id in sorted(df['subcluster'].unique()):
        subcluster_df = df[df['subcluster'] == subcluster_id]

        row = {
            'Sub-cluster': subcluster_id,
            'Songs': len(subcluster_df),
        }

        # Top genre
        if 'top_genre' in subcluster_df.columns:
            top_genre = subcluster_df['top_genre'].value_counts().index[0] if len(subcluster_df) > 0 else 'N/A'
            # Simplify genre name if it has "---"
            if '---' in str(top_genre):
                top_genre = str(top_genre).split('---')[0]
            row['Top Genre'] = top_genre

        # Average BPM
        if 'bpm' in subcluster_df.columns:
            row['Avg BPM'] = f"{subcluster_df['bpm'].mean():.0f}"

        # Dominant mood
        mood_cols = ['mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed', 'mood_party']
        available_moods = [col for col in mood_cols if col in subcluster_df.columns]
        if available_moods:
            mood_means = {col.replace('mood_', ''): subcluster_df[col].mean() for col in available_moods}
            dominant_mood = max(mood_means, key=mood_means.get)
            row['Dominant Mood'] = dominant_mood.capitalize()

        # Average danceability
        if 'danceability' in subcluster_df.columns:
            row['Avg Danceability'] = f"{subcluster_df['danceability'].mean():.2f}"

        stats_data.append(row)

    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)


def _render_subcluster_tracks(df: pd.DataFrame) -> None:
    """
    Render expandable track lists for each sub-cluster.

    Args:
        df: DataFrame with 'subcluster' column
    """
    display_cols = ['track_name', 'artist']

    # Add optional columns if available
    optional_cols = ['top_genre', 'bpm', 'danceability', 'dominant_mood']
    for col in optional_cols:
        if col in df.columns:
            display_cols.append(col)

    for subcluster_id in sorted(df['subcluster'].unique()):
        subcluster_df = df[df['subcluster'] == subcluster_id]
        available_cols = [c for c in display_cols if c in subcluster_df.columns]

        with st.expander(f"Sub-cluster {subcluster_id} ({len(subcluster_df)} songs)"):
            # Format display DataFrame
            display_df = subcluster_df[available_cols].copy()

            # Round numeric columns
            for col in display_df.columns:
                if display_df[col].dtype in ['float64', 'float32']:
                    display_df[col] = display_df[col].round(2)

            st.dataframe(display_df, use_container_width=True, hide_index=True)
