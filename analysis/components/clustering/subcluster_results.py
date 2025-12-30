"""Sub-cluster results display with 3D visualization.

This module renders the results of sub-clustering, including:
- Metrics summary
- 3D UMAP visualization colored by sub-cluster
- Track tables grouped by sub-cluster
- Optimal k analysis visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict

from analysis.components.visualization.color_palette import CLUSTER_COLORS, SPOTIFY_GREEN


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
    st.caption(f"Cluster {parent_cluster} Sub-Clusters (3D UMAP)")
    fig = _create_subcluster_3d_plot(df, coords, parent_cluster)
    st.plotly_chart(fig, use_container_width=True)

    # Sub-cluster statistics
    st.markdown("### Sub-Cluster Statistics")
    _render_subcluster_stats(df)

    # Top artists by sub-cluster
    st.markdown("### 🎤 Top Artists by Sub-Cluster")
    _render_subcluster_top_artists(df)

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
    colors = CLUSTER_COLORS

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


def _render_subcluster_top_artists(df: pd.DataFrame) -> None:
    """
    Render top artists for each sub-cluster with bar charts.

    Args:
        df: DataFrame with 'subcluster' column
    """
    if "artist" not in df.columns:
        st.info("Artist data not available in dataset")
        return

    # Create columns for side-by-side display
    subcluster_ids = sorted(df['subcluster'].unique())
    num_cols = min(len(subcluster_ids), 3)
    cols = st.columns(num_cols)

    colors = CLUSTER_COLORS

    for i, subcluster_id in enumerate(subcluster_ids):
        subcluster_df = df[df['subcluster'] == subcluster_id]
        col_idx = i % num_cols

        with cols[col_idx]:
            st.write(f"**Sub-cluster {subcluster_id} - Top 5 Artists**")

            # Count songs per artist
            artist_counts = subcluster_df["artist"].value_counts().head(5)

            if len(artist_counts) > 0:
                # Create bar chart
                fig = px.bar(
                    x=artist_counts.values,
                    y=artist_counts.index,
                    orientation="h",
                    labels={"x": "Song Count", "y": "Artist"},
                    color_discrete_sequence=[colors[i % len(colors)]],
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key=f"subcluster_results_artist_chart_{subcluster_id}")

                # Show percentage
                total_songs = len(subcluster_df)
                st.caption(
                    f"Top: {artist_counts.index[0]} ({artist_counts.values[0]} songs, {artist_counts.values[0]/total_songs*100:.1f}%)"
                )
            else:
                st.info("No artist data available for this sub-cluster")


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


def render_optimal_k_results(optimal_k_data: Dict) -> None:
    """
    Display optimal k analysis results with multiple quality metrics.

    Args:
        optimal_k_data: Dictionary returned by find_optimal_subclusters()
    """
    parent_cluster = optimal_k_data['parent_cluster']
    k_values = optimal_k_data['k_values']
    silhouette_scores = optimal_k_data['silhouette_scores']
    calinski_harabasz_scores = optimal_k_data.get('calinski_harabasz_scores', [])
    davies_bouldin_scores = optimal_k_data.get('davies_bouldin_scores', [])
    optimal_k = optimal_k_data['optimal_k']
    optimal_score = optimal_k_data['optimal_score']
    cluster_size = optimal_k_data['cluster_size']

    st.markdown("---")
    st.subheader(f"📊 Optimal k Analysis for Cluster {parent_cluster}")

    if not k_values:
        st.warning(f"Cluster {parent_cluster} is too small for sub-clustering ({cluster_size} songs)")
        return

    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cluster Size", f"{cluster_size} songs")
    with col2:
        st.metric("Optimal k", optimal_k, help="Number of sub-clusters with highest silhouette score")
    with col3:
        st.metric("Best Silhouette", f"{optimal_score:.3f}", help="Higher is better (max 1.0)")

    # Metric selection for visualization
    metric_option = st.radio(
        "Select metric to visualize",
        ["Silhouette Score", "Calinski-Harabasz Index", "Davies-Bouldin Index", "All Metrics"],
        horizontal=True,
        help="Silhouette: higher is better | CH: higher is better | DB: lower is better",
    )

    # Create plot based on selection
    st.caption(f"Clustering Quality vs k (Cluster {parent_cluster})")
    fig = go.Figure()

    if metric_option in ["Silhouette Score", "All Metrics"]:
        fig.add_trace(go.Scatter(
            x=k_values,
            y=silhouette_scores,
            mode='lines+markers',
            name='Silhouette',
            line=dict(color=SPOTIFY_GREEN, width=3),
            marker=dict(size=10),
            hovertemplate='k=%{x}<br>Silhouette=%{y:.3f}<extra></extra>',
            yaxis='y1' if metric_option == "All Metrics" else None,
        ))

    if metric_option == "Calinski-Harabasz Index" and calinski_harabasz_scores:
        # Normalize CH scores for display
        fig.add_trace(go.Scatter(
            x=k_values,
            y=calinski_harabasz_scores,
            mode='lines+markers',
            name='Calinski-Harabasz',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=10),
            hovertemplate='k=%{x}<br>CH=%{y:.1f}<extra></extra>',
        ))

    if metric_option == "Davies-Bouldin Index" and davies_bouldin_scores:
        fig.add_trace(go.Scatter(
            x=k_values,
            y=davies_bouldin_scores,
            mode='lines+markers',
            name='Davies-Bouldin',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=10),
            hovertemplate='k=%{x}<br>DB=%{y:.3f}<extra></extra>',
        ))

    if metric_option == "All Metrics" and calinski_harabasz_scores:
        # Normalize all metrics to 0-1 for comparison
        sil_norm = np.array(silhouette_scores)
        ch_norm = np.array(calinski_harabasz_scores)
        db_norm = np.array(davies_bouldin_scores)

        # Normalize
        if ch_norm.max() > ch_norm.min():
            ch_norm = (ch_norm - ch_norm.min()) / (ch_norm.max() - ch_norm.min())
        if db_norm.max() > db_norm.min():
            # Invert DB since lower is better
            db_norm = 1 - (db_norm - db_norm.min()) / (db_norm.max() - db_norm.min())

        fig.add_trace(go.Scatter(
            x=k_values,
            y=ch_norm,
            mode='lines+markers',
            name='CH (normalized)',
            line=dict(color='#FF6B6B', width=2, dash='dash'),
            marker=dict(size=8),
            hovertemplate='k=%{x}<br>CH (norm)=%{y:.3f}<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=k_values,
            y=db_norm,
            mode='lines+markers',
            name='DB (inverted, normalized)',
            line=dict(color='#4ECDC4', width=2, dash='dot'),
            marker=dict(size=8),
            hovertemplate='k=%{x}<br>DB (inv norm)=%{y:.3f}<extra></extra>',
        ))

    # Highlight optimal point (based on silhouette)
    if metric_option in ["Silhouette Score", "All Metrics"]:
        fig.add_trace(go.Scatter(
            x=[optimal_k],
            y=[optimal_score],
            mode='markers',
            name=f'Optimal (k={optimal_k})',
            marker=dict(
                size=20,
                color='red',
                symbol='star',
                line=dict(width=2, color='white'),
            ),
            hovertemplate=f'<b>OPTIMAL</b><br>k={optimal_k}<br>Silhouette={optimal_score:.3f}<extra></extra>',
        ))

        # Add reference line at optimal
        fig.add_vline(
            x=optimal_k,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            annotation_text=f"Optimal k={optimal_k}",
            annotation_position="top",
        )

    # Set y-axis title based on metric
    if metric_option == "Silhouette Score":
        yaxis_title = "Silhouette Score (higher = better)"
    elif metric_option == "Calinski-Harabasz Index":
        yaxis_title = "Calinski-Harabasz Index (higher = better)"
    elif metric_option == "Davies-Bouldin Index":
        yaxis_title = "Davies-Bouldin Index (lower = better)"
    else:
        yaxis_title = "Score (normalized)"

    fig.update_layout(
        xaxis_title="Number of Sub-Clusters (k)",
        yaxis_title=yaxis_title,
        height=400,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        xaxis=dict(tickmode='linear', tick0=2, dtick=1),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    st.markdown("#### 💡 Interpretation")

    if optimal_score >= 0.5:
        quality = "strong"
    elif optimal_score >= 0.25:
        quality = "moderate"
    else:
        quality = "weak"

    st.markdown(
        f"The optimal number of sub-clusters is **k={optimal_k}** with a "
        f"**{quality}** silhouette score of **{optimal_score:.3f}**."
    )

    # Show metric guide
    with st.expander("📖 Metric Guide"):
        st.markdown("""
**Silhouette Score** (primary metric, higher is better)
- Measures how similar points are to their own cluster vs other clusters
- Range: -1 to 1 (>0.5 = good, >0.7 = strong)

**Calinski-Harabasz Index** (higher is better)
- Ratio of between-cluster to within-cluster variance
- No fixed range; compare relative values across k

**Davies-Bouldin Index** (lower is better)
- Average similarity between clusters
- Lower values indicate better separation
        """)

    # Show all scores in a table
    with st.expander("📋 All Scores by k"):
        scores_data = {
            'k': k_values,
            'Silhouette': [f"{s:.4f}" for s in silhouette_scores],
        }
        if calinski_harabasz_scores:
            scores_data['Calinski-Harabasz'] = [f"{s:.1f}" for s in calinski_harabasz_scores]
        if davies_bouldin_scores:
            scores_data['Davies-Bouldin'] = [f"{s:.4f}" for s in davies_bouldin_scores]
        scores_data['Optimal'] = ['⭐' if k == optimal_k else '' for k in k_values]

        scores_df = pd.DataFrame(scores_data)
        st.dataframe(scores_df, use_container_width=True, hide_index=True)

    # Show applied feature weights
    feature_weights = optimal_k_data.get('feature_weights')
    if feature_weights:
        with st.expander("🎛️ Applied Feature Weights"):
            weight_cols = st.columns(3)
            weight_items = list(feature_weights.items())
            for i, (name, weight) in enumerate(weight_items):
                col_idx = i % 3
                with weight_cols[col_idx]:
                    # Show weight with visual indicator
                    if weight > 1.0:
                        indicator = "🔼"
                    elif weight < 1.0:
                        indicator = "🔽"
                    else:
                        indicator = "➖"
                    st.write(f"{indicator} **{name}**: {weight:.1f}x")


def render_auto_tune_results(auto_tune_data: Dict) -> None:
    """
    Display auto-tune weight results with comparison of all presets.

    Args:
        auto_tune_data: Dictionary returned by auto_tune_subcluster_weights()
    """
    parent_cluster = auto_tune_data['parent_cluster']
    best_preset = auto_tune_data['best_preset']
    best_weights = auto_tune_data['best_weights']
    best_k = auto_tune_data['best_k']
    best_score = auto_tune_data['best_score']
    all_results = auto_tune_data['all_results']

    st.markdown("---")
    st.subheader(f"🎯 Auto-Tune Results for Cluster {parent_cluster}")

    # Winner announcement
    st.success(
        f"**Best Configuration:** {best_preset} preset with **k={best_k}** sub-clusters "
        f"(silhouette score: **{best_score:.3f}**)"
    )

    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Preset", best_preset)
    with col2:
        st.metric("Optimal k", best_k)
    with col3:
        st.metric("Best Silhouette", f"{best_score:.3f}")

    # Comparison chart
    st.markdown("### 📊 Preset Comparison")
    st.caption("Silhouette Score by Weight Preset")

    # Prepare data for chart
    valid_results = [r for r in all_results if r.get('optimal_score', 0) > 0]
    if valid_results:
        # Sort by score descending
        valid_results_sorted = sorted(valid_results, key=lambda x: x['optimal_score'], reverse=True)

        presets = [r['preset'] for r in valid_results_sorted]
        scores = [r['optimal_score'] for r in valid_results_sorted]
        optimal_ks = [r['optimal_k'] for r in valid_results_sorted]

        # Create bar chart
        import plotly.graph_objects as go

        colors = [SPOTIFY_GREEN if p == best_preset else 'rgba(100, 100, 100, 0.6)' for p in presets]

        fig = go.Figure(data=[
            go.Bar(
                x=presets,
                y=scores,
                marker_color=colors,
                text=[f"k={k}<br>{s:.3f}" for k, s in zip(optimal_ks, scores)],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<br>k=%{customdata}<extra></extra>',
                customdata=optimal_ks,
            )
        ])

        fig.update_layout(
            xaxis_title="Preset",
            yaxis_title="Best Silhouette Score",
            height=400,
            showlegend=False,
            xaxis_tickangle=-45,
        )

        st.plotly_chart(fig, use_container_width=True)

    # Detailed results table
    with st.expander("📋 Detailed Results"):
        table_data = []
        for r in sorted(all_results, key=lambda x: x.get('optimal_score', 0), reverse=True):
            row = {
                'Preset': r['preset'],
                'Best k': r.get('optimal_k', 'N/A'),
                'Silhouette': f"{r.get('optimal_score', 0):.4f}",
                'Winner': '🏆' if r['preset'] == best_preset else '',
            }
            table_data.append(row)

        results_df = pd.DataFrame(table_data)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

    # Best weights breakdown
    st.markdown("### 🎛️ Best Weight Configuration")
    st.caption(f"Use these weights with the '{best_preset}' preset for optimal results")

    if best_weights:
        weight_cols = st.columns(3)
        weight_items = list(best_weights.items())
        for i, (name, weight) in enumerate(weight_items):
            col_idx = i % 3
            with weight_cols[col_idx]:
                if weight > 1.0:
                    indicator = "🔼"
                    color = "green"
                elif weight < 1.0:
                    indicator = "🔽"
                    color = "red"
                else:
                    indicator = "➖"
                    color = "gray"
                st.write(f"{indicator} **{name}**: {weight:.1f}x")

    # Apply button hint
    st.info(
        "💡 **To apply these weights:**\n"
        "1. Expand '🎛️ Sub-Cluster Feature Weights' in the sidebar\n"
        "2. Manually adjust sliders to match the values above\n"
        f"3. Set 'Number of sub-clusters' to **{best_k}**\n"
        "4. Click '🔍 Run Sub-Clustering'"
    )
