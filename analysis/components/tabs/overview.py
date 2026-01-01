"""Overview Tab Component

Global overview and summary of clustering results:
- Summary statistics (songs, clusters, lyric coverage)
- Cluster size distribution and percentages
- Cluster similarity matrix (heatmap visualization)
- Most/least similar cluster pairs
- Key feature summary across all clusters
- Genre distribution analysis
- Mood profiles radar chart
- Export functionality for similarity matrix and cluster summary
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from analysis.interpretability.cluster_comparison import (
    compute_cluster_similarity_matrix,
)
from analysis.components.visualization.color_palette import CLUSTER_COLORS, get_cluster_color
from analysis.pipeline.config import get_cluster_name
from analysis.components.export.chart_export import render_chart_with_export, render_export_section


def render_overview(df: pd.DataFrame):
    """Render Overview view."""
    st.header("🔍 Global Overview")

    st.write("High-level summary of all clusters and their relationships.")

    # Summary Statistics
    st.subheader("📊 Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Songs", len(df))

    with col2:
        st.metric("Number of Clusters", df["cluster"].nunique())

    with col3:
        if "has_lyrics" in df.columns:
            lyric_pct = (df["has_lyrics"].sum() / len(df) * 100) if len(df) > 0 else 0
            st.metric("Songs with Lyrics", f"{lyric_pct:.1f}%")
        else:
            st.metric("Songs with Lyrics", "N/A")

    with col4:
        # Calculate average silhouette score if available
        st.metric("Clustering Mode", "Combined")

    # Cluster sizes
    st.markdown("---")
    st.subheader("📏 Cluster Size Distribution")

    cluster_sizes = df["cluster"].value_counts().sort_index()
    bar_colors = [get_cluster_color(idx) for idx in cluster_sizes.index]

    st.caption("Songs per Cluster")
    fig = px.bar(
        x=cluster_sizes.index,
        y=cluster_sizes.values,
        labels={"x": "Cluster", "y": "Number of Songs"},
        color_discrete_sequence=bar_colors,
    )
    fig.update_traces(marker_color=bar_colors)
    fig.update_layout(height=500, showlegend=False, margin=dict(t=0, l=0, r=0, b=0))
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)

    # Show cluster size table
    cluster_info = []
    for cluster_id in sorted(df["cluster"].unique()):
        size = len(df[df["cluster"] == cluster_id])
        percentage = size / len(df) * 100

        cluster_info.append(
            {
                "Cluster": get_cluster_name(cluster_id),
                "Size": size,
                "Percentage": f"{percentage:.1f}%",
            }
        )

    st.dataframe(pd.DataFrame(cluster_info), use_container_width=True, hide_index=True)

    # Cluster similarity matrix
    st.markdown("---")
    st.subheader("🔗 Cluster Similarity Matrix")

    st.write("Lower values = more similar clusters (based on average effect sizes)")

    with st.spinner("Computing cluster similarities..."):
        similarity_matrix = compute_cluster_similarity_matrix(df)

        # Use actual cluster names
        cluster_ids = sorted(df["cluster"].unique())
        labels = [get_cluster_name(cid) for cid in cluster_ids]

        fig = px.imshow(
            similarity_matrix.values,
            labels=dict(color="Dissimilarity"),
            x=labels,
            y=labels,
            color_continuous_scale="YlOrRd",
            aspect="equal",
            text_auto=".2f",
        )

        fig.update_traces(textfont_size=12)
        fig.update_layout(
            height=600,
            margin=dict(t=0, l=0, r=0, b=0),
            xaxis_title="",
            yaxis_title="",
        )
        render_chart_with_export(fig, "cluster_similarity_matrix", "Cluster Similarity Matrix", "overview")

    # Most and least similar cluster pairs
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Most Similar Cluster Pairs**")

        # Get all pairs with their similarity scores using cluster names
        pairs = []
        cluster_ids = sorted(df["cluster"].unique())
        cluster_names = [get_cluster_name(cid) for cid in cluster_ids]

        for i, (cluster_a, name_a) in enumerate(zip(cluster_ids, cluster_names)):
            for cluster_b, name_b in zip(cluster_ids[i + 1:], cluster_names[i + 1:]):
                dissimilarity = similarity_matrix.loc[name_a, name_b]
                pairs.append(
                    {
                        "Pair": f"{name_a} & {name_b}",
                        "Dissimilarity": f"{dissimilarity:.3f}",
                    }
                )

        pairs_df = pd.DataFrame(pairs)
        pairs_df = pairs_df.sort_values("Dissimilarity")
        st.dataframe(pairs_df.head(5), use_container_width=True, hide_index=True)

    with col2:
        st.write("**Most Different Cluster Pairs**")
        st.dataframe(
            pairs_df.tail(5).iloc[::-1], use_container_width=True, hide_index=True
        )

    # Key Feature Summary
    st.markdown("---")
    st.subheader("🎯 Key Feature Summary Across All Clusters")

    # Select key features to summarize (representative from 33-dim vector)
    key_features = [
        # Audio core
        "bpm", "danceability", "instrumentalness", "valence", "arousal",
        # Audio moods
        "mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed", "mood_party",
        # Lyric core
        "lyric_valence", "lyric_arousal",
        # Meta
        "popularity",
    ]
    key_features = [f for f in key_features if f in df.columns]

    if key_features:
        summary_data = []

        for feature in key_features:
            feature_data = {"Feature": feature}

            for cluster_id in sorted(df["cluster"].unique()):
                cluster_df = df[df["cluster"] == cluster_id]
                mean_val = cluster_df[feature].mean()
                feature_data[get_cluster_name(cluster_id)] = f"{mean_val:.3f}"

            summary_data.append(feature_data)

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Genre distribution across clusters
    if "top_genre" in df.columns:
        st.markdown("---")
        st.subheader("🎸 Top Genre Distribution Across Clusters")

        # Get top 5 overall genres
        top_genres = df["top_genre"].value_counts().head(5).index

        genre_cluster_data = []
        for genre in top_genres:
            row = {"Genre": genre}
            for cluster_id in sorted(df["cluster"].unique()):
                cluster_df = df[df["cluster"] == cluster_id]
                count = (cluster_df["top_genre"] == genre).sum()
                percentage = (
                    (count / len(cluster_df) * 100) if len(cluster_df) > 0 else 0
                )
                row[get_cluster_name(cluster_id)] = f"{percentage:.1f}%"

            genre_cluster_data.append(row)

        genre_dist_df = pd.DataFrame(genre_cluster_data)
        st.dataframe(genre_dist_df, use_container_width=True, hide_index=True)

    # Mood radar comparison
    mood_cols = [
        "mood_happy",
        "mood_sad",
        "mood_aggressive",
        "mood_relaxed",
        "mood_party",
    ]

    if all(col in df.columns for col in mood_cols):
        st.markdown("---")
        st.subheader("😊 Mood Profiles by Cluster")

        fig = go.Figure()

        cluster_ids = sorted(df["cluster"].unique())
        colors = CLUSTER_COLORS

        for i, cluster_id in enumerate(cluster_ids):
            cluster_df = df[df["cluster"] == cluster_id]

            mood_means = [
                cluster_df["mood_happy"].mean() * 100,
                cluster_df["mood_sad"].mean() * 100,
                cluster_df["mood_aggressive"].mean() * 100,
                cluster_df["mood_relaxed"].mean() * 100,
                cluster_df["mood_party"].mean() * 100,
            ]

            fig.add_trace(
                go.Scatterpolar(
                    r=mood_means,
                    theta=["Happy", "Sad", "Aggressive", "Relaxed", "Party"],
                    fill="toself",
                    name=get_cluster_name(cluster_id),
                    line_color=colors[i % len(colors)],
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=700,
            margin=dict(t=0, l=0, r=0, b=0),
        )

        render_chart_with_export(fig, "mood_profiles_radar", "Mood Profiles by Cluster", "overview")

    # Export all overview data
    st.markdown("---")
    st.subheader("📥 Export Data")

    col1, col2 = st.columns(2)

    with col1:
        # Export similarity matrix
        csv = similarity_matrix.to_csv().encode("utf-8")
        st.download_button(
            label="Download Similarity Matrix",
            data=csv,
            file_name="cluster_similarity_matrix.csv",
            mime="text/csv",
        )

    with col2:
        # Export cluster summary
        summary_csv = pd.DataFrame(cluster_info).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cluster Summary",
            data=summary_csv,
            file_name="cluster_summary.csv",
            mime="text/csv",
        )

    # Chart export section for HTML exports
    render_export_section(default_dir="export/dimensions-of-taste-viz", section_key="overview")
