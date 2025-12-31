"""Genre analysis sections for EDA explorer."""

import streamlit as st
import pandas as pd
import plotly.express as px

from analysis.components.visualization.color_palette import CLUSTER_COLORS, SPOTIFY_GREEN
from analysis.pipeline.config import get_cluster_name
from .utils import group_small_slices, get_pie_colors


def render_genre_analysis(df: pd.DataFrame):
    """Render genre distribution analysis section."""
    with st.expander("🎸 Genre Analysis", expanded=False):
        if "top_genre" not in df.columns:
            st.warning("Genre information not available in this dataset")
            return

        # Parent genre distribution
        st.subheader("Genre Family Distribution")
        st.caption("Genres grouped by parent category (e.g., all Hip Hop subgenres combined)")

        def extract_parent_genre(genre_str):
            if pd.isna(genre_str) or not isinstance(genre_str, str):
                return "Unknown"
            if "---" in genre_str:
                return genre_str.split("---")[0]
            return genre_str

        df_genre = df.copy()
        df_genre["parent_genre"] = df_genre["top_genre"].apply(extract_parent_genre)
        parent_counts = df_genre["parent_genre"].value_counts()
        parent_counts_grouped, _ = group_small_slices(parent_counts, threshold_pct=2.0)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.caption("Genre Family Share")
            fig_pie = px.pie(
                values=parent_counts_grouped.values,
                names=parent_counts_grouped.index,
                hole=0.3,
                color_discrete_sequence=get_pie_colors(parent_counts_grouped.index, CLUSTER_COLORS),
            )
            fig_pie.update_layout(height=500, margin=dict(t=0, l=0, r=0, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.caption("Genre Family Counts")
            fig_bar = px.bar(
                x=parent_counts.values,
                y=parent_counts.index,
                orientation="h",
                labels={"x": "Number of Songs", "y": "Genre Family"},
                color_discrete_sequence=[SPOTIFY_GREEN],
            )
            fig_bar.update_layout(height=500, showlegend=False, margin=dict(t=0, l=0, r=0, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)

        # Detailed subgenre breakdown
        st.markdown("---")
        st.subheader("Top 20 Specific Subgenres")
        st.caption("Individual subgenres (e.g., Hip Hop---Trap, Electronic---House)")

        genre_counts = df["top_genre"].value_counts().head(20)

        fig = px.bar(
            x=genre_counts.values,
            y=genre_counts.index,
            orientation="h",
            labels={"x": "Number of Songs", "y": "Genre"},
            color_discrete_sequence=[SPOTIFY_GREEN],
        )
        fig.update_layout(height=700, showlegend=False, margin=dict(t=0, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Genre distribution across clusters - grouped by parent genre
        st.subheader("Genre Distribution Across Clusters")
        st.caption("All genres grouped by parent category (e.g., all Hip Hop subgenres combined)")

        # Create dataframe with parent genres for clustering analysis
        df_cluster_genre = df.copy()
        df_cluster_genre["parent_genre"] = df_cluster_genre["top_genre"].apply(extract_parent_genre)

        # Get all unique parent genres
        all_parent_genres = sorted(df_cluster_genre["parent_genre"].unique())

        genre_cluster_data = []
        for cluster_id in sorted(df_cluster_genre["cluster"].unique()):
            cluster_df = df_cluster_genre[df_cluster_genre["cluster"] == cluster_id]
            cluster_name = get_cluster_name(cluster_id)  # Use cluster names instead of indices

            for parent_genre in all_parent_genres:
                count = (cluster_df["parent_genre"] == parent_genre).sum()
                if count > 0:  # Only include genres that have songs in this cluster
                    genre_cluster_data.append({
                        "Cluster": cluster_name,
                        "Genre Family": parent_genre,
                        "Count": count,
                    })

        genre_cluster_df = pd.DataFrame(genre_cluster_data)

        # Create stacked bar chart
        fig = px.bar(
            genre_cluster_df,
            x="Cluster",
            y="Count",
            color="Genre Family",
            barmode="stack",
            color_discrete_sequence=CLUSTER_COLORS,
        )
        fig.update_layout(height=500, margin=dict(t=0, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Show detailed breakdown in expandable section
        with st.expander("View Detailed Genre Distribution"):
            # Pivot table for better readability
            pivot_df = genre_cluster_df.pivot_table(
                index="Genre Family",
                columns="Cluster",
                values="Count",
                fill_value=0
            )

            # Calculate totals
            pivot_df["Total"] = pivot_df.sum(axis=1)
            pivot_df = pivot_df.sort_values("Total", ascending=False)

            # Format as integers
            st.dataframe(
                pivot_df.astype(int),
                use_container_width=True
            )


def render_genre_fusion_analysis(df: pd.DataFrame):
    """Render genre fusion analysis section."""
    with st.expander("🎸 Genre Fusion Analysis", expanded=False):
        if "genre_fusion" not in df.columns:
            st.warning("Genre fusion information not available")
            return

        st.subheader("Pure Genre ↔ Genre Fusion Distribution")
        st.caption("Genre fusion measures genre entropy (0=pure single genre, 1=genre fusion/hybrid)")

        fig = px.histogram(
            df,
            x="genre_fusion",
            nbins=50,
            labels={"genre_fusion": "Genre Fusion Score", "count": "Number of Songs"},
            color_discrete_sequence=[SPOTIFY_GREEN],
        )
        fig.update_layout(height=500, margin=dict(t=0, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Most Pure Genre Songs**")
            pure = df.nsmallest(10, "genre_fusion")[["track_name", "artist", "top_genre", "genre_fusion"]]
            st.dataframe(pure, use_container_width=True, hide_index=True)

        with col2:
            st.write("**Most Genre-Fusion Songs**")
            fusion = df.nlargest(10, "genre_fusion")[["track_name", "artist", "top_genre", "genre_fusion"]]
            st.dataframe(fusion, use_container_width=True, hide_index=True)

        # Compare genre fusion with actual acoustic/electronic production
        if "electronic_acoustic" in df.columns:
            st.markdown("---")
            st.subheader("Genre Fusion vs Acoustic/Electronic Production")
            st.caption("Does genre fusion correlate with acoustic or electronic production style?")

            # Add cluster names to dataframe for visualization
            df_with_names = df.copy()
            df_with_names["cluster_name"] = df_with_names["cluster"].apply(get_cluster_name)

            fig = px.scatter(
                df_with_names,
                x="genre_fusion",
                y="electronic_acoustic",
                color="cluster_name",
                hover_data=["track_name", "artist", "top_genre"],
                labels={
                    "genre_fusion": "Genre Fusion (0=Pure, 1=Fusion)",
                    "electronic_acoustic": "Production (0=Electronic, 1=Acoustic)",
                    "cluster_name": "Cluster",
                },
                color_discrete_sequence=CLUSTER_COLORS,
            )
            fig.update_layout(height=500, margin=dict(t=0, l=0, r=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

            correlation = df["genre_fusion"].corr(df["electronic_acoustic"])
            st.metric("Genre Fusion ↔ Acoustic Production Correlation", f"{correlation:.3f}")
