"""Genre analysis sections for EDA explorer."""

import streamlit as st
import pandas as pd
import plotly.express as px

from analysis.components.visualization.color_palette import CLUSTER_COLORS, SPOTIFY_GREEN
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
            fig_pie = px.pie(
                values=parent_counts_grouped.values,
                names=parent_counts_grouped.index,
                title="Genre Family Share",
                hole=0.3,
                color_discrete_sequence=get_pie_colors(parent_counts_grouped.index, CLUSTER_COLORS),
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            fig_bar = px.bar(
                x=parent_counts.values,
                y=parent_counts.index,
                orientation="h",
                labels={"x": "Number of Songs", "y": "Genre Family"},
                title="Genre Family Counts",
                color_discrete_sequence=[SPOTIFY_GREEN],
            )
            fig_bar.update_layout(height=400, showlegend=False)
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
            title="Top 20 Subgenres in Your Library",
            color_discrete_sequence=[SPOTIFY_GREEN],
        )
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Genre distribution across clusters
        st.subheader("Genre Distribution Across Clusters")
        top_genres = df["top_genre"].value_counts().head(10).index

        genre_cluster_data = []
        for cluster_id in sorted(df["cluster"].unique()):
            cluster_df = df[df["cluster"] == cluster_id]
            for genre in top_genres:
                count = (cluster_df["top_genre"] == genre).sum()
                genre_cluster_data.append({
                    "Cluster": f"Cluster {cluster_id}",
                    "Genre": genre,
                    "Count": count,
                })

        genre_cluster_df = pd.DataFrame(genre_cluster_data)

        fig = px.bar(
            genre_cluster_df,
            x="Cluster",
            y="Count",
            color="Genre",
            title="Top 10 Genres by Cluster",
            barmode="stack",
            color_discrete_sequence=CLUSTER_COLORS,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_genre_ladder_analysis(df: pd.DataFrame):
    """Render genre ladder analysis section."""
    with st.expander("🎸 Genre Ladder Analysis", expanded=False):
        if "genre_ladder" not in df.columns:
            st.warning("Genre ladder information not available")
            return

        st.subheader("Acoustic ↔ Electronic Distribution (Genre-Based)")
        st.caption("Genre ladder captures stylistic intent (0=acoustic/traditional, 1=electronic/synthetic)")

        fig = px.histogram(
            df,
            x="genre_ladder",
            nbins=50,
            title="Genre Ladder Distribution (0=Acoustic, 1=Electronic)",
            labels={"genre_ladder": "Genre Ladder Score", "count": "Number of Songs"},
            color_discrete_sequence=[SPOTIFY_GREEN],
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Hybrid")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Most Acoustic Songs (by Genre)**")
            acoustic = df.nsmallest(10, "genre_ladder")[["track_name", "artist", "top_genre", "genre_ladder"]]
            st.dataframe(acoustic, use_container_width=True, hide_index=True)

        with col2:
            st.write("**Most Electronic Songs (by Genre)**")
            electronic = df.nlargest(10, "genre_ladder")[["track_name", "artist", "top_genre", "genre_ladder"]]
            st.dataframe(electronic, use_container_width=True, hide_index=True)

        # Compare with mood_acoustic/mood_electronic
        if "mood_acoustic" in df.columns and "mood_electronic" in df.columns:
            st.markdown("---")
            st.subheader("Genre Ladder vs Audio Production Analysis")

            fig = px.scatter(
                df,
                x="genre_ladder",
                y="mood_electronic",
                color="cluster",
                hover_data=["track_name", "artist", "top_genre"],
                labels={
                    "genre_ladder": "Genre Ladder (0=Acoustic, 1=Electronic)",
                    "mood_electronic": "Audio Electronic Score",
                },
                title="Genre Taxonomy vs Audio Production",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig, use_container_width=True)

            correlation = df["genre_ladder"].corr(df["mood_electronic"])
            st.metric("Correlation", f"{correlation:.3f}")
