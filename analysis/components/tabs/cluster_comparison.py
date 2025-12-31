"""Cluster Comparison Tab Component

Statistical comparison and visualization of multiple clusters:
- Pairwise statistical tests (t-tests, Cohen's d)
- Multi-cluster radar plots showing feature profiles
- Genre comparison and overlap analysis
- Sample songs from each cluster
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from analysis.interpretability.cluster_comparison import (
    compare_two_clusters,
    compute_cluster_similarity_matrix,
)
from analysis.components.visualization.color_palette import (
    get_cluster_color,
    CLUSTER_COLORS,
)
from analysis.pipeline.config import get_cluster_name
from analysis.components.export.chart_export import render_chart_with_export


def render_cluster_comparison(df: pd.DataFrame):
    """Render Cluster Comparison view."""
    st.header("⚖️ Statistical Cluster Comparison")

    st.write("Compare multiple clusters using statistical tests and visualizations.")

    # Cluster selection - allow multiple
    cluster_ids = sorted(df["cluster"].unique())

    # Multi-select for clusters
    selected_clusters = st.multiselect(
        "Select Clusters to Compare (select 2 or more)",
        options=cluster_ids,
        default=cluster_ids[: min(2, len(cluster_ids))],
        format_func=lambda x: get_cluster_name(x),
        help="Select 2 or more clusters to compare. The radar plot will show all selected clusters.",
    )

    if len(selected_clusters) < 2:
        st.warning("⚠️ Please select at least 2 clusters to compare")
        return

    # Show number of clusters being compared
    st.info(
        f"📊 Comparing {len(selected_clusters)} clusters: {', '.join([f'Cluster {c}' for c in selected_clusters])}"
    )

    # Basic cluster information
    st.markdown("---")
    st.subheader("📊 Cluster Overview")

    # Create overview table for all selected clusters
    overview_data = []
    for cluster_id in selected_clusters:
        cluster_df = df[df["cluster"] == cluster_id]
        row = {
            "Cluster": get_cluster_name(cluster_id),
            "Size": len(cluster_df),
            "Percentage": f"{len(cluster_df) / len(df) * 100:.1f}%",
        }

        if "bpm" in df.columns:
            row["Avg BPM"] = f"{cluster_df['bpm'].mean():.1f}"
        if "danceability" in df.columns:
            row["Avg Danceability"] = f"{cluster_df['danceability'].mean():.2f}"
        if "mood_happy" in df.columns:
            row["Avg Happiness"] = f"{cluster_df['mood_happy'].mean():.2f}"
        if "valence" in df.columns:
            row["Avg Valence"] = f"{cluster_df['valence'].mean():.2f}"

        overview_data.append(row)

    overview_df = pd.DataFrame(overview_data)
    st.dataframe(overview_df, use_container_width=True, hide_index=True)

    # Pairwise Statistical Comparisons
    st.markdown("---")
    st.subheader("📈 Pairwise Statistical Comparisons")

    if len(selected_clusters) == 2:
        # For 2 clusters, show detailed comparison
        with st.spinner("Running statistical tests..."):
            comparison_df = compare_two_clusters(
                df, selected_clusters[0], selected_clusters[1]
            )

            if len(comparison_df) > 0:
                # Show only significant differences by default
                show_all = st.checkbox(
                    "Show all features (including non-significant)", value=False
                )

                if not show_all:
                    display_df = comparison_df[comparison_df["significant"]].copy()
                    st.write(
                        f"**Showing {len(display_df)} significant differences (p < 0.05)**"
                    )
                else:
                    display_df = comparison_df.copy()
                    st.write(f"**Showing all {len(display_df)} features**")

                if len(display_df) > 0:
                    # Format for display
                    display_df["cluster_a_mean"] = display_df["cluster_a_mean"].round(3)
                    display_df["cluster_b_mean"] = display_df["cluster_b_mean"].round(3)
                    display_df["difference"] = display_df["difference"].round(3)
                    display_df["effect_size"] = display_df["effect_size"].round(3)
                    display_df["t_statistic"] = display_df["t_statistic"].round(3)
                    display_df["p_value"] = display_df["p_value"].apply(
                        lambda x: f"{x:.4f}"
                    )

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=400,
                        column_config={
                            "feature": "Feature",
                            "cluster_a_mean": f"{get_cluster_name(selected_clusters[0])} Mean",
                            "cluster_b_mean": f"{get_cluster_name(selected_clusters[1])} Mean",
                            "difference": "Difference",
                            "effect_size": st.column_config.NumberColumn(
                                "Effect Size",
                                help="Cohen's d - measures practical significance",
                                format="%.3f",
                            ),
                            "t_statistic": "t-statistic",
                            "p_value": "p-value",
                            "significant": st.column_config.CheckboxColumn(
                                "Significant?"
                            ),
                        },
                        hide_index=True,
                    )

                    # Download button
                    csv = comparison_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"📥 Download Comparison Data",
                        data=csv,
                        file_name=f"cluster_{selected_clusters[0]}_vs_{selected_clusters[1]}_comparison.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No significant differences found between these clusters")
            else:
                st.warning("Unable to compare these clusters")
    else:
        # For 3+ clusters, show summary of all pairwise comparisons
        st.write(
            f"**Showing summary of all pairwise comparisons for {len(selected_clusters)} clusters**"
        )

        with st.spinner("Computing all pairwise comparisons..."):
            # Compute all pairs
            comparison_summaries = []

            for i, cluster_a in enumerate(selected_clusters):
                for cluster_b in selected_clusters[i + 1 :]:
                    comparison_df = compare_two_clusters(df, cluster_a, cluster_b)

                    if len(comparison_df) > 0:
                        # Count significant differences
                        n_significant = comparison_df["significant"].sum()
                        avg_effect_size = comparison_df["effect_size"].abs().mean()

                        comparison_summaries.append(
                            {
                                "Cluster A": get_cluster_name(cluster_a),
                                "Cluster B": get_cluster_name(cluster_b),
                                "Significant Differences": n_significant,
                                "Avg Effect Size": f"{avg_effect_size:.3f}",
                                "Most Different Feature": comparison_df.iloc[0][
                                    "feature"
                                ],
                            }
                        )

            summary_df = pd.DataFrame(comparison_summaries)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.info(
                "💡 Select exactly 2 clusters to see detailed statistical comparison"
            )

    # Radar plot comparison - now supports multiple clusters!
    st.markdown("---")
    st.subheader("🎯 Multi-Dimensional Comparison")

    # Select key features for radar plot (representative features from all 33 dims)
    radar_features = [
        # Audio core
        "danceability",
        "instrumentalness",
        "valence",
        "arousal",
        # Audio moods
        "mood_happy",
        "mood_sad",
        "mood_aggressive",
        "mood_relaxed",
        "mood_party",
        # Audio texture
        "emb_acoustic_electronic",
        "emb_timbre_brightness",
        # Lyric core
        "lyric_valence",
        "lyric_arousal",
        # Lyric content
        "lyric_explicit",
        "lyric_narrative",
    ]
    radar_features = [f for f in radar_features if f in df.columns]

    if len(radar_features) >= 3:
        # Normalize features to 0-1 scale for fair comparison
        normalized_df = df[radar_features].copy()
        for col in radar_features:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val > min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (
                    max_val - min_val
                )

        # Add cluster column back
        normalized_df["cluster"] = df["cluster"].values

        # Create radar plot with all selected clusters
        fig = go.Figure()

        colors = CLUSTER_COLORS

        for i, cluster_id in enumerate(selected_clusters):
            cluster_means = normalized_df[normalized_df["cluster"] == cluster_id][
                radar_features
            ].mean()

            fig.add_trace(
                go.Scatterpolar(
                    r=cluster_means.values,
                    theta=radar_features,
                    fill="toself",
                    name=get_cluster_name(cluster_id),
                    line_color=colors[i % len(colors)],
                    opacity=0.6,
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=700,
            margin=dict(t=0, l=0, r=0, b=0),
        )

        render_chart_with_export(fig, "cluster_comparison_radar", "Multi-Cluster Comparison", "comparison")
        st.caption(
            "💡 All features normalized to 0-1 scale for fair comparison. Each cluster shown as a different color."
        )

    # Genre comparison
    st.markdown("---")
    st.subheader("🎸 Genre Comparison")

    if "top_genre" in df.columns:
        # Show top genres for each selected cluster
        num_cols = min(len(selected_clusters), 3)  # Max 3 columns
        cols = st.columns(num_cols)

        for i, cluster_id in enumerate(selected_clusters):
            cluster_df = df[df["cluster"] == cluster_id]
            col_idx = i % num_cols

            with cols[col_idx]:
                st.write(f"**Cluster {cluster_id} - Top 10 Genres**")
                cluster_genres = cluster_df["top_genre"].value_counts().head(10)

                fig = px.bar(
                    x=cluster_genres.values,
                    y=cluster_genres.index,
                    orientation="h",
                    labels={"x": "Count", "y": "Genre"},
                    color_discrete_sequence=[
                        get_cluster_color(i)
                    ],
                )
                fig.update_layout(height=500, showlegend=False, margin=dict(t=0, l=0, r=0, b=0))
                st.plotly_chart(fig, use_container_width=True)

        # Genre overlap analysis for all selected clusters
        st.markdown("---")
        st.write(f"**Genre Overlap Analysis ({len(selected_clusters)} clusters)**")

        genre_sets = {
            cluster_id: set(df[df["cluster"] == cluster_id]["top_genre"].unique())
            for cluster_id in selected_clusters
        }

        # Find shared genres across all clusters
        shared_genres = (
            set.intersection(*genre_sets.values()) if len(genre_sets) > 0 else set()
        )

        overlap_data = []
        for cluster_id in selected_clusters:
            # Genres unique to this cluster
            other_clusters = [c for c in selected_clusters if c != cluster_id]
            other_genres = (
                set.union(*[genre_sets[c] for c in other_clusters])
                if other_clusters
                else set()
            )
            unique_genres = genre_sets[cluster_id] - other_genres

            overlap_data.append(
                {
                    "Cluster": get_cluster_name(cluster_id),
                    "Total Genres": len(genre_sets[cluster_id]),
                    "Unique Genres": len(unique_genres),
                    "Shared with All": len(shared_genres),
                }
            )

        overlap_df = pd.DataFrame(overlap_data)
        st.dataframe(overlap_df, use_container_width=True, hide_index=True)

        if len(shared_genres) > 0:
            st.info(
                f"🎵 {len(shared_genres)} genres appear in all selected clusters: {', '.join(list(shared_genres)[:10])}{('...' if len(shared_genres) > 10 else '')}"
            )

    # Artist comparison
    st.markdown("---")
    st.subheader("🎤 Top Artists by Cluster")

    if "artist" in df.columns:
        # For each selected cluster, show top N artists
        num_cols = min(len(selected_clusters), 3)
        cols = st.columns(num_cols)

        for i, cluster_id in enumerate(selected_clusters):
            cluster_df = df[df["cluster"] == cluster_id]
            col_idx = i % num_cols

            with cols[col_idx]:
                st.write(f"**Cluster {cluster_id} - Top 10 Artists**")

                # Count songs per artist
                artist_counts = cluster_df["artist"].value_counts().head(10)

                if len(artist_counts) > 0:
                    # Create bar chart
                    fig = px.bar(
                        x=artist_counts.values,
                        y=artist_counts.index,
                        orientation="h",
                        labels={"x": "Song Count", "y": "Artist"},
                        color_discrete_sequence=[get_cluster_color(i)],
                    )
                    fig.update_layout(height=500, showlegend=False, margin=dict(t=0, l=0, r=0, b=0))
                    st.plotly_chart(fig, use_container_width=True, key=f"cluster_artist_chart_{cluster_id}")

                    # Show percentage
                    total_songs = len(cluster_df)
                    st.caption(
                        f"Top artist: {artist_counts.index[0]} ({artist_counts.values[0]} songs, {artist_counts.values[0]/total_songs*100:.1f}%)"
                    )
                else:
                    st.info("No artist data available for this cluster")
    else:
        st.info("Artist data not available in dataset")

    # Sample songs from each cluster
    st.markdown("---")
    st.subheader("🎵 Sample Songs from Each Cluster")

    num_cols = min(len(selected_clusters), 3)  # Max 3 columns
    cols = st.columns(num_cols)

    for i, cluster_id in enumerate(selected_clusters):
        cluster_df = df[df["cluster"] == cluster_id]
        col_idx = i % num_cols

        with cols[col_idx]:
            st.write(f"**Cluster {cluster_id} - Random Sample**")
            sample_df = cluster_df.sample(min(10, len(cluster_df)))[
                ["track_name", "artist"]
            ]
            st.dataframe(sample_df, use_container_width=True, hide_index=True)
