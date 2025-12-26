"""Sub-cluster Comparison Component

Statistical comparison and visualization of subclusters within a parent cluster:
- Pairwise statistical tests (t-tests, Cohen's d)
- Multi-subcluster radar plots showing feature profiles
- Genre comparison and overlap analysis
- Sample songs from each subcluster
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict
from scipy import stats

from analysis.components.visualization.color_palette import CLUSTER_COLORS


def compute_subcluster_defining_features(
    df: pd.DataFrame, n_top: int = 3
) -> dict[int, list[tuple[str, float, str]]]:
    """Compute the top defining features for each subcluster.

    For each subcluster, finds the features where it deviates most from
    the overall parent cluster mean (using standardized z-scores).

    Args:
        df: DataFrame with 'subcluster' column and emb_* feature columns
        n_top: Number of top features to return per subcluster

    Returns:
        Dict mapping subcluster_id -> list of (feature_name, z_score, direction)
        where direction is "high" or "low" relative to parent cluster mean
    """
    # Get embedding feature columns
    emb_cols = [col for col in df.columns if col.startswith("emb_")]
    if not emb_cols:
        return {}

    # Compute overall mean and std for each feature
    overall_means = df[emb_cols].mean()
    overall_stds = df[emb_cols].std()

    # Avoid division by zero
    overall_stds = overall_stds.replace(0, 1)

    results = {}
    for sc_id in sorted(df["subcluster"].unique()):
        sc_df = df[df["subcluster"] == sc_id]
        sc_means = sc_df[emb_cols].mean()

        # Compute z-scores (how many stds away from overall mean)
        z_scores = (sc_means - overall_means) / overall_stds

        # Get top features by absolute z-score
        top_features = []
        for feature in z_scores.abs().nlargest(n_top).index:
            z = z_scores[feature]
            direction = "high" if z > 0 else "low"
            # Clean up feature name (remove emb_ prefix)
            display_name = feature.replace("emb_", "")
            top_features.append((display_name, z, direction))

        results[sc_id] = top_features

    return results


def compute_dissimilarity_matrix(
    df: pd.DataFrame,
    subcluster_ids: list,
    method: str = "centroid"
) -> pd.DataFrame:
    """
    Compute pairwise dissimilarity matrix between subclusters.

    Args:
        df: DataFrame with subcluster assignments and feature columns
        subcluster_ids: List of subcluster IDs to include
        method: "centroid" (Euclidean between centroids) or "effect_size" (avg absolute Cohen's d)

    Returns:
        DataFrame with dissimilarity values (symmetric matrix)
    """
    # Get numeric feature columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['cluster', 'subcluster', 'umap_x', 'umap_y', 'umap_z', 'track_id']
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    n = len(subcluster_ids)
    matrix = np.zeros((n, n))

    if method == "centroid":
        # Compute centroids for each subcluster
        centroids = {}
        for sc_id in subcluster_ids:
            sc_df = df[df['subcluster'] == sc_id][feature_cols]
            centroids[sc_id] = sc_df.mean().values

        # Compute pairwise Euclidean distances
        for i, sc_a in enumerate(subcluster_ids):
            for j, sc_b in enumerate(subcluster_ids):
                if i == j:
                    matrix[i, j] = 0.0
                else:
                    dist = np.linalg.norm(centroids[sc_a] - centroids[sc_b])
                    matrix[i, j] = dist

    elif method == "effect_size":
        # Use average absolute Cohen's d across all features
        for i, sc_a in enumerate(subcluster_ids):
            for j, sc_b in enumerate(subcluster_ids):
                if i == j:
                    matrix[i, j] = 0.0
                elif j > i:
                    # Compute average absolute effect size
                    comparison = compare_two_subclusters(df, sc_a, sc_b)
                    if len(comparison) > 0:
                        avg_effect = comparison['effect_size'].abs().mean()
                        matrix[i, j] = avg_effect
                        matrix[j, i] = avg_effect
                    else:
                        matrix[i, j] = 0.0
                        matrix[j, i] = 0.0

    # Create DataFrame with labels
    labels = [f"SC {sc_id}" for sc_id in subcluster_ids]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def render_dissimilarity_matrix(
    df: pd.DataFrame,
    subcluster_ids: list,
) -> None:
    """
    Render dissimilarity matrix heatmap for subclusters.

    Args:
        df: DataFrame with subcluster assignments
        subcluster_ids: List of subcluster IDs to compare
    """
    st.markdown("### 🔲 Dissimilarity Matrix")
    st.caption("Pairwise distances between sub-cluster centroids in feature space")

    method = st.radio(
        "Distance metric",
        options=["centroid", "effect_size"],
        format_func=lambda x: "Centroid Distance" if x == "centroid" else "Avg Effect Size",
        horizontal=True,
        key="dissimilarity_method",
    )

    with st.spinner("Computing dissimilarity matrix..."):
        dissim_matrix = compute_dissimilarity_matrix(df, subcluster_ids, method=method)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=dissim_matrix.values,
        x=dissim_matrix.columns,
        y=dissim_matrix.index,
        colorscale="Viridis",
        text=np.round(dissim_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Distance: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        height=400,
        xaxis_title="Sub-cluster",
        yaxis_title="Sub-cluster",
        yaxis=dict(autorange="reversed"),  # Match matrix orientation
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show interpretation
    if method == "centroid":
        st.caption("💡 Higher values = more dissimilar sub-clusters (Euclidean distance in standardized feature space)")
    else:
        st.caption("💡 Higher values = more statistically different sub-clusters (average |Cohen's d| across features)")

    # Identify most/least similar pairs
    n = len(subcluster_ids)
    if n >= 2:
        # Extract upper triangle (excluding diagonal)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((
                    subcluster_ids[i],
                    subcluster_ids[j],
                    dissim_matrix.iloc[i, j]
                ))

        if pairs:
            pairs_sorted = sorted(pairs, key=lambda x: x[2])
            most_similar = pairs_sorted[0]
            most_different = pairs_sorted[-1]

            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Most similar:** Sub-cluster {most_similar[0]} & {most_similar[1]} (dist: {most_similar[2]:.2f})")
            with col2:
                st.error(f"**Most different:** Sub-cluster {most_different[0]} & {most_different[1]} (dist: {most_different[2]:.2f})")


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (group1.mean() - group2.mean()) / pooled_std


def compare_two_subclusters(
    df: pd.DataFrame,
    subcluster_a: int,
    subcluster_b: int,
) -> pd.DataFrame:
    """
    Compare two subclusters using statistical tests.

    Returns DataFrame with feature comparisons including:
    - Mean values for each subcluster
    - Difference
    - t-statistic and p-value
    - Effect size (Cohen's d)
    """
    df_a = df[df["subcluster"] == subcluster_a]
    df_b = df[df["subcluster"] == subcluster_b]

    # Features to compare
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['cluster', 'subcluster', 'umap_x', 'umap_y', 'umap_z', 'track_id']
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    results = []
    for feature in feature_cols:
        vals_a = df_a[feature].dropna().values
        vals_b = df_b[feature].dropna().values

        if len(vals_a) < 2 or len(vals_b) < 2:
            continue

        # T-test
        t_stat, p_value = stats.ttest_ind(vals_a, vals_b, equal_var=False)

        # Effect size
        effect_size = compute_cohens_d(vals_a, vals_b)

        results.append({
            'feature': feature,
            'subcluster_a_mean': vals_a.mean(),
            'subcluster_b_mean': vals_b.mean(),
            'difference': vals_a.mean() - vals_b.mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05,
        })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values('effect_size', key=abs, ascending=False)

    return results_df


def render_subcluster_comparison(subcluster_data: Dict) -> None:
    """
    Render subcluster comparison interface.

    Args:
        subcluster_data: Dictionary from run_subcluster_pipeline()
    """
    df = subcluster_data['subcluster_df']
    parent_cluster = subcluster_data['parent_cluster']
    n_subclusters = subcluster_data['n_subclusters']

    st.markdown("### ⚖️ Sub-Cluster Comparison")
    st.write(f"Compare sub-clusters within Cluster {parent_cluster}")

    # Subcluster selection
    subcluster_ids = sorted(df['subcluster'].unique())

    selected_subclusters = st.multiselect(
        "Select Sub-Clusters to Compare (2 or more)",
        options=subcluster_ids,
        default=subcluster_ids[:min(2, len(subcluster_ids))],
        format_func=lambda x: f"Sub-cluster {x}",
        key="subcluster_comparison_select",
    )

    if len(selected_subclusters) < 2:
        st.warning("⚠️ Please select at least 2 sub-clusters to compare")
        return

    st.info(
        f"📊 Comparing {len(selected_subclusters)} sub-clusters: "
        f"{', '.join([f'Sub-cluster {c}' for c in selected_subclusters])}"
    )

    # Overview table
    st.markdown("---")
    st.subheader("📊 Sub-Cluster Overview")

    overview_data = []
    for sc_id in selected_subclusters:
        sc_df = df[df['subcluster'] == sc_id]
        row = {
            "Sub-cluster": f"Sub-cluster {sc_id}",
            "Size": len(sc_df),
            "Percentage": f"{len(sc_df) / len(df) * 100:.1f}%",
        }

        if "bpm" in df.columns:
            row["Avg BPM"] = f"{sc_df['bpm'].mean():.1f}"
        if "danceability" in df.columns:
            row["Avg Danceability"] = f"{sc_df['danceability'].mean():.2f}"
        if "mood_happy" in df.columns:
            row["Avg Happiness"] = f"{sc_df['mood_happy'].mean():.2f}"
        if "valence" in df.columns:
            row["Avg Valence"] = f"{sc_df['valence'].mean():.2f}"

        overview_data.append(row)

    overview_df = pd.DataFrame(overview_data)
    st.dataframe(overview_df, use_container_width=True, hide_index=True)

    # Defining Characteristics
    st.markdown("---")
    st.subheader("🎯 Top 3 Defining Characteristics")
    st.caption(
        "Features where each sub-cluster deviates most from the parent cluster average "
        "(z-score = standard deviations from mean)"
    )

    defining_features = compute_subcluster_defining_features(df, n_top=3)

    if defining_features:
        # Create columns for selected subclusters
        n_cols = min(len(selected_subclusters), 4)
        cols = st.columns(n_cols)

        for idx, sc_id in enumerate(selected_subclusters):
            col_idx = idx % n_cols
            features = defining_features.get(sc_id, [])

            with cols[col_idx]:
                st.markdown(f"**Sub-cluster {sc_id}**")
                if features:
                    for feature_name, z_score, direction in features:
                        arrow = "↑" if direction == "high" else "↓"
                        st.markdown(
                            f"- {arrow} **{feature_name}** ({z_score:+.2f}σ)"
                        )
                else:
                    st.caption("No features found")
    else:
        st.info("No embedding features (emb_*) found for characteristic analysis")

    # Dissimilarity Matrix
    st.markdown("---")
    render_dissimilarity_matrix(df, selected_subclusters)

    # Pairwise Statistical Comparisons
    st.markdown("---")
    st.subheader("📈 Pairwise Statistical Comparisons")

    if len(selected_subclusters) == 2:
        # Detailed comparison for 2 subclusters
        with st.spinner("Running statistical tests..."):
            comparison_df = compare_two_subclusters(
                df, selected_subclusters[0], selected_subclusters[1]
            )

        if len(comparison_df) > 0:
            show_all = st.checkbox(
                "Show all features (including non-significant)",
                value=False,
                key="subcluster_show_all_features",
            )

            if not show_all:
                display_df = comparison_df[comparison_df["significant"]].copy()
                st.write(f"**Showing {len(display_df)} significant differences (p < 0.05)**")
            else:
                display_df = comparison_df.copy()
                st.write(f"**Showing all {len(display_df)} features**")

            if len(display_df) > 0:
                # Format for display
                display_df["subcluster_a_mean"] = display_df["subcluster_a_mean"].round(3)
                display_df["subcluster_b_mean"] = display_df["subcluster_b_mean"].round(3)
                display_df["difference"] = display_df["difference"].round(3)
                display_df["effect_size"] = display_df["effect_size"].round(3)
                display_df["t_statistic"] = display_df["t_statistic"].round(3)
                display_df["p_value"] = display_df["p_value"].apply(lambda x: f"{x:.4f}")

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400,
                    column_config={
                        "feature": "Feature",
                        "subcluster_a_mean": f"Sub-cluster {selected_subclusters[0]} Mean",
                        "subcluster_b_mean": f"Sub-cluster {selected_subclusters[1]} Mean",
                        "difference": "Difference",
                        "effect_size": st.column_config.NumberColumn(
                            "Effect Size",
                            help="Cohen's d - measures practical significance",
                            format="%.3f",
                        ),
                        "t_statistic": "t-statistic",
                        "p_value": "p-value",
                        "significant": st.column_config.CheckboxColumn("Significant?"),
                    },
                    hide_index=True,
                )
            else:
                st.info("No significant differences found between these sub-clusters")
        else:
            st.warning("Unable to compare these sub-clusters")
    else:
        # Summary for 3+ subclusters
        st.write(f"**Showing summary of all pairwise comparisons for {len(selected_subclusters)} sub-clusters**")

        with st.spinner("Computing all pairwise comparisons..."):
            comparison_summaries = []

            for i, sc_a in enumerate(selected_subclusters):
                for sc_b in selected_subclusters[i + 1:]:
                    comparison_df = compare_two_subclusters(df, sc_a, sc_b)

                    if len(comparison_df) > 0:
                        n_significant = comparison_df["significant"].sum()
                        avg_effect_size = comparison_df["effect_size"].abs().mean()

                        comparison_summaries.append({
                            "Sub-cluster A": f"Sub-cluster {sc_a}",
                            "Sub-cluster B": f"Sub-cluster {sc_b}",
                            "Significant Differences": n_significant,
                            "Avg Effect Size": f"{avg_effect_size:.3f}",
                            "Most Different Feature": comparison_df.iloc[0]["feature"],
                        })

            if comparison_summaries:
                summary_df = pd.DataFrame(comparison_summaries)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.info("💡 Select exactly 2 sub-clusters to see detailed statistical comparison")

    # Radar plot comparison
    st.markdown("---")
    st.subheader("🎯 Multi-Dimensional Comparison")

    radar_features = [
        "bpm", "danceability", "valence", "arousal",
        "mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed",
    ]
    radar_features = [f for f in radar_features if f in df.columns]

    if len(radar_features) >= 3:
        # Normalize features
        normalized_df = df[radar_features].copy()
        for col in radar_features:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val > min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)

        normalized_df["subcluster"] = df["subcluster"].values

        # Create radar plot
        fig = go.Figure()
        colors = CLUSTER_COLORS

        for i, sc_id in enumerate(selected_subclusters):
            sc_means = normalized_df[normalized_df["subcluster"] == sc_id][radar_features].mean()

            fig.add_trace(
                go.Scatterpolar(
                    r=sc_means.values,
                    theta=radar_features,
                    fill="toself",
                    name=f"Sub-cluster {sc_id}",
                    line_color=colors[i % len(colors)],
                    opacity=0.6,
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title=f"Sub-Cluster Comparison: {len(selected_subclusters)} Sub-clusters (Normalized)",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 All features normalized to 0-1 scale for fair comparison.")

    # Genre comparison
    st.markdown("---")
    st.subheader("🎸 Genre Comparison")

    if "top_genre" in df.columns:
        num_cols = min(len(selected_subclusters), 3)
        cols = st.columns(num_cols)

        for i, sc_id in enumerate(selected_subclusters):
            sc_df = df[df["subcluster"] == sc_id]
            col_idx = i % num_cols

            with cols[col_idx]:
                st.write(f"**Sub-cluster {sc_id} - Top 5 Genres**")
                sc_genres = sc_df["top_genre"].value_counts().head(5)

                fig = px.bar(
                    x=sc_genres.values,
                    y=sc_genres.index,
                    orientation="h",
                    labels={"x": "Count", "y": "Genre"},
                    color_discrete_sequence=[colors[i % len(colors)]],
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # Sample songs
    st.markdown("---")
    st.subheader("🎵 Sample Songs from Each Sub-Cluster")

    num_cols = min(len(selected_subclusters), 3)
    cols = st.columns(num_cols)

    for i, sc_id in enumerate(selected_subclusters):
        sc_df = df[df["subcluster"] == sc_id]
        col_idx = i % num_cols

        with cols[col_idx]:
            st.write(f"**Sub-cluster {sc_id} - Sample**")
            sample_df = sc_df.sample(min(8, len(sc_df)))[["track_name", "artist"]]
            st.dataframe(sample_df, use_container_width=True, hide_index=True)
