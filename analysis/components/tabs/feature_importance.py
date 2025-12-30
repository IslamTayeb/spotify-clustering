"""Feature Importance Tab Component

Analyzes which features make each cluster distinctive using:
- Cohen's d effect sizes
- Feature importance rankings
- Cross-cluster heatmaps
- Violin plots showing feature distributions
- Statistical summaries
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from analysis.interpretability.feature_importance import (
    get_top_features,
    get_feature_interpretation,
)
from analysis.components.visualization.color_palette import get_cluster_color


@st.cache_data
def compute_all_cluster_importance(df: pd.DataFrame):
    """Compute feature importance for all clusters (cached)."""
    importance_data = {}
    cluster_ids = sorted(df["cluster"].unique())

    for cluster_id in cluster_ids:
        result = get_top_features(df, cluster_id, n=20)
        importance_data[cluster_id] = result

    return importance_data


def render_feature_importance(df: pd.DataFrame):
    """Render Feature Importance view."""
    st.header("🎯 Feature Importance Analysis")

    st.write(
        "Identify which features make each cluster distinctive using Cohen's d effect sizes."
    )

    # Cluster selection
    cluster_ids = sorted(df["cluster"].unique())
    selected_cluster = st.selectbox(
        "Select Cluster to Analyze",
        options=cluster_ids,
        format_func=lambda x: f"Cluster {x}",
    )

    with st.spinner("Computing feature importance..."):
        # Get feature importance for selected cluster
        cluster_info = get_top_features(df, selected_cluster, n=20)

        # Display cluster info metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Cluster Size", cluster_info["cluster_size"])

        with col2:
            st.metric(
                "Percentage of Library", f"{cluster_info['cluster_percentage']:.1f}%"
            )

        with col3:
            # Get top feature
            if len(cluster_info["top_features"]) > 0:
                top_feature = cluster_info["top_features"].iloc[0]
                st.metric(
                    f"Top Feature: {top_feature['feature']}",
                    f"Effect size: {top_feature['effect_size']:.2f}",
                )

        st.markdown("---")

        # Top 3 distinctive features with interpretations
        st.subheader(
            f"🌟 Top 3 Most Distinctive Features for Cluster {selected_cluster}"
        )

        if len(cluster_info["top_features"]) >= 3:
            for i, row in cluster_info["top_features"].head(3).iterrows():
                feature = row["feature"]
                effect_size = row["effect_size"]
                cluster_mean = row["cluster_mean"]
                global_mean = row["global_mean"]

                interpretation = get_feature_interpretation(effect_size)

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**{i}. {feature}**")
                    st.write(
                        f"Cluster mean: {cluster_mean:.3f} | Global mean: {global_mean:.3f}"
                    )
                    st.write(f"Effect size: **{effect_size:.2f}** ({interpretation})")

                with col2:
                    # Simple visual indicator
                    if abs(effect_size) >= 0.8:
                        st.success("🔥 Large effect")
                    elif abs(effect_size) >= 0.5:
                        st.info("⚡ Medium effect")
                    else:
                        st.warning("✨ Small effect")

        # Full feature importance table
        st.markdown("---")
        st.subheader("📊 Full Feature Importance Ranking")

        # Prepare dataframe for display
        display_df = cluster_info["all_features"].copy()
        display_df["effect_size"] = display_df["effect_size"].round(3)
        display_df["cluster_mean"] = display_df["cluster_mean"].round(3)
        display_df["global_mean"] = display_df["global_mean"].round(3)

        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            column_config={
                "feature": "Feature",
                "effect_size": st.column_config.NumberColumn(
                    "Effect Size (Cohen's d)",
                    help="How many standard deviations this cluster differs from average",
                    format="%.3f",
                ),
                "cluster_mean": "Cluster Mean",
                "global_mean": "Global Mean",
                "importance_rank": "Rank",
            },
            hide_index=True,
        )

        # Download button
        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download Cluster {selected_cluster} Importance Data",
            data=csv,
            file_name=f"cluster_{selected_cluster}_importance.csv",
            mime="text/csv",
        )

    # Feature importance heatmap for all clusters
    st.markdown("---")
    st.subheader("🔥 Feature Importance Heatmap (All Clusters)")

    with st.spinner("Computing importance for all clusters..."):
        all_importance = compute_all_cluster_importance(df)

        # Build heatmap data
        top_n_features = 15  # Top features to show
        feature_names = set()

        # Collect top features from all clusters
        for cluster_id, data in all_importance.items():
            top_features = data["top_features"].head(top_n_features)
            feature_names.update(top_features["feature"].tolist())

        feature_names = sorted(list(feature_names))[:20]  # Limit to 20 features

        # Build matrix
        heatmap_data = []
        for feature in feature_names:
            row = [feature]
            for cluster_id in cluster_ids:
                importance_df = all_importance[cluster_id]["all_features"]
                feature_row = importance_df[importance_df["feature"] == feature]

                if len(feature_row) > 0:
                    effect_size = feature_row.iloc[0]["effect_size"]
                    row.append(effect_size)
                else:
                    row.append(0.0)

            heatmap_data.append(row)

        # Create heatmap
        heatmap_df = pd.DataFrame(
            heatmap_data,
            columns=["Feature"] + [f"Cluster {cid}" for cid in cluster_ids],
        )

        fig = px.imshow(
            heatmap_df.set_index("Feature").T,
            labels=dict(x="Feature", y="Cluster", color="Effect Size"),
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            aspect="auto",
        )

        fig.update_xaxes(side="bottom")
        fig.update_layout(height=400 + len(cluster_ids) * 50)

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "💡 Red = higher than average, Blue = lower than average, White = near average"
        )

    # Distribution violin plots for top features
    st.markdown("---")
    st.subheader("📊 Feature Distribution Comparison")

    # Let user select a feature to visualize
    all_features = cluster_info["all_features"]["feature"].tolist()
    selected_feature = st.selectbox(
        "Select feature to visualize distribution", options=all_features, index=0
    )

    if selected_feature in df.columns:
        st.caption(f"Distribution of '{selected_feature}' Across Clusters")
        # Create violin plot
        fig = go.Figure()

        for cluster_id in cluster_ids:
            cluster_values = df[df["cluster"] == cluster_id][selected_feature].dropna()

            cluster_color = get_cluster_color(cluster_id)
            fig.add_trace(
                go.Violin(
                    y=cluster_values,
                    name=f"Cluster {cluster_id}",
                    box_visible=True,
                    meanline_visible=True,
                    line_color=cluster_color,
                    fillcolor=cluster_color,
                    opacity=0.6,
                )
            )

        fig.update_layout(
            yaxis_title=selected_feature,
            xaxis_title="Cluster",
            showlegend=True,
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add statistical summary
        st.subheader("Statistical Summary")

        summary_data = []
        for cluster_id in cluster_ids:
            cluster_values = df[df["cluster"] == cluster_id][selected_feature].dropna()

            summary_data.append(
                {
                    "Cluster": f"Cluster {cluster_id}",
                    "Mean": cluster_values.mean(),
                    "Median": cluster_values.median(),
                    "Std Dev": cluster_values.std(),
                    "Min": cluster_values.min(),
                    "Max": cluster_values.max(),
                    "Count": len(cluster_values),
                }
            )

        summary_df = pd.DataFrame(summary_data)

        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    else:
        st.warning(f"Feature '{selected_feature}' not found in dataframe")
