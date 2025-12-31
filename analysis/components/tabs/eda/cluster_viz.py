"""3D cluster visualization and data export for EDA explorer."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from analysis.components.visualization.color_palette import CLUSTER_COLORS
from analysis.pipeline.config import get_cluster_name
from analysis.components.export.chart_export import render_chart_with_export


def _format_embedding_table(row, emb_cols):
    """Format embedding features in 2 columns using | separator."""
    if not emb_cols:
        return ""

    valid_embs = [(col, row[col]) for col in emb_cols if col in row and pd.notna(row[col])]

    if not valid_embs:
        return ""

    text = "<br><b>Embedding Vector:</b><br>"

    mid = (len(valid_embs) + 1) // 2
    left_cols = valid_embs[:mid]
    right_cols = valid_embs[mid:]

    for i in range(max(len(left_cols), len(right_cols))):
        left_text = ""
        right_text = ""

        if i < len(left_cols):
            col_name, col_val = left_cols[i]
            display_name = col_name.replace("emb_", "")
            left_text = f"{display_name}: {col_val:.3f}"

        if i < len(right_cols):
            col_name, col_val = right_cols[i]
            display_name = col_name.replace("emb_", "")
            right_text = f"{display_name}: {col_val:.3f}"

        if right_text:
            text += f"{left_text} │ {right_text}<br>"
        elif left_text:
            text += f"{left_text}<br>"

    return text


def render_3d_cluster_visualization(df: pd.DataFrame):
    """Render 3D cluster visualization section."""
    with st.expander("🗺️ Interactive 3D Cluster Map", expanded=False):
        st.subheader("3D UMAP Visualization of Clusters")

        if "umap_x" not in df.columns or "umap_y" not in df.columns or "umap_z" not in df.columns:
            st.warning("UMAP coordinates not found")
            return

        # Check for subcluster data
        subcluster_data = st.session_state.get("subcluster_data")
        show_subclusters = False

        if subcluster_data is not None:
            show_subclusters = st.checkbox(
                f"🔍 Show Sub-Clusters of Cluster {subcluster_data['parent_cluster']}",
                value=True,
                help="Color parent cluster by subcluster, dim other clusters",
            )

        fig = go.Figure()
        unique_clusters = sorted(df["cluster"].unique())
        colors = CLUSTER_COLORS
        emb_cols = [col for col in df.columns if col.startswith("emb_")]

        if show_subclusters and subcluster_data is not None:
            parent_cluster = subcluster_data["parent_cluster"]
            subcluster_df = subcluster_data["subcluster_df"]

            for i, cluster_id in enumerate(unique_clusters):
                cluster_df = df[df["cluster"] == cluster_id]

                if cluster_id == parent_cluster:
                    # Plot each subcluster with distinct colors
                    subcluster_ids = sorted(subcluster_df["subcluster"].unique())
                    for j, sc_id in enumerate(subcluster_ids):
                        sc_mask = subcluster_df["subcluster"] == sc_id
                        sc_tracks = subcluster_df[sc_mask]

                        track_ids = sc_tracks["track_id"].values
                        main_mask = cluster_df["track_id"].isin(track_ids)
                        sc_cluster_df = cluster_df[main_mask]

                        if len(sc_cluster_df) == 0:
                            continue

                        hover_texts = []
                        for _, row in sc_cluster_df.iterrows():
                            text = (
                                f"<b>{row['track_name']}</b><br>"
                                f"Artist: {row['artist']}<br>"
                                f"Cluster: {get_cluster_name(row['cluster'])}<br>"
                                f"<b>Sub-cluster: {sc_id}</b><br>"
                            )
                            if "top_genre" in row:
                                text += f"Genre: {row['top_genre']}<br>"
                            if "bpm" in row:
                                text += f"BPM: {row['bpm']:.0f}<br>"
                            text += _format_embedding_table(row, emb_cols)
                            hover_texts.append(text)

                        fig.add_trace(go.Scatter3d(
                            x=sc_cluster_df["umap_x"],
                            y=sc_cluster_df["umap_y"],
                            z=sc_cluster_df["umap_z"],
                            mode="markers",
                            name=f"Sub-cluster {sc_id} ({len(sc_cluster_df)})",
                            marker=dict(size=5, color=colors[j % len(colors)], opacity=0.9),
                            text=hover_texts,
                            hovertemplate="%{text}<extra></extra>",
                        ))
                else:
                    # Gray out non-parent clusters
                    hover_texts = []
                    for _, row in cluster_df.iterrows():
                        text = (
                            f"<b>{row['track_name']}</b><br>"
                            f"Artist: {row['artist']}<br>"
                            f"Cluster: {get_cluster_name(row['cluster'])}<br>"
                        )
                        if "top_genre" in row:
                            text += f"Genre: {row['top_genre']}<br>"
                        hover_texts.append(text)

                    fig.add_trace(go.Scatter3d(
                        x=cluster_df["umap_x"],
                        y=cluster_df["umap_y"],
                        z=cluster_df["umap_z"],
                        mode="markers",
                        name=f"{get_cluster_name(cluster_id)} ({len(cluster_df)})",
                        marker=dict(size=3, color="rgba(128,128,128,0.3)", opacity=0.3),
                        text=hover_texts,
                        hovertemplate="%{text}<extra></extra>",
                    ))

            st.caption(f"Sub-Clusters of Cluster {parent_cluster}")
        else:
            # Standard view: all clusters with full colors
            for i, cluster_id in enumerate(unique_clusters):
                cluster_df = df[df["cluster"] == cluster_id]

                hover_texts = []
                for _, row in cluster_df.iterrows():
                    text = (
                        f"<b>{row['track_name']}</b><br>"
                        f"Artist: {row['artist']}<br>"
                        f"Cluster: {get_cluster_name(row['cluster'])}<br>"
                    )
                    if "top_genre" in row:
                        text += f"Genre: {row['top_genre']}<br>"
                    if "bpm" in row:
                        text += f"BPM: {row['bpm']:.0f}<br>"
                    text += _format_embedding_table(row, emb_cols)
                    hover_texts.append(text)

                fig.add_trace(go.Scatter3d(
                    x=cluster_df["umap_x"],
                    y=cluster_df["umap_y"],
                    z=cluster_df["umap_z"],
                    mode="markers",
                    name=f"{get_cluster_name(cluster_id)} ({len(cluster_df)})",
                    marker=dict(size=4, color=colors[i % len(colors)], opacity=0.8),
                    text=hover_texts,
                    hovertemplate="%{text}<extra></extra>",
                ))

            st.caption("3D Cluster Visualization")
        fig.update_layout(
            height=800,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ),
            showlegend=True,
            margin=dict(t=0, l=0, r=0, b=0),
        )

        render_chart_with_export(fig, "cluster_3d_map", "3D Cluster Map", "cluster")


def render_data_preview_export(df: pd.DataFrame):
    """Render data preview and export section."""
    with st.expander("🔍 Data Preview & Export", expanded=False):
        st.subheader("Full Dataset Preview")

        display_df = df.copy()
        for col in display_df.columns:
            if display_df[col].dtype == "object":
                display_df[col] = display_df[col].astype(str)

        st.dataframe(display_df, use_container_width=True, height=400)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="cluster_data.csv",
            mime="text/csv",
        )
