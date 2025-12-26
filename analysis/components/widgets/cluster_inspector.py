"""Cluster inspector widget - filterable table view of tracks.

This module provides an interactive table for browsing and filtering tracks by cluster.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Dict, Any

from analysis.components.visualization.color_palette import MOOD_COLORS


def render_cluster_filter(df: pd.DataFrame) -> Optional[int]:
    """Render cluster filter dropdown.

    Args:
        df: DataFrame with 'cluster' column

    Returns:
        Selected cluster ID or None for "All"
    """
    unique_labels = sorted(df["cluster"].unique())

    options = ["All"] + [
        f"Cluster {lbl}" if lbl != -1 else "Outliers"
        for lbl in unique_labels
    ]

    selected = st.selectbox(
        "Filter List by Cluster",
        options,
        help="Filter the track list to show only tracks from a specific cluster",
    )

    if selected == "All":
        return None
    elif selected == "Outliers":
        return -1
    else:
        return int(selected.split(" ")[1])


def render_cluster_table(
    df: pd.DataFrame,
    selected_cluster: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Render interactive cluster table with track selection.

    Args:
        df: DataFrame with track data
        selected_cluster: Cluster ID to filter by (None = all clusters)

    Returns:
        Dictionary of selected track data or None
    """
    # Filter by cluster if specified
    if selected_cluster is not None:
        view_df = df[df["cluster"] == selected_cluster]
    else:
        view_df = df

    # Columns to display
    cols_to_show = [
        "track_name",
        "artist",
        "cluster",
        "top_genre",
        "bpm",
        "key",
        "mood_happy",
        "mood_sad",
        "mood_party",
        "danceability",
    ]

    # Ensure columns exist
    display_df = view_df[[c for c in cols_to_show if c in view_df.columns]].copy()

    st.caption("👇 Click on a row to view track details")

    # Render interactive dataframe
    selection = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="cluster_inspector_table",
        column_config={
            "track_name": "Track",
            "artist": "Artist",
            "cluster": "Cluster",
            "top_genre": "Genre",
            "bpm": st.column_config.NumberColumn("BPM", format="%d"),
            "key": "Key",
            "mood_happy": st.column_config.ProgressColumn(
                "Happy", min_value=0, max_value=1
            ),
            "mood_sad": st.column_config.ProgressColumn(
                "Sad", min_value=0, max_value=1
            ),
            "mood_party": st.column_config.ProgressColumn(
                "Party", min_value=0, max_value=1
            ),
            "danceability": st.column_config.ProgressColumn(
                "Danceability", min_value=0, max_value=1
            ),
        },
    )

    # Return selected track
    if selection and selection.get("selection", {}).get("rows"):
        row_idx = selection["selection"]["rows"][0]
        if row_idx < len(view_df):
            return view_df.iloc[row_idx].to_dict()

    return None


def render_track_details(track: Dict[str, Any]) -> None:
    """Render detailed view of a selected track.

    Args:
        track: Dictionary of track metadata
    """
    st.markdown("---")
    st.subheader("🎵 Track Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{track.get('track_name', 'Unknown')}**")
        st.write(f"Artist: {track.get('artist', 'Unknown')}")
        st.write(f"Cluster: {track.get('cluster', 'N/A')}")
        st.write(f"Genre: {track.get('top_genre', 'Unknown')}")

    with col2:
        st.write(f"BPM: {track.get('bpm', 0):.0f}")
        st.write(f"Key: {track.get('key', 'Unknown')}")
        st.write(f"Danceability: {track.get('danceability', 0):.2f}")
        st.write(f"Valence: {track.get('valence', 0):.2f}")

    # Mood profile
    st.markdown("**Mood Profile:**")
    mood_data = {
        "Happy": track.get("mood_happy", 0),
        "Sad": track.get("mood_sad", 0),
        "Aggressive": track.get("mood_aggressive", 0),
        "Relaxed": track.get("mood_relaxed", 0),
        "Party": track.get("mood_party", 0),
    }

    mood_colors = [
        MOOD_COLORS["happy"],
        MOOD_COLORS["sad"],
        MOOD_COLORS["aggressive"],
        MOOD_COLORS["relaxed"],
        MOOD_COLORS["party"],
    ]

    fig = go.Figure(data=[
        go.Bar(
            x=list(mood_data.keys()),
            y=list(mood_data.values()),
            marker_color=mood_colors,
        )
    ])
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(range=[0, 1]),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Lyric features if available
    if track.get("lyric_theme") and track.get("lyric_theme") != "other":
        st.markdown("**Lyric Features:**")
        st.write(f"Theme: {track.get('lyric_theme', 'N/A')}")
        st.write(f"Language: {track.get('lyric_language', 'Unknown')}")
        st.write(f"Explicit: {track.get('lyric_explicit', 0):.2f}")
