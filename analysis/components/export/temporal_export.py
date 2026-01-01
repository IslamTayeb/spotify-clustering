"""Temporal chart export functionality for Netlify/web hosting.

This module handles exporting temporal analysis visualizations as standalone HTML files.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional, Callable
from pathlib import Path

from analysis.components.visualization.color_palette import (
    CLUSTER_COLORS, MOOD_COLORS, GENRE_FAMILY_COLORS, SPOTIFY_GREEN
)
from analysis.pipeline.config import get_cluster_name

# Genre families for temporal analysis (copied from temporal.py)
GENRE_FAMILIES = {
    "Hip Hop": [
        "hip hop", "rap", "trap", "drill", "boom bap", "conscious hip hop",
        "southern hip hop", "west coast", "east coast", "gangsta", "mumble",
        "cloud rap", "phonk", "memphis", "crunk", "grime", "uk hip hop",
    ],
    "Electronic": [
        "electronic", "edm", "house", "techno", "trance", "dubstep",
        "drum and bass", "dnb", "ambient", "downtempo", "idm", "electro",
        "synthwave", "retrowave", "future bass", "garage", "breakbeat",
    ],
    "Rock": [
        "rock", "alternative", "indie", "punk", "metal", "grunge",
        "hard rock", "classic rock", "progressive", "post-rock",
        "shoegaze", "emo", "hardcore",
    ],
    "R&B/Soul": [
        "r&b", "rnb", "soul", "neo soul", "funk", "motown",
        "quiet storm", "contemporary r&b", "new jack swing",
    ],
    "Pop": [
        "pop", "synth pop", "dance pop", "electropop", "indie pop",
        "art pop", "dream pop", "k-pop", "j-pop", "latin pop",
    ],
    "Jazz/Blues": [
        "jazz", "blues", "smooth jazz", "bebop", "swing", "fusion",
        "acid jazz", "nu jazz", "contemporary jazz",
    ],
    "Latin": [
        "latin", "reggaeton", "salsa", "bachata", "cumbia", "dembow",
        "urbano", "latin trap", "spanish", "brazilian", "bossa nova",
    ],
    "World/Folk": [
        "world", "folk", "acoustic", "country", "bluegrass", "celtic",
        "african", "middle eastern", "indian", "asian",
    ],
}


def _get_genre_family(genre_str):
    """Map a genre to its family."""
    if pd.isna(genre_str):
        return "Other"
    genre_lower = str(genre_str).lower()
    for family, keywords in GENRE_FAMILIES.items():
        if any(kw in genre_lower for kw in keywords):
            return family
    return "Other"


def _prepare_temporal_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Prepare dataframe for temporal analysis."""
    temporal_cols = ["added_at", "release_date"]
    has_temporal = all(col in df.columns for col in temporal_cols)

    if not has_temporal:
        return None

    df_temp = df.copy()
    df_temp["added_at"] = pd.to_datetime(df_temp["added_at"], errors="coerce")
    df_temp["release_date"] = pd.to_datetime(df_temp["release_date"], errors="coerce")

    valid_temporal = df_temp["added_at"].notna()
    if valid_temporal.sum() == 0:
        return None

    df_temp = df_temp[valid_temporal].copy()

    # Filter out data before June 4, 2024
    cutoff_date = pd.Timestamp("2024-06-04")
    if df_temp["added_at"].dt.tz is not None:
        cutoff_date = cutoff_date.tz_localize(df_temp["added_at"].dt.tz)
    df_temp = df_temp[df_temp["added_at"] >= cutoff_date].copy()

    # Calculate age_at_add_years if needed
    if "age_at_add_years" not in df_temp.columns:
        if df_temp["added_at"].dt.tz is not None and df_temp["release_date"].dt.tz is None:
            df_temp["release_date"] = df_temp["release_date"].dt.tz_localize("UTC")
        elif df_temp["added_at"].dt.tz is None and df_temp["release_date"].dt.tz is not None:
            df_temp["added_at"] = df_temp["added_at"].dt.tz_localize("UTC")
        df_temp["age_at_add_years"] = (df_temp["added_at"] - df_temp["release_date"]).dt.days / 365.25

    df_temp["release_year"] = df_temp["release_date"].dt.year
    df_temp["added_year"] = df_temp["added_at"].dt.year

    return df_temp


def _apply_web_layout(fig: go.Figure, height: int = 600) -> go.Figure:
    """Apply web-friendly layout to a plotly figure."""
    fig.update_layout(
        autosize=True,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=20, t=40, b=60),
        font=dict(size=12),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
    )
    return fig


def _fig_to_html(fig: go.Figure, title: str = "") -> str:
    """Convert a plotly figure to standalone HTML."""
    html = fig.to_html(
        include_plotlyjs='cdn',
        config={
            'displayModeBar': 'hover',
            'displaylogo': False,
            'responsive': True,
            'fillFrame': True
        },
        div_id="plotly-div"
    )
    return html


# =============================================================================
# CHART GENERATION FUNCTIONS
# =============================================================================

def generate_library_growth_chart(df_temp: pd.DataFrame) -> go.Figure:
    """Generate library growth timeline chart."""
    df_sorted = df_temp.sort_values("added_at").reset_index(drop=True)
    df_sorted["cumulative_additions"] = range(1, len(df_sorted) + 1)

    fig = px.line(
        df_sorted,
        x="added_at",
        y="cumulative_additions",
        labels={"cumulative_additions": "New Additions", "added_at": "Date"},
        color_discrete_sequence=[SPOTIFY_GREEN],
    )
    fig = _apply_web_layout(fig)
    fig.update_layout(title="Library Growth Since June 2024")
    return fig


def generate_monthly_additions_chart(df_temp: pd.DataFrame) -> go.Figure:
    """Generate monthly addition patterns chart."""
    monthly_additions = df_temp.groupby(df_temp["added_at"].dt.to_period("M")).size()

    fig = px.bar(
        x=monthly_additions.index.astype(str),
        y=monthly_additions.values,
        labels={"x": "Month", "y": "Songs Added"},
        color_discrete_sequence=[SPOTIFY_GREEN],
    )
    fig = _apply_web_layout(fig)
    fig.update_layout(title="Monthly Song Additions", showlegend=False)
    return fig


def generate_song_age_distribution_chart(df_temp: pd.DataFrame) -> Optional[go.Figure]:
    """Generate song age distribution histogram."""
    if "age_at_add_years" not in df_temp.columns:
        return None

    valid_ages = df_temp["age_at_add_years"].between(-1, 100)
    df_age = df_temp[valid_ages].copy()

    if len(df_age) == 0:
        return None

    fig = px.histogram(df_age, x="age_at_add_years", nbins=50, color_discrete_sequence=[SPOTIFY_GREEN])
    fig = _apply_web_layout(fig)
    fig.update_layout(
        title="Song Age When Added (Years)",
        xaxis_title="Age (Years)",
        yaxis_title="Count",
    )
    return fig


def generate_release_year_distribution_chart(df_temp: pd.DataFrame) -> Optional[go.Figure]:
    """Generate release year distribution histogram."""
    if df_temp["release_year"].notna().sum() == 0:
        return None

    valid_years = df_temp["release_year"].between(1900, 2030)
    df_year = df_temp[valid_years].copy()

    if len(df_year) == 0:
        return None

    fig = px.histogram(df_year, x="release_year", nbins=50, color_discrete_sequence=[SPOTIFY_GREEN])
    fig = _apply_web_layout(fig)
    fig.update_layout(
        title="Release Year Distribution",
        xaxis_title="Release Year",
        yaxis_title="Count",
    )
    return fig


def generate_cluster_evolution_chart(df_temp: pd.DataFrame) -> Optional[go.Figure]:
    """Generate cluster evolution stacked bar chart."""
    if len(df_temp) < 4 or "cluster" not in df_temp.columns:
        return None

    df_sorted_cluster = df_temp.sort_values("added_at").copy()

    try:
        df_sorted_cluster["time_period"] = pd.qcut(
            df_sorted_cluster["added_at"].astype(int) / 10**9,
            q=4,
            labels=["Period 1", "Period 2", "Period 3", "Period 4"],
            duplicates="drop",
        )

        period_cluster = df_sorted_cluster.groupby(["time_period", "cluster"]).size().unstack(fill_value=0)
        period_cluster_pct = period_cluster.div(period_cluster.sum(axis=1), axis=0) * 100

        # Rename columns to use cluster names
        period_cluster_pct.columns = [get_cluster_name(c) for c in period_cluster_pct.columns]

        fig = px.bar(
            period_cluster_pct,
            barmode="stack",
            color_discrete_sequence=CLUSTER_COLORS,
        )
        fig = _apply_web_layout(fig, height=500)
        fig.update_layout(
            title="Taste Evolution: Clusters Over Time",
            xaxis_title="Time Period",
            yaxis_title="Percentage",
        )
        return fig
    except Exception:
        return None


def generate_cluster_trends_chart(df_temp: pd.DataFrame, show_trendlines: bool = True) -> Optional[go.Figure]:
    """Generate rolling cluster distribution line chart."""
    if len(df_temp) < 30 or "cluster" not in df_temp.columns:
        return None

    df_sorted = df_temp.sort_values("added_at").copy()
    cluster_dummies = pd.get_dummies(df_sorted["cluster"], prefix="cluster")
    rolling_clusters = cluster_dummies.rolling(window=30, min_periods=10).mean() * 100
    rolling_clusters["added_at"] = df_sorted["added_at"].values

    cluster_cols = [col for col in rolling_clusters.columns if col.startswith("cluster_")]
    rolling_melted = rolling_clusters.melt(
        id_vars=["added_at"], value_vars=cluster_cols, var_name="Cluster", value_name="Percentage"
    )
    rolling_melted["Cluster"] = rolling_melted["Cluster"].str.replace("cluster_", "").astype(int).apply(get_cluster_name)

    fig = px.line(
        rolling_melted, x="added_at", y="Percentage", color="Cluster",
        labels={"Percentage": "Proportion (%)", "added_at": "Date Added"},
        color_discrete_sequence=CLUSTER_COLORS,
    )

    if show_trendlines:
        for i, cluster in enumerate(rolling_melted["Cluster"].unique()):
            cluster_data = rolling_melted[rolling_melted["Cluster"] == cluster].dropna()
            if len(cluster_data) > 1:
                x_numeric = (cluster_data["added_at"] - cluster_data["added_at"].min()).dt.total_seconds()
                z = np.polyfit(x_numeric, cluster_data["Percentage"], 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=cluster_data["added_at"], y=p(x_numeric), mode="lines",
                    line=dict(dash="dash", width=2, color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)]),
                    name=f"{cluster} trend", showlegend=False, opacity=0.7,
                ))

    fig = _apply_web_layout(fig)
    fig.update_layout(title="Cluster Trends Over Time (30-song rolling window)")
    return fig


def generate_mood_trends_chart(df_temp: pd.DataFrame, show_trendlines: bool = True) -> Optional[go.Figure]:
    """Generate rolling mood trends line chart."""
    mood_cols = ["mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed", "mood_party"]
    available_moods = [col for col in mood_cols if col in df_temp.columns]

    if not available_moods or len(df_temp) < 30:
        return None

    df_sorted = df_temp.sort_values("added_at").copy()
    rolling_moods = df_sorted[available_moods].rolling(window=30, min_periods=10).mean()
    rolling_moods["added_at"] = df_sorted["added_at"].values

    rolling_melted = rolling_moods.melt(id_vars=["added_at"], var_name="Mood", value_name="Score")

    fig = px.line(
        rolling_melted, x="added_at", y="Score", color="Mood",
        color_discrete_sequence=list(MOOD_COLORS.values()),
    )

    if show_trendlines:
        mood_color_list = list(MOOD_COLORS.values())
        for i, mood in enumerate(rolling_melted["Mood"].unique()):
            mood_data = rolling_melted[rolling_melted["Mood"] == mood].dropna()
            if len(mood_data) > 1:
                x_numeric = (mood_data["added_at"] - mood_data["added_at"].min()).dt.total_seconds()
                z = np.polyfit(x_numeric, mood_data["Score"], 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=mood_data["added_at"], y=p(x_numeric), mode="lines",
                    line=dict(dash="dash", width=2, color=mood_color_list[i % len(mood_color_list)]),
                    name=f"{mood} trend", showlegend=False, opacity=0.7,
                ))

    fig = _apply_web_layout(fig)
    fig.update_layout(title="Mood Trends Over Time (30-song rolling window)")
    return fig


def generate_genre_family_trends_chart(
    df_temp: pd.DataFrame,
    view: str = "proportion",
    show_trendlines: bool = True
) -> Optional[go.Figure]:
    """Generate genre family trends chart.

    Args:
        df_temp: Prepared temporal dataframe
        view: One of 'proportion', 'delta', or 'cumulative'
        show_trendlines: Whether to show trend lines
    """
    if "top_genre" not in df_temp.columns or len(df_temp) == 0:
        return None

    df_temp = df_temp.copy()
    df_temp["genre_family"] = df_temp["top_genre"].apply(_get_genre_family)
    df_temp["quarter"] = df_temp["added_at"].dt.to_period("Q")

    top_families = df_temp["genre_family"].value_counts().head(6).index.tolist()
    if "Other" in top_families and len(top_families) > 5:
        top_families.remove("Other")

    quarters = sorted(df_temp["quarter"].dropna().unique())
    timeline_data = []
    cumulative_counts = {family: 0 for family in top_families}

    for quarter in quarters:
        quarter_df = df_temp[df_temp["quarter"] == quarter]
        total_in_quarter = len(quarter_df)

        for family in top_families:
            added_this_quarter = (quarter_df["genre_family"] == family).sum()
            cumulative_counts[family] += added_this_quarter
            quarter_pct = (added_this_quarter / total_in_quarter * 100) if total_in_quarter > 0 else 0

            timeline_data.append({
                "Quarter": str(quarter),
                "Genre Family": family,
                "Added": added_this_quarter,
                "Cumulative": cumulative_counts[family],
                "Quarter %": quarter_pct,
            })

    if not timeline_data:
        return None

    timeline_df = pd.DataFrame(timeline_data)

    for family in top_families:
        family_mask = timeline_df["Genre Family"] == family
        timeline_df.loc[family_mask, "Delta"] = timeline_df.loc[family_mask, "Quarter %"].diff().fillna(0)

    if view == "proportion":
        fig = px.line(
            timeline_df, x="Quarter", y="Quarter %", color="Genre Family",
            labels={"Quarter %": "Share of Quarter (%)"},
            color_discrete_sequence=GENRE_FAMILY_COLORS,
        )
        if show_trendlines:
            _add_trendlines_to_fig(fig, timeline_df, top_families, "Quarter %", GENRE_FAMILY_COLORS)
        title = "Genre Family Share of Quarterly Additions"

    elif view == "delta":
        fig = px.bar(
            timeline_df, x="Quarter", y="Delta", color="Genre Family",
            barmode="group",
            labels={"Delta": "Change in Share (pp)"},
            color_discrete_sequence=GENRE_FAMILY_COLORS,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        if show_trendlines:
            _add_trendlines_to_fig(fig, timeline_df, top_families, "Delta", GENRE_FAMILY_COLORS, dash="dot")
        title = "Genre Family Delta (Quarter-over-Quarter Change)"

    else:  # cumulative
        fig = px.area(
            timeline_df, x="Quarter", y="Cumulative", color="Genre Family",
            labels={"Cumulative": "Total Songs"},
            color_discrete_sequence=GENRE_FAMILY_COLORS,
        )
        if show_trendlines:
            _add_trendlines_to_fig(fig, timeline_df, top_families, "Cumulative", GENRE_FAMILY_COLORS)
        title = "Cumulative Genre Family Growth"

    fig = _apply_web_layout(fig)
    fig.update_layout(title=title)
    return fig


def _add_trendlines_to_fig(fig, timeline_df, families, y_col, colors, dash="dash"):
    """Add trendlines to a plotly figure."""
    for i, family in enumerate(families):
        family_data = timeline_df[timeline_df["Genre Family"] == family]
        if len(family_data) > 1:
            x_numeric = np.arange(len(family_data))
            z = np.polyfit(x_numeric, family_data[y_col], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=family_data["Quarter"], y=p(x_numeric), mode="lines",
                line=dict(dash=dash, width=2, color=colors[i % len(colors)]),
                name=f"{family} trend", showlegend=False, opacity=0.7,
            ))


def generate_cluster_heatmap(df_temp: pd.DataFrame) -> Optional[go.Figure]:
    """Generate cluster distribution heatmap."""
    if len(df_temp) == 0 or "cluster" not in df_temp.columns:
        return None

    df_temp = df_temp.copy()
    df_temp["month"] = df_temp["added_at"].dt.to_period("M")

    try:
        cluster_month_matrix = df_temp.groupby(["month", "cluster"]).size().unstack(fill_value=0)
        cluster_month_matrix.columns = [get_cluster_name(c) for c in cluster_month_matrix.columns]

        if len(cluster_month_matrix) > 1 and len(cluster_month_matrix.columns) > 1:
            fig = px.imshow(
                cluster_month_matrix.T,
                labels=dict(x="Month", y="Cluster", color="Songs Added"),
                aspect="auto",
                color_continuous_scale="Viridis",
            )
            fig.update_xaxes(side="bottom")
            fig = _apply_web_layout(fig, height=400 + len(cluster_month_matrix.columns) * 30)
            fig.update_layout(title="Cluster Activity Heatmap")
            return fig
    except Exception:
        return None

    return None


# =============================================================================
# AVAILABLE CHARTS REGISTRY
# =============================================================================

AVAILABLE_CHARTS = {
    "library_growth": {
        "name": "Library Growth Timeline",
        "description": "Cumulative additions over time",
        "generator": generate_library_growth_chart,
        "filename": "library_growth",
    },
    "monthly_additions": {
        "name": "Monthly Additions",
        "description": "Bar chart of monthly song additions",
        "generator": generate_monthly_additions_chart,
        "filename": "monthly_additions",
    },
    "song_age": {
        "name": "Song Age Distribution",
        "description": "Histogram of song age when added",
        "generator": generate_song_age_distribution_chart,
        "filename": "song_age",
    },
    "release_year": {
        "name": "Release Year Distribution",
        "description": "Histogram of release years",
        "generator": generate_release_year_distribution_chart,
        "filename": "release_year",
    },
    "cluster_evolution": {
        "name": "Cluster Evolution",
        "description": "Stacked bar of clusters across time periods",
        "generator": generate_cluster_evolution_chart,
        "filename": "cluster_evolution",
    },
    "cluster_trends": {
        "name": "Cluster Trends",
        "description": "Rolling cluster distribution with trendlines",
        "generator": generate_cluster_trends_chart,
        "filename": "cluster_trends",
    },
    "mood_trends": {
        "name": "Mood Trends",
        "description": "Rolling mood scores with trendlines",
        "generator": generate_mood_trends_chart,
        "filename": "mood_trends",
    },
    "genre_trends_proportion": {
        "name": "Genre Trends (Proportion)",
        "description": "Genre family share per quarter",
        "generator": lambda df: generate_genre_family_trends_chart(df, view="proportion"),
        "filename": "genre_trends_proportion",
    },
    "genre_trends_delta": {
        "name": "Genre Trends (Delta)",
        "description": "Genre family change rate",
        "generator": lambda df: generate_genre_family_trends_chart(df, view="delta"),
        "filename": "genre_trends_delta",
    },
    "genre_trends_cumulative": {
        "name": "Genre Trends (Cumulative)",
        "description": "Genre family cumulative growth",
        "generator": lambda df: generate_genre_family_trends_chart(df, view="cumulative"),
        "filename": "genre_trends_cumulative",
    },
    "cluster_heatmap": {
        "name": "Cluster Heatmap",
        "description": "Month x Cluster activity matrix",
        "generator": generate_cluster_heatmap,
        "filename": "cluster_heatmap",
    },
}


def export_temporal_charts(
    df: pd.DataFrame,
    chart_ids: List[str],
    output_dir: str = "export/dimensions-of-taste-viz/temporal"
) -> List[str]:
    """Export selected temporal charts as HTML files.

    Args:
        df: Raw dataframe with temporal columns
        chart_ids: List of chart IDs to export (keys from AVAILABLE_CHARTS)
        output_dir: Directory to save HTML files

    Returns:
        List of exported file paths
    """
    # Prepare temporal data
    df_temp = _prepare_temporal_df(df)
    if df_temp is None:
        return []

    os.makedirs(output_dir, exist_ok=True)
    exported = []

    for chart_id in chart_ids:
        if chart_id not in AVAILABLE_CHARTS:
            continue

        chart_info = AVAILABLE_CHARTS[chart_id]
        generator = chart_info["generator"]
        filename = chart_info["filename"]

        try:
            fig = generator(df_temp)
            if fig is None:
                continue

            html = _fig_to_html(fig)
            output_path = os.path.join(output_dir, f"{filename}.html")

            with open(output_path, 'w') as f:
                f.write(html)

            exported.append(output_path)
        except Exception as e:
            print(f"Error exporting {chart_id}: {e}")
            continue

    return exported


def render_temporal_export_ui(df: pd.DataFrame) -> None:
    """Render the temporal export UI in Streamlit."""
    st.markdown("---")
    st.subheader("Export Temporal Charts for Web")
    st.caption("Select charts to export as standalone HTML files for Netlify/web hosting")

    # Check if temporal data is available
    df_temp = _prepare_temporal_df(df)
    if df_temp is None:
        st.warning("No temporal data available for export")
        return

    # Chart selection
    st.write("**Select charts to export:**")

    selected_charts = []
    cols = st.columns(2)

    chart_items = list(AVAILABLE_CHARTS.items())
    for i, (chart_id, chart_info) in enumerate(chart_items):
        col = cols[i % 2]
        with col:
            if st.checkbox(
                chart_info["name"],
                help=chart_info["description"],
                key=f"export_chart_{chart_id}"
            ):
                selected_charts.append(chart_id)

    # Quick select buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Select All", key="select_all_temporal"):
            st.session_state["selected_temporal_charts"] = list(AVAILABLE_CHARTS.keys())
            st.rerun()
    with col2:
        if st.button("Select Trends Only", key="select_trends_temporal"):
            trends = ["cluster_trends", "mood_trends", "genre_trends_proportion"]
            st.session_state["selected_temporal_charts"] = trends
            st.rerun()
    with col3:
        if st.button("Clear Selection", key="clear_temporal"):
            st.session_state["selected_temporal_charts"] = []
            st.rerun()

    # Override with session state if set
    if "selected_temporal_charts" in st.session_state:
        selected_charts = st.session_state["selected_temporal_charts"]
        # Clear after use
        del st.session_state["selected_temporal_charts"]

    # Output directory
    output_dir = st.text_input(
        "Output directory",
        value="export/dimensions-of-taste-viz/temporal",
        key="temporal_export_dir"
    )

    # Export button
    if st.button(
        f"Export {len(selected_charts)} Chart(s)",
        type="primary",
        disabled=len(selected_charts) == 0,
        key="export_temporal_btn"
    ):
        with st.spinner("Exporting charts..."):
            exported_paths = export_temporal_charts(df, selected_charts, output_dir)

        if exported_paths:
            st.success(f"Exported {len(exported_paths)} chart(s) to `{output_dir}/`")

            st.markdown("**Exported files:**")
            for path in exported_paths:
                filename = os.path.basename(path)
                file_size = os.path.getsize(path) / 1024
                st.text(f"  {filename} ({file_size:.1f} KB)")

            st.info("""
**Next steps:**
1. Go to [Netlify Drop](https://app.netlify.com/drop)
2. Drag the `{output_dir}` folder into the browser
3. Get instant URL like: `https://amazing-viz-123.netlify.app`
4. Add to Bear Blog: `<iframe src="YOUR_URL/chart_name.html" width="100%" height="600"></iframe>`
            """.format(output_dir=output_dir))
        else:
            st.error("No charts were exported. Check that the data contains valid temporal information.")
