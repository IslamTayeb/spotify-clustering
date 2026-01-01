"""Temporal analysis section for EDA explorer."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from analysis.components.visualization.color_palette import (
    CLUSTER_COLORS, MOOD_COLORS, GENRE_FAMILY_COLORS, SPOTIFY_GREEN
)
from analysis.pipeline.config import get_cluster_name
from analysis.components.export.chart_export import render_chart_with_export, render_export_section
from .utils import group_small_slices, get_pie_colors

# Genre families for temporal analysis
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


def render_temporal_analysis(df: pd.DataFrame):
    """Render temporal analysis section with 11 visualizations."""
    with st.expander("⏰ Temporal Analysis", expanded=False):
        st.subheader("Temporal Analysis")
        st.caption("Explore how your music taste evolved over time | Check boxes on right to select charts for export")

        temporal_cols = ["added_at", "release_date"]
        has_temporal = all(col in df.columns for col in temporal_cols)

        if not has_temporal:
            st.warning("Temporal information not available")
            st.info("💡 Data loaded from spotify/saved_tracks.json")
            return

        df_temp = df.copy()
        df_temp["added_at"] = pd.to_datetime(df_temp["added_at"], errors="coerce")
        df_temp["release_date"] = pd.to_datetime(df_temp["release_date"], errors="coerce")

        valid_temporal = df_temp["added_at"].notna()
        if valid_temporal.sum() == 0:
            st.warning("No valid temporal data found")
            return

        df_temp = df_temp[valid_temporal].copy()

        # Filter out data before June 4, 2024
        cutoff_date = pd.Timestamp("2024-06-04")
        if df_temp["added_at"].dt.tz is not None:
            cutoff_date = cutoff_date.tz_localize(df_temp["added_at"].dt.tz)
        df_temp = df_temp[df_temp["added_at"] >= cutoff_date].copy()

        songs_before_cutoff = len(df) - len(df_temp)

        st.info(
            f"**Note on Temporal Data:** Spotify only reliably tracks `added_at` timestamps "
            f"after June 4, 2024. Your library had **{songs_before_cutoff} songs** before this date "
            f"(shown in other analyses but excluded here). The temporal analysis below covers "
            f"**{len(df_temp)} songs** added since then."
        )

        # Calculate age_at_add_years if needed
        if "age_at_add_years" not in df_temp.columns:
            if df_temp["added_at"].dt.tz is not None and df_temp["release_date"].dt.tz is None:
                df_temp["release_date"] = df_temp["release_date"].dt.tz_localize("UTC")
            elif df_temp["added_at"].dt.tz is None and df_temp["release_date"].dt.tz is not None:
                df_temp["added_at"] = df_temp["added_at"].dt.tz_localize("UTC")
            df_temp["age_at_add_years"] = (df_temp["added_at"] - df_temp["release_date"]).dt.days / 365.25

        df_temp["release_year"] = df_temp["release_date"].dt.year
        df_temp["added_year"] = df_temp["added_at"].dt.year

        # 1. Overview Metrics
        _render_overview_metrics(df_temp, songs_before_cutoff)

        # 2. Library Growth Timeline
        _render_library_growth(df_temp, songs_before_cutoff)

        # 3. Monthly Addition Patterns
        _render_monthly_additions(df_temp)

        # 4. Song Age Distribution
        _render_song_age_distribution(df_temp)

        # 5. Release Year Distribution
        _render_release_year_distribution(df_temp)

        # 6. Cluster Evolution Over Time
        _render_cluster_evolution(df_temp)

        # 7. Temporal Extremes
        _render_temporal_extremes(df_temp)

        # 8. Cluster Trends Over Time
        _render_cluster_trends(df_temp)

        # 9. Mood Evolution
        _render_mood_trends(df_temp)

        # 10. Genre Family Trends
        _render_genre_family_trends(df_temp)

        # 11. Cluster Timeline Heatmap
        _render_cluster_heatmap(df_temp)

        # Export section
        render_export_section("export/visualizations", "temporal")

        st.markdown("---")
        st.success("✨ Temporal analysis complete!")


def _render_overview_metrics(df_temp, songs_before_cutoff):
    """Render overview metrics."""
    st.markdown("---")
    st.subheader("📊 Overview (Post June 4, 2024)")
    col1, col2, col3 = st.columns(3)

    with col1:
        min_date = df_temp["added_at"].min()
        max_date = df_temp["added_at"].max()
        date_range = (max_date - min_date).days
        st.metric("Tracked Period", f"{date_range} days")
        st.caption(f"{min_date.strftime('%Y-%m-%d')} → {max_date.strftime('%Y-%m-%d')}")

    with col2:
        if "age_at_add_years" in df_temp.columns:
            median_age = df_temp["age_at_add_years"].median()
            st.metric("Median Song Age at Add", f"{median_age:.1f} years")
        else:
            st.metric("Median Song Age at Add", "N/A")

    with col3:
        df_temp["added_month"] = df_temp["added_at"].dt.to_period("M")
        most_active = df_temp["added_month"].value_counts().index[0]
        most_active_count = df_temp["added_month"].value_counts().iloc[0]
        st.metric("Most Active Month", str(most_active))
        st.caption(f"{most_active_count} songs added")


def _render_library_growth(df_temp, songs_before_cutoff):
    """Render library growth timeline."""
    st.markdown("---")
    st.subheader("📈 Library Growth Since June 4, 2024")

    df_sorted = df_temp.sort_values("added_at").reset_index(drop=True)
    df_sorted["cumulative_additions"] = range(1, len(df_sorted) + 1)

    st.caption(f"Songs Added Since June 4, 2024 (started with {songs_before_cutoff} songs)")
    fig = px.line(
        df_sorted,
        x="added_at",
        y="cumulative_additions",
        labels={"cumulative_additions": "New Additions", "added_at": "Date"},
        color_discrete_sequence=[SPOTIFY_GREEN],
    )
    fig.update_layout(height=500, margin=dict(t=0, l=0, r=0, b=0))

    render_chart_with_export(fig, "library_growth", "Library Growth Timeline", "temporal")

    st.caption(
        f"This chart shows the {len(df_temp)} songs added after June 4, 2024. "
        f"The {songs_before_cutoff} songs added before that date are not shown."
    )


def _render_monthly_additions(df_temp):
    """Render monthly addition patterns."""
    st.markdown("---")
    st.subheader("📅 Monthly Addition Patterns")

    monthly_additions = df_temp.groupby(df_temp["added_at"].dt.to_period("M")).size()

    st.caption("Monthly Song Additions (Since June 2024)")
    fig = px.bar(
        x=monthly_additions.index.astype(str),
        y=monthly_additions.values,
        labels={"x": "Month", "y": "Songs Added"},
        color_discrete_sequence=[SPOTIFY_GREEN],
    )
    fig.update_layout(height=500, showlegend=False, margin=dict(t=0, l=0, r=0, b=0))

    render_chart_with_export(fig, "monthly_additions", "Monthly Additions", "temporal")


def _render_song_age_distribution(df_temp):
    """Render song age distribution."""
    st.markdown("---")
    st.subheader("🕰️ Song Age When Added")

    if "age_at_add_years" not in df_temp.columns:
        st.info("Age data not available")
        return

    valid_ages = df_temp["age_at_add_years"].between(-1, 100)
    df_age = df_temp[valid_ages].copy()

    df_age["age_category"] = pd.cut(
        df_age["age_at_add_years"],
        bins=[-float("inf"), 1, 5, 10, float("inf")],
        labels=["New (<1yr)", "Recent (1-5yr)", "Classic (5-10yr)", "Vintage (>10yr)"],
    )

    fig = px.histogram(df_age, x="age_at_add_years", nbins=50, color_discrete_sequence=[SPOTIFY_GREEN])
    fig.update_layout(height=500, margin=dict(t=0, l=0, r=0, b=0))

    render_chart_with_export(fig, "song_age", "Song Age Distribution", "temporal")

    category_counts = df_age["age_category"].value_counts()
    st.write("**Age Categories:**")
    for cat in ["New (<1yr)", "Recent (1-5yr)", "Classic (5-10yr)", "Vintage (>10yr)"]:
        if cat in category_counts.index:
            count = category_counts[cat]
            pct = count / len(df_age) * 100
            st.write(f"- {cat}: {count} songs ({pct:.1f}%)")


def _render_release_year_distribution(df_temp):
    """Render release year distribution."""
    st.markdown("---")
    st.subheader("🎵 Release Year Distribution")

    if df_temp["release_year"].notna().sum() == 0:
        st.info("Release year data not available")
        return

    valid_years = df_temp["release_year"].between(1900, 2030)
    df_year = df_temp[valid_years].copy()

    fig = px.histogram(df_year, x="release_year", nbins=50, color_discrete_sequence=[SPOTIFY_GREEN])
    fig.update_layout(height=500, margin=dict(t=0, l=0, r=0, b=0))

    render_chart_with_export(fig, "release_year", "Release Year Distribution", "temporal")

    df_year["decade"] = (df_year["release_year"] // 10) * 10
    decade_counts = df_year["decade"].value_counts().sort_index()

    st.write("**By Decade:**")
    for decade, count in decade_counts.items():
        pct = count / len(df_year) * 100
        st.write(f"- {int(decade)}s: {count} songs ({pct:.1f}%)")

    decade_counts_grouped, _ = group_small_slices(decade_counts)
    decade_names = [f"{int(d)}s" if isinstance(d, (int, float)) else d for d in decade_counts_grouped.index]
    st.caption("Decade Distribution")
    fig_pie = px.pie(
        values=decade_counts_grouped.values,
        names=decade_names,
        color_discrete_sequence=get_pie_colors(decade_names, CLUSTER_COLORS),
        hole=0.4,
    )
    fig_pie.update_layout(height=500, margin=dict(t=0, l=0, r=0, b=0))

    render_chart_with_export(fig_pie, "release_decade_pie", "Release Decade Pie Chart", "temporal")


def _render_cluster_evolution(df_temp):
    """Render cluster evolution over time."""
    st.markdown("---")
    st.subheader("🎭 Taste Evolution: Clusters Over Time")

    if len(df_temp) < 4:
        st.info("Need at least 4 songs to show cluster evolution")
        return

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

        st.caption("Cluster Distribution Across Time Periods")
        fig = px.bar(
            period_cluster_pct,
            barmode="stack",
            color_discrete_sequence=CLUSTER_COLORS,
        )
        fig.update_layout(height=600, margin=dict(t=0, l=0, r=0, b=0))

        render_chart_with_export(fig, "cluster_evolution", "Cluster Evolution", "temporal")

        st.write("**Time Period Breakdown:**")
        for period in ["Period 1", "Period 2", "Period 3", "Period 4"]:
            if period in df_sorted_cluster["time_period"].values:
                period_df = df_sorted_cluster[df_sorted_cluster["time_period"] == period]
                min_d = period_df["added_at"].min().strftime("%Y-%m-%d")
                max_d = period_df["added_at"].max().strftime("%Y-%m-%d")
                st.write(f"- {period}: {min_d} → {max_d} ({len(period_df)} songs)")
    except Exception as e:
        st.warning(f"Could not split into time periods: {e}")


def _render_temporal_extremes(df_temp):
    """Render temporal extremes."""
    st.markdown("---")
    st.subheader("🏆 Temporal Extremes")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**By Release Date:**")
        if df_temp["release_year"].notna().sum() > 0:
            valid_years_mask = df_temp["release_year"].between(1900, 2030)
            df_valid_years = df_temp[valid_years_mask]

            if len(df_valid_years) > 0:
                oldest = df_valid_years.loc[df_valid_years["release_year"].idxmin()]
                st.text(f"Oldest: {oldest['track_name']}")
                st.caption(f"Released: {int(oldest['release_year'])} | by {oldest['artist']}")

                newest = df_valid_years.loc[df_valid_years["release_year"].idxmax()]
                st.text(f"Newest: {newest['track_name']}")
                st.caption(f"Released: {int(newest['release_year'])} | by {newest['artist']}")
        else:
            st.info("Release year data not available")

    with col2:
        st.write("**By Addition Date:**")
        first = df_temp.loc[df_temp["added_at"].idxmin()]
        st.text(f"First Added: {first['track_name']}")
        st.caption(f"on {first['added_at'].strftime('%Y-%m-%d')} | by {first['artist']}")

        last = df_temp.loc[df_temp["added_at"].idxmax()]
        st.text(f"Last Added: {last['track_name']}")
        st.caption(f"on {last['added_at'].strftime('%Y-%m-%d')} | by {last['artist']}")

    # Age extremes
    st.markdown("---")
    st.write("**By Age at Addition:**")
    if "age_at_add_years" in df_temp.columns and df_temp["age_at_add_years"].notna().sum() > 0:
        valid_ages_mask = df_temp["age_at_add_years"].between(0, 100)
        df_valid_ages = df_temp[valid_ages_mask]

        if len(df_valid_ages) > 0:
            vintage = df_valid_ages.loc[df_valid_ages["age_at_add_years"].idxmax()]
            st.text(f"Most Vintage: {vintage['track_name']} ({vintage['age_at_add_years']:.1f} yrs old)")

            brand_new = df_valid_ages.loc[df_valid_ages["age_at_add_years"].idxmin()]
            st.text(f"Newest Release: {brand_new['track_name']} ({brand_new['age_at_add_years']:.1f} yrs old)")
    else:
        st.info("Age data not available")

    # Top 5 oldest and newest songs tables
    st.markdown("---")
    st.write("**Top 5 Oldest & Newest Songs by Release Year:**")
    if df_temp["release_year"].notna().sum() > 0:
        valid_years_mask = df_temp["release_year"].between(1900, 2030)
        df_valid_years = df_temp[valid_years_mask]

        if len(df_valid_years) >= 5:
            col1, col2 = st.columns(2)

            with col1:
                st.caption("Top 5 Oldest")
                oldest_5 = df_valid_years.nsmallest(5, "release_year")[["track_name", "artist", "release_year"]].copy()
                oldest_5["release_year"] = oldest_5["release_year"].astype(int)
                oldest_5.columns = ["Track", "Artist", "Year"]
                st.dataframe(oldest_5, use_container_width=True, hide_index=True)

            with col2:
                st.caption("Top 5 Newest")
                newest_5 = df_valid_years.nlargest(5, "release_year")[["track_name", "artist", "release_year"]].copy()
                newest_5["release_year"] = newest_5["release_year"].astype(int)
                newest_5.columns = ["Track", "Artist", "Year"]
                st.dataframe(newest_5, use_container_width=True, hide_index=True)
        else:
            st.info("Need at least 5 songs with valid release years")
    else:
        st.info("Release year data not available")


def _render_cluster_trends(df_temp):
    """Render cluster trends over time."""
    st.markdown("---")
    st.subheader("🎭 Cluster Trends Over Time")
    show_cluster_trendlines = st.checkbox("Show trend lines", value=True, key="cluster_trends_trendlines")

    if len(df_temp) < 30 or "cluster" not in df_temp.columns:
        if len(df_temp) < 30:
            st.info("Need at least 30 songs for rolling cluster trends")
        else:
            st.info("Cluster info not available")
        return

    df_sorted = df_temp.sort_values("added_at").copy()
    cluster_dummies = pd.get_dummies(df_sorted["cluster"], prefix="cluster")
    rolling_clusters = cluster_dummies.rolling(window=30, min_periods=10).mean() * 100
    rolling_clusters["added_at"] = df_sorted["added_at"].values

    cluster_cols = [col for col in rolling_clusters.columns if col.startswith("cluster_")]
    rolling_melted = rolling_clusters.melt(
        id_vars=["added_at"], value_vars=cluster_cols, var_name="Cluster", value_name="Percentage"
    )
    # Map cluster IDs to human-readable names
    rolling_melted["Cluster"] = rolling_melted["Cluster"].str.replace("cluster_", "").astype(int).apply(get_cluster_name)

    st.caption("Rolling Cluster Distribution (30-song window)")
    fig = px.line(
        rolling_melted, x="added_at", y="Percentage", color="Cluster",
        labels={"Percentage": "Proportion (%)", "added_at": "Date Added"},
        color_discrete_sequence=CLUSTER_COLORS,
    )

    if show_cluster_trendlines:
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

    fig.update_layout(height=600, margin=dict(t=0, l=0, r=0, b=0))

    render_chart_with_export(fig, "cluster_trends", "Cluster Trends", "temporal")

    caption = "Shows how the proportion of each cluster changes as you add songs over time"
    if show_cluster_trendlines:
        caption += " (dashed = trendline)"
    st.caption(caption)


def _render_mood_trends(df_temp):
    """Render mood trends over time."""
    st.markdown("---")
    st.subheader("😊 Mood Trends Over Time")
    show_mood_trendlines = st.checkbox("Show trend lines", value=True, key="mood_trends_trendlines")

    mood_cols = ["mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed", "mood_party"]
    available_moods = [col for col in mood_cols if col in df_temp.columns]

    if not available_moods or len(df_temp) < 30:
        if len(df_temp) < 30:
            st.info("Need at least 30 songs")
        else:
            st.info("Mood info not available")
        return

    df_sorted = df_temp.sort_values("added_at").copy()
    rolling_moods = df_sorted[available_moods].rolling(window=30, min_periods=10).mean()
    rolling_moods["added_at"] = df_sorted["added_at"].values

    rolling_melted = rolling_moods.melt(id_vars=["added_at"], var_name="Mood", value_name="Score")

    fig = px.line(
        rolling_melted, x="added_at", y="Score", color="Mood",
        color_discrete_sequence=list(MOOD_COLORS.values()),
    )

    if show_mood_trendlines:
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

    fig.update_layout(height=600, margin=dict(t=0, l=0, r=0, b=0))

    render_chart_with_export(fig, "mood_trends", "Mood Trends", "temporal")

    if show_mood_trendlines:
        st.caption("Dashed lines show overall trend direction for each mood")


def _render_genre_family_trends(df_temp):
    """Render genre family trends over time."""
    st.markdown("---")
    st.subheader("🎸 Genre Family Trends Over Time")
    show_genre_trendlines = st.checkbox("Show trend lines", value=True, key="genre_family_trendlines")

    if "top_genre" not in df_temp.columns or len(df_temp) == 0:
        st.info("Genre info not available")
        return

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
        st.info("Not enough data for genre trends")
        return

    timeline_df = pd.DataFrame(timeline_data)

    for family in top_families:
        family_mask = timeline_df["Genre Family"] == family
        timeline_df.loc[family_mask, "Delta"] = timeline_df.loc[family_mask, "Quarter %"].diff().fillna(0)

    genre_view = st.radio(
        "View",
        ["Quarterly Proportion", "Delta (Rate of Change)", "Cumulative Growth"],
        horizontal=True,
        key="genre_view_selector",
    )

    if genre_view == "Quarterly Proportion":
        st.caption("Genre Family Share of Quarterly Additions")
        fig = px.line(
            timeline_df, x="Quarter", y="Quarter %", color="Genre Family",
            labels={"Quarter %": "Share of Quarter (%)"},
            color_discrete_sequence=GENRE_FAMILY_COLORS,
        )
        if show_genre_trendlines:
            _add_trendlines_to_fig(fig, timeline_df, top_families, "Quarter %", GENRE_FAMILY_COLORS)
        caption = "What proportion of songs added each quarter belong to each genre family"
        if show_genre_trendlines:
            caption += " (dashed = trendline)"
        chart_id = "genre_trends_proportion"
        chart_title = "Genre Trends (Quarterly Proportion)"

    elif genre_view == "Delta (Rate of Change)":
        st.caption("Genre Family Delta (Quarter-over-Quarter Change)")
        fig = px.bar(
            timeline_df, x="Quarter", y="Delta", color="Genre Family",
            barmode="group",
            labels={"Delta": "Change in Share (pp)"},
            color_discrete_sequence=GENRE_FAMILY_COLORS,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        if show_genre_trendlines:
            _add_trendlines_to_fig(fig, timeline_df, top_families, "Delta", GENRE_FAMILY_COLORS, dash="dot")
        caption = "Positive = growing interest, Negative = declining interest"
        if show_genre_trendlines:
            caption += " (dotted = trendline)"
        chart_id = "genre_trends_delta"
        chart_title = "Genre Trends (Delta)"

    else:  # Cumulative Growth
        st.caption("Cumulative Genre Family Growth")
        fig = px.area(
            timeline_df, x="Quarter", y="Cumulative", color="Genre Family",
            labels={"Cumulative": "Total Songs"},
            color_discrete_sequence=GENRE_FAMILY_COLORS,
        )
        if show_genre_trendlines:
            _add_trendlines_to_fig(fig, timeline_df, top_families, "Cumulative", GENRE_FAMILY_COLORS)
        caption = "How your collection of each genre family has grown over time"
        if show_genre_trendlines:
            caption += " (dashed = linear growth rate)"
        chart_id = "genre_trends_cumulative"
        chart_title = "Genre Trends (Cumulative)"

    fig.update_layout(height=600, margin=dict(t=0, l=0, r=0, b=0))

    render_chart_with_export(fig, chart_id, chart_title, "temporal")

    st.caption(caption)

    # Summary statistics
    st.markdown("**Genre Family Summary:**")
    summary_cols = st.columns(min(len(top_families), 4))
    total_songs = len(df_temp)

    for i, family in enumerate(top_families[:4]):
        family_total = cumulative_counts[family]
        family_pct = family_total / total_songs * 100 if total_songs > 0 else 0
        family_deltas = timeline_df[timeline_df["Genre Family"] == family]["Delta"]
        avg_delta = family_deltas.mean()
        trend = "📈" if avg_delta > 0.5 else ("📉" if avg_delta < -0.5 else "➡️")

        with summary_cols[i]:
            st.metric(family, f"{family_total} songs", f"{avg_delta:+.1f}pp avg {trend}")
            st.caption(f"{family_pct:.1f}% of library")


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


def _render_cluster_heatmap(df_temp):
    """Render cluster distribution heatmap."""
    st.markdown("---")
    st.subheader("🔥 Cluster Distribution Heatmap")

    if len(df_temp) == 0 or "cluster" not in df_temp.columns:
        st.info("Cluster info not available")
        return

    df_temp["month"] = df_temp["added_at"].dt.to_period("M")

    try:
        cluster_month_matrix = df_temp.groupby(["month", "cluster"]).size().unstack(fill_value=0)

        # Rename columns to use cluster names
        cluster_month_matrix.columns = [get_cluster_name(c) for c in cluster_month_matrix.columns]

        if len(cluster_month_matrix) > 1 and len(cluster_month_matrix.columns) > 1:
            st.caption("Cluster Activity Heatmap")
            fig = px.imshow(
                cluster_month_matrix.T,
                labels=dict(x="Month", y="Cluster", color="Songs Added"),
                aspect="auto",
                color_continuous_scale="Viridis",
            )
            fig.update_xaxes(side="bottom")
            fig.update_layout(height=500 + len(cluster_month_matrix.columns) * 20, margin=dict(t=0, l=0, r=0, b=0))

            render_chart_with_export(fig, "cluster_heatmap", "Cluster Heatmap", "temporal")
        else:
            st.info("Not enough time periods")
    except Exception as e:
        st.warning(f"Could not generate heatmap: {e}")
