#!/usr/bin/env python3
"""
Temporal analysis of music taste evolution.

Analyzes when songs were added to the library and how taste has evolved over time.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_temporal_metadata(saved_tracks_path: str = 'spotify/saved_tracks.json') -> pd.DataFrame:
    """Load temporal metadata from saved_tracks.json"""
    with open(saved_tracks_path, 'r') as f:
        tracks = json.load(f)

    # Extract temporal fields
    temporal_data = []
    for track in tracks:
        temporal_data.append({
            'track_id': track['track_id'],
            'track_name': track['track_name'],
            'artist': ', '.join(track['artists']),
            'added_at': track['added_at'],
            'release_date': track.get('release_date'),
            'popularity': track.get('popularity', 0),
            'duration_ms': track.get('duration_ms', 0)
        })

    df = pd.DataFrame(temporal_data)

    # Parse dates
    df['added_at'] = pd.to_datetime(df['added_at'])
    df['added_date'] = df['added_at'].dt.date
    df['added_year'] = df['added_at'].dt.year
    df['added_month'] = df['added_at'].dt.to_period('M')
    df['added_week'] = df['added_at'].dt.to_period('W')

    # Parse release dates (handle different formats)
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year

    # Ensure added_at and release_date have compatible timezones for subtraction
    if df['added_at'].dt.tz is not None:
        # If added_at is tz-aware, make release_date tz-aware (UTC) to match
        df['release_date'] = df['release_date'].dt.tz_localize('UTC')
    elif df['release_date'].dt.tz is not None:
         # If release_date is tz-aware but added_at is not (unlikely but possible), make added_at tz-aware
        df['added_at'] = df['added_at'].dt.tz_localize('UTC')

    # Calculate "age at add" - how old was the song when you added it?
    df['age_at_add_years'] = (df['added_at'] - df['release_date']).dt.days / 365.25

    logger.info(f"Loaded temporal data for {len(df)} tracks")
    logger.info(f"Date range: {df['added_at'].min()} to {df['added_at'].max()}")

    return df


def create_taste_evolution_viz(df_clustered: pd.DataFrame, output_dir: str = 'analysis/outputs/eda'):
    """
    Create comprehensive temporal visualizations.

    Args:
        df_clustered: Dataframe with clustering results AND temporal data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Songs added over time (cumulative)
    df_sorted = df_clustered.sort_values('added_at')
    df_sorted['cumulative_songs'] = range(1, len(df_sorted) + 1)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_sorted['added_at'],
        y=df_sorted['cumulative_songs'],
        mode='lines',
        name='Total Songs',
        line=dict(color='#1DB954', width=2),
        fill='tozeroy'
    ))

    fig1.update_layout(
        title='Your Music Library Growth Over Time',
        xaxis_title='Date',
        yaxis_title='Cumulative Songs',
        template='plotly_white',
        hovermode='x unified',
        height=500
    )

    fig1.write_html(output_path / 'library_growth.html')
    logger.info(f"Saved library growth chart to {output_path / 'library_growth.html'}")

    # 2. Songs added per month/week
    songs_per_month = df_clustered.groupby('added_month').size().reset_index(name='count')
    songs_per_month['added_month'] = songs_per_month['added_month'].astype(str)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=songs_per_month['added_month'],
        y=songs_per_month['count'],
        marker_color='#1DB954',
        name='Songs Added'
    ))

    fig2.update_layout(
        title='Songs Added Per Month',
        xaxis_title='Month',
        yaxis_title='Number of Songs',
        template='plotly_white',
        height=500,
        xaxis={'tickangle': -45}
    )

    fig2.write_html(output_path / 'songs_per_month.html')
    logger.info(f"Saved monthly additions chart to {output_path / 'songs_per_month.html'}")

    # 3. Cluster discovery timeline
    fig3 = go.Figure()

    for cluster_id in sorted(df_clustered['cluster'].unique()):
        if cluster_id == -1:
            continue

        cluster_df = df_clustered[df_clustered['cluster'] == cluster_id].sort_values('added_at')
        cluster_df['cumulative_in_cluster'] = range(1, len(cluster_df) + 1)

        fig3.add_trace(go.Scatter(
            x=cluster_df['added_at'],
            y=cluster_df['cumulative_in_cluster'],
            mode='lines',
            name=f'Cluster {cluster_id}',
            stackgroup='one'
        ))

    fig3.update_layout(
        title='Cluster Discovery Timeline (Stacked)',
        xaxis_title='Date',
        yaxis_title='Cumulative Songs in Cluster',
        template='plotly_white',
        hovermode='x unified',
        height=600
    )

    fig3.write_html(output_path / 'cluster_timeline.html')
    logger.info(f"Saved cluster timeline to {output_path / 'cluster_timeline.html'}")

    # 4. Age of songs when added (distribution)
    fig4 = go.Figure()

    fig4.add_trace(go.Histogram(
        x=df_clustered['age_at_add_years'],
        nbinsx=50,
        marker_color='#1DB954',
        name='Song Age at Add'
    ))

    fig4.update_layout(
        title='How Old Were Songs When You Added Them?',
        xaxis_title='Years Since Release',
        yaxis_title='Number of Songs',
        template='plotly_white',
        height=500
    )

    fig4.write_html(output_path / 'song_age_at_add.html')
    logger.info(f"Saved song age distribution to {output_path / 'song_age_at_add.html'}")

    # 5. Cluster characteristics by time period
    # Split library into time periods (quartiles)
    df_clustered['time_period'] = pd.qcut(
        df_clustered['added_at'].astype(int) / 10**9,
        q=4,
        labels=['Early Adds', 'Mid-Early Adds', 'Mid-Late Adds', 'Recent Adds']
    )

    cluster_period_matrix = pd.crosstab(
        df_clustered['cluster'],
        df_clustered['time_period'],
        normalize='columns'
    ) * 100

    fig5 = go.Figure(data=go.Heatmap(
        z=cluster_period_matrix.values,
        x=cluster_period_matrix.columns,
        y=[f'Cluster {i}' if i != -1 else 'Outliers' for i in cluster_period_matrix.index],
        colorscale='Viridis',
        text=cluster_period_matrix.values.round(1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title='% of Period')
    ))

    fig5.update_layout(
        title='Cluster Distribution Across Time Periods',
        xaxis_title='Time Period',
        yaxis_title='Cluster',
        template='plotly_white',
        height=600
    )

    fig5.write_html(output_path / 'cluster_time_heatmap.html')
    logger.info(f"Saved cluster-time heatmap to {output_path / 'cluster_time_heatmap.html'}")

    # 6. Genre evolution over time
    # Extract top genre for each song
    df_clustered['genre_for_timeline'] = df_clustered['top_genre']

    # Get top 10 genres overall
    top_10_genres = df_clustered['genre_for_timeline'].value_counts().head(10).index.tolist()

    # Filter to top genres and group by month
    df_top_genres = df_clustered[df_clustered['genre_for_timeline'].isin(top_10_genres)].copy()
    genre_by_month = df_top_genres.groupby(['added_month', 'genre_for_timeline']).size().reset_index(name='count')
    genre_by_month['added_month'] = genre_by_month['added_month'].astype(str)

    fig6 = px.area(
        genre_by_month,
        x='added_month',
        y='count',
        color='genre_for_timeline',
        title='Genre Evolution Over Time (Top 10 Genres)',
        labels={'count': 'Songs Added', 'added_month': 'Month', 'genre_for_timeline': 'Genre'},
        template='plotly_white',
        height=600
    )

    fig6.update_xaxes(tickangle=-45)
    fig6.write_html(output_path / 'genre_evolution.html')
    logger.info(f"Saved genre evolution chart to {output_path / 'genre_evolution.html'}")

    # 7. Mood trends over time
    mood_cols = ['mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed', 'mood_party']

    # Calculate rolling average of moods over time
    df_sorted = df_clustered.sort_values('added_at').copy()
    window_size = 50  # 50-song rolling window

    fig7 = go.Figure()

    for mood in mood_cols:
        rolling_avg = df_sorted[mood].rolling(window=window_size, center=True).mean()
        mood_name = mood.replace('mood_', '').capitalize()

        fig7.add_trace(go.Scatter(
            x=df_sorted['added_at'],
            y=rolling_avg * 100,
            mode='lines',
            name=mood_name,
            line=dict(width=2)
        ))

    fig7.update_layout(
        title=f'Mood Trends Over Time (Rolling {window_size}-Song Average)',
        xaxis_title='Date Added',
        yaxis_title='Mood Score (%)',
        template='plotly_white',
        hovermode='x unified',
        height=600
    )

    fig7.write_html(output_path / 'mood_trends.html')
    logger.info(f"Saved mood trends chart to {output_path / 'mood_trends.html'}")


def generate_temporal_report(df_clustered: pd.DataFrame, output_dir: str = 'analysis/outputs/eda'):
    """Generate markdown report with temporal statistics"""

    output_path = Path(output_dir)

    n_songs = len(df_clustered)
    date_range = (df_clustered['added_at'].min(), df_clustered['added_at'].max())
    duration_days = (date_range[1] - date_range[0]).days

    songs_per_day = n_songs / duration_days if duration_days > 0 else 0

    # Time period analysis
    df_clustered['time_period'] = pd.qcut(
        df_clustered['added_at'].astype(int) / 10**9,
        q=4,
        labels=['Early Adds (Q1)', 'Mid-Early (Q2)', 'Mid-Late (Q3)', 'Recent Adds (Q4)']
    )

    period_stats = []
    for period in ['Early Adds (Q1)', 'Mid-Early (Q2)', 'Mid-Late (Q3)', 'Recent Adds (Q4)']:
        period_df = df_clustered[df_clustered['time_period'] == period]
        period_stats.append({
            'period': period,
            'n_songs': len(period_df),
            'date_range': f"{period_df['added_at'].min().date()} to {period_df['added_at'].max().date()}",
            'top_genre': period_df['top_genre'].mode()[0] if len(period_df) > 0 else 'N/A',
            'avg_mood_happy': period_df['mood_happy'].mean() * 100,
            'avg_mood_sad': period_df['mood_sad'].mean() * 100,
            'avg_age_at_add': period_df['age_at_add_years'].median()
        })

    # Most active months
    songs_per_month = df_clustered.groupby('added_month').size().reset_index(name='count')
    songs_per_month = songs_per_month.sort_values('count', ascending=False)
    top_5_months = songs_per_month.head(5)

    # Age at add statistics
    median_age = df_clustered['age_at_add_years'].median()
    recent_adds = (df_clustered['age_at_add_years'] < 1).sum()
    old_adds = (df_clustered['age_at_add_years'] > 5).sum()

    report = f"""# Temporal Analysis Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Library Timeline

**Total Songs**: {n_songs}
**Date Range**: {date_range[0].date()} to {date_range[1].date()}
**Duration**: {duration_days} days ({duration_days/365.25:.1f} years)
**Average Rate**: {songs_per_day:.2f} songs/day

---

## Most Active Months

Your top 5 months for adding music:

"""

    for i, row in top_5_months.iterrows():
        report += f"{i+1}. **{row['added_month']}**: {row['count']} songs\n"

    report += f"""

---

## Taste Evolution by Time Period

Your library divided into 4 equal time periods:

"""

    for stats in period_stats:
        report += f"""### {stats['period']}

- **Songs Added**: {stats['n_songs']}
- **Date Range**: {stats['date_range']}
- **Top Genre**: {stats['top_genre']}
- **Mood Profile**: {stats['avg_mood_happy']:.1f}% happy, {stats['avg_mood_sad']:.1f}% sad
- **Median Song Age at Add**: {stats['avg_age_at_add']:.1f} years

"""

    report += f"""---

## Song Discovery Patterns

**Median Song Age When Added**: {median_age:.1f} years

- **Recent Music** (< 1 year old): {recent_adds} songs ({recent_adds/n_songs*100:.1f}%)
- **Older Music** (> 5 years old): {old_adds} songs ({old_adds/n_songs*100:.1f}%)

This shows whether you tend to add new releases or discover older music.

---

## Cluster Evolution

Distribution of clusters across time periods:

"""

    cluster_period_counts = pd.crosstab(
        df_clustered['cluster'],
        df_clustered['time_period']
    )

    for cluster_id in sorted(df_clustered['cluster'].unique()):
        if cluster_id == -1:
            continue

        cluster_name = f"Cluster {cluster_id}"
        cluster_over_time = cluster_period_counts.loc[cluster_id]

        report += f"**{cluster_name}**: "
        report += " → ".join([f"{count} in {period.split('(')[0].strip()}"
                              for period, count in cluster_over_time.items()])
        report += "\n\n"

    report += """---

## Visualizations

Explore the interactive temporal visualizations:

1. **library_growth.html** - How your library grew over time
2. **songs_per_month.html** - Activity patterns by month
3. **cluster_timeline.html** - When you discovered each cluster
4. **song_age_at_add.html** - Distribution of song ages when added
5. **cluster_time_heatmap.html** - Cluster preferences across time periods
6. **genre_evolution.html** - How your genre preferences evolved
7. **mood_trends.html** - Mood trends over your library timeline

"""

    with open(output_path / 'temporal_analysis_report.md', 'w') as f:
        f.write(report)

    logger.info(f"Temporal report saved to {output_path / 'temporal_analysis_report.md'}")


def run_temporal_analysis(results_pkl: str = 'analysis/outputs/analysis_data.pkl',
                         saved_tracks_path: str = 'spotify/saved_tracks.json',
                         output_dir: str = 'analysis/outputs/eda'):
    """
    Run complete temporal analysis.

    Args:
        results_pkl: Path to analysis_data.pkl with clustering results
        saved_tracks_path: Path to saved_tracks.json with temporal metadata
        output_dir: Output directory for temporal analysis
    """
    import pickle

    logger.info("Loading clustering results...")
    with open(results_pkl, 'rb') as f:
        results = pickle.load(f)

    df_clustered = results['dataframe']

    logger.info("Loading temporal metadata...")
    df_temporal = load_temporal_metadata(saved_tracks_path)

    logger.info("Merging temporal data with clustering results...")
    df_merged = df_clustered.merge(df_temporal, on='track_id', how='left', suffixes=('', '_temporal'))

    # Use temporal columns, keep original track_name and artist from clustering
    temporal_cols = ['added_at', 'added_date', 'added_year', 'added_month', 'added_week',
                     'release_date', 'release_year', 'age_at_add_years', 'popularity', 'duration_ms']

    for col in temporal_cols:
        if col in df_merged.columns:
            df_clustered[col] = df_merged[col]

    logger.info(f"Merged data: {len(df_clustered)} tracks with temporal information")

    logger.info("Creating temporal visualizations...")
    create_taste_evolution_viz(df_clustered, output_dir)

    logger.info("Generating temporal report...")
    generate_temporal_report(df_clustered, output_dir)

    logger.info("Temporal analysis complete!")

    return df_clustered


if __name__ == '__main__':
    run_temporal_analysis()
