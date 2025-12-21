#!/usr/bin/env python3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_interactive_map(df: pd.DataFrame, results: Dict) -> go.Figure:
    # Auto-detect 2D vs 3D based on dataframe columns
    is_3d = 'umap_z' in df.columns
    logger.info(f"Creating {'3D' if is_3d else '2D'} visualization")

    fig = go.Figure()

    for cluster_id in sorted(df['cluster'].unique()):
        if cluster_id == -1:
            name = 'Outliers'
            color = 'lightgray'
            showlegend = True
        else:
            name = f"Cluster {cluster_id}"
            color = None
            showlegend = True

        cluster_df = df[df['cluster'] == cluster_id]

        # Build hover text with temporal info if available
        def build_hover_text(row):
            text = (
                f"<b>{row['track_name']}</b><br>"
                f"Artist: {row['artist']}<br>"
                f"Cluster: {row['cluster']}<br>"
                f"Genre: {row['top_genre']}<br>"
                f"BPM: {row['bpm']:.0f} | Key: {row['key']}<br>"
                f"Danceability: {row['danceability']:.2f}<br>"
                f"Instrumentalness: {row['instrumentalness']:.2f}<br>"
                f"Valence: {row.get('valence', 0):.2f} | Arousal: {row.get('arousal', 0):.2f}<br>"
                f"Engagement: {row.get('engagement_score', 0):.2f} | Approachability: {row.get('approachability_score', 0):.2f}<br>"
                f"Moods:<br>"
                f"- Happy: {row['mood_happy']:.2f}<br>"
                f"- Sad: {row['mood_sad']:.2f}<br>"
                f"- Aggressive: {row['mood_aggressive']:.2f}<br>"
                f"- Relaxed: {row['mood_relaxed']:.2f}<br>"
                f"- Party: {row['mood_party']:.2f}<br>"
                f"Language: {row['language'] if row['has_lyrics'] else 'No lyrics'}"
            )

            # Add temporal info if available
            if 'added_at' in row and pd.notna(row['added_at']):
                text += f"<br>Added: {row['added_at'].strftime('%Y-%m-%d')}"
            if 'release_date' in row and pd.notna(row['release_date']):
                text += f"<br>Released: {row['release_date'].year}"
            if 'age_at_add_years' in row and pd.notna(row['age_at_add_years']):
                text += f"<br>Age when added: {row['age_at_add_years']:.1f} years"

            return text

        hover_text = cluster_df.apply(build_hover_text, axis=1)

        if is_3d:
            # 3D scatter plot
            fig.add_trace(go.Scatter3d(
                x=cluster_df['umap_x'],
                y=cluster_df['umap_y'],
                z=cluster_df['umap_z'],
                mode='markers',
                name=name,
                marker=dict(
                    size=5,
                    color=color if cluster_id == -1 else cluster_id,
                    colorscale='Viridis' if cluster_id != -1 else None,
                    showscale=False,
                    line=dict(width=0.5, color='white'),
                    opacity=0.8
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                showlegend=showlegend
            ))
        else:
            # 2D scatter plot (original)
            fig.add_trace(go.Scatter(
                x=cluster_df['umap_x'],
                y=cluster_df['umap_y'],
                mode='markers',
                name=name,
                marker=dict(
                    size=8,
                    color=color if cluster_id == -1 else cluster_id,
                    colorscale='Viridis' if cluster_id != -1 else None,
                    showscale=False,
                    line=dict(width=0.5, color='white')
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                showlegend=showlegend
            ))

    if is_3d:
        # 3D layout
        fig.update_layout(
            title='My Music Taste Map (3D)',
            scene=dict(
                xaxis=dict(title='UMAP Dimension 1', showgrid=True),
                yaxis=dict(title='UMAP Dimension 2', showgrid=True),
                zaxis=dict(title='UMAP Dimension 3', showgrid=True),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            hovermode='closest',
            template='plotly_white',
            width=1400,
            height=900,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
    else:
        # 2D layout (original)
        fig.update_layout(
            title='My Music Taste Map',
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            hovermode='closest',
            template='plotly_white',
            width=1400,
            height=900,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

    return fig


def generate_report(results: Dict, output_dir: str = 'outputs'):
    df = results['dataframe']
    cluster_stats = results['cluster_stats']
    outliers = results['outlier_songs']

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_songs = len(df)
    n_clusters = results['n_clusters']
    n_outliers = results['n_outliers']
    outlier_pct = (n_outliers / n_songs * 100) if n_songs > 0 else 0

    genre_counts = {}
    for top_3 in df['top_3_genres'].values:
        for genre, prob in top_3:
            if genre not in genre_counts:
                genre_counts[genre] = []
            genre_counts[genre].append(prob)

    top_10_genres = sorted(
        [(g, np.mean(p), len(p)) for g, p in genre_counts.items()],
        key=lambda x: (x[2], x[1]),
        reverse=True
    )[:10]

    mood_cols = ['mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed', 'mood_party']
    avg_moods = {col: df[col].mean() * 100 for col in mood_cols}

    median_bpm = df['bpm'].median()
    major_pct = (df['key'].str.contains('major', case=False, na=False).sum() / n_songs * 100)
    minor_pct = (df['key'].str.contains('minor', case=False, na=False).sum() / n_songs * 100)
    avg_danceability = df['danceability'].mean()
    vocal_pct = (df['is_vocal'].sum() / n_songs * 100)
    with_lyrics_pct = (df['has_lyrics'].sum() / n_songs * 100)

    lang_dist = df['language'].value_counts()

    report = f"""# My Music Taste Analysis Report

**Generated**: {timestamp}
**Total Songs**: {n_songs}
**Clusters Found**: {n_clusters}
**Outliers**: {n_outliers} ({outlier_pct:.1f}%)
**Silhouette Score**: {results['silhouette_score']:.3f}

---

## Overview Statistics

### Genre Distribution
Top 10 genres across entire library:
"""

    for i, (genre, avg_prob, count) in enumerate(top_10_genres, 1):
        report += f"{i}. {genre}: {avg_prob*100:.1f}% (appears in {count} songs)\n"

    report += f"""
### Mood Profile
- Happy: {avg_moods['mood_happy']:.1f}%
- Sad: {avg_moods['mood_sad']:.1f}%
- Aggressive: {avg_moods['mood_aggressive']:.1f}%
- Relaxed: {avg_moods['mood_relaxed']:.1f}%
- Party: {avg_moods['mood_party']:.1f}%

### Musical Characteristics
- **Median BPM**: {median_bpm:.0f}
- **Key Distribution**: {major_pct:.1f}% major, {minor_pct:.1f}% minor
- **Average Danceability**: {avg_danceability:.2f}
- **Vocal vs Instrumental**: {vocal_pct:.1f}% vocal, {100-vocal_pct:.1f}% instrumental
- **Songs with Lyrics**: {with_lyrics_pct:.1f}%

### Language Distribution
"""

    for lang, count in lang_dist.head(10).items():
        pct = count / n_songs * 100
        report += f"- {lang}: {count} songs ({pct:.1f}%)\n"

    report += "\n---\n\n## Cluster Breakdown\n\n"

    for cluster_id in sorted(cluster_stats.keys()):
        stats = cluster_stats[cluster_id]

        top_genres_str = "\n".join([
            f"{i}. {genre} ({prob*100:.1f}%)"
            for i, (genre, prob) in enumerate(stats['top_3_genres'], 1)
        ])

        rep_songs_str = "\n".join([
            f"{i}. {song}"
            for i, song in enumerate(stats['representative_songs'], 1)
        ])

        lang_dist_str = ", ".join([
            f"{lang}: {count}" for lang, count in
            sorted(stats['language_distribution'].items(), key=lambda x: x[1], reverse=True)[:3]
        ])

        report += f"""### Cluster {cluster_id}

**Size**: {stats['n_songs']} songs ({stats['percentage']:.1f}% of library)

**Top Genres**:
{top_genres_str}

**Musical Profile**:
- Median BPM: {stats['median_bpm']:.0f}
- Key: {stats['key_distribution']['major']*100:.1f}% major, {stats['key_distribution']['minor']*100:.1f}% minor
- Avg Danceability: {stats['avg_danceability']:.2f}

**Mood**:
- Happy: {stats['mood_distribution']['mood_happy']*100:.1f}%
- Sad: {stats['mood_distribution']['mood_sad']*100:.1f}%
- Aggressive: {stats['mood_distribution']['mood_aggressive']*100:.1f}%
- Relaxed: {stats['mood_distribution']['mood_relaxed']*100:.1f}%
- Party: {stats['mood_distribution']['mood_party']*100:.1f}%

**Languages**: {lang_dist_str}

"""

        # Add lyric themes if available
        if stats.get('lyric_themes') and stats['lyric_themes']:
            themes = stats['lyric_themes']
            keywords = ', '.join([kw[0] for kw in themes['top_keywords'].get('unigrams', [])[:5]])

            report += f"""**Lyric Themes** ({themes['n_lyrics']} songs with lyrics):
- Top Keywords: {keywords}
- Sentiment: {themes['sentiment_label']} ({themes['avg_sentiment']:.2f})
- Vocabulary Richness: {themes['avg_complexity']:.2f}

"""

        report += f"""**Representative Songs**:
{rep_songs_str}

---

"""

    report += f"""## Outliers

{n_outliers} songs didn't fit into any cluster. These might be:
- Unique experiments in your taste
- Guilty pleasures
- Songs that bridge multiple clusters

See `outliers.txt` for the full list.

---

## Extremes

- **Highest BPM**: {df.loc[df['bpm'].idxmax(), 'filename']} ({df['bpm'].max():.0f} BPM)
- **Lowest BPM**: {df.loc[df['bpm'].idxmin(), 'filename']} ({df['bpm'].min():.0f} BPM)
- **Most Danceable**: {df.loc[df['danceability'].idxmax(), 'filename']}
- **Least Danceable**: {df.loc[df['danceability'].idxmin(), 'filename']}
- **Happiest**: {df.loc[df['mood_happy'].idxmax(), 'filename']}
- **Saddest**: {df.loc[df['mood_sad'].idxmax(), 'filename']}

---

## Next Steps

1. **Listen to clusters**: Start with Cluster 0 and see if it makes intuitive sense
2. **Explore outliers**: Check `outliers.txt` for surprising inclusions
3. **Use the interactive map**: Open `music_taste_map.html` to explore visually
"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / 'music_taste_report.md', 'w') as f:
        f.write(report)

    with open(output_path / 'outliers.txt', 'w') as f:
        f.write('\n'.join(outliers))

    logger.info(f"Report saved to {output_path / 'music_taste_report.md'}")
    logger.info(f"Outliers saved to {output_path / 'outliers.txt'}")


def create_combined_map(all_results: Dict[str, Dict]) -> go.Figure:
    """
    Create a combined visualization with 3 subplots (audio, lyrics, combined).
    Each subplot is a 3D scatter plot showing clustering results for that mode.
    """
    from plotly.subplots import make_subplots

    modes = ['audio', 'lyrics', 'combined']
    mode_titles = {
        'audio': 'Audio-Only Clustering',
        'lyrics': 'Lyrics-Only Clustering',
        'combined': 'Combined (Audio + Lyrics) Clustering'
    }

    # Create subplots with 1 row, 3 columns
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[mode_titles.get(mode, mode.title()) for mode in modes if mode in all_results],
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        horizontal_spacing=0.05
    )

    for col_idx, mode in enumerate(modes, start=1):
        if mode not in all_results:
            continue

        results = all_results[mode]
        df = results['dataframe']

        # Check if 3D coordinates exist
        if 'umap_z' not in df.columns:
            logger.warning(f"No 3D coordinates for {mode} mode, skipping")
            continue

        for cluster_id in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster_id]

            if cluster_id == -1:
                name = f'Outliers ({mode})'
                color = 'lightgray'
                size = 3
                opacity = 0.3
            else:
                name = f'{mode.title()} C{cluster_id}'
                color = cluster_id
                size = 4
                opacity = 0.7

            # Build hover text
            def build_hover_text(row):
                return (
                    f"<b>{row['track_name']}</b><br>"
                    f"Artist: {row['artist']}<br>"
                    f"Cluster: {row['cluster']}<br>"
                    f"Genre: {row['top_genre']}<br>"
                    f"BPM: {row['bpm']:.0f} | Key: {row['key']}<br>"
                    f"Moods: Happy={row['mood_happy']:.2f}, Sad={row['mood_sad']:.2f}"
                )

            hover_text = cluster_df.apply(build_hover_text, axis=1)

            fig.add_trace(
                go.Scatter3d(
                    x=cluster_df['umap_x'],
                    y=cluster_df['umap_y'],
                    z=cluster_df['umap_z'],
                    mode='markers',
                    name=name,
                    marker=dict(
                        size=size,
                        color=color if cluster_id == -1 else cluster_id,
                        colorscale='Viridis' if cluster_id != -1 else None,
                        showscale=False,
                        opacity=opacity,
                    ),
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>',
                    legendgroup=f'{mode}_{cluster_id}',
                    showlegend=(col_idx == 1)  # Only show legend for first subplot
                ),
                row=1, col=col_idx
            )

    # Update layout
    fig.update_layout(
        title_text='Music Taste Analysis: Comparison Across Modes',
        height=700,
        width=1800,
        template='plotly_white',
        showlegend=True
    )

    # Update 3D scene settings for all subplots
    for i in range(1, 4):
        fig.update_scenes(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            row=1, col=i
        )

    return fig


if __name__ == '__main__':
    import pickle

    with open('outputs/analysis_data.pkl', 'rb') as f:
        results = pickle.load(f)

    fig = create_interactive_map(results['dataframe'], results)
    fig.write_html('outputs/music_taste_map.html')
    print("Visualization saved to outputs/music_taste_map.html")

    generate_report(results)
    print("Report generated")
