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


# NOTE: generate_report() function removed - use Streamlit dashboard for interactive analysis
# Static markdown reports are no longer generated. All analysis is now done through:
# streamlit run analysis/interactive_interpretability.py


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
