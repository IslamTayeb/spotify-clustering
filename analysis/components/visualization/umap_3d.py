"""UMAP 3D visualization utilities.

This module handles UMAP dimensionality reduction and 3D scatter plot creation.
"""

import umap
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from typing import Optional

from analysis.components.visualization.color_palette import (
    get_cluster_color,
    OUTLIER_COLOR,
)


def compute_umap_embedding(
    features: np.ndarray,
    n_neighbors: int = 20,
    min_dist: float = 0.2,
    random_state: int = 42
) -> np.ndarray:
    """Compute 3D UMAP embedding for visualization.

    Args:
        features: Feature matrix (n_samples x n_features)
        n_neighbors: Controls local vs global structure balance
        min_dist: How tightly points are packed
        random_state: Random seed for reproducibility

    Returns:
        UMAP coordinates (n_samples x 3)
    """
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def build_hover_text(row: pd.Series) -> str:
    """Build hover text for a single track.

    Args:
        row: DataFrame row with track metadata

    Returns:
        HTML-formatted hover text
    """
    # Basic info
    text = (
        f"<b>{row['track_name']}</b><br>"
        f"Artist: {row['artist']}<br>"
        f"Cluster: {row['label']}"
    )

    # Add year and popularity if available
    if 'release_year' in row and row.get('release_year', 0) > 0:
        year_val = row['release_year']
        # Convert from decade scale back to year
        decade = int(year_val * 10 + 1950) // 10 * 10
        text += f" • {decade}s"
    if 'popularity' in row:
        text += f" • Pop: {row['popularity']:.0f}%"
    text += "<br>"

    # Genre and key
    text += (
        f"Genre: {row.get('genre', 'Unknown')}<br>"
        f"BPM: {row.get('bpm', 0):.0f} | Key: {row.get('key', 'Unknown')}<br>"
    )

    # Audio characteristics
    text += (
        f"Dance: {row.get('danceability', 0):.2f} | Instrumental: {row.get('instrumentalness', 0):.2f}<br>"
        f"Valence: {row.get('valence', 0):.2f} | Arousal: {row.get('arousal', 0):.2f}<br>"
    )

    # Electronic/Acoustic and Timbre if available
    if 'electronic_acoustic' in row:
        ea_val = row['electronic_acoustic']
        ea_type = 'Acoustic' if ea_val > 0.7 else 'Electronic' if ea_val < 0.3 else 'Mixed'
        text += f"Sound: {ea_type} ({ea_val:.2f})"
        if 'timbre_brightness' in row:
            tb_val = row['timbre_brightness']
            tb_type = 'Bright' if tb_val > 0.6 else 'Dark' if tb_val < 0.4 else 'Neutral'
            text += f" | Timbre: {tb_type} ({tb_val:.2f})"
        text += "<br>"

    # Engagement and approachability
    text += f"Engagement: {row.get('engagement_score', 0):.2f} | Approachability: {row.get('approachability_score', 0):.2f}<br>"

    # Audio moods (compact)
    mood_values = []
    for mood in ['happy', 'sad', 'aggressive', 'relaxed', 'party']:
        val = row.get(f'mood_{mood}', 0)
        if val > 0.3:  # Only show significant moods
            mood_values.append(f"{mood.capitalize()}: {val:.2f}")
    if mood_values:
        text += f"Audio Moods: {' | '.join(mood_values)}<br>"

    # Genre fusion
    if 'genre_fusion' in row:
        genre_type = 'Pure' if row['genre_fusion'] < 0.4 else 'Fusion' if row['genre_fusion'] > 0.6 else 'Mixed'
        text += f"Genre Fusion: {row['genre_fusion']:.2f} ({genre_type})<br>"

    # Lyric features (only if not instrumental)
    if row.get('instrumentalness', 0) < 0.5:
        # Lyric valence/arousal if different from audio
        lyric_val = row.get('lyric_valence', 0.5)
        lyric_aro = row.get('lyric_arousal', 0.5)
        if abs(lyric_val - row.get('valence', 0)) > 0.2 or abs(lyric_aro - row.get('arousal', 0)) > 0.2:
            text += f"Lyric Emotion: V={lyric_val:.2f} A={lyric_aro:.2f}<br>"

        # Lyric moods (compact, only significant ones)
        lyric_mood_values = []
        for mood in ['happy', 'sad', 'angry', 'relaxed']:
            val = row.get(f'lyric_mood_{mood}', 0)
            if val > 0.3:
                lyric_mood_values.append(f"{mood.capitalize()}: {val:.2f}")
        if lyric_mood_values:
            text += f"Lyric Moods: {' | '.join(lyric_mood_values)}<br>"

        # Lyric attributes (compact)
        lyric_attrs = []
        if row.get('explicit', 0) > 0.5:
            lyric_attrs.append("Explicit")
        if row.get('narrative', 0) > 0.6:
            lyric_attrs.append("Narrative")
        if row.get('vocabulary_complexity', 0) > 0.7:
            lyric_attrs.append("Complex Vocab")
        elif row.get('vocabulary_complexity', 0) < 0.3:
            lyric_attrs.append("Simple Vocab")
        if row.get('repetition', 0) > 0.7:
            lyric_attrs.append("Repetitive")

        if lyric_attrs:
            text += f"Lyrics: {' | '.join(lyric_attrs)}<br>"

        # Theme and language (if not default)
        theme = row.get('theme', 0.5)
        # Handle both string and float formats
        if isinstance(theme, str):
            # If it's already a string theme name, use it directly
            if theme and theme.lower() not in ['none', 'other']:
                text += f"Theme: {theme}<br>"
        elif theme != 0.5:  # Not "none" (numeric format)
            # Map theme value back to name (approximate)
            if theme > 0.9:
                theme_name = "Party"
            elif theme > 0.8:
                theme_name = "Flex"
            elif theme > 0.7:
                theme_name = "Love"
            elif theme > 0.6:
                theme_name = "Social"
            elif theme > 0.5:
                theme_name = "Spirituality"
            elif theme > 0.4:
                theme_name = "Introspection"
            elif theme > 0.3:
                theme_name = "Street"
            elif theme > 0.2:
                theme_name = "Heartbreak"
            else:
                theme_name = "Struggle"
            text += f"Theme: {theme_name}<br>"

        language = row.get('language', 0.5)
        # Handle both string and float formats
        if isinstance(language, str):
            # If it's already a string language name, use it directly
            if language and language.lower() != 'none':
                text += f"Language: {language}<br>"
        elif language != 0.5:  # Not "none" (numeric format)
            # Map language value back to name (approximate)
            if language > 0.9:
                lang_name = "English"
            elif language > 0.75:
                lang_name = "Romance"
            elif language > 0.6:
                lang_name = "Germanic"
            elif language > 0.4:
                lang_name = "Slavic"
            elif language > 0.35:
                lang_name = "Middle Eastern"
            elif language > 0.25:
                lang_name = "South Asian"
            elif language > 0.15:
                lang_name = "East Asian"
            else:
                lang_name = "African"
            text += f"Language: {lang_name}<br>"

    return text


def create_umap_3d_plot(
    df: pd.DataFrame,
    color_by: str = "label",
) -> go.Figure:
    """Create 3D UMAP scatter plot.

    Args:
        df: DataFrame with columns 'x', 'y', 'z', and 'label'
        color_by: Column name to color points by (default: 'label')

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Get unique labels
    unique_labels = sorted(df[color_by].unique())

    # Add trace for each cluster
    for label in unique_labels:
        cluster_points = df[df[color_by] == label]
        if cluster_points.empty:
            continue

        if label == -1:
            name = f"Outliers ({len(cluster_points)})"
            color_val = OUTLIER_COLOR  # Consistent light gray
            size = 3
            opacity = 0.3
        else:
            name = f"Cluster {label} ({len(cluster_points)})"
            color_val = get_cluster_color(label)  # Consistent color from palette
            size = 4
            opacity = 0.8

        # Build hover text
        hover_texts = cluster_points.apply(build_hover_text, axis=1)

        fig.add_trace(
            go.Scatter3d(
                x=cluster_points["x"],
                y=cluster_points["y"],
                z=cluster_points["z"],
                mode="markers",
                name=name,
                marker=dict(
                    size=size,
                    color=color_val,  # Direct hex color (no colorscale needed)
                    opacity=opacity,
                ),
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        height=900,
        margin=dict(t=0, l=0, r=0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
    )

    return fig


def create_interactive_cluster_map(
    df: pd.DataFrame,
) -> go.Figure:
    """Create interactive 3D cluster map with selection support.

    Args:
        df: DataFrame with UMAP coordinates and cluster labels

    Returns:
        Plotly Figure with interactive selection
    """
    return create_umap_3d_plot(df)
