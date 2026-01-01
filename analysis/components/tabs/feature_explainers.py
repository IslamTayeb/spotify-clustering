"""Feature Explainers Tab - Comprehensive documentation of the 33-dimensional feature vector.

This tab provides interactive visualizations and explanations for all features
used in the clustering pipeline, making the methodology transparent and
accessible to both technical and non-technical audiences.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.pipeline.config import THEME_SCALE, LANGUAGE_SCALE
from analysis.components.export.chart_export import (
    render_chart_with_export,
    render_export_section,
)


# =============================================================================
# CIRCULAR KEY ENCODING
# =============================================================================


def render_key_encoding_explainer():
    """Render the circular key encoding explanation with interactive visualization."""

    st.subheader("🎹 Circular Key Encoding")

    # Music theory primer for non-musicians
    with st.expander(
        "🎓 Music Theory Primer (click if you're not a musician)", expanded=False
    ):
        st.markdown("""
        **What are musical keys?**

        Western music uses **12 notes** that repeat in a cycle:

        `C → C# → D → D# → E → F → F# → G → G# → A → A# → B → (back to C)`

        - The distance between adjacent notes is called a **semitone** (the smallest step in Western music)
        - The **key** of a song tells you which note feels like "home" - the note the melody gravitates toward
        - **Major keys** sound happy/bright (think: "Happy Birthday")
        - **Minor keys** sound sad/dark (think: most movie villain themes)

        **Why does this matter for clustering?**

        Songs in similar keys often sound harmonically compatible. If you've ever noticed that
        certain songs "flow" well together in a playlist, it's often because they share the same
        key or are in closely related keys.
        """)

    st.markdown("""
    **The Problem:** If we encode keys as simple numbers (C=0, C#=1, ... B=11), the algorithm
    thinks C and B are far apart (distance = 11). But musically, they're neighbors—just **one
    semitone apart**!

    This is like encoding compass directions as N=0°, E=90°, S=180°, W=270° and concluding
    that North and West are far apart (270°) when you could just turn 90° the other way.
    """)

    # Create the visualization
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**❌ Linear Encoding (Wrong)**")
        fig_linear = create_linear_key_figure()
        render_chart_with_export(
            fig_linear, "key_linear", "Linear Key Encoding", "explainers"
        )

    with col2:
        st.markdown("**✅ Circular Encoding (Correct)**")
        fig_circular = create_circular_key_figure()
        render_chart_with_export(
            fig_circular, "key_circular", "Circular Key Encoding", "explainers"
        )

    # Distance comparison
    st.markdown("### Distance Comparison")
    fig_distances = create_key_distance_comparison()
    render_chart_with_export(
        fig_distances, "key_distances", "Key Distance Comparison", "explainers"
    )

    # Formula
    st.markdown("### The Encoding Formula")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.latex(
            r"\text{key\_sin} = \sin\left(\frac{2\pi \times \text{pitch}}{12}\right) \times 0.33"
        )
    with col2:
        st.latex(
            r"\text{key\_cos} = \cos\left(\frac{2\pi \times \text{pitch}}{12}\right) \times 0.33"
        )
    with col3:
        st.latex(
            r"\text{key\_scale} = \begin{cases} 0.33 & \text{major} \\ 0 & \text{minor} \end{cases}"
        )

    st.info("""
    **Why multiply by 0.33?**

    We use 3 dimensions for key (sin, cos, scale). The 0.33 weight ensures these 3 dimensions
    together contribute roughly the same influence as 1 dimension of another feature. Without
    this, key would be overrepresented in the 33-dim vector.
    """)


def create_linear_key_figure():
    """Create linear key encoding visualization."""
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    fig = go.Figure()

    # Number line FIRST (so it's behind the circles)
    fig.add_trace(
        go.Scatter(
            x=[-0.5, 11.5],
            y=[0, 0],
            mode="lines",
            line=dict(color="#ccc", width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Then add circles on top
    for i, key in enumerate(keys):
        color = "#E74C3C" if key in ["C", "B"] else "#3498DB"
        fig.add_trace(
            go.Scatter(
                x=[i],
                y=[0],
                mode="markers+text",
                marker=dict(size=25, color=color, line=dict(color="white", width=2)),
                text=[key],
                textposition="top center",
                textfont=dict(size=11),
                showlegend=False,
                hovertemplate=f"<b>{key}</b><br>Index: {i}<extra></extra>",
            )
        )

    # Distance arrow
    fig.add_annotation(
        x=5.5,
        y=-0.4,
        text="<b>C to B: Distance = 11 (maximum!)</b>",
        showarrow=False,
        font=dict(size=11, color="#E74C3C"),
    )

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=60),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, range=[-1, 12]
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_circular_key_figure():
    """Create circular key encoding visualization."""
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    colors = [
        "#FF6B6B",
        "#FF8E72",
        "#FFB347",
        "#FFD700",
        "#9ACD32",
        "#3CB371",
        "#20B2AA",
        "#4169E1",
        "#6A5ACD",
        "#9932CC",
        "#FF1493",
        "#FF69B4",
    ]

    fig = go.Figure()

    # Circle FIRST (behind everything)
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            line=dict(color="#ddd", width=2, dash="solid"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Draw arc between B and C to show adjacency
    # B is at pitch 11, C is at pitch 0
    arc_theta = np.linspace(2 * np.pi * 11 / 12, 2 * np.pi * 12 / 12, 20)
    fig.add_trace(
        go.Scatter(
            x=1.15 * np.cos(arc_theta),
            y=1.15 * np.sin(arc_theta),
            mode="lines",
            line=dict(color="#27AE60", width=4),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Keys on circle
    for i, (key, color) in enumerate(zip(keys, colors)):
        angle = 2 * np.pi * i / 12
        x, y = np.cos(angle), np.sin(angle)

        # Highlight C and B
        marker_color = "#27AE60" if key in ["C", "B"] else color
        size = 28 if key in ["C", "B"] else 22

        textpos = (
            "top center"
            if y > 0.3
            else "bottom center"
            if y < -0.3
            else "middle right"
            if x > 0
            else "middle left"
        )

        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(
                    size=size, color=marker_color, line=dict(color="white", width=2)
                ),
                text=[key],
                textposition=textpos,
                textfont=dict(size=11, color="#333"),
                showlegend=False,
                hovertemplate=f"<b>{key}</b><br>sin: {y:.3f}<br>cos: {x:.3f}<extra></extra>",
            )
        )

    # Label for B-C adjacency
    fig.add_annotation(
        x=1.55,
        y=-0.35,
        text="<b>B → C</b><br><i>1 semitone</i>",
        showarrow=False,
        font=dict(color="#27AE60", size=11),
        align="left",
    )

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1.5, 1.6],
            scaleanchor="y",
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_key_distance_comparison():
    """Create bar chart comparing linear vs circular distances."""
    pairs = ["C → D", "C → G", "C → B", "B → C", "F# → G"]
    linear = [2, 7, 11, 11, 1]
    circular = [2, 5, 1, 1, 1]
    musical = ["2 semitones", "7 semitones", "1 semitone", "1 semitone", "1 semitone"]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=pairs,
            y=linear,
            name="Linear (0-11)",
            marker_color="#E74C3C",
            text=linear,
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            x=pairs,
            y=circular,
            name="Circular (sin/cos)",
            marker_color="#27AE60",
            text=circular,
            textposition="auto",
        )
    )

    # Add musical reality annotations
    for i, (pair, real) in enumerate(zip(pairs, musical)):
        fig.add_annotation(
            x=pair,
            y=max(linear[i], circular[i]) + 1.5,
            text=f"<i>Reality: {real}</i>",
            showarrow=False,
            font=dict(size=10, color="#666"),
        )

    fig.update_layout(
        barmode="group",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Key Pair",
        yaxis_title="Encoded Distance",
        yaxis=dict(range=[0, 14]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# =============================================================================
# ORDINAL ENCODING (LANGUAGE & THEME)
# =============================================================================


def render_ordinal_encoding_explainer():
    """Render the ordinal encoding explanation for language and theme."""

    st.subheader("🌍 Ordinal Encoding: Language & Theme")

    st.markdown("""
    Language and theme are **categorical** features, but we encode them as **ordinal scales**
    rather than one-hot vectors. Here's why:
    """)

    with st.expander("🤔 Why not One-Hot Encoding?", expanded=True):
        st.markdown("""
        **One-hot encoding** would represent each category as a separate dimension:
        - English = [1, 0, 0, 0, ...]
        - Spanish = [0, 1, 0, 0, ...]
        - Arabic = [0, 0, 1, 0, ...]

        **Problems with one-hot:**

        1. **Dimensionality explosion**: One-hot would add ~8 dimensions for language families
           and ~10 for themes. That's 18 dimensions just for categorical metadata, which could
           dominate the 33-dim vector even with weighting.

        2. **Equal distance assumption is wrong**: One-hot assumes all categories are
           equidistant (Euclidean distance = √2 between any pair). But why should Spanish
           be equally distant from English and Arabic? Musical traditions don't work that way—
           Romance languages share melodic patterns with English pop more than Arabic maqam scales.

        3. **PCA collapses to ordinal anyway**: If you apply PCA to one-hot encoded language
           data, the first principal component typically recovers an ordinal-like scale based
           on the underlying structure. We're just doing this explicitly.

        4. **Ordinal encodes musical tradition similarity**: The language scale groups by
           musical tradition (Romance=0.85, East Asian=0.20) rather than linguistic family.
           This is a deliberate heuristic based on production style similarity.
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Language Scale")
        fig_lang = create_language_scale_figure()
        st.plotly_chart(fig_lang, use_container_width=True)

    with col2:
        st.markdown("### Theme Scale")
        fig_theme = create_theme_scale_figure()
        st.plotly_chart(fig_theme, use_container_width=True)

    st.info("""
    **Note on "none" = 0.5**: For instrumental tracks (instrumentalness > 0.5), we set
    language and theme to 0.5 (centered). This ensures "none" is equidistant from all
    categories rather than being at an edge of the scale.
    """)


def create_language_scale_figure():
    """Create language scale visualization."""
    # Group by family for display
    families = {
        "English": ["english"],
        "Romance": ["spanish", "portuguese", "french"],
        "Germanic": ["german", "swedish", "norwegian"],
        "Slavic": ["russian", "ukrainian", "serbian", "czech"],
        "Middle Eastern": ["arabic", "hebrew", "turkish"],
        "South Asian": ["punjabi"],
        "East Asian": ["korean", "japanese", "chinese", "vietnamese"],
        "African": ["luganda"],
    }

    family_values = {
        "English": 1.0,
        "Romance": 0.857,
        "Germanic": 0.714,
        "Slavic": 0.571,
        "Middle Eastern": 0.429,
        "South Asian": 0.286,
        "East Asian": 0.143,
        "African": 0.0,
    }

    colors = [
        "#1DB954",
        "#2ECC71",
        "#58D68D",
        "#82E0AA",
        "#ABEBC6",
        "#D5F5E3",
        "#E8F8F5",
        "#F0F4F0",
    ]

    fig = go.Figure()

    y_labels = list(family_values.keys())[::-1]
    x_vals = [family_values[f] for f in y_labels]

    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=y_labels,
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:.2f}" for v in x_vals],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Value: %{x:.3f}<extra></extra>",
        )
    )

    # Add "none" marker
    fig.add_trace(
        go.Scatter(
            x=[0.5],
            y=["East Asian"],
            mode="markers",
            marker=dict(size=15, color="#E74C3C", symbol="diamond"),
            name="none/instrumental",
            showlegend=True,
            hovertemplate="<b>none/instrumental</b><br>Value: 0.5 (centered)<extra></extra>",
        )
    )

    fig.update_layout(
        height=350,
        margin=dict(l=100, r=20, t=20, b=40),
        xaxis=dict(title="Ordinal Value", range=[0, 1.1]),
        yaxis=dict(title=""),
        showlegend=True,
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_theme_scale_figure():
    """Create theme scale visualization."""
    themes = {
        "party": 1.0,
        "flex": 0.889,
        "love": 0.778,
        "social": 0.667,
        "spirituality": 0.556,
        "introspection": 0.444,
        "street": 0.333,
        "heartbreak": 0.222,
        "struggle": 0.111,
        "other": 0.0,
    }

    # Color gradient from bright to dark
    colors = [
        "#FFD700",
        "#FFC107",
        "#FF9800",
        "#FF5722",
        "#E91E63",
        "#9C27B0",
        "#673AB7",
        "#3F51B5",
        "#2196F3",
        "#607D8B",
    ]

    fig = go.Figure()

    y_labels = list(themes.keys())[::-1]
    x_vals = [themes[t] for t in y_labels]

    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=y_labels,
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:.2f}" for v in x_vals],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Value: %{x:.3f}<extra></extra>",
        )
    )

    # Add "none" marker
    fig.add_trace(
        go.Scatter(
            x=[0.5],
            y=["introspection"],
            mode="markers",
            marker=dict(size=15, color="#E74C3C", symbol="diamond"),
            name="none/instrumental",
            showlegend=True,
            hovertemplate="<b>none/instrumental</b><br>Value: 0.5 (centered)<extra></extra>",
        )
    )

    fig.update_layout(
        height=400,
        margin=dict(l=100, r=20, t=20, b=40),
        xaxis=dict(title="Ordinal Value", range=[0, 1.1]),
        yaxis=dict(title=""),
        showlegend=True,
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# =============================================================================
# INSTRUMENTALNESS WEIGHTING
# =============================================================================


def render_instrumentalness_explainer():
    """Render the instrumentalness weighting explanation."""

    st.subheader("🎸 Instrumentalness & Lyric Feature Weighting")

    st.markdown("""
    **Instrumentalness** measures whether a track has vocals (0) or is purely instrumental (1).
    This score is crucial because it determines how we weight lyric-derived features.

    **The challenge:** If a track is instrumental, GPT can't analyze lyrics that don't exist.
    But we can't just leave lyric features blank—that would create missing data issues.
    Instead, we use **semantic weighting** to pull lyric features toward neutral values.
    """)

    # Create weighting strategy visualization
    fig = create_weighting_strategy_figure()
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Three Weighting Strategies")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **1. Bipolar Scales**

        *Features:* valence, arousal

        *Formula:*
        ```
        0.5 + (raw - 0.5) × (1 - inst)
        ```

        *Rationale:* These are negative↔positive scales.
        An instrumental track isn't "lyrically negative"—
        it's lyrically **absent**, which should be **neutral (0.5)**.
        """)

    with col2:
        st.markdown("""
        **2. Presence/Absence**

        *Features:* moods, explicit, narrative, vocabulary, repetition

        *Formula:*
        ```
        raw × (1 - inst)
        ```

        *Rationale:* These measure presence (happy/sad/explicit/etc).
        An instrumental track is definitively **non-happy**, **non-sad**,
        **non-explicit**—pull toward **0 (absent)**.
        """)

    with col3:
        st.markdown("""
        **3. Categorical**

        *Features:* theme, language

        *Formula:*
        ```
        if inst > 0.5: use 0.5
        else: use ordinal value
        ```

        *Rationale:* Themes are categorical. A track at inst=0.9
        doesn't have "10% Japanese"—it either has lyrics or doesn't.
        **Hard threshold** at 0.5.
        """)

    # Interactive example
    st.markdown("### Interactive Example")

    col1, col2 = st.columns([1, 2])

    with col1:
        inst_value = st.slider("Instrumentalness", 0.0, 1.0, 0.3, 0.1)
        raw_valence = st.slider("Raw Lyric Valence", 0.0, 1.0, 0.8, 0.1)
        raw_mood = st.slider("Raw Lyric Mood (happy)", 0.0, 1.0, 0.7, 0.1)

    with col2:
        # Calculate weighted values
        weighted_valence = 0.5 + (raw_valence - 0.5) * (1 - inst_value)
        weighted_mood = raw_mood * (1 - inst_value)
        theme_value = 0.5 if inst_value > 0.5 else 0.778  # Assuming "love" theme

        st.metric("Weighted Valence (bipolar → 0.5)", f"{weighted_valence:.3f}")
        st.metric("Weighted Mood (presence → 0)", f"{weighted_mood:.3f}")
        st.metric(
            "Theme (categorical)",
            f"{theme_value:.3f}"
            + (" (centered none)" if inst_value > 0.5 else " (love)"),
        )


def create_weighting_strategy_figure():
    """Create visualization of the three weighting strategies."""

    inst_range = np.linspace(0, 1, 50)
    raw_value = 0.8  # Example raw value

    # Calculate weighted values for each strategy
    bipolar = 0.5 + (raw_value - 0.5) * (1 - inst_range)
    presence = raw_value * (1 - inst_range)
    categorical = np.where(inst_range > 0.5, 0.5, raw_value)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=inst_range,
            y=bipolar,
            mode="lines",
            name="Bipolar (→ 0.5)",
            line=dict(color="#3498DB", width=3),
            hovertemplate="Inst: %{x:.2f}<br>Value: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=inst_range,
            y=presence,
            mode="lines",
            name="Presence (→ 0)",
            line=dict(color="#E74C3C", width=3),
            hovertemplate="Inst: %{x:.2f}<br>Value: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=inst_range,
            y=categorical,
            mode="lines",
            name="Categorical (hard threshold)",
            line=dict(color="#27AE60", width=3),
            hovertemplate="Inst: %{x:.2f}<br>Value: %{y:.3f}<extra></extra>",
        )
    )

    # Add threshold line
    fig.add_vline(
        x=0.5, line_dash="dash", line_color="gray", annotation_text="threshold"
    )

    fig.update_layout(
        title=f"Weighting Strategies (raw value = {raw_value})",
        xaxis_title="Instrumentalness",
        yaxis_title="Weighted Feature Value",
        height=350,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# =============================================================================
# FULL 33-DIMENSION OVERVIEW
# =============================================================================


def render_vector_overview():
    """Render an overview of all 33 dimensions."""

    st.subheader("📐 The 33-Dimensional Feature Vector")

    st.markdown("""
    Every song is represented as a 33-dimensional vector where **each dimension has
    human-readable meaning**. This is the key innovation—unlike black-box embeddings,
    you can explain *why* songs cluster together.
    """)

    # Create dimension table
    dimensions = [
        # Audio Features (0-15)
        (0, "bpm", "Audio", "Tempo normalized to [0,1]", "Essentia RhythmExtractor"),
        (1, "danceability", "Audio", "How suitable for dancing", "Essentia classifier"),
        (
            2,
            "instrumentalness",
            "Audio",
            "0=vocal, 1=instrumental",
            "voice_instrumental classifier",
        ),
        (3, "valence", "Audio", "Emotional positivity", "DEAM model (MusiCNN)"),
        (4, "arousal", "Audio", "Energy/activation level", "DEAM model (MusiCNN)"),
        (
            5,
            "engagement",
            "Audio",
            "Active vs background listening",
            "engagement_regression",
        ),
        (
            6,
            "approachability",
            "Audio",
            "Mainstream vs niche",
            "approachability_regression",
        ),
        (7, "mood_happy", "Audio", "Joy/celebration presence", "mood_happy classifier"),
        (8, "mood_sad", "Audio", "Melancholy/grief presence", "mood_sad classifier"),
        (
            9,
            "mood_aggressive",
            "Audio",
            "Anger/intensity presence",
            "mood_aggressive classifier",
        ),
        (
            10,
            "mood_relaxed",
            "Audio",
            "Calm/peaceful presence",
            "mood_relaxed classifier",
        ),
        (
            11,
            "mood_party",
            "Audio",
            "Upbeat/celebratory presence",
            "mood_party classifier",
        ),
        (
            12,
            "voice_gender",
            "Audio",
            "0=female, 1=male (0 if instrumental)",
            "gender classifier",
        ),
        (
            13,
            "genre_fusion",
            "Audio",
            "0=pure genre, 1=genre fusion",
            "Entropy of genre probs",
        ),
        (
            14,
            "electronic_acoustic",
            "Audio",
            "0=electronic, 1=acoustic",
            "mood_acoustic - mood_electronic",
        ),
        (
            15,
            "timbre_brightness",
            "Audio",
            "0=dark/mellow, 1=bright/crisp",
            "timbre classifier",
        ),
        # Key Features (16-18)
        (16, "key_sin", "Key", "sin(2π × pitch/12) × 0.33", "Circular encoding"),
        (17, "key_cos", "Key", "cos(2π × pitch/12) × 0.33", "Circular encoding"),
        (18, "key_scale", "Key", "0=minor, 0.33=major", "Scale type"),
        # Lyric Features (19-28)
        (
            19,
            "lyric_valence",
            "Lyrics",
            "Emotional tone (→ 0.5 if instrumental)",
            "GPT + bipolar weighting",
        ),
        (
            20,
            "lyric_arousal",
            "Lyrics",
            "Energy level (→ 0.5 if instrumental)",
            "GPT + bipolar weighting",
        ),
        (
            21,
            "lyric_mood_happy",
            "Lyrics",
            "Joy in lyrics (→ 0 if instrumental)",
            "GPT + presence weighting",
        ),
        (
            22,
            "lyric_mood_sad",
            "Lyrics",
            "Grief in lyrics (→ 0 if instrumental)",
            "GPT + presence weighting",
        ),
        (
            23,
            "lyric_mood_aggressive",
            "Lyrics",
            "Anger in lyrics (→ 0 if instrumental)",
            "GPT + presence weighting",
        ),
        (
            24,
            "lyric_mood_relaxed",
            "Lyrics",
            "Peace in lyrics (→ 0 if instrumental)",
            "GPT + presence weighting",
        ),
        (
            25,
            "lyric_explicit",
            "Lyrics",
            "Holistic explicit score (→ 0 if instrumental)",
            "GPT + presence weighting",
        ),
        (
            26,
            "lyric_narrative",
            "Lyrics",
            "0=vibes → 1=specific story",
            "GPT + presence weighting",
        ),
        (27, "lyric_vocabulary", "Lyrics", "Type-token ratio", "Local computation"),
        (
            28,
            "lyric_repetition",
            "Lyrics",
            "1 - (unique lines / total)",
            "Local computation",
        ),
        # Metadata Features (29-32)
        (
            29,
            "theme",
            "Meta",
            "Ordinal theme scale (0.5 if instrumental)",
            "GPT + categorical",
        ),
        (
            30,
            "language",
            "Meta",
            "Ordinal language scale (0.5 if instrumental)",
            "GPT + categorical",
        ),
        (
            31,
            "popularity",
            "Meta",
            "Spotify popularity [0-100] normalized",
            "Spotify API",
        ),
        (
            32,
            "release_year",
            "Meta",
            "Decade bucket [0=1950s, 1=2020s]",
            "Spotify metadata",
        ),
    ]

    df = pd.DataFrame(
        dimensions, columns=["Dim", "Name", "Category", "Description", "Source"]
    )

    # Color by category
    category_colors = {
        "Audio": "#3498DB",
        "Key": "#9B59B6",
        "Lyrics": "#27AE60",
        "Meta": "#E67E22",
    }

    # Display as styled table
    st.dataframe(
        df.style.apply(
            lambda row: [
                f"background-color: {category_colors.get(row['Category'], '#fff')}20"
            ]
            * len(row),
            axis=1,
        ),
        use_container_width=True,
        height=600,
    )

    # Category breakdown
    st.markdown("### Category Breakdown")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Audio Features", "16 dims", "0-15")
    with col2:
        st.metric("Key Features", "3 dims", "16-18")
    with col3:
        st.metric("Lyric Features", "10 dims", "19-28")
    with col4:
        st.metric("Metadata", "4 dims", "29-32")


# =============================================================================
# ESSENTIA MODEL ARCHITECTURE
# =============================================================================


def render_essentia_explainer():
    """Render explanation of Essentia model architecture."""

    st.subheader("🧠 Essentia Model Architecture")

    st.markdown("""
    [Essentia](https://essentia.upf.edu/) is an open-source library for audio analysis.
    We use its pre-trained neural networks to extract interpretable audio features.
    """)

    st.markdown("### Two-Stage Pipeline")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Stage 1: Embedding Extraction**

        A backbone neural network converts raw audio waveform into a high-dimensional embedding:

        | Backbone | Dimensions | Used For |
        |----------|------------|----------|
        | **EffNet-Discogs** | 1,280 | Most classifiers |
        | **MusiCNN** | 200 | Valence/arousal (DEAM) |

        These embeddings capture acoustic patterns but aren't directly interpretable.
        """)

    with col2:
        st.markdown("""
        **Stage 2: Classification Heads**

        Smaller neural networks trained on top of embeddings for specific tasks:

        - `genre_discogs400` - 400 genre probabilities
        - `mood_happy/sad/aggressive/relaxed/party` - Binary mood classifiers
        - `danceability` - Dance suitability
        - `voice_instrumental` - Vocal vs instrumental
        - `gender` - Voice male/female
        - `timbre` - Bright vs dark
        - `mood_acoustic/electronic` - Production style
        """)

    # Pipeline diagram
    st.markdown("### Pipeline Flow")

    fig = create_essentia_pipeline_diagram()
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Why classifiers instead of raw embeddings?**

    Raw 1,280-dim embeddings are powerful but opaque—you can't explain what each dimension means.
    Classifier outputs are interpretable: "this song has 0.8 danceability and 0.3 aggressiveness"
    is actionable information that explains clustering results.
    """)


def create_essentia_pipeline_diagram():
    """Create a visual diagram of the Essentia pipeline."""

    fig = go.Figure()

    # Nodes
    nodes = [
        ("Audio\nWaveform", 0, 0.5, "#3498DB"),
        ("EffNet\nBackbone", 1, 0.7, "#9B59B6"),
        ("MusiCNN\nBackbone", 1, 0.3, "#9B59B6"),
        ("1280-dim\nEmbedding", 2, 0.7, "#E67E22"),
        ("200-dim\nEmbedding", 2, 0.3, "#E67E22"),
        ("Genre\nClassifier", 3, 0.85, "#27AE60"),
        ("Mood\nClassifiers", 3, 0.65, "#27AE60"),
        ("Danceability", 3, 0.45, "#27AE60"),
        ("Valence/\nArousal", 3, 0.25, "#27AE60"),
        ("16 Audio\nFeatures", 4, 0.5, "#E74C3C"),
    ]

    for name, x, y, color in nodes:
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(size=50, color=color, line=dict(color="white", width=2)),
                text=[name],
                textposition="middle center",
                textfont=dict(size=9, color="white"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Arrows (simplified as lines)
    arrows = [
        (0, 0.5, 1, 0.7),  # Audio -> EffNet
        (0, 0.5, 1, 0.3),  # Audio -> MusiCNN
        (1, 0.7, 2, 0.7),  # EffNet -> embedding
        (1, 0.3, 2, 0.3),  # MusiCNN -> embedding
        (2, 0.7, 3, 0.85),  # embedding -> genre
        (2, 0.7, 3, 0.65),  # embedding -> mood
        (2, 0.7, 3, 0.45),  # embedding -> dance
        (2, 0.3, 3, 0.25),  # MusiCNN -> valence
        (3, 0.85, 4, 0.5),  # classifiers -> features
        (3, 0.65, 4, 0.5),
        (3, 0.45, 4, 0.5),
        (3, 0.25, 4, 0.5),
    ]

    for x1, y1, x2, y2 in arrows:
        fig.add_trace(
            go.Scatter(
                x=[x1 + 0.1, x2 - 0.1],
                y=[y1, y2],
                mode="lines",
                line=dict(color="#ccc", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, range=[-0.3, 4.5]
        ),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================


def render_feature_explainers(df: pd.DataFrame = None):
    """Main render function for the Feature Explainers tab."""

    st.header("📚 Feature Explainers")
    st.markdown("""
    This tab documents how the 33-dimensional interpretable feature vector is constructed.
    Each section explains a key design decision with interactive visualizations.
    """)

    # Navigation
    sections = [
        "📐 Vector Overview",
        "🎹 Circular Key Encoding",
        "🌍 Ordinal Encoding (Language & Theme)",
        "🎸 Instrumentalness Weighting",
        "🧠 Essentia Model Architecture",
    ]

    selected = st.radio("Jump to section:", sections, horizontal=True)

    st.markdown("---")

    if selected == "📐 Vector Overview":
        render_vector_overview()
    elif selected == "🎹 Circular Key Encoding":
        render_key_encoding_explainer()
    elif selected == "🌍 Ordinal Encoding (Language & Theme)":
        render_ordinal_encoding_explainer()
    elif selected == "🎸 Instrumentalness Weighting":
        render_instrumentalness_explainer()
    elif selected == "🧠 Essentia Model Architecture":
        render_essentia_explainer()

    # Export section for any selected charts
    render_export_section("export/visualizations", "explainers")
