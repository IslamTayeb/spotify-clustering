"""Mood and vocal analysis sections for EDA explorer."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analysis.components.visualization.color_palette import CLUSTER_COLORS
from analysis.pipeline.config import get_cluster_name
from analysis.components.export.chart_export import render_chart_with_export


def render_audio_extremes(df: pd.DataFrame):
    """Render audio extremes section."""
    with st.expander("🔊 Audio Extremes", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            if "approachability_score" in df.columns:
                st.subheader("Most Approachable Songs")
                top_approachable = df.nlargest(10, "approachability_score")[
                    ["track_name", "artist", "approachability_score"]
                ]
                st.dataframe(top_approachable, use_container_width=True, hide_index=True)

                st.subheader("Least Approachable Songs (Most Niche)")
                bottom_approachable = df.nsmallest(10, "approachability_score")[
                    ["track_name", "artist", "approachability_score"]
                ]
                st.dataframe(bottom_approachable, use_container_width=True, hide_index=True)

        with col2:
            if "engagement_score" in df.columns:
                st.subheader("Most Engaging Songs")
                top_engaging = df.nlargest(10, "engagement_score")[
                    ["track_name", "artist", "engagement_score"]
                ]
                st.dataframe(top_engaging, use_container_width=True, hide_index=True)

                st.subheader("Least Engaging Songs")
                bottom_engaging = df.nsmallest(10, "engagement_score")[
                    ["track_name", "artist", "engagement_score"]
                ]
                st.dataframe(bottom_engaging, use_container_width=True, hide_index=True)

        # BPM Extremes
        if "bpm" in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Fastest Songs (Highest BPM)")
                fastest = df.nlargest(10, "bpm")[["track_name", "artist", "bpm"]]
                st.dataframe(fastest, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Slowest Songs (Lowest BPM)")
                slowest = df.nsmallest(10, "bpm")[["track_name", "artist", "bpm"]]
                st.dataframe(slowest, use_container_width=True, hide_index=True)

        # Danceability Extremes
        if "danceability" in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Most Danceable Songs")
                danceable = df.nlargest(10, "danceability")[["track_name", "artist", "danceability"]]
                st.dataframe(danceable, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Least Danceable Songs")
                not_danceable = df.nsmallest(10, "danceability")[["track_name", "artist", "danceability"]]
                st.dataframe(not_danceable, use_container_width=True, hide_index=True)


def render_mood_analysis(df: pd.DataFrame):
    """Render mood analysis section."""
    with st.expander("😊 Mood Analysis", expanded=False):
        mood_cols = ["mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed", "mood_party"]

        if not all(col in df.columns for col in mood_cols):
            st.warning("Mood information not available in this dataset")
            return

        # Radar plot
        st.subheader("Overall Library Mood Profile")

        avg_moods = {
            "Happy": df["mood_happy"].mean() * 100,
            "Sad": df["mood_sad"].mean() * 100,
            "Aggressive": df["mood_aggressive"].mean() * 100,
            "Relaxed": df["mood_relaxed"].mean() * 100,
            "Party": df["mood_party"].mean() * 100,
        }

        fig = go.Figure(data=go.Scatterpolar(
            r=list(avg_moods.values()),
            theta=list(avg_moods.keys()),
            fill="toself",
            name="Library Average",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=500,
            margin=dict(t=0, l=0, r=0, b=0),
        )
        render_chart_with_export(fig, "mood_radar", "Library Mood Profile", "mood")
        st.caption("Average Mood Distribution (%)")

        # Valence vs Arousal scatter plot
        if "valence" in df.columns and "arousal" in df.columns:
            st.subheader("Emotional Quadrants (Valence vs Arousal)")
            st.caption("Songs by Emotional Content")

            # Add cluster names to dataframe for visualization
            df_with_names = df.copy()
            df_with_names["cluster_name"] = df_with_names["cluster"].apply(get_cluster_name)

            fig = px.scatter(
                df_with_names,
                x="valence",
                y="arousal",
                color="cluster_name",
                hover_data=["track_name", "artist"],
                labels={
                    "valence": "Valence (Pleasant)",
                    "arousal": "Arousal (Energy)",
                    "cluster_name": "Cluster",
                },
                color_discrete_sequence=CLUSTER_COLORS,
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_annotation(x=0.75, y=0.75, text="Happy/Energetic", showarrow=False, opacity=0.5)
            fig.add_annotation(x=0.25, y=0.75, text="Angry/Tense", showarrow=False, opacity=0.5)
            fig.add_annotation(x=0.25, y=0.25, text="Sad/Depressed", showarrow=False, opacity=0.5)
            fig.add_annotation(x=0.75, y=0.25, text="Calm/Peaceful", showarrow=False, opacity=0.5)
            fig.update_layout(height=500, margin=dict(t=0, l=0, r=0, b=0))
            render_chart_with_export(fig, "valence_arousal", "Emotional Quadrants", "mood")

        # Mood extremes
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Happiest Songs")
            happiest = df.nlargest(10, "mood_happy")[["track_name", "artist", "mood_happy"]]
            st.dataframe(happiest, use_container_width=True, hide_index=True)

            st.subheader("Most Aggressive Songs")
            aggressive = df.nlargest(10, "mood_aggressive")[["track_name", "artist", "mood_aggressive"]]
            st.dataframe(aggressive, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Saddest Songs")
            saddest = df.nlargest(10, "mood_sad")[["track_name", "artist", "mood_sad"]]
            st.dataframe(saddest, use_container_width=True, hide_index=True)

            st.subheader("Most Relaxed Songs")
            relaxed = df.nlargest(10, "mood_relaxed")[["track_name", "artist", "mood_relaxed"]]
            st.dataframe(relaxed, use_container_width=True, hide_index=True)


def render_vocal_analysis(df: pd.DataFrame):
    """Render vocal analysis section."""
    with st.expander("🎤 Vocal Analysis", expanded=False):
        if "instrumentalness" in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Most Instrumental Songs")
                instrumental = df.nlargest(10, "instrumentalness")[
                    ["track_name", "artist", "instrumentalness"]
                ]
                st.dataframe(instrumental, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Most Vocal Songs")
                vocal = df.nsmallest(10, "instrumentalness")[
                    ["track_name", "artist", "instrumentalness"]
                ]
                st.dataframe(vocal, use_container_width=True, hide_index=True)

        # Voice gender distribution
        if "voice_gender_male" in df.columns and "voice_gender_female" in df.columns:
            st.subheader("Voice Gender Distribution")

            df_vocal = df.copy()
            df_vocal["dominant_gender"] = "Mixed"
            df_vocal.loc[df_vocal["voice_gender_male"] > 0.6, "dominant_gender"] = "Male"
            df_vocal.loc[df_vocal["voice_gender_female"] > 0.6, "dominant_gender"] = "Female"

            gender_counts = df_vocal["dominant_gender"].value_counts()

            fig = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                color_discrete_sequence=CLUSTER_COLORS,
            )
            fig.update_layout(height=500, margin=dict(t=0, l=0, r=0, b=0))
            render_chart_with_export(fig, "voice_gender", "Voice Gender Distribution", "vocal")

        # Acoustic vs Electronic (using the actual clustering dimension)
        if "electronic_acoustic" in df.columns:
            st.subheader("Acoustic ↔ Electronic Distribution")
            st.caption("Production style dimension (0=electronic/synthetic, 1=acoustic/organic)")

            from analysis.components.visualization.color_palette import SPOTIFY_GREEN

            fig = px.histogram(
                df,
                x="electronic_acoustic",
                nbins=50,
                labels={"electronic_acoustic": "Production Style", "count": "Number of Songs"},
                color_discrete_sequence=[SPOTIFY_GREEN],
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Hybrid")
            fig.update_layout(height=500, margin=dict(t=0, l=0, r=0, b=0))
            render_chart_with_export(fig, "acoustic_electronic", "Acoustic/Electronic Distribution", "vocal")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Most Electronic Songs**")
                electronic = df.nsmallest(10, "electronic_acoustic")[["track_name", "artist", "top_genre", "electronic_acoustic"]]
                st.dataframe(electronic, use_container_width=True, hide_index=True)

            with col2:
                st.write("**Most Acoustic Songs**")
                acoustic = df.nlargest(10, "electronic_acoustic")[["track_name", "artist", "top_genre", "electronic_acoustic"]]
                st.dataframe(acoustic, use_container_width=True, hide_index=True)

        # Fallback: show raw mood_acoustic/mood_electronic if electronic_acoustic not available
        elif "mood_acoustic" in df.columns and "mood_electronic" in df.columns:
            st.subheader("Acoustic vs Electronic Distribution (Raw Scores)")
            st.caption("⚠️ Using raw mood scores - electronic_acoustic dimension not found")

            df_production = df.copy()
            df_production["production_style"] = "Mixed"
            df_production.loc[df_production["mood_acoustic"] > 0.6, "production_style"] = "Acoustic"
            df_production.loc[df_production["mood_electronic"] > 0.6, "production_style"] = "Electronic"

            production_counts = df_production["production_style"].value_counts()

            st.caption("Production Style Distribution")
            fig = px.pie(
                values=production_counts.values,
                names=production_counts.index,
                color_discrete_sequence=CLUSTER_COLORS,
            )
            fig.update_layout(height=500, margin=dict(t=0, l=0, r=0, b=0))
            render_chart_with_export(fig, "production_style_pie", "Production Style Distribution", "vocal")


def render_language_analysis(df: pd.DataFrame):
    """Render language analysis section."""
    with st.expander("🌍 Language Analysis", expanded=False):
        if "language" not in df.columns and "lyric_language" not in df.columns:
            st.warning("Language information not available")
            return

        from .utils import group_small_slices, get_pie_colors

        lang_col = "language" if "language" in df.columns else "lyric_language"

        st.subheader("Language Distribution")
        st.caption("Language detection from lyric text using langdetect")

        lang_counts = df[lang_col].value_counts()
        lang_counts_grouped, _ = group_small_slices(lang_counts)

        st.caption("Language Distribution Across Library")
        fig = px.pie(
            values=lang_counts_grouped.values,
            names=lang_counts_grouped.index,
            hole=0.3,
            color_discrete_sequence=get_pie_colors(lang_counts_grouped.index, CLUSTER_COLORS),
        )
        fig.update_layout(height=500, margin=dict(t=0, l=0, r=0, b=0))
        render_chart_with_export(fig, "language_pie", "Language Distribution", "language")

        # Statistics table
        st.subheader("Language Statistics")
        lang_stats = pd.DataFrame({
            "Language": lang_counts.index,
            "Songs": lang_counts.values,
            "Percentage": (lang_counts.values / len(df) * 100).round(1),
        })
        st.dataframe(lang_stats, use_container_width=True, hide_index=True)

        # Sample songs
        st.markdown("---")
        st.subheader("Sample Songs by Language")

        top_langs = [
            lang for lang in lang_counts.index[:5]
            if lang not in ["none", "unknown", "None", "Unknown"]
        ][:3]

        if top_langs:
            cols = st.columns(len(top_langs))
            for i, lang in enumerate(top_langs):
                with cols[i]:
                    st.write(f"**{lang.title()}**")
                    lang_songs = df[df[lang_col] == lang].sample(min(5, len(df[df[lang_col] == lang])))
                    for _, row in lang_songs.iterrows():
                        st.text(f"• {row['track_name']}")
                        st.caption(f"  by {row['artist']}")

        # Language Ladder explanation
        st.markdown("---")
        st.subheader("Language Encoding in Features")
        st.caption("Ordinal scale for clustering (English=1.0 → None=0.0)")

        st.info(
            "**Language Ladder** (1 dimension):\n\n"
            "- English: 1.0\n"
            "- Spanish: 0.86\n"
            "- French: 0.71\n"
            "- Arabic: 0.57\n"
            "- Korean: 0.43\n"
            "- Japanese: 0.29\n"
            "- Unknown: 0.14\n"
            "- None (instrumental): 0.0\n\n"
            "Helps clustering understand language relationships while keeping 1 dimension."
        )

        # Language vs Instrumentalness
        if "instrumentalness" in df.columns:
            st.markdown("---")
            st.subheader("Language vs Instrumentalness")

            st.caption("Instrumentalness by Language")
            fig = px.box(
                df[df[lang_col].isin(top_langs + ["none", "unknown"])],
                x=lang_col,
                y="instrumentalness",
                color_discrete_sequence=CLUSTER_COLORS,
            )
            fig.update_layout(height=500, margin=dict(t=0, l=0, r=0, b=0))
            render_chart_with_export(fig, "language_instrumentalness", "Language vs Instrumentalness", "language")
