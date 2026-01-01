"""Feature statistics sections for EDA explorer."""

import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from analysis.components.visualization.color_palette import SPOTIFY_GREEN


def render_feature_vector_analysis(df: pd.DataFrame):
    """Render the interpretable embedding features section."""
    with st.expander("🧮 Interpretable Embedding Features", expanded=True):
        st.subheader("Feature Vector Analysis - Clustering Features")
        st.caption("These normalized features are the EXACT values used for clustering")

        embedding_feature_cols = [col for col in df.columns if col.startswith("emb_")]

        if not embedding_feature_cols:
            st.warning(
                "No embedding columns (emb_*) found in dataframe. "
                "This data may have been generated before embedding columns were added. "
                "Re-run the analysis pipeline to include embedding dimensions."
            )
            return

        # Overview Metrics
        st.markdown("### 📊 Feature Statistics")
        st.caption(f"Analyzing all {len(embedding_feature_cols)} embedding dimensions used for clustering")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Embedding Dimensions", len(embedding_feature_cols))
            st.caption("(All clustering features)")

        with col2:
            df_features = df[embedding_feature_cols].fillna(0)
            vector_magnitudes = np.sqrt((df_features**2).sum(axis=1))
            st.metric("Avg Magnitude (subset)", f"{vector_magnitudes.mean():.2f}")

        with col3:
            feature_ranges = df_features.max() - df_features.min()
            avg_range = feature_ranges.mean()
            st.metric("Avg Feature Range", f"{avg_range:.2f}")

        # Feature Statistics Table
        st.markdown("---")
        st.markdown("### 📈 Feature Statistics")
        st.caption("Statistics for features that are part of the clustering embedding")

        stats_df = pd.DataFrame({
            "Feature": embedding_feature_cols,
            "Mean": [df[col].mean() for col in embedding_feature_cols],
            "Std": [df[col].std() for col in embedding_feature_cols],
            "Min": [df[col].min() for col in embedding_feature_cols],
            "Max": [df[col].max() for col in embedding_feature_cols],
            "Range": [df[col].max() - df[col].min() for col in embedding_feature_cols],
        })

        stats_df["Relative Importance"] = (
            stats_df["Range"] / stats_df["Range"].max()
        ) * (stats_df["Std"] / stats_df["Std"].max())
        stats_df = stats_df.sort_values("Relative Importance", ascending=False)

        for col in ["Mean", "Std", "Min", "Max", "Range", "Relative Importance"]:
            stats_df[col] = stats_df[col].round(4)

        st.dataframe(stats_df, use_container_width=True, height=400)

        # Feature Distribution Visualizations
        st.markdown("---")
        st.markdown("### 📊 Feature Distributions (Top 9)")

        top_features = stats_df["Feature"].head(9).tolist()
        n_cols = 3
        n_rows = 3

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=top_features,
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        for idx, feature in enumerate(top_features):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            hist_data = df[feature].dropna()

            fig.add_trace(
                go.Histogram(x=hist_data, name=feature, showlegend=False, marker_color=SPOTIFY_GREEN),
                row=row, col=col,
            )

        fig.update_layout(height=800, showlegend=False, margin=dict(t=0, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Download Embedding Features
        st.markdown("---")
        st.markdown("### 💾 Export Embedding Features")

        export_df = df[["track_name", "artist", "cluster"] + embedding_feature_cols].copy()
        csv_features = export_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Embedding Features (CSV)",
            data=csv_features,
            file_name="embedding_features.csv",
            mime="text/csv",
        )

        st.success("✨ Interpretable embedding analysis complete!")


def render_raw_features(df: pd.DataFrame):
    """Render the raw dataframe features section."""
    embedding_feature_cols = [col for col in df.columns if col.startswith("emb_")]

    with st.expander("📊 Raw Dataframe Features (not used in clustering)", expanded=False):
        st.subheader("Additional Numeric Columns")
        st.caption("These raw values are stored for display but NOT used in clustering")

        exclude_cols = [
            "track_name", "artist", "cluster", "umap_x", "umap_y", "umap_z",
            "added_at", "release_date", "spotify_uri", "track_id", "album",
            "top_genre", "language", "lyric_language", "has_lyrics",
            "age_at_add_years", "release_year", "added_year", "added_month",
            "time_period", "quarter", "month", "decade", "age_category",
            "dominant_gender", "production_style", "is_vocal", "album_name",
            "album_type", "explicit", "dominant_mood", "top_3_genres",
            "genre_probs", "mtg_jamendo_probs",
        ] + embedding_feature_cols

        raw_feature_cols = [
            col for col in df.columns
            if df[col].dtype in ["float64", "int64", "float32", "int32"]
            and col not in exclude_cols
        ]

        if not raw_feature_cols:
            st.info("No additional raw features found (all numeric columns are in the embedding)")
            return

        st.warning(
            f"⚠️ Found {len(raw_feature_cols)} raw columns that are NOT part of the embedding. "
            "These are metadata stored for display purposes only."
        )

        raw_stats_df = pd.DataFrame({
            "Feature": raw_feature_cols,
            "Mean": [df[col].mean() for col in raw_feature_cols],
            "Std": [df[col].std() for col in raw_feature_cols],
            "Min": [df[col].min() for col in raw_feature_cols],
            "Max": [df[col].max() for col in raw_feature_cols],
        })

        for col in ["Mean", "Std", "Min", "Max"]:
            raw_stats_df[col] = raw_stats_df[col].round(4)

        st.dataframe(raw_stats_df, use_container_width=True, height=300)

        st.info(
            "**Note:** Raw features like `bpm`, `valence`, `arousal` are shown here in their unnormalized form. "
            "In the actual clustering, these are normalized to [0,1] within the embedding."
        )


# Feature category definitions matching interpretable_features.py structure
FEATURE_CATEGORIES = {
    "Audio Features (16)": [
        "emb_bpm", "emb_danceability", "emb_instrumentalness", "emb_valence",
        "emb_arousal", "emb_engagement", "emb_approachability",
        "emb_mood_happy", "emb_mood_sad", "emb_mood_aggressive",
        "emb_mood_relaxed", "emb_mood_party", "emb_voice_gender",
        "emb_genre_fusion", "emb_acoustic_electronic", "emb_timbre_brightness",
    ],
    "Key Features (3)": [
        "emb_key_sin", "emb_key_cos", "emb_key_scale",
    ],
    "Lyric Features (10)": [
        "emb_lyric_valence", "emb_lyric_arousal",
        "emb_lyric_mood_happy", "emb_lyric_mood_sad",
        "emb_lyric_mood_aggressive", "emb_lyric_mood_relaxed",
        "emb_lyric_explicit", "emb_lyric_narrative",
        "emb_lyric_vocabulary", "emb_lyric_repetition",
    ],
    "Metadata Features (4)": [
        "emb_theme", "emb_language", "emb_popularity", "emb_release_year",
    ],
}

# Friendly display names for features
FEATURE_DISPLAY_NAMES = {
    "emb_bpm": "BPM",
    "emb_danceability": "Danceability",
    "emb_instrumentalness": "Instrumentalness",
    "emb_valence": "Audio Valence",
    "emb_arousal": "Audio Arousal",
    "emb_engagement": "Engagement",
    "emb_approachability": "Approachability",
    "emb_mood_happy": "Mood: Happy",
    "emb_mood_sad": "Mood: Sad",
    "emb_mood_aggressive": "Mood: Aggressive",
    "emb_mood_relaxed": "Mood: Relaxed",
    "emb_mood_party": "Mood: Party",
    "emb_voice_gender": "Voice Gender (0=F, 1=M)",
    "emb_genre_fusion": "Genre Fusion",
    "emb_acoustic_electronic": "Acoustic↔Electronic",
    "emb_timbre_brightness": "Timbre Brightness",
    "emb_key_sin": "Key (sin)",
    "emb_key_cos": "Key (cos)",
    "emb_key_scale": "Major/Minor",
    "emb_lyric_valence": "Lyric Valence",
    "emb_lyric_arousal": "Lyric Arousal",
    "emb_lyric_mood_happy": "Lyric: Happy",
    "emb_lyric_mood_sad": "Lyric: Sad",
    "emb_lyric_mood_aggressive": "Lyric: Aggressive",
    "emb_lyric_mood_relaxed": "Lyric: Relaxed",
    "emb_lyric_explicit": "Explicit Content",
    "emb_lyric_narrative": "Narrative Style",
    "emb_lyric_vocabulary": "Vocabulary Richness",
    "emb_lyric_repetition": "Repetition",
    "emb_theme": "Theme",
    "emb_language": "Language",
    "emb_popularity": "Popularity",
    "emb_release_year": "Release Year",
}


def render_feature_distributions_by_category(df: pd.DataFrame):
    """Render all feature distributions organized by category (Audio, Lyrics, Metadata)."""
    with st.expander("📊 Feature Distributions by Category", expanded=True):
        st.subheader("All 33 Embedding Features")
        st.caption("Distributions split by Audio, Key, Lyrics, and Metadata categories")

        # Check for embedding columns
        embedding_cols = [col for col in df.columns if col.startswith("emb_")]
        if not embedding_cols:
            st.warning("No embedding columns found. Re-run analysis to include embeddings.")
            return

        # Color palette for each category
        category_colors = {
            "Audio Features (16)": "#1DB954",      # Spotify green
            "Key Features (3)": "#E91E63",         # Pink
            "Lyric Features (10)": "#2196F3",      # Blue
            "Metadata Features (4)": "#FF9800",    # Orange
        }

        for category_name, features in FEATURE_CATEGORIES.items():
            # Filter to features that exist in the dataframe
            available_features = [f for f in features if f in df.columns]

            if not available_features:
                continue

            st.markdown(f"### {category_name}")
            color = category_colors.get(category_name, SPOTIFY_GREEN)

            # Calculate grid dimensions
            n_features = len(available_features)
            n_cols = min(4, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols

            # Create subplot titles with friendly names
            titles = [FEATURE_DISPLAY_NAMES.get(f, f.replace("emb_", "")) for f in available_features]

            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=titles,
                vertical_spacing=0.15,
                horizontal_spacing=0.08,
            )

            for idx, feature in enumerate(available_features):
                row = idx // n_cols + 1
                col = idx % n_cols + 1
                hist_data = df[feature].dropna()

                fig.add_trace(
                    go.Histogram(
                        x=hist_data,
                        name=FEATURE_DISPLAY_NAMES.get(feature, feature),
                        showlegend=False,
                        marker_color=color,
                        opacity=0.8,
                    ),
                    row=row, col=col,
                )

            height = max(300, n_rows * 200)
            fig.update_layout(
                height=height,
                showlegend=False,
                margin=dict(t=40, l=40, r=20, b=20),
            )
            fig.update_xaxes(title_text="", tickfont=dict(size=9))
            fig.update_yaxes(title_text="", tickfont=dict(size=9))

            st.plotly_chart(fig, use_container_width=True)

            # Summary stats for this category
            with st.container():
                stats_cols = st.columns(min(5, len(available_features)))
                for i, feature in enumerate(available_features[:5]):  # Show first 5
                    with stats_cols[i]:
                        mean_val = df[feature].mean()
                        std_val = df[feature].std()
                        display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
                        st.metric(
                            display_name[:15],  # Truncate long names
                            f"{mean_val:.2f}",
                            delta=f"σ={std_val:.2f}",
                            delta_color="off",
                        )

            st.markdown("---")


def render_overall_statistics(df: pd.DataFrame):
    """Render overall statistics section."""
    with st.expander("📈 Overall Statistics", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Songs", len(df))

        with col2:
            st.metric("Number of Clusters", df["cluster"].nunique())

        with col3:
            st.metric("Songs", len(df))

        with col4:
            if "added_at" in df.columns:
                try:
                    min_date = pd.to_datetime(df["added_at"]).min()
                    max_date = pd.to_datetime(df["added_at"]).max()
                    date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
                    st.metric("Date Range", date_range)
                except:
                    st.metric("Date Range", "N/A")
            else:
                st.metric("Date Range", "N/A")
