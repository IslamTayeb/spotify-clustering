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
