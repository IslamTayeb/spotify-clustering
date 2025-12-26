"""EDA Explorer Tab Component

Comprehensive exploratory data analysis with:
- Overall statistics
- Genre analysis  
- Audio/mood/vocal extremes
- Genre ladder analysis
- Language analysis
- Temporal analysis (10 visualizations)
- 3D cluster visualization
- Data preview & export
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def render_eda_explorer(df: pd.DataFrame):
    """Render EDA Explorer view with comprehensive statistics."""
    st.header("📊 Exploratory Data Analysis")

    # Vector Analysis Section
    with st.expander("🧮 Interpretable Embedding Features (30 dimensions)", expanded=True):
        st.subheader("Feature Vector Analysis - Clustering Features")
        st.caption("These 30 normalized features are the EXACT values used for clustering")

        # IMPORTANT: This section explains the interpretable vector structure
        st.info(
            "**Vector Structure (30 dimensions):**\n\n"
            "- **Audio (14):** BPM, danceability, instrumentalness, valence, arousal, "
            "engagement, approachability, moods (5), voice gender, genre ladder\n"
            "- **Key (3):** Circular encoding (sin/cos) + major/minor scale (weighted 0.33)\n"
            "- **Lyrics (10):** valence, arousal, moods (4), explicit, narrative, vocabulary, repetition\n"
            "- **Theme (1):** Semantic scale (0=none → 1=party)\n"
            "- **Language (1):** Ordinal encoding (0=none → 1=english)\n"
            "- **Popularity (1):** Spotify popularity normalized to [0,1]\n\n"
            "All values in [0,1]. Lyric features weighted by (1 - instrumentalness)."
        )

        # Look for emb_* columns (the actual 30-dim embedding values)
        embedding_feature_cols = [col for col in df.columns if col.startswith("emb_")]

        if not embedding_feature_cols:
            st.warning(
                "No embedding columns (emb_*) found in dataframe. "
                "This data may have been generated before embedding columns were added. "
                "Re-run the analysis pipeline to include all 30 dimensions."
            )
        else:
            # 1. Overview Metrics
            st.markdown("### 📊 Feature Statistics")
            st.caption(f"Analyzing all {len(embedding_feature_cols)} embedding dimensions used for clustering")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Embedding Dimensions", len(embedding_feature_cols))
                st.caption("(All clustering features)" if len(embedding_feature_cols) == 30 else f"(Expected 30, got {len(embedding_feature_cols)})")

            with col2:
                # Calculate average vector magnitude for available features
                df_features = df[embedding_feature_cols].fillna(0)
                vector_magnitudes = np.sqrt((df_features ** 2).sum(axis=1))
                st.metric("Avg Magnitude (subset)", f"{vector_magnitudes.mean():.2f}")

            with col3:
                # Dynamic range
                feature_ranges = df_features.max() - df_features.min()
                avg_range = feature_ranges.mean()
                st.metric("Avg Feature Range", f"{avg_range:.2f}")

            # 2. Feature Statistics Table
            st.markdown("---")
            st.markdown("### 📈 Feature Statistics")
            st.caption("Statistics for features that are part of the clustering embedding")

            stats_df = pd.DataFrame({
                'Feature': embedding_feature_cols,
                'Mean': [df[col].mean() for col in embedding_feature_cols],
                'Std': [df[col].std() for col in embedding_feature_cols],
                'Min': [df[col].min() for col in embedding_feature_cols],
                'Max': [df[col].max() for col in embedding_feature_cols],
                'Range': [df[col].max() - df[col].min() for col in embedding_feature_cols],
            })

            # Add relative importance (normalized range * std)
            stats_df['Relative Importance'] = (
                (stats_df['Range'] / stats_df['Range'].max()) *
                (stats_df['Std'] / stats_df['Std'].max())
            )

            # Sort by importance
            stats_df = stats_df.sort_values('Relative Importance', ascending=False)

            # Format numeric columns
            for col in ['Mean', 'Std', 'Min', 'Max', 'Range', 'Relative Importance']:
                stats_df[col] = stats_df[col].round(4)

            st.dataframe(stats_df, use_container_width=True, height=400)

            # 3. Feature Distribution Visualizations (simplified)
            st.markdown("---")
            st.markdown("### 📊 Feature Distributions (Top 9)")

            # Select top features by importance
            top_features = stats_df['Feature'].head(9).tolist()

            # Create grid of histograms
            n_cols = 3
            n_rows = 3

            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=top_features,
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )

            for idx, feature in enumerate(top_features):
                row = idx // n_cols + 1
                col = idx % n_cols + 1

                hist_data = df[feature].dropna()

                fig.add_trace(
                    go.Histogram(
                        x=hist_data,
                        name=feature,
                        showlegend=False,
                        marker_color='rgb(29, 185, 84)'
                    ),
                    row=row, col=col
                )

            fig.update_layout(
                height=700,
                title_text="Top 9 Embedding Feature Distributions",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # 4. Download Embedding Features
            st.markdown("---")
            st.markdown("### 💾 Export Embedding Features")

            export_df = df[['track_name', 'artist', 'cluster'] + embedding_feature_cols].copy()
            csv_features = export_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Download Embedding Features (CSV)",
                data=csv_features,
                file_name="embedding_features.csv",
                mime="text/csv",
            )

            st.success("✨ Interpretable embedding analysis complete!")

    # Raw Features Section (Collapsed)
    with st.expander("📊 Raw Dataframe Features (not used in clustering)", expanded=False):
        st.subheader("Additional Numeric Columns")
        st.caption("These raw values are stored for display but NOT used in clustering")

        # Identify NON-embedding numeric columns
        exclude_cols = [
            'track_name', 'artist', 'cluster', 'umap_x', 'umap_y', 'umap_z',
            'added_at', 'release_date', 'spotify_uri', 'track_id', 'album',
            'top_genre', 'language', 'lyric_language', 'has_lyrics',
            'age_at_add_years', 'release_year', 'added_year', 'added_month',
            'time_period', 'quarter', 'month', 'decade', 'age_category',
            'dominant_gender', 'production_style', 'is_vocal', 'album_name',
            'album_type', 'explicit', 'dominant_mood', 'top_3_genres',
            'genre_probs', 'mtg_jamendo_probs'
        ] + embedding_feature_cols  # Exclude embedding features already shown

        raw_feature_cols = [
            col for col in df.columns
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']
            and col not in exclude_cols
        ]

        if not raw_feature_cols:
            st.info("No additional raw features found (all numeric columns are in the embedding)")
        else:
            st.warning(
                f"⚠️ Found {len(raw_feature_cols)} raw columns that are NOT part of the 30-dim embedding. "
                "These are metadata stored for display purposes only."
            )

            # Simple table view
            raw_stats_df = pd.DataFrame({
                'Feature': raw_feature_cols,
                'Mean': [df[col].mean() for col in raw_feature_cols],
                'Std': [df[col].std() for col in raw_feature_cols],
                'Min': [df[col].min() for col in raw_feature_cols],
                'Max': [df[col].max() for col in raw_feature_cols],
            })

            for col in ['Mean', 'Std', 'Min', 'Max']:
                raw_stats_df[col] = raw_stats_df[col].round(4)

            st.dataframe(raw_stats_df, use_container_width=True, height=300)

            st.info(
                "**Note:** Raw features like `bpm`, `valence`, `arousal` are shown here in their unnormalized form. "
                "In the actual clustering, these are normalized to [0,1] within the 30-dim embedding."
            )

    # Overall Statistics
    with st.expander("📈 Overall Statistics", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Songs", len(df))

        with col2:
            st.metric("Number of Clusters", df["cluster"].nunique())

        with col3:
            if "has_lyrics" in df.columns:
                st.metric("Songs", len(df))
            else:
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

    # Genre Analysis
    with st.expander("🎸 Genre Analysis", expanded=False):
        st.subheader("Top 20 Most Common Genres")

        if "top_genre" in df.columns:
            genre_counts = df["top_genre"].value_counts().head(20)

            fig = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation="h",
                labels={"x": "Number of Songs", "y": "Genre"},
                title="Top 20 Genres in Your Library",
            )
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Genre distribution across clusters
            st.subheader("Genre Distribution Across Clusters")
            top_genres = df["top_genre"].value_counts().head(10).index

            genre_cluster_data = []
            for cluster_id in sorted(df["cluster"].unique()):
                cluster_df = df[df["cluster"] == cluster_id]
                for genre in top_genres:
                    count = (cluster_df["top_genre"] == genre).sum()
                    genre_cluster_data.append({
                        "Cluster": f"Cluster {cluster_id}",
                        "Genre": genre,
                        "Count": count,
                    })

            genre_cluster_df = pd.DataFrame(genre_cluster_data)

            fig = px.bar(
                genre_cluster_df,
                x="Cluster",
                y="Count",
                color="Genre",
                title="Top 10 Genres by Cluster",
                barmode="stack",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Genre information not available in this dataset")

    # Audio Extremes
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
                danceable = df.nlargest(10, "danceability")[
                    ["track_name", "artist", "danceability"]
                ]
                st.dataframe(danceable, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Least Danceable Songs")
                not_danceable = df.nsmallest(10, "danceability")[
                    ["track_name", "artist", "danceability"]
                ]
                st.dataframe(not_danceable, use_container_width=True, hide_index=True)

    # Mood Analysis
    with st.expander("😊 Mood Analysis", expanded=False):
        mood_cols = [
            "mood_happy",
            "mood_sad",
            "mood_aggressive",
            "mood_relaxed",
            "mood_party",
        ]

        if all(col in df.columns for col in mood_cols):
            # 5D Radar plot
            st.subheader("Overall Library Mood Profile")

            avg_moods = {
                "Happy": df["mood_happy"].mean() * 100,
                "Sad": df["mood_sad"].mean() * 100,
                "Aggressive": df["mood_aggressive"].mean() * 100,
                "Relaxed": df["mood_relaxed"].mean() * 100,
                "Party": df["mood_party"].mean() * 100,
            }

            fig = go.Figure(
                data=go.Scatterpolar(
                    r=list(avg_moods.values()),
                    theta=list(avg_moods.keys()),
                    fill="toself",
                    name="Library Average",
                )
            )

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title="Average Mood Distribution (%)",
            )

            st.plotly_chart(fig, use_container_width=True)

            # Valence vs Arousal scatter plot
            if "valence" in df.columns and "arousal" in df.columns:
                st.subheader("Emotional Quadrants (Valence vs Arousal)")

                fig = px.scatter(
                    df,
                    x="valence",
                    y="arousal",
                    color="cluster",
                    hover_data=["track_name", "artist"],
                    labels={
                        "valence": "Valence (Pleasant)",
                        "arousal": "Arousal (Energy)",
                    },
                    title="Songs by Emotional Content",
                    color_continuous_scale="Viridis",
                )

                # Add quadrant lines
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)

                # Add quadrant labels
                fig.add_annotation(
                    x=0.75, y=0.75, text="Happy/Energetic", showarrow=False, opacity=0.5
                )
                fig.add_annotation(
                    x=0.25, y=0.75, text="Angry/Tense", showarrow=False, opacity=0.5
                )
                fig.add_annotation(
                    x=0.25, y=0.25, text="Sad/Depressed", showarrow=False, opacity=0.5
                )
                fig.add_annotation(
                    x=0.75, y=0.25, text="Calm/Peaceful", showarrow=False, opacity=0.5
                )

                st.plotly_chart(fig, use_container_width=True)

            # Mood extremes
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Happiest Songs")
                happiest = df.nlargest(10, "mood_happy")[
                    ["track_name", "artist", "mood_happy"]
                ]
                st.dataframe(happiest, use_container_width=True, hide_index=True)

                st.subheader("Most Aggressive Songs")
                aggressive = df.nlargest(10, "mood_aggressive")[
                    ["track_name", "artist", "mood_aggressive"]
                ]
                st.dataframe(aggressive, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Saddest Songs")
                saddest = df.nlargest(10, "mood_sad")[
                    ["track_name", "artist", "mood_sad"]
                ]
                st.dataframe(saddest, use_container_width=True, hide_index=True)

                st.subheader("Most Relaxed Songs")
                relaxed = df.nlargest(10, "mood_relaxed")[
                    ["track_name", "artist", "mood_relaxed"]
                ]
                st.dataframe(relaxed, use_container_width=True, hide_index=True)
        else:
            st.warning("Mood information not available in this dataset")

    # Vocal Analysis
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
                title="Voice Gender Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Acoustic vs Electronic
        if "mood_acoustic" in df.columns and "mood_electronic" in df.columns:
            st.subheader("Acoustic vs Electronic Distribution")

            df_production = df.copy()
            df_production["production_style"] = "Mixed"
            df_production.loc[
                df_production["mood_acoustic"] > 0.6, "production_style"
            ] = "Acoustic"
            df_production.loc[
                df_production["mood_electronic"] > 0.6, "production_style"
            ] = "Electronic"

            production_counts = df_production["production_style"].value_counts()

            fig = px.pie(
                values=production_counts.values,
                names=production_counts.index,
                title="Production Style Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Genre Ladder Analysis
    with st.expander("🎸 Genre Ladder Analysis", expanded=False):
        if "genre_ladder" in df.columns:
            st.subheader("Acoustic ↔ Electronic Distribution (Genre-Based)")
            st.caption(
                "Genre ladder captures stylistic intent (0=acoustic/traditional, 1=electronic/synthetic)"
            )

            fig = px.histogram(
                df,
                x="genre_ladder",
                nbins=50,
                title="Genre Ladder Distribution (0=Acoustic, 1=Electronic)",
                labels={
                    "genre_ladder": "Genre Ladder Score",
                    "count": "Number of Songs",
                },
            )
            fig.add_vline(
                x=0.5, line_dash="dash", line_color="gray", annotation_text="Hybrid"
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Most Acoustic Songs (by Genre)**")
                acoustic = df.nsmallest(10, "genre_ladder")[
                    ["track_name", "artist", "top_genre", "genre_ladder"]
                ]
                st.dataframe(acoustic, use_container_width=True, hide_index=True)

            with col2:
                st.write("**Most Electronic Songs (by Genre)**")
                electronic = df.nlargest(10, "genre_ladder")[
                    ["track_name", "artist", "top_genre", "genre_ladder"]
                ]
                st.dataframe(electronic, use_container_width=True, hide_index=True)

            # Compare with mood_acoustic/mood_electronic
            if "mood_acoustic" in df.columns and "mood_electronic" in df.columns:
                st.markdown("---")
                st.subheader("Genre Ladder vs Audio Production Analysis")

                fig = px.scatter(
                    df,
                    x="genre_ladder",
                    y="mood_electronic",
                    color="cluster",
                    hover_data=["track_name", "artist", "top_genre"],
                    labels={
                        "genre_ladder": "Genre Ladder (0=Acoustic, 1=Electronic)",
                        "mood_electronic": "Audio Electronic Score",
                    },
                    title="Genre Taxonomy vs Audio Production",
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig, use_container_width=True)

                correlation = df["genre_ladder"].corr(df["mood_electronic"])
                st.metric("Correlation", f"{correlation:.3f}")
        else:
            st.warning("Genre ladder information not available")

    # Language Analysis  
    with st.expander("🌍 Language Analysis", expanded=False):
        if "language" in df.columns or "lyric_language" in df.columns:
            lang_col = "language" if "language" in df.columns else "lyric_language"

            st.subheader("Language Distribution")
            st.caption("Language detection from lyric text using langdetect")

            lang_counts = df[lang_col].value_counts()

            fig = px.pie(
                values=lang_counts.values,
                names=lang_counts.index,
                title="Language Distribution Across Library",
                hole=0.3,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Statistics table
            st.subheader("Language Statistics")
            lang_stats = pd.DataFrame({
                "Language": lang_counts.index,
                "Songs": lang_counts.values,
                "Percentage": (lang_counts.values / len(df) * 100).round(1)
            })
            st.dataframe(lang_stats, use_container_width=True, hide_index=True)

            # Sample songs
            st.markdown("---")
            st.subheader("Sample Songs by Language")

            top_langs = [lang for lang in lang_counts.index[:5]
                        if lang not in ['none', 'unknown', 'None', 'Unknown']][:3]

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

                fig = px.box(
                    df[df[lang_col].isin(top_langs + ['none', 'unknown'])],
                    x=lang_col,
                    y="instrumentalness",
                    title="Instrumentalness by Language",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Language information not available")

    # Temporal Analysis (10 visualizations)
    with st.expander("⏰ Temporal Analysis", expanded=False):
        st.subheader("Temporal Analysis")
        st.caption("Explore how your music taste evolved over time")

        temporal_cols = ['added_at', 'release_date']
        has_temporal = all(col in df.columns for col in temporal_cols)

        if not has_temporal:
            st.warning("Temporal information not available")
            st.info("💡 Data loaded from spotify/saved_tracks.json")
        else:
            df_temp = df.copy()
            df_temp['added_at'] = pd.to_datetime(df_temp['added_at'], errors='coerce')
            df_temp['release_date'] = pd.to_datetime(df_temp['release_date'], errors='coerce')

            valid_temporal = df_temp['added_at'].notna()
            if valid_temporal.sum() == 0:
                st.warning("No valid temporal data found")
            else:
                df_temp = df_temp[valid_temporal].copy()

                # Calculate age_at_add_years if needed
                if 'age_at_add_years' not in df_temp.columns:
                    if df_temp['added_at'].dt.tz is not None and df_temp['release_date'].dt.tz is None:
                        df_temp['release_date'] = df_temp['release_date'].dt.tz_localize('UTC')
                    elif df_temp['added_at'].dt.tz is None and df_temp['release_date'].dt.tz is not None:
                        df_temp['added_at'] = df_temp['added_at'].dt.tz_localize('UTC')

                    df_temp['age_at_add_years'] = (df_temp['added_at'] - df_temp['release_date']).dt.days / 365.25

                df_temp['release_year'] = df_temp['release_date'].dt.year
                df_temp['added_year'] = df_temp['added_at'].dt.year

                # 1. Overview Metrics
                st.markdown("---")
                st.subheader("📊 Overview")
                col1, col2, col3 = st.columns(3)

                with col1:
                    min_date = df_temp['added_at'].min()
                    max_date = df_temp['added_at'].max()
                    date_range = (max_date - min_date).days
                    st.metric("Library Timespan", f"{date_range} days")
                    st.caption(f"{min_date.strftime('%Y-%m-%d')} → {max_date.strftime('%Y-%m-%d')}")

                with col2:
                    if 'age_at_add_years' in df_temp.columns:
                        median_age = df_temp['age_at_add_years'].median()
                        st.metric("Median Song Age at Add", f"{median_age:.1f} years")
                    else:
                        st.metric("Median Song Age at Add", "N/A")

                with col3:
                    df_temp['added_month'] = df_temp['added_at'].dt.to_period('M')
                    most_active = df_temp['added_month'].value_counts().index[0]
                    most_active_count = df_temp['added_month'].value_counts().iloc[0]
                    st.metric("Most Active Month", str(most_active))
                    st.caption(f"{most_active_count} songs added")

                # 2. Library Growth Timeline
                st.markdown("---")
                st.subheader("📈 Library Growth Over Time")

                df_sorted = df_temp.sort_values('added_at').reset_index(drop=True)
                df_sorted['cumulative_songs'] = range(1, len(df_sorted) + 1)

                fig = px.line(
                    df_sorted,
                    x='added_at',
                    y='cumulative_songs',
                    title='Cumulative Songs Added to Library',
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                # 3. Addition Patterns (Monthly)
                st.markdown("---")
                st.subheader("📅 Addition Patterns")

                monthly_additions = df_temp.groupby(df_temp['added_at'].dt.to_period('M')).size()

                fig = px.bar(
                    x=monthly_additions.index.astype(str),
                    y=monthly_additions.values,
                    title='Monthly Song Additions',
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                # 4. Song Age Distribution
                st.markdown("---")
                st.subheader("🕰️ Song Age When Added")

                if 'age_at_add_years' in df_temp.columns:
                    valid_ages = df_temp['age_at_add_years'].between(-1, 100)
                    df_age = df_temp[valid_ages].copy()

                    df_age['age_category'] = pd.cut(
                        df_age['age_at_add_years'],
                        bins=[-float('inf'), 1, 5, 10, float('inf')],
                        labels=['New (<1yr)', 'Recent (1-5yr)', 'Classic (5-10yr)', 'Vintage (>10yr)']
                    )

                    fig = px.histogram(df_age, x='age_at_add_years', nbins=50)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    category_counts = df_age['age_category'].value_counts()
                    st.write("**Age Categories:**")
                    for cat in ['New (<1yr)', 'Recent (1-5yr)', 'Classic (5-10yr)', 'Vintage (>10yr)']:
                        if cat in category_counts.index:
                            count = category_counts[cat]
                            pct = count / len(df_age) * 100
                            st.write(f"- {cat}: {count} songs ({pct:.1f}%)")
                else:
                    st.info("Age data not available")

                # 5. Release Year Distribution
                st.markdown("---")
                st.subheader("🎵 Release Year Distribution")

                if df_temp['release_year'].notna().sum() > 0:
                    valid_years = df_temp['release_year'].between(1900, 2030)
                    df_year = df_temp[valid_years].copy()

                    fig = px.histogram(df_year, x='release_year', nbins=50)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    df_year['decade'] = (df_year['release_year'] // 10) * 10
                    decade_counts = df_year['decade'].value_counts().sort_index()

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.write("**By Decade:**")
                        for decade, count in decade_counts.items():
                            pct = count / len(df_year) * 100
                            st.write(f"- {int(decade)}s: {count} songs ({pct:.1f}%)")

                    with col2:
                        fig = px.pie(
                            values=decade_counts.values,
                            names=[f"{int(d)}s" for d in decade_counts.index],
                            title="Decade Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Release year data not available")

                # 6. Cluster Evolution Over Time
                st.markdown("---")
                st.subheader("🎭 Taste Evolution: Clusters Over Time")

                if len(df_temp) >= 4:
                    df_sorted_cluster = df_temp.sort_values('added_at').copy()

                    try:
                        df_sorted_cluster['time_period'] = pd.qcut(
                            df_sorted_cluster['added_at'].astype(int) / 10**9,
                            q=4,
                            labels=['Period 1', 'Period 2', 'Period 3', 'Period 4'],
                            duplicates='drop'
                        )

                        period_cluster = df_sorted_cluster.groupby(['time_period', 'cluster']).size().unstack(fill_value=0)
                        period_cluster_pct = period_cluster.div(period_cluster.sum(axis=1), axis=0) * 100

                        fig = px.bar(
                            period_cluster_pct,
                            barmode='stack',
                            title='Cluster Distribution Across Time Periods',
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)

                        st.write("**Time Period Breakdown:**")
                        for period in ['Period 1', 'Period 2', 'Period 3', 'Period 4']:
                            if period in df_sorted_cluster['time_period'].values:
                                period_df = df_sorted_cluster[df_sorted_cluster['time_period'] == period]
                                min_d = period_df['added_at'].min().strftime('%Y-%m-%d')
                                max_d = period_df['added_at'].max().strftime('%Y-%m-%d')
                                st.write(f"- {period}: {min_d} → {max_d} ({len(period_df)} songs)")
                    except Exception as e:
                        st.warning(f"Could not split into time periods: {e}")
                else:
                    st.info("Need at least 4 songs to show cluster evolution")

                # 7. Temporal Extremes
                st.markdown("---")
                st.subheader("🏆 Temporal Extremes")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**By Release Date:**")
                    if df_temp['release_year'].notna().sum() > 0:
                        valid_years_mask = df_temp['release_year'].between(1900, 2030)
                        df_valid_years = df_temp[valid_years_mask]

                        if len(df_valid_years) > 0:
                            oldest = df_valid_years.loc[df_valid_years['release_year'].idxmin()]
                            st.text(f"Oldest: {oldest['track_name']}")
                            st.caption(f"Released: {int(oldest['release_year'])} | by {oldest['artist']}")

                            newest = df_valid_years.loc[df_valid_years['release_year'].idxmax()]
                            st.text(f"Newest: {newest['track_name']}")
                            st.caption(f"Released: {int(newest['release_year'])} | by {newest['artist']}")
                    else:
                        st.info("Release year data not available")

                with col2:
                    st.write("**By Addition Date:**")
                    first = df_temp.loc[df_temp['added_at'].idxmin()]
                    st.text(f"First Added: {first['track_name']}")
                    st.caption(f"on {first['added_at'].strftime('%Y-%m-%d')} | by {first['artist']}")

                    last = df_temp.loc[df_temp['added_at'].idxmax()]
                    st.text(f"Last Added: {last['track_name']}")
                    st.caption(f"on {last['added_at'].strftime('%Y-%m-%d')} | by {last['artist']}")

                # Age extremes
                st.markdown("---")
                st.write("**By Age at Addition:**")
                if 'age_at_add_years' in df_temp.columns and df_temp['age_at_add_years'].notna().sum() > 0:
                    valid_ages_mask = df_temp['age_at_add_years'].between(0, 100)
                    df_valid_ages = df_temp[valid_ages_mask]

                    if len(df_valid_ages) > 0:
                        vintage = df_valid_ages.loc[df_valid_ages['age_at_add_years'].idxmax()]
                        st.text(f"Most Vintage: {vintage['track_name']} ({vintage['age_at_add_years']:.1f} yrs old)")

                        brand_new = df_valid_ages.loc[df_valid_ages['age_at_add_years'].idxmin()]
                        st.text(f"Newest Release: {brand_new['track_name']} ({brand_new['age_at_add_years']:.1f} yrs old)")
                else:
                    st.info("Age data not available")

                # 8. Cluster Trends Over Time (Rolling Window)
                st.markdown("---")
                st.subheader("🎭 Cluster Trends Over Time")

                if len(df_temp) >= 30 and 'cluster' in df_temp.columns:
                    df_sorted_cluster_trend = df_temp.sort_values('added_at').copy()

                    # Create one-hot encoding for clusters
                    cluster_dummies = pd.get_dummies(df_sorted_cluster_trend['cluster'], prefix='cluster')

                    # Apply rolling window (same as mood trends)
                    rolling_clusters = cluster_dummies.rolling(window=30, min_periods=10).mean() * 100
                    rolling_clusters['added_at'] = df_sorted_cluster_trend['added_at'].values

                    # Melt for plotting
                    cluster_cols = [col for col in rolling_clusters.columns if col.startswith('cluster_')]
                    rolling_melted = rolling_clusters.melt(
                        id_vars=['added_at'],
                        value_vars=cluster_cols,
                        var_name='Cluster',
                        value_name='Percentage'
                    )
                    rolling_melted['Cluster'] = rolling_melted['Cluster'].str.replace('cluster_', 'Cluster ')

                    fig = px.line(
                        rolling_melted,
                        x='added_at',
                        y='Percentage',
                        color='Cluster',
                        title='Rolling Cluster Distribution (30-song window)',
                        labels={'Percentage': 'Proportion (%)', 'added_at': 'Date Added'}
                    )

                    # Add trendlines for each cluster
                    colors = px.colors.qualitative.Plotly
                    for i, cluster in enumerate(rolling_melted['Cluster'].unique()):
                        cluster_data = rolling_melted[rolling_melted['Cluster'] == cluster].dropna()
                        if len(cluster_data) > 1:
                            x_numeric = (cluster_data['added_at'] - cluster_data['added_at'].min()).dt.total_seconds()
                            z = np.polyfit(x_numeric, cluster_data['Percentage'], 1)
                            p = np.poly1d(z)
                            fig.add_trace(go.Scatter(
                                x=cluster_data['added_at'],
                                y=p(x_numeric),
                                mode='lines',
                                line=dict(dash='dash', width=2, color=colors[i % len(colors)]),
                                name=f'{cluster} trend',
                                showlegend=False,
                                opacity=0.7
                            ))

                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    st.caption("Shows how the proportion of each cluster changes as you add songs over time (dashed = trendline)")
                elif len(df_temp) < 30:
                    st.info("Need at least 30 songs for rolling cluster trends")
                else:
                    st.info("Cluster info not available")

                # 9. Mood Evolution (Rolling Window)
                st.markdown("---")
                st.subheader("😊 Mood Trends Over Time")

                mood_cols = ['mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed', 'mood_party']
                available_moods = [col for col in mood_cols if col in df_temp.columns]

                if available_moods and len(df_temp) >= 30:
                    df_sorted_mood = df_temp.sort_values('added_at').copy()

                    rolling_moods = df_sorted_mood[available_moods].rolling(window=30, min_periods=10).mean()
                    rolling_moods['added_at'] = df_sorted_mood['added_at'].values

                    rolling_melted = rolling_moods.melt(id_vars=['added_at'], var_name='Mood', value_name='Score')

                    fig = px.line(rolling_melted, x='added_at', y='Score', color='Mood')

                    # Add trendlines for each mood
                    colors = px.colors.qualitative.Plotly
                    for i, mood in enumerate(rolling_melted['Mood'].unique()):
                        mood_data = rolling_melted[rolling_melted['Mood'] == mood].dropna()
                        if len(mood_data) > 1:
                            x_numeric = (mood_data['added_at'] - mood_data['added_at'].min()).dt.total_seconds()
                            z = np.polyfit(x_numeric, mood_data['Score'], 1)
                            p = np.poly1d(z)
                            fig.add_trace(go.Scatter(
                                x=mood_data['added_at'],
                                y=p(x_numeric),
                                mode='lines',
                                line=dict(dash='dash', width=2, color=colors[i % len(colors)]),
                                name=f'{mood} trend',
                                showlegend=False,
                                opacity=0.7
                            ))

                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Dashed lines show overall trend direction for each mood")
                elif len(df_temp) < 30:
                    st.info("Need at least 30 songs")
                else:
                    st.info("Mood info not available")

                # 10. Genre Family Trends (Grouped + Delta Analysis)
                st.markdown("---")
                st.subheader("🎸 Genre Family Trends Over Time")

                if 'top_genre' in df_temp.columns and len(df_temp) > 0:
                    # Define genre families - group similar genres together
                    genre_families = {
                        'Hip Hop': ['hip hop', 'rap', 'trap', 'drill', 'boom bap', 'conscious hip hop',
                                   'southern hip hop', 'west coast', 'east coast', 'gangsta', 'mumble',
                                   'cloud rap', 'phonk', 'memphis', 'crunk', 'grime', 'uk hip hop'],
                        'Electronic': ['electronic', 'edm', 'house', 'techno', 'trance', 'dubstep',
                                      'drum and bass', 'dnb', 'ambient', 'downtempo', 'idm', 'electro',
                                      'synthwave', 'retrowave', 'future bass', 'garage', 'breakbeat'],
                        'Rock': ['rock', 'alternative', 'indie', 'punk', 'metal', 'grunge', 'hard rock',
                                'classic rock', 'progressive', 'post-rock', 'shoegaze', 'emo', 'hardcore'],
                        'R&B/Soul': ['r&b', 'rnb', 'soul', 'neo soul', 'funk', 'motown', 'quiet storm',
                                    'contemporary r&b', 'new jack swing'],
                        'Pop': ['pop', 'synth pop', 'dance pop', 'electropop', 'indie pop', 'art pop',
                               'dream pop', 'k-pop', 'j-pop', 'latin pop'],
                        'Jazz/Blues': ['jazz', 'blues', 'smooth jazz', 'bebop', 'swing', 'fusion',
                                      'acid jazz', 'nu jazz', 'contemporary jazz'],
                        'Latin': ['latin', 'reggaeton', 'salsa', 'bachata', 'cumbia', 'dembow',
                                 'urbano', 'latin trap', 'spanish', 'brazilian', 'bossa nova'],
                        'World/Folk': ['world', 'folk', 'acoustic', 'country', 'bluegrass', 'celtic',
                                      'african', 'middle eastern', 'indian', 'asian'],
                    }

                    def get_genre_family(genre_str):
                        """Map a genre to its family."""
                        if pd.isna(genre_str):
                            return 'Other'
                        genre_lower = str(genre_str).lower()
                        for family, keywords in genre_families.items():
                            if any(kw in genre_lower for kw in keywords):
                                return family
                        return 'Other'

                    df_temp['genre_family'] = df_temp['top_genre'].apply(get_genre_family)
                    df_temp['quarter'] = df_temp['added_at'].dt.to_period('Q')

                    # Get top genre families by total count
                    top_families = df_temp['genre_family'].value_counts().head(6).index.tolist()
                    if 'Other' in top_families and len(top_families) > 5:
                        top_families.remove('Other')

                    # Build timeline data with cumulative counts and quarterly additions
                    quarters = sorted(df_temp['quarter'].dropna().unique())
                    timeline_data = []
                    cumulative_counts = {family: 0 for family in top_families}

                    for quarter in quarters:
                        quarter_df = df_temp[df_temp['quarter'] == quarter]
                        total_in_quarter = len(quarter_df)

                        for family in top_families:
                            added_this_quarter = (quarter_df['genre_family'] == family).sum()
                            cumulative_counts[family] += added_this_quarter

                            # Proportion of this family in this quarter's additions
                            quarter_pct = (added_this_quarter / total_in_quarter * 100) if total_in_quarter > 0 else 0

                            timeline_data.append({
                                'Quarter': str(quarter),
                                'Genre Family': family,
                                'Added': added_this_quarter,
                                'Cumulative': cumulative_counts[family],
                                'Quarter %': quarter_pct,
                            })

                    if timeline_data:
                        timeline_df = pd.DataFrame(timeline_data)

                        # Calculate delta (change from previous quarter)
                        for family in top_families:
                            family_mask = timeline_df['Genre Family'] == family
                            timeline_df.loc[family_mask, 'Delta'] = timeline_df.loc[family_mask, 'Quarter %'].diff().fillna(0)

                        # Tab selection for different views
                        genre_view = st.radio(
                            "View",
                            ["Quarterly Proportion", "Delta (Rate of Change)", "Cumulative Growth"],
                            horizontal=True,
                            key="genre_view_selector"
                        )

                        if genre_view == "Quarterly Proportion":
                            fig = px.line(
                                timeline_df,
                                x='Quarter',
                                y='Quarter %',
                                color='Genre Family',
                                title='Genre Family Share of Quarterly Additions',
                                labels={'Quarter %': 'Share of Quarter (%)'}
                            )
                            # Add trendlines
                            colors = px.colors.qualitative.Plotly
                            for i, family in enumerate(top_families):
                                family_data = timeline_df[timeline_df['Genre Family'] == family]
                                if len(family_data) > 1:
                                    x_numeric = np.arange(len(family_data))
                                    z = np.polyfit(x_numeric, family_data['Quarter %'], 1)
                                    p = np.poly1d(z)
                                    fig.add_trace(go.Scatter(
                                        x=family_data['Quarter'],
                                        y=p(x_numeric),
                                        mode='lines',
                                        line=dict(dash='dash', width=2, color=colors[i % len(colors)]),
                                        name=f'{family} trend',
                                        showlegend=False,
                                        opacity=0.7
                                    ))
                            st.caption("What proportion of songs added each quarter belong to each genre family (dashed = trendline)")

                        elif genre_view == "Delta (Rate of Change)":
                            fig = px.bar(
                                timeline_df,
                                x='Quarter',
                                y='Delta',
                                color='Genre Family',
                                barmode='group',
                                title='Genre Family Delta (Quarter-over-Quarter Change)',
                                labels={'Delta': 'Change in Share (pp)'}
                            )
                            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                            # Add trendlines for delta
                            colors = px.colors.qualitative.Plotly
                            for i, family in enumerate(top_families):
                                family_data = timeline_df[timeline_df['Genre Family'] == family]
                                if len(family_data) > 1:
                                    x_numeric = np.arange(len(family_data))
                                    z = np.polyfit(x_numeric, family_data['Delta'], 1)
                                    p = np.poly1d(z)
                                    fig.add_trace(go.Scatter(
                                        x=family_data['Quarter'],
                                        y=p(x_numeric),
                                        mode='lines',
                                        line=dict(dash='dot', width=3, color=colors[i % len(colors)]),
                                        name=f'{family} trend',
                                        showlegend=False,
                                        opacity=0.8
                                    ))
                            st.caption("Positive = growing interest, Negative = declining interest (dotted = trendline)")

                        else:  # Cumulative Growth
                            fig = px.area(
                                timeline_df,
                                x='Quarter',
                                y='Cumulative',
                                color='Genre Family',
                                title='Cumulative Genre Family Growth',
                                labels={'Cumulative': 'Total Songs'}
                            )
                            # Add trendlines for cumulative
                            colors = px.colors.qualitative.Plotly
                            for i, family in enumerate(top_families):
                                family_data = timeline_df[timeline_df['Genre Family'] == family]
                                if len(family_data) > 1:
                                    x_numeric = np.arange(len(family_data))
                                    z = np.polyfit(x_numeric, family_data['Cumulative'], 1)
                                    p = np.poly1d(z)
                                    fig.add_trace(go.Scatter(
                                        x=family_data['Quarter'],
                                        y=p(x_numeric),
                                        mode='lines',
                                        line=dict(dash='dash', width=2, color='white'),
                                        name=f'{family} trend',
                                        showlegend=False,
                                        opacity=0.8
                                    ))
                            st.caption("How your collection of each genre family has grown over time (dashed = linear growth rate)")

                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)

                        # Summary statistics
                        st.markdown("**Genre Family Summary:**")
                        summary_cols = st.columns(min(len(top_families), 4))
                        total_songs = len(df_temp)

                        for i, family in enumerate(top_families[:4]):
                            family_total = cumulative_counts[family]
                            family_pct = family_total / total_songs * 100 if total_songs > 0 else 0
                            family_deltas = timeline_df[timeline_df['Genre Family'] == family]['Delta']
                            avg_delta = family_deltas.mean()
                            trend = "📈" if avg_delta > 0.5 else ("📉" if avg_delta < -0.5 else "➡️")

                            with summary_cols[i]:
                                st.metric(
                                    family,
                                    f"{family_total} songs",
                                    f"{avg_delta:+.1f}pp avg {trend}"
                                )
                                st.caption(f"{family_pct:.1f}% of library")
                    else:
                        st.info("Not enough data for genre trends")
                else:
                    st.info("Genre info not available")

                # 11. Cluster Timeline Heatmap
                st.markdown("---")
                st.subheader("🔥 Cluster Distribution Heatmap")

                if len(df_temp) > 0 and 'cluster' in df_temp.columns:
                    df_temp['month'] = df_temp['added_at'].dt.to_period('M')

                    try:
                        cluster_month_matrix = df_temp.groupby(['month', 'cluster']).size().unstack(fill_value=0)

                        if len(cluster_month_matrix) > 1 and len(cluster_month_matrix.columns) > 1:
                            fig = px.imshow(
                                cluster_month_matrix.T,
                                labels=dict(x="Month", y="Cluster", color="Songs Added"),
                                title="Cluster Activity Heatmap",
                                aspect="auto",
                                color_continuous_scale="Viridis"
                            )
                            fig.update_xaxes(side="bottom")
                            fig.update_layout(height=400 + len(cluster_month_matrix.columns) * 20)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Not enough time periods")
                    except Exception as e:
                        st.warning(f"Could not generate heatmap: {e}")
                else:
                    st.info("Cluster info not available")

                st.markdown("---")
                st.success("✨ Temporal analysis complete!")

    # 3D Cluster Visualization
    with st.expander("🗺️ Interactive 3D Cluster Map", expanded=False):
        st.subheader("3D UMAP Visualization of Clusters")

        if "umap_x" in df.columns and "umap_y" in df.columns and "umap_z" in df.columns:
            # Check for subcluster data
            subcluster_data = st.session_state.get("subcluster_data")
            show_subclusters = False

            if subcluster_data is not None:
                show_subclusters = st.checkbox(
                    f"🔍 Show Sub-Clusters of Cluster {subcluster_data['parent_cluster']}",
                    value=True,
                    help="Color parent cluster by subcluster, dim other clusters"
                )

            fig = go.Figure()

            unique_clusters = sorted(df["cluster"].unique())
            colors = px.colors.qualitative.Plotly

            # Get embedding columns for hover display
            emb_cols = [col for col in df.columns if col.startswith("emb_")]

            if show_subclusters and subcluster_data is not None:
                # Subcluster view: highlight parent cluster subclusters, dim others
                parent_cluster = subcluster_data['parent_cluster']
                subcluster_df = subcluster_data['subcluster_df']

                for i, cluster_id in enumerate(unique_clusters):
                    cluster_df = df[df["cluster"] == cluster_id]

                    if cluster_id == parent_cluster:
                        # Plot each subcluster with distinct colors
                        subcluster_ids = sorted(subcluster_df['subcluster'].unique())
                        for j, sc_id in enumerate(subcluster_ids):
                            sc_mask = subcluster_df['subcluster'] == sc_id
                            sc_tracks = subcluster_df[sc_mask]

                            # Find matching rows in main df by track_id
                            track_ids = sc_tracks['track_id'].values
                            main_mask = cluster_df['track_id'].isin(track_ids)
                            sc_cluster_df = cluster_df[main_mask]

                            if len(sc_cluster_df) == 0:
                                continue

                            # Build hover texts with subcluster info
                            hover_texts = []
                            for _, row in sc_cluster_df.iterrows():
                                text = (
                                    f"<b>{row['track_name']}</b><br>"
                                    f"Artist: {row['artist']}<br>"
                                    f"Cluster: {row['cluster']}<br>"
                                    f"<b>Sub-cluster: {sc_id}</b><br>"
                                )
                                if "top_genre" in row:
                                    text += f"Genre: {row['top_genre']}<br>"
                                if "bpm" in row:
                                    text += f"BPM: {row['bpm']:.0f}<br>"

                                if emb_cols:
                                    text += "<br><b>Embedding Vector:</b><br>"
                                    for emb_col in emb_cols:
                                        if emb_col in row and pd.notna(row[emb_col]):
                                            display_name = emb_col.replace("emb_", "")
                                            text += f"{display_name}: {row[emb_col]:.3f}<br>"

                                hover_texts.append(text)

                            fig.add_trace(
                                go.Scatter3d(
                                    x=sc_cluster_df["umap_x"],
                                    y=sc_cluster_df["umap_y"],
                                    z=sc_cluster_df["umap_z"],
                                    mode="markers",
                                    name=f"Sub-cluster {sc_id} ({len(sc_cluster_df)})",
                                    marker=dict(size=5, color=colors[j % len(colors)], opacity=0.9),
                                    text=hover_texts,
                                    hovertemplate="%{text}<extra></extra>",
                                )
                            )
                    else:
                        # Gray out non-parent clusters
                        hover_texts = []
                        for _, row in cluster_df.iterrows():
                            text = (
                                f"<b>{row['track_name']}</b><br>"
                                f"Artist: {row['artist']}<br>"
                                f"Cluster: {row['cluster']}<br>"
                            )
                            if "top_genre" in row:
                                text += f"Genre: {row['top_genre']}<br>"
                            hover_texts.append(text)

                        fig.add_trace(
                            go.Scatter3d(
                                x=cluster_df["umap_x"],
                                y=cluster_df["umap_y"],
                                z=cluster_df["umap_z"],
                                mode="markers",
                                name=f"Cluster {cluster_id} ({len(cluster_df)})",
                                marker=dict(size=3, color='rgba(128,128,128,0.3)', opacity=0.3),
                                text=hover_texts,
                                hovertemplate="%{text}<extra></extra>",
                            )
                        )

                plot_title = f"3D Sub-Clusters of Cluster {parent_cluster} (UMAP)"
            else:
                # Standard view: all clusters with full colors
                for i, cluster_id in enumerate(unique_clusters):
                    cluster_df = df[df["cluster"] == cluster_id]

                    hover_texts = []
                    for _, row in cluster_df.iterrows():
                        text = (
                            f"<b>{row['track_name']}</b><br>"
                            f"Artist: {row['artist']}<br>"
                            f"Cluster: {row['cluster']}<br>"
                        )
                        if "top_genre" in row:
                            text += f"Genre: {row['top_genre']}<br>"
                        if "bpm" in row:
                            text += f"BPM: {row['bpm']:.0f}<br>"

                        # Add full embedding vector
                        if emb_cols:
                            text += "<br><b>Embedding Vector:</b><br>"
                            for emb_col in emb_cols:
                                if emb_col in row and pd.notna(row[emb_col]):
                                    # Clean up column name for display (remove emb_ prefix)
                                    display_name = emb_col.replace("emb_", "")
                                    text += f"{display_name}: {row[emb_col]:.3f}<br>"

                        hover_texts.append(text)

                    fig.add_trace(
                        go.Scatter3d(
                            x=cluster_df["umap_x"],
                            y=cluster_df["umap_y"],
                            z=cluster_df["umap_z"],
                            mode="markers",
                            name=f"Cluster {cluster_id} ({len(cluster_df)})",
                            marker=dict(size=4, color=colors[i % len(colors)], opacity=0.8),
                            text=hover_texts,
                            hovertemplate="%{text}<extra></extra>",
                        )
                    )

                plot_title = "3D Cluster Visualization (UMAP)"

            fig.update_layout(
                height=700,
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                ),
                title=plot_title,
                showlegend=True,
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("UMAP coordinates not found")

    # Data Preview & Export
    with st.expander("🔍 Data Preview & Export", expanded=False):
        st.subheader("Full Dataset Preview")

        display_df = df.copy()
        for col in display_df.columns:
            if display_df[col].dtype == "object":
                display_df[col] = display_df[col].astype(str)

        st.dataframe(display_df, use_container_width=True, height=400)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="cluster_data.csv",
            mime="text/csv",
        )
