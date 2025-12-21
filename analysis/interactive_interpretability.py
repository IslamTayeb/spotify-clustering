"""
Interactive Cluster Interpretability Dashboard

Streamlit app for exploring cluster interpretability with:
- EDA Explorer: Overall statistics and extremes
- Feature Importance: What makes clusters unique
- Cluster Comparison: Statistical comparison of clusters
- Lyric Themes: Keyword extraction, sentiment, complexity
- Overview: Global similarity and summary
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path
from typing import Dict, List
import umap

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our interpretability modules
from analysis.interpretability.feature_importance import (
    compute_feature_importance,
    get_top_features,
    get_feature_interpretation,
)
from analysis.interpretability.cluster_comparison import (
    compare_two_clusters,
    compute_cluster_similarity_matrix,
    find_most_different_pairs,
)
from analysis.interpretability.lyric_themes import (
    load_lyrics_for_cluster,
    extract_tfidf_keywords,
    analyze_sentiment,
    compute_lyric_complexity,
    extract_common_phrases,
    compare_cluster_keywords,
)

# Page configuration
st.set_page_config(
    page_title="Cluster Interpretability Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1DB954;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1DB954;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_analysis_data(file_path: str = 'analysis/outputs/analysis_data.pkl'):
    """Load analysis data with caching."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


@st.cache_data
def get_dataframe(data: Dict, mode: str = 'combined'):
    """Extract dataframe for selected mode."""
    return data[mode]['dataframe']


def main():
    # Header
    st.markdown('<div class="main-header">🎵 Music Cluster Interpretability Dashboard</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Load data
        data_file = st.text_input(
            "Data file path",
            value="analysis/outputs/analysis_data.pkl",
            help="Path to analysis_data.pkl file"
        )

        if not Path(data_file).exists():
            st.error(f"File not found: {data_file}")
            st.stop()

        try:
            all_data = load_analysis_data(data_file)
            st.success("✓ Data loaded successfully")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

        # Mode selection
        mode = st.selectbox(
            "Clustering mode",
            options=['combined', 'audio', 'lyrics'],
            index=0,
            help="Select which clustering mode to analyze"
        )

        df = get_dataframe(all_data, mode)

        # Display basic stats
        st.markdown("---")
        st.metric("Total Songs", len(df))
        st.metric("Number of Clusters", df['cluster'].nunique())

        if 'n_outliers' in all_data[mode]:
            st.metric("Outliers", all_data[mode]['n_outliers'])

        st.markdown("---")
        st.caption(f"📊 Analyzing {mode} mode")

    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 EDA Explorer",
        "🎯 Feature Importance",
        "⚖️ Cluster Comparison",
        "📝 Lyric Themes",
        "🔍 Overview"
    ])

    with tab1:
        render_eda_explorer(df)

    with tab2:
        render_feature_importance(df)

    with tab3:
        render_cluster_comparison(df)

    with tab4:
        render_lyric_themes(df)

    with tab5:
        render_overview(df)


def render_eda_explorer(df: pd.DataFrame):
    """Render EDA Explorer view with comprehensive statistics."""
    st.header("📊 Exploratory Data Analysis")

    # Overall Statistics
    with st.expander("📈 Overall Statistics", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Songs", len(df))

        with col2:
            st.metric("Number of Clusters", df['cluster'].nunique())

        with col3:
            if 'has_lyrics' in df.columns:
                lyric_pct = (df['has_lyrics'].sum() / len(df) * 100) if len(df) > 0 else 0
                st.metric("Songs with Lyrics", f"{lyric_pct:.1f}%")
            else:
                st.metric("Songs with Lyrics", "N/A")

        with col4:
            if 'added_at' in df.columns:
                try:
                    min_date = pd.to_datetime(df['added_at']).min()
                    max_date = pd.to_datetime(df['added_at']).max()
                    date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
                    st.metric("Date Range", date_range)
                except:
                    st.metric("Date Range", "N/A")
            else:
                st.metric("Date Range", "N/A")

    # Genre Analysis
    with st.expander("🎸 Genre Analysis", expanded=False):
        st.subheader("Top 20 Most Common Genres")

        if 'top_genre' in df.columns:
            genre_counts = df['top_genre'].value_counts().head(20)

            fig = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation='h',
                labels={'x': 'Number of Songs', 'y': 'Genre'},
                title="Top 20 Genres in Your Library"
            )
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Genre distribution across clusters
            st.subheader("Genre Distribution Across Clusters")
            # Get top 10 genres
            top_genres = df['top_genre'].value_counts().head(10).index

            genre_cluster_data = []
            for cluster_id in sorted(df['cluster'].unique()):
                cluster_df = df[df['cluster'] == cluster_id]
                for genre in top_genres:
                    count = (cluster_df['top_genre'] == genre).sum()
                    genre_cluster_data.append({
                        'Cluster': f"Cluster {cluster_id}",
                        'Genre': genre,
                        'Count': count
                    })

            genre_cluster_df = pd.DataFrame(genre_cluster_data)

            fig = px.bar(
                genre_cluster_df,
                x='Cluster',
                y='Count',
                color='Genre',
                title="Top 10 Genres by Cluster",
                barmode='stack'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Rarest genres
            st.subheader("Rarest Genres (appear only once)")
            rare_genres = df['top_genre'].value_counts()
            rare_genres = rare_genres[rare_genres == 1]
            st.write(f"Found {len(rare_genres)} genres that appear in only 1 song:")
            st.write(", ".join(rare_genres.index.tolist()[:30]) + ("..." if len(rare_genres) > 30 else ""))

        else:
            st.warning("Genre information not available in this dataset")

    # Audio Extremes
    with st.expander("🔊 Audio Extremes", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Most/Least Approachable
            if 'approachability_score' in df.columns:
                st.subheader("Most Approachable Songs")
                top_approachable = df.nlargest(10, 'approachability_score')[['track_name', 'artist', 'approachability_score']]
                st.dataframe(top_approachable, use_container_width=True, hide_index=True)

                st.subheader("Least Approachable Songs (Most Niche)")
                bottom_approachable = df.nsmallest(10, 'approachability_score')[['track_name', 'artist', 'approachability_score']]
                st.dataframe(bottom_approachable, use_container_width=True, hide_index=True)

        with col2:
            # Most/Least Engaging
            if 'engagement_score' in df.columns:
                st.subheader("Most Engaging Songs")
                top_engaging = df.nlargest(10, 'engagement_score')[['track_name', 'artist', 'engagement_score']]
                st.dataframe(top_engaging, use_container_width=True, hide_index=True)

                st.subheader("Least Engaging Songs")
                bottom_engaging = df.nsmallest(10, 'engagement_score')[['track_name', 'artist', 'engagement_score']]
                st.dataframe(bottom_engaging, use_container_width=True, hide_index=True)

        # BPM Extremes
        if 'bpm' in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Fastest Songs (Highest BPM)")
                fastest = df.nlargest(10, 'bpm')[['track_name', 'artist', 'bpm']]
                st.dataframe(fastest, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Slowest Songs (Lowest BPM)")
                slowest = df.nsmallest(10, 'bpm')[['track_name', 'artist', 'bpm']]
                st.dataframe(slowest, use_container_width=True, hide_index=True)

        # Danceability Extremes
        if 'danceability' in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Most Danceable Songs")
                danceable = df.nlargest(10, 'danceability')[['track_name', 'artist', 'danceability']]
                st.dataframe(danceable, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Least Danceable Songs")
                not_danceable = df.nsmallest(10, 'danceability')[['track_name', 'artist', 'danceability']]
                st.dataframe(not_danceable, use_container_width=True, hide_index=True)

    # Mood Analysis
    with st.expander("😊 Mood Analysis", expanded=False):
        mood_cols = ['mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed', 'mood_party']

        if all(col in df.columns for col in mood_cols):
            # 5D Radar plot for library mood profile
            st.subheader("Overall Library Mood Profile")

            avg_moods = {
                'Happy': df['mood_happy'].mean() * 100,
                'Sad': df['mood_sad'].mean() * 100,
                'Aggressive': df['mood_aggressive'].mean() * 100,
                'Relaxed': df['mood_relaxed'].mean() * 100,
                'Party': df['mood_party'].mean() * 100,
            }

            fig = go.Figure(data=go.Scatterpolar(
                r=list(avg_moods.values()),
                theta=list(avg_moods.keys()),
                fill='toself',
                name='Library Average'
            ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title="Average Mood Distribution (%)"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Valence vs Arousal scatter plot (emotional quadrants)
            if 'valence' in df.columns and 'arousal' in df.columns:
                st.subheader("Emotional Quadrants (Valence vs Arousal)")

                fig = px.scatter(
                    df,
                    x='valence',
                    y='arousal',
                    color='cluster',
                    hover_data=['track_name', 'artist'],
                    labels={'valence': 'Valence (Pleasant)', 'arousal': 'Arousal (Energy)'},
                    title="Songs by Emotional Content",
                    color_continuous_scale='Viridis'
                )

                # Add quadrant lines
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)

                # Add quadrant labels
                fig.add_annotation(x=0.75, y=0.75, text="Happy/Energetic", showarrow=False, opacity=0.5)
                fig.add_annotation(x=0.25, y=0.75, text="Angry/Tense", showarrow=False, opacity=0.5)
                fig.add_annotation(x=0.25, y=0.25, text="Sad/Depressed", showarrow=False, opacity=0.5)
                fig.add_annotation(x=0.75, y=0.25, text="Calm/Peaceful", showarrow=False, opacity=0.5)

                st.plotly_chart(fig, use_container_width=True)

            # Mood extremes
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Happiest Songs")
                happiest = df.nlargest(10, 'mood_happy')[['track_name', 'artist', 'mood_happy']]
                st.dataframe(happiest, use_container_width=True, hide_index=True)

                st.subheader("Most Aggressive Songs")
                aggressive = df.nlargest(10, 'mood_aggressive')[['track_name', 'artist', 'mood_aggressive']]
                st.dataframe(aggressive, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Saddest Songs")
                saddest = df.nlargest(10, 'mood_sad')[['track_name', 'artist', 'mood_sad']]
                st.dataframe(saddest, use_container_width=True, hide_index=True)

                st.subheader("Most Relaxed Songs")
                relaxed = df.nlargest(10, 'mood_relaxed')[['track_name', 'artist', 'mood_relaxed']]
                st.dataframe(relaxed, use_container_width=True, hide_index=True)

        else:
            st.warning("Mood information not available in this dataset")

    # Vocal Analysis
    with st.expander("🎤 Vocal Analysis", expanded=False):
        if 'instrumentalness' in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Most Instrumental Songs")
                instrumental = df.nlargest(10, 'instrumentalness')[['track_name', 'artist', 'instrumentalness']]
                st.dataframe(instrumental, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Most Vocal Songs")
                vocal = df.nsmallest(10, 'instrumentalness')[['track_name', 'artist', 'instrumentalness']]
                st.dataframe(vocal, use_container_width=True, hide_index=True)

        # Voice gender distribution
        if 'voice_gender_male' in df.columns and 'voice_gender_female' in df.columns:
            st.subheader("Voice Gender Distribution")

            # Classify songs as predominantly male, female, or mixed
            df_vocal = df.copy()
            df_vocal['dominant_gender'] = 'Mixed'
            df_vocal.loc[df_vocal['voice_gender_male'] > 0.6, 'dominant_gender'] = 'Male'
            df_vocal.loc[df_vocal['voice_gender_female'] > 0.6, 'dominant_gender'] = 'Female'

            gender_counts = df_vocal['dominant_gender'].value_counts()

            fig = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title="Voice Gender Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Acoustic vs Electronic
        if 'mood_acoustic' in df.columns and 'mood_electronic' in df.columns:
            st.subheader("Acoustic vs Electronic Distribution")

            # Classify songs
            df_production = df.copy()
            df_production['production_style'] = 'Mixed'
            df_production.loc[df_production['mood_acoustic'] > 0.6, 'production_style'] = 'Acoustic'
            df_production.loc[df_production['mood_electronic'] > 0.6, 'production_style'] = 'Electronic'

            production_counts = df_production['production_style'].value_counts()

            fig = px.pie(
                values=production_counts.values,
                names=production_counts.index,
                title="Production Style Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    # 3D Cluster Visualization
    with st.expander("🗺️ Interactive 3D Cluster Map", expanded=False):
        st.subheader("3D UMAP Visualization of Clusters")
        st.write("Explore your music clusters in 3D space. Points are colored by cluster assignment.")

        # UMAP parameters
        col1, col2 = st.columns(2)

        with col1:
            n_neighbors_viz = st.slider(
                "n_neighbors (visualization)",
                5, 100, 20,
                help="Controls balance between local and global structure",
                key="umap_neighbors_eda"
            )

        with col2:
            min_dist_viz = st.slider(
                "min_dist (visualization)",
                0.0, 1.0, 0.2, step=0.01,
                help="How tightly points are packed",
                key="umap_mindist_eda"
            )

        # Check if we have UMAP coordinates already
        if 'umap_x' in df.columns and 'umap_y' in df.columns and 'umap_z' in df.columns:
            st.info("Using pre-computed UMAP coordinates from clustering pipeline")
            umap_coords = df[['umap_x', 'umap_y', 'umap_z']].values
        else:
            st.warning("UMAP coordinates not found in data. Cannot display cluster map.")
            st.info("Run the analysis pipeline with UMAP enabled to see cluster visualization.")
            return

        # Create 3D scatter plot
        fig = go.Figure()

        unique_clusters = sorted(df['cluster'].unique())
        colors = px.colors.qualitative.Plotly

        for i, cluster_id in enumerate(unique_clusters):
            cluster_df = df[df['cluster'] == cluster_id]

            # Build hover text
            hover_texts = []
            for _, row in cluster_df.iterrows():
                text = (
                    f"<b>{row['track_name']}</b><br>"
                    f"Artist: {row['artist']}<br>"
                    f"Cluster: {row['cluster']}<br>"
                )

                if 'top_genre' in row:
                    text += f"Genre: {row['top_genre']}<br>"
                if 'bpm' in row:
                    text += f"BPM: {row['bpm']:.0f}<br>"
                if 'danceability' in row:
                    text += f"Danceability: {row['danceability']:.2f}<br>"
                if 'valence' in row and 'arousal' in row:
                    text += f"Valence: {row['valence']:.2f} | Arousal: {row['arousal']:.2f}<br>"
                if 'mood_happy' in row:
                    text += f"Moods: Happy {row['mood_happy']:.2f}, Sad {row.get('mood_sad', 0):.2f}<br>"

                hover_texts.append(text)

            fig.add_trace(go.Scatter3d(
                x=cluster_df['umap_x'],
                y=cluster_df['umap_y'],
                z=cluster_df['umap_z'],
                mode='markers',
                name=f'Cluster {cluster_id} ({len(cluster_df)})',
                marker=dict(
                    size=4,
                    color=colors[i % len(colors)],
                    opacity=0.8
                ),
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>'
            ))

        fig.update_layout(
            height=700,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ),
            title="3D Cluster Visualization (UMAP)",
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # 2D projection option
        st.subheader("2D Projection")

        projection = st.selectbox(
            "Select 2D projection",
            ["X vs Y", "X vs Z", "Y vs Z"],
            key="projection_select"
        )

        fig_2d = go.Figure()

        for i, cluster_id in enumerate(unique_clusters):
            cluster_df = df[df['cluster'] == cluster_id]

            if projection == "X vs Y":
                x_data, y_data = cluster_df['umap_x'], cluster_df['umap_y']
                x_label, y_label = "UMAP X", "UMAP Y"
            elif projection == "X vs Z":
                x_data, y_data = cluster_df['umap_x'], cluster_df['umap_z']
                x_label, y_label = "UMAP X", "UMAP Z"
            else:  # Y vs Z
                x_data, y_data = cluster_df['umap_y'], cluster_df['umap_z']
                x_label, y_label = "UMAP Y", "UMAP Z"

            # Build simple hover text for 2D
            hover_texts_2d = [f"{row['track_name']} - {row['artist']}<br>Cluster: {row['cluster']}"
                             for _, row in cluster_df.iterrows()]

            fig_2d.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(
                    size=6,
                    color=colors[i % len(colors)],
                    opacity=0.7
                ),
                text=hover_texts_2d,
                hovertemplate='%{text}<extra></extra>'
            ))

        fig_2d.update_layout(
            height=600,
            xaxis_title=x_label,
            yaxis_title=y_label,
            title=f"2D Cluster Visualization - {projection}",
            showlegend=True
        )

        st.plotly_chart(fig_2d, use_container_width=True)

    # Data Preview
    with st.expander("🔍 Data Preview & Export", expanded=False):
        st.subheader("Full Dataset Preview")

        # Create a copy for display and convert problematic columns
        display_df = df.copy()

        # Convert object columns that might have mixed types to strings
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                display_df[col] = display_df[col].astype(str)

        st.dataframe(display_df, use_container_width=True, height=400)

        # Export button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="cluster_data.csv",
            mime="text/csv",
        )


@st.cache_data
def compute_all_cluster_importance(df: pd.DataFrame):
    """Compute feature importance for all clusters (cached)."""
    importance_data = {}
    cluster_ids = sorted(df['cluster'].unique())

    for cluster_id in cluster_ids:
        result = get_top_features(df, cluster_id, n=20)
        importance_data[cluster_id] = result

    return importance_data


def render_feature_importance(df: pd.DataFrame):
    """Render Feature Importance view."""
    st.header("🎯 Feature Importance Analysis")

    st.write("Identify which features make each cluster distinctive using Cohen's d effect sizes.")

    # Cluster selection
    cluster_ids = sorted(df['cluster'].unique())
    selected_cluster = st.selectbox(
        "Select Cluster to Analyze",
        options=cluster_ids,
        format_func=lambda x: f"Cluster {x}"
    )

    with st.spinner("Computing feature importance..."):
        # Get feature importance for selected cluster
        cluster_info = get_top_features(df, selected_cluster, n=20)

        # Display cluster info metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Cluster Size", cluster_info['cluster_size'])

        with col2:
            st.metric("Percentage of Library", f"{cluster_info['cluster_percentage']:.1f}%")

        with col3:
            # Get top feature
            if len(cluster_info['top_features']) > 0:
                top_feature = cluster_info['top_features'].iloc[0]
                st.metric(
                    f"Top Feature: {top_feature['feature']}",
                    f"Effect size: {top_feature['effect_size']:.2f}"
                )

        st.markdown("---")

        # Top 3 distinctive features with interpretations
        st.subheader(f"🌟 Top 3 Most Distinctive Features for Cluster {selected_cluster}")

        if len(cluster_info['top_features']) >= 3:
            for i, row in cluster_info['top_features'].head(3).iterrows():
                feature = row['feature']
                effect_size = row['effect_size']
                cluster_mean = row['cluster_mean']
                global_mean = row['global_mean']

                interpretation = get_feature_interpretation(effect_size)

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**{i}. {feature}**")
                    st.write(f"Cluster mean: {cluster_mean:.3f} | Global mean: {global_mean:.3f}")
                    st.write(f"Effect size: **{effect_size:.2f}** ({interpretation})")

                with col2:
                    # Simple visual indicator
                    if abs(effect_size) >= 0.8:
                        st.success("🔥 Large effect")
                    elif abs(effect_size) >= 0.5:
                        st.info("⚡ Medium effect")
                    else:
                        st.warning("✨ Small effect")

        # Full feature importance table
        st.markdown("---")
        st.subheader("📊 Full Feature Importance Ranking")

        # Prepare dataframe for display
        display_df = cluster_info['all_features'].copy()
        display_df['effect_size'] = display_df['effect_size'].round(3)
        display_df['cluster_mean'] = display_df['cluster_mean'].round(3)
        display_df['global_mean'] = display_df['global_mean'].round(3)

        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            column_config={
                "feature": "Feature",
                "effect_size": st.column_config.NumberColumn(
                    "Effect Size (Cohen's d)",
                    help="How many standard deviations this cluster differs from average",
                    format="%.3f"
                ),
                "cluster_mean": "Cluster Mean",
                "global_mean": "Global Mean",
                "importance_rank": "Rank"
            },
            hide_index=True
        )

        # Download button
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download Cluster {selected_cluster} Importance Data",
            data=csv,
            file_name=f"cluster_{selected_cluster}_importance.csv",
            mime="text/csv",
        )

    # Feature importance heatmap for all clusters
    st.markdown("---")
    st.subheader("🔥 Feature Importance Heatmap (All Clusters)")

    with st.spinner("Computing importance for all clusters..."):
        all_importance = compute_all_cluster_importance(df)

        # Build heatmap data
        top_n_features = 15  # Top features to show
        feature_names = set()

        # Collect top features from all clusters
        for cluster_id, data in all_importance.items():
            top_features = data['top_features'].head(top_n_features)
            feature_names.update(top_features['feature'].tolist())

        feature_names = sorted(list(feature_names))[:20]  # Limit to 20 features

        # Build matrix
        heatmap_data = []
        for feature in feature_names:
            row = [feature]
            for cluster_id in cluster_ids:
                importance_df = all_importance[cluster_id]['all_features']
                feature_row = importance_df[importance_df['feature'] == feature]

                if len(feature_row) > 0:
                    effect_size = feature_row.iloc[0]['effect_size']
                    row.append(effect_size)
                else:
                    row.append(0.0)

            heatmap_data.append(row)

        # Create heatmap
        heatmap_df = pd.DataFrame(
            heatmap_data,
            columns=['Feature'] + [f"Cluster {cid}" for cid in cluster_ids]
        )

        fig = px.imshow(
            heatmap_df.set_index('Feature').T,
            labels=dict(x="Feature", y="Cluster", color="Effect Size"),
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            aspect="auto",
            title="Feature Effect Sizes Across All Clusters"
        )

        fig.update_xaxes(side="bottom")
        fig.update_layout(height=400 + len(cluster_ids) * 50)

        st.plotly_chart(fig, use_container_width=True)

        st.caption("💡 Red = higher than average, Blue = lower than average, White = near average")

    # Distribution violin plots for top features
    st.markdown("---")
    st.subheader("📊 Feature Distribution Comparison")

    # Let user select a feature to visualize
    all_features = cluster_info['all_features']['feature'].tolist()
    selected_feature = st.selectbox(
        "Select feature to visualize distribution",
        options=all_features,
        index=0
    )

    if selected_feature in df.columns:
        # Create violin plot
        fig = go.Figure()

        for cluster_id in cluster_ids:
            cluster_values = df[df['cluster'] == cluster_id][selected_feature].dropna()

            fig.add_trace(go.Violin(
                y=cluster_values,
                name=f"Cluster {cluster_id}",
                box_visible=True,
                meanline_visible=True
            ))

        fig.update_layout(
            title=f"Distribution of '{selected_feature}' Across Clusters",
            yaxis_title=selected_feature,
            xaxis_title="Cluster",
            showlegend=True,
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add statistical summary
        st.subheader("Statistical Summary")

        summary_data = []
        for cluster_id in cluster_ids:
            cluster_values = df[df['cluster'] == cluster_id][selected_feature].dropna()

            summary_data.append({
                'Cluster': f"Cluster {cluster_id}",
                'Mean': cluster_values.mean(),
                'Median': cluster_values.median(),
                'Std Dev': cluster_values.std(),
                'Min': cluster_values.min(),
                'Max': cluster_values.max(),
                'Count': len(cluster_values)
            })

        summary_df = pd.DataFrame(summary_data)

        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    else:
        st.warning(f"Feature '{selected_feature}' not found in dataframe")


def render_cluster_comparison(df: pd.DataFrame):
    """Render Cluster Comparison view."""
    st.header("⚖️ Statistical Cluster Comparison")

    st.write("Compare multiple clusters using statistical tests and visualizations.")

    # Cluster selection - allow multiple
    cluster_ids = sorted(df['cluster'].unique())

    # Multi-select for clusters
    selected_clusters = st.multiselect(
        "Select Clusters to Compare (select 2 or more)",
        options=cluster_ids,
        default=cluster_ids[:min(2, len(cluster_ids))],
        format_func=lambda x: f"Cluster {x}",
        help="Select 2 or more clusters to compare. The radar plot will show all selected clusters."
    )

    if len(selected_clusters) < 2:
        st.warning("⚠️ Please select at least 2 clusters to compare")
        return

    # Show number of clusters being compared
    st.info(f"📊 Comparing {len(selected_clusters)} clusters: {', '.join([f'Cluster {c}' for c in selected_clusters])}")

    # Basic cluster information
    st.markdown("---")
    st.subheader("📊 Cluster Overview")

    # Create overview table for all selected clusters
    overview_data = []
    for cluster_id in selected_clusters:
        cluster_df = df[df['cluster'] == cluster_id]
        row = {
            'Cluster': f"Cluster {cluster_id}",
            'Size': len(cluster_df),
            'Percentage': f"{len(cluster_df) / len(df) * 100:.1f}%"
        }

        if 'bpm' in df.columns:
            row['Avg BPM'] = f"{cluster_df['bpm'].mean():.1f}"
        if 'danceability' in df.columns:
            row['Avg Danceability'] = f"{cluster_df['danceability'].mean():.2f}"
        if 'mood_happy' in df.columns:
            row['Avg Happiness'] = f"{cluster_df['mood_happy'].mean():.2f}"
        if 'valence' in df.columns:
            row['Avg Valence'] = f"{cluster_df['valence'].mean():.2f}"

        overview_data.append(row)

    overview_df = pd.DataFrame(overview_data)
    st.dataframe(overview_df, use_container_width=True, hide_index=True)

    # Pairwise Statistical Comparisons
    st.markdown("---")
    st.subheader("📈 Pairwise Statistical Comparisons")

    if len(selected_clusters) == 2:
        # For 2 clusters, show detailed comparison
        with st.spinner("Running statistical tests..."):
            comparison_df = compare_two_clusters(df, selected_clusters[0], selected_clusters[1])

            if len(comparison_df) > 0:
                # Show only significant differences by default
                show_all = st.checkbox("Show all features (including non-significant)", value=False)

                if not show_all:
                    display_df = comparison_df[comparison_df['significant']].copy()
                    st.write(f"**Showing {len(display_df)} significant differences (p < 0.05)**")
                else:
                    display_df = comparison_df.copy()
                    st.write(f"**Showing all {len(display_df)} features**")

                if len(display_df) > 0:
                    # Format for display
                    display_df['cluster_a_mean'] = display_df['cluster_a_mean'].round(3)
                    display_df['cluster_b_mean'] = display_df['cluster_b_mean'].round(3)
                    display_df['difference'] = display_df['difference'].round(3)
                    display_df['effect_size'] = display_df['effect_size'].round(3)
                    display_df['t_statistic'] = display_df['t_statistic'].round(3)
                    display_df['p_value'] = display_df['p_value'].apply(lambda x: f"{x:.4f}")

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=400,
                        column_config={
                            "feature": "Feature",
                            "cluster_a_mean": f"Cluster {selected_clusters[0]} Mean",
                            "cluster_b_mean": f"Cluster {selected_clusters[1]} Mean",
                            "difference": "Difference",
                            "effect_size": st.column_config.NumberColumn(
                                "Effect Size",
                                help="Cohen's d - measures practical significance",
                                format="%.3f"
                            ),
                            "t_statistic": "t-statistic",
                            "p_value": "p-value",
                            "significant": st.column_config.CheckboxColumn("Significant?")
                        },
                        hide_index=True
                    )

                    # Download button
                    csv = comparison_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"📥 Download Comparison Data",
                        data=csv,
                        file_name=f"cluster_{selected_clusters[0]}_vs_{selected_clusters[1]}_comparison.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No significant differences found between these clusters")
            else:
                st.warning("Unable to compare these clusters")
    else:
        # For 3+ clusters, show summary of all pairwise comparisons
        st.write(f"**Showing summary of all pairwise comparisons for {len(selected_clusters)} clusters**")

        with st.spinner("Computing all pairwise comparisons..."):
            # Compute all pairs
            comparison_summaries = []

            for i, cluster_a in enumerate(selected_clusters):
                for cluster_b in selected_clusters[i+1:]:
                    comparison_df = compare_two_clusters(df, cluster_a, cluster_b)

                    if len(comparison_df) > 0:
                        # Count significant differences
                        n_significant = comparison_df['significant'].sum()
                        avg_effect_size = comparison_df['effect_size'].abs().mean()

                        comparison_summaries.append({
                            'Cluster A': f"Cluster {cluster_a}",
                            'Cluster B': f"Cluster {cluster_b}",
                            'Significant Differences': n_significant,
                            'Avg Effect Size': f"{avg_effect_size:.3f}",
                            'Most Different Feature': comparison_df.iloc[0]['feature']
                        })

            summary_df = pd.DataFrame(comparison_summaries)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.info("💡 Select exactly 2 clusters to see detailed statistical comparison")

    # Radar plot comparison - now supports multiple clusters!
    st.markdown("---")
    st.subheader("🎯 Multi-Dimensional Comparison")

    # Select key features for radar plot
    radar_features = [
        'bpm', 'danceability', 'valence', 'arousal',
        'mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed'
    ]
    radar_features = [f for f in radar_features if f in df.columns]

    if len(radar_features) >= 3:
        # Normalize features to 0-1 scale for fair comparison
        normalized_df = df[radar_features].copy()
        for col in radar_features:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val > min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)

        # Add cluster column back
        normalized_df['cluster'] = df['cluster'].values

        # Create radar plot with all selected clusters
        fig = go.Figure()

        colors = px.colors.qualitative.Plotly

        for i, cluster_id in enumerate(selected_clusters):
            cluster_means = normalized_df[normalized_df['cluster'] == cluster_id][radar_features].mean()

            fig.add_trace(go.Scatterpolar(
                r=cluster_means.values,
                theta=radar_features,
                fill='toself',
                name=f'Cluster {cluster_id}',
                line_color=colors[i % len(colors)],
                opacity=0.6
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title=f"Multi-Cluster Comparison: {len(selected_clusters)} Clusters (Normalized Features)",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 All features normalized to 0-1 scale for fair comparison. Each cluster shown as a different color.")

    # Genre comparison
    st.markdown("---")
    st.subheader("🎸 Genre Comparison")

    if 'top_genre' in df.columns:
        # Show top genres for each selected cluster
        num_cols = min(len(selected_clusters), 3)  # Max 3 columns
        cols = st.columns(num_cols)

        for i, cluster_id in enumerate(selected_clusters):
            cluster_df = df[df['cluster'] == cluster_id]
            col_idx = i % num_cols

            with cols[col_idx]:
                st.write(f"**Cluster {cluster_id} - Top 10 Genres**")
                cluster_genres = cluster_df['top_genre'].value_counts().head(10)

                fig = px.bar(
                    x=cluster_genres.values,
                    y=cluster_genres.index,
                    orientation='h',
                    labels={'x': 'Count', 'y': 'Genre'},
                    color_discrete_sequence=[px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]]
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Genre overlap analysis for all selected clusters
        st.markdown("---")
        st.write(f"**Genre Overlap Analysis ({len(selected_clusters)} clusters)**")

        genre_sets = {cluster_id: set(df[df['cluster'] == cluster_id]['top_genre'].unique())
                     for cluster_id in selected_clusters}

        # Find shared genres across all clusters
        shared_genres = set.intersection(*genre_sets.values()) if len(genre_sets) > 0 else set()

        overlap_data = []
        for cluster_id in selected_clusters:
            # Genres unique to this cluster
            other_clusters = [c for c in selected_clusters if c != cluster_id]
            other_genres = set.union(*[genre_sets[c] for c in other_clusters]) if other_clusters else set()
            unique_genres = genre_sets[cluster_id] - other_genres

            overlap_data.append({
                'Cluster': f"Cluster {cluster_id}",
                'Total Genres': len(genre_sets[cluster_id]),
                'Unique Genres': len(unique_genres),
                'Shared with All': len(shared_genres)
            })

        overlap_df = pd.DataFrame(overlap_data)
        st.dataframe(overlap_df, use_container_width=True, hide_index=True)

        if len(shared_genres) > 0:
            st.info(f"🎵 {len(shared_genres)} genres appear in all selected clusters: {', '.join(list(shared_genres)[:10])}{('...' if len(shared_genres) > 10 else '')}")

    # Sample songs from each cluster
    st.markdown("---")
    st.subheader("🎵 Sample Songs from Each Cluster")

    num_cols = min(len(selected_clusters), 3)  # Max 3 columns
    cols = st.columns(num_cols)

    for i, cluster_id in enumerate(selected_clusters):
        cluster_df = df[df['cluster'] == cluster_id]
        col_idx = i % num_cols

        with cols[col_idx]:
            st.write(f"**Cluster {cluster_id} - Random Sample**")
            sample_df = cluster_df.sample(min(10, len(cluster_df)))[['track_name', 'artist']]
            st.dataframe(sample_df, use_container_width=True, hide_index=True)


@st.cache_data
def load_all_lyrics(df: pd.DataFrame, lyrics_dir: str = 'lyrics/temp/'):
    """Load all lyrics for the dataset (cached)."""
    all_lyrics = []

    for _, row in df.iterrows():
        filename = row.get('filename', '')
        if not filename:
            continue

        lyric_filename = filename.replace('.mp3', '.txt')
        lyric_file = Path(lyrics_dir) / lyric_filename

        if lyric_file.exists():
            try:
                with open(lyric_file, 'r', encoding='utf-8') as f:
                    lyrics_text = f.read().strip()
                    if lyrics_text:
                        all_lyrics.append(lyrics_text)
            except:
                pass

    return all_lyrics


def render_lyric_themes(df: pd.DataFrame):
    """Render Lyric Themes view."""
    st.header("📝 Lyric Theme Analysis")

    st.write("Explore lyric themes, sentiment, and complexity across clusters.")

    # Lyrics directory configuration
    lyrics_dir = st.text_input(
        "Lyrics directory",
        value="lyrics/temp/",
        help="Directory containing lyric .txt files"
    )

    if not Path(lyrics_dir).exists():
        st.error(f"Lyrics directory not found: {lyrics_dir}")
        st.info("💡 Update the path above to point to your lyrics directory")
        return

    # Cluster selection
    cluster_ids = sorted(df['cluster'].unique())
    selected_cluster = st.selectbox(
        "Select Cluster to Analyze",
        options=cluster_ids,
        format_func=lambda x: f"Cluster {x}",
        key="lyric_cluster"
    )

    with st.spinner("Loading lyrics..."):
        # Load lyrics for selected cluster
        cluster_lyrics_data = load_lyrics_for_cluster(df, selected_cluster, lyrics_dir)

        # Load all lyrics for TF-IDF comparison
        all_lyrics = load_all_lyrics(df, lyrics_dir)

        if not cluster_lyrics_data:
            st.warning(f"No lyrics found for Cluster {selected_cluster}")
            st.info("Make sure lyrics are stored as .txt files matching the MP3 filenames")
            return

        cluster_lyrics = [lyrics for _, lyrics in cluster_lyrics_data]

        # Basic metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            coverage_pct = (len(cluster_lyrics_data) / len(df[df['cluster'] == selected_cluster]) * 100)
            st.metric("Songs with Lyrics", f"{len(cluster_lyrics_data)}")
            st.caption(f"{coverage_pct:.1f}% coverage")

        with col2:
            if cluster_lyrics:
                avg_word_count = sum(len(lyrics.split()) for lyrics in cluster_lyrics) / len(cluster_lyrics)
                st.metric("Avg Word Count", f"{avg_word_count:.0f}")

        with col3:
            if cluster_lyrics:
                total_words = sum(len(lyrics.split()) for lyrics in cluster_lyrics)
                unique_words = len(set(' '.join(cluster_lyrics).lower().split()))
                st.metric("Unique Words", unique_words)

    # Keyword Analysis
    st.markdown("---")
    st.subheader("🔑 Keyword Analysis (TF-IDF)")

    with st.spinner("Extracting keywords..."):
        if all_lyrics and cluster_lyrics:
            keywords_data = extract_tfidf_keywords(
                all_lyrics,
                cluster_lyrics,
                top_n=30,
                ngram_range=(1, 3)
            )

            if keywords_data['unigrams']:
                tab1, tab2, tab3 = st.tabs(["Unigrams", "Bigrams", "Trigrams"])

                with tab1:
                    st.write("**Top 30 Single Words**")
                    unigrams_df = pd.DataFrame(keywords_data['unigrams'], columns=['Word', 'TF-IDF Score'])
                    unigrams_df['TF-IDF Score'] = unigrams_df['TF-IDF Score'].round(4)

                    # Bar chart
                    fig = px.bar(
                        unigrams_df.head(20),
                        x='TF-IDF Score',
                        y='Word',
                        orientation='h',
                        title="Top 20 Keywords"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(unigrams_df, use_container_width=True, hide_index=True)

                with tab2:
                    if keywords_data['bigrams']:
                        st.write("**Top 30 Two-Word Phrases**")
                        bigrams_df = pd.DataFrame(keywords_data['bigrams'], columns=['Phrase', 'TF-IDF Score'])
                        bigrams_df['TF-IDF Score'] = bigrams_df['TF-IDF Score'].round(4)
                        st.dataframe(bigrams_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No significant bigrams found")

                with tab3:
                    if keywords_data['trigrams']:
                        st.write("**Top 30 Three-Word Phrases**")
                        trigrams_df = pd.DataFrame(keywords_data['trigrams'], columns=['Phrase', 'TF-IDF Score'])
                        trigrams_df['TF-IDF Score'] = trigrams_df['TF-IDF Score'].round(4)
                        st.dataframe(trigrams_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No significant trigrams found")

                # Word Cloud
                st.markdown("---")
                st.subheader("☁️ Word Cloud")

                try:
                    from wordcloud import WordCloud
                    import matplotlib.pyplot as plt

                    # Create word cloud from keywords
                    word_freq = {word: score for word, score in keywords_data['unigrams'][:50]}

                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        colormap='viridis',
                        relative_scaling=0.5,
                        min_font_size=10
                    ).generate_from_frequencies(word_freq)

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()

                except ImportError:
                    st.warning("WordCloud library not available. Install with: pip install wordcloud")

            else:
                st.warning("No keywords extracted. Try adjusting the lyrics directory path.")

    # Sentiment Analysis
    st.markdown("---")
    st.subheader("😊 Sentiment Analysis")

    with st.spinner("Analyzing sentiment..."):
        if cluster_lyrics:
            sentiments = []

            for lyrics in cluster_lyrics:
                sentiment = analyze_sentiment(lyrics)
                sentiments.append(sentiment)

            # Average sentiment
            avg_compound = np.mean([s['compound_score'] for s in sentiments])
            avg_positive = np.mean([s['positive'] for s in sentiments])
            avg_negative = np.mean([s['negative'] for s in sentiments])
            avg_neutral = np.mean([s['neutral'] for s in sentiments])

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Compound Score", f"{avg_compound:.3f}",
                         help="Overall sentiment: -1 (very negative) to +1 (very positive)")

            with col2:
                st.metric("Positive", f"{avg_positive:.3f}")

            with col3:
                st.metric("Negative", f"{avg_negative:.3f}")

            with col4:
                st.metric("Neutral", f"{avg_neutral:.3f}")

            # Sentiment distribution
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=[s['compound_score'] for s in sentiments],
                nbinsx=20,
                name="Sentiment Distribution"
            ))

            fig.update_layout(
                title="Distribution of Sentiment Scores",
                xaxis_title="Compound Sentiment Score",
                yaxis_title="Number of Songs",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Most positive/negative songs
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Most Positive Songs**")
                # Get top 5 most positive
                sorted_sentiments = sorted(enumerate(sentiments), key=lambda x: x[1]['compound_score'], reverse=True)
                positive_songs = []
                for idx, sentiment in sorted_sentiments[:5]:
                    track_name, _ = cluster_lyrics_data[idx]
                    positive_songs.append({
                        'Song': track_name,
                        'Sentiment': f"{sentiment['compound_score']:.3f}"
                    })
                st.dataframe(pd.DataFrame(positive_songs), use_container_width=True, hide_index=True)

            with col2:
                st.write("**Most Negative Songs**")
                negative_songs = []
                for idx, sentiment in sorted_sentiments[-5:][::-1]:
                    track_name, _ = cluster_lyrics_data[idx]
                    negative_songs.append({
                        'Song': track_name,
                        'Sentiment': f"{sentiment['compound_score']:.3f}"
                    })
                st.dataframe(pd.DataFrame(negative_songs), use_container_width=True, hide_index=True)

    # Lyric Complexity
    st.markdown("---")
    st.subheader("📚 Lyric Complexity")

    with st.spinner("Computing complexity metrics..."):
        if cluster_lyrics:
            complexities = []

            for lyrics in cluster_lyrics:
                complexity = compute_lyric_complexity(lyrics)
                complexities.append(complexity)

            # Average complexity metrics
            avg_richness = np.mean([c['vocabulary_richness'] for c in complexities])
            avg_word_length = np.mean([c['avg_word_length'] for c in complexities])
            avg_flesch = np.mean([c['flesch_reading_ease'] for c in complexities if c['flesch_reading_ease'] > 0])

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Avg Vocabulary Richness", f"{avg_richness:.3f}",
                         help="Unique words / Total words (higher = more diverse vocabulary)")

            with col2:
                st.metric("Avg Word Length", f"{avg_word_length:.1f} chars")

            with col3:
                if avg_flesch > 0:
                    st.metric("Flesch Reading Ease", f"{avg_flesch:.1f}",
                             help="0-100 scale, higher = easier to read")

            # Complexity extremes
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Most Complex Lyrics**")
                sorted_complexity = sorted(enumerate(complexities),
                                         key=lambda x: x[1]['vocabulary_richness'], reverse=True)
                complex_songs = []
                for idx, complexity in sorted_complexity[:5]:
                    track_name, _ = cluster_lyrics_data[idx]
                    complex_songs.append({
                        'Song': track_name,
                        'Richness': f"{complexity['vocabulary_richness']:.3f}",
                        'Words': complexity['word_count']
                    })
                st.dataframe(pd.DataFrame(complex_songs), use_container_width=True, hide_index=True)

            with col2:
                st.write("**Simplest Lyrics**")
                simple_songs = []
                for idx, complexity in sorted_complexity[-5:][::-1]:
                    track_name, _ = cluster_lyrics_data[idx]
                    simple_songs.append({
                        'Song': track_name,
                        'Richness': f"{complexity['vocabulary_richness']:.3f}",
                        'Words': complexity['word_count']
                    })
                st.dataframe(pd.DataFrame(simple_songs), use_container_width=True, hide_index=True)

    # Common Phrases
    st.markdown("---")
    st.subheader("🔁 Common Repeated Phrases")

    with st.spinner("Finding common phrases..."):
        if cluster_lyrics:
            common_phrases = extract_common_phrases(cluster_lyrics, top_n=20)

            if common_phrases:
                phrases_df = pd.DataFrame(common_phrases, columns=['Phrase', 'Count'])
                st.write("**Top 20 Repeated Phrases (appear more than once)**")
                st.dataframe(phrases_df, use_container_width=True, hide_index=True)
            else:
                st.info("No commonly repeated phrases found")


def render_overview(df: pd.DataFrame):
    """Render Overview view."""
    st.header("🔍 Global Overview")

    st.write("High-level summary of all clusters and their relationships.")

    # Summary Statistics
    st.subheader("📊 Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Songs", len(df))

    with col2:
        st.metric("Number of Clusters", df['cluster'].nunique())

    with col3:
        if 'has_lyrics' in df.columns:
            lyric_pct = (df['has_lyrics'].sum() / len(df) * 100) if len(df) > 0 else 0
            st.metric("Songs with Lyrics", f"{lyric_pct:.1f}%")
        else:
            st.metric("Songs with Lyrics", "N/A")

    with col4:
        # Calculate average silhouette score if available
        st.metric("Clustering Mode", "Combined")

    # Cluster sizes
    st.markdown("---")
    st.subheader("📏 Cluster Size Distribution")

    cluster_sizes = df['cluster'].value_counts().sort_index()

    fig = px.bar(
        x=cluster_sizes.index,
        y=cluster_sizes.values,
        labels={'x': 'Cluster', 'y': 'Number of Songs'},
        title="Songs per Cluster"
    )
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(type='category')
    st.plotly_chart(fig, use_container_width=True)

    # Show cluster size table
    cluster_info = []
    for cluster_id in sorted(df['cluster'].unique()):
        size = len(df[df['cluster'] == cluster_id])
        percentage = (size / len(df) * 100)

        cluster_info.append({
            'Cluster': f"Cluster {cluster_id}",
            'Size': size,
            'Percentage': f"{percentage:.1f}%"
        })

    st.dataframe(pd.DataFrame(cluster_info), use_container_width=True, hide_index=True)

    # Cluster similarity matrix
    st.markdown("---")
    st.subheader("🔗 Cluster Similarity Matrix")

    st.write("Lower values = more similar clusters (based on average effect sizes)")

    with st.spinner("Computing cluster similarities..."):
        similarity_matrix = compute_cluster_similarity_matrix(df)

        fig = px.imshow(
            similarity_matrix,
            labels=dict(x="Cluster", y="Cluster", color="Dissimilarity"),
            x=similarity_matrix.columns,
            y=similarity_matrix.index,
            color_continuous_scale='YlOrRd',
            title="Cluster Dissimilarity Matrix (Lower = More Similar)",
            aspect="auto"
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Most and least similar cluster pairs
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Most Similar Cluster Pairs**")

        # Get all pairs with their similarity scores
        pairs = []
        cluster_ids = sorted(df['cluster'].unique())

        for i, cluster_a in enumerate(cluster_ids):
            for cluster_b in cluster_ids[i+1:]:
                dissimilarity = similarity_matrix.loc[cluster_a, cluster_b]
                pairs.append({
                    'Pair': f"Cluster {cluster_a} & {cluster_b}",
                    'Dissimilarity': f"{dissimilarity:.3f}"
                })

        pairs_df = pd.DataFrame(pairs)
        pairs_df = pairs_df.sort_values('Dissimilarity')
        st.dataframe(pairs_df.head(5), use_container_width=True, hide_index=True)

    with col2:
        st.write("**Most Different Cluster Pairs**")
        st.dataframe(pairs_df.tail(5).iloc[::-1], use_container_width=True, hide_index=True)

    # Key Feature Summary
    st.markdown("---")
    st.subheader("🎯 Key Feature Summary Across All Clusters")

    # Select a few key features to summarize
    key_features = ['bpm', 'danceability', 'valence', 'arousal', 'mood_happy', 'mood_sad']
    key_features = [f for f in key_features if f in df.columns]

    if key_features:
        summary_data = []

        for feature in key_features:
            feature_data = {'Feature': feature}

            for cluster_id in sorted(df['cluster'].unique()):
                cluster_df = df[df['cluster'] == cluster_id]
                mean_val = cluster_df[feature].mean()
                feature_data[f'Cluster {cluster_id}'] = f"{mean_val:.3f}"

            summary_data.append(feature_data)

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Genre distribution across clusters
    if 'top_genre' in df.columns:
        st.markdown("---")
        st.subheader("🎸 Top Genre Distribution Across Clusters")

        # Get top 5 overall genres
        top_genres = df['top_genre'].value_counts().head(5).index

        genre_cluster_data = []
        for genre in top_genres:
            row = {'Genre': genre}
            for cluster_id in sorted(df['cluster'].unique()):
                cluster_df = df[df['cluster'] == cluster_id]
                count = (cluster_df['top_genre'] == genre).sum()
                percentage = (count / len(cluster_df) * 100) if len(cluster_df) > 0 else 0
                row[f'Cluster {cluster_id}'] = f"{percentage:.1f}%"

            genre_cluster_data.append(row)

        genre_dist_df = pd.DataFrame(genre_cluster_data)
        st.dataframe(genre_dist_df, use_container_width=True, hide_index=True)

    # Mood radar comparison
    mood_cols = ['mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed', 'mood_party']

    if all(col in df.columns for col in mood_cols):
        st.markdown("---")
        st.subheader("😊 Mood Profiles by Cluster")

        fig = go.Figure()

        cluster_ids = sorted(df['cluster'].unique())
        colors = px.colors.qualitative.Plotly

        for i, cluster_id in enumerate(cluster_ids):
            cluster_df = df[df['cluster'] == cluster_id]

            mood_means = [
                cluster_df['mood_happy'].mean() * 100,
                cluster_df['mood_sad'].mean() * 100,
                cluster_df['mood_aggressive'].mean() * 100,
                cluster_df['mood_relaxed'].mean() * 100,
                cluster_df['mood_party'].mean() * 100,
            ]

            fig.add_trace(go.Scatterpolar(
                r=mood_means,
                theta=['Happy', 'Sad', 'Aggressive', 'Relaxed', 'Party'],
                fill='toself',
                name=f'Cluster {cluster_id}',
                line_color=colors[i % len(colors)]
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Mood Profiles by Cluster",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

    # Export all overview data
    st.markdown("---")
    st.subheader("📥 Export Data")

    col1, col2 = st.columns(2)

    with col1:
        # Export similarity matrix
        csv = similarity_matrix.to_csv().encode('utf-8')
        st.download_button(
            label="Download Similarity Matrix",
            data=csv,
            file_name="cluster_similarity_matrix.csv",
            mime="text/csv",
        )

    with col2:
        # Export cluster summary
        summary_csv = pd.DataFrame(cluster_info).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cluster Summary",
            data=summary_csv,
            file_name="cluster_summary.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
