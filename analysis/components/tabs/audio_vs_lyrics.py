"""Audio vs Lyrics Tab Component

Comprehensive analysis of audio-based vs lyric-based clustering:
- Contingency matrix heatmap
- Clustering agreement metrics (ARI, NMI, FMI)
- Where audio clusters map to lyric space (and vice versa)
- Cross-domain feature correlations
- Genre-specific cluster movement analysis
- Feature combination experiments
- Example tracks (stable vs. high movement)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix

from analysis.components.visualization.color_palette import CLUSTER_COLORS
from analysis.components.export.chart_export import render_chart_with_export, render_export_section


# Feature column definitions
AUDIO_FEATURES = [
    'emb_bpm', 'emb_danceability', 'emb_instrumentalness', 'emb_valence', 'emb_arousal',
    'emb_engagement', 'emb_approachability', 'emb_mood_aggressive', 'emb_mood_happy',
    'emb_mood_sad', 'emb_mood_relaxed', 'emb_mood_party', 'emb_voice_gender',
    'emb_genre_ladder', 'emb_acoustic_electronic', 'emb_timbre_brightness'
]

LYRIC_FEATURES = [
    'emb_lyric_valence', 'emb_lyric_arousal', 'emb_lyric_mood_happy',
    'emb_lyric_mood_sad', 'emb_lyric_mood_aggressive', 'emb_lyric_mood_relaxed',
    'emb_lyric_explicit', 'emb_lyric_narrative', 'emb_lyric_vocabulary', 'emb_lyric_repetition'
]


def get_available_features(df: pd.DataFrame, feature_list: list) -> list:
    """Return only features that exist in the dataframe."""
    return [f for f in feature_list if f in df.columns]


def compute_audio_lyric_clusters(df: pd.DataFrame, n_clusters: int = 5):
    """Compute separate audio-only and lyric-only clusterings."""
    audio_cols = get_available_features(df, AUDIO_FEATURES)
    lyric_cols = get_available_features(df, LYRIC_FEATURES)

    if len(audio_cols) < 3 or len(lyric_cols) < 3:
        return None, None, None, None

    X_audio = df[audio_cols].values
    X_lyrics = df[lyric_cols].values

    # Handle any NaN values
    X_audio = np.nan_to_num(X_audio, nan=0.0)
    X_lyrics = np.nan_to_num(X_lyrics, nan=0.0)

    # Cluster on audio features only
    clusterer_audio = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels_audio = clusterer_audio.fit_predict(X_audio)

    # Cluster on lyric features only
    clusterer_lyrics = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels_lyrics = clusterer_lyrics.fit_predict(X_lyrics)

    return labels_audio, labels_lyrics, X_audio, X_lyrics


def compute_agreement_metrics(labels_audio, labels_lyrics, X_audio, X_lyrics):
    """Compute clustering agreement metrics between audio and lyric clusterings."""
    ari = adjusted_rand_score(labels_audio, labels_lyrics)
    nmi = normalized_mutual_info_score(labels_audio, labels_lyrics)
    fmi = fowlkes_mallows_score(labels_audio, labels_lyrics)

    sil_audio = silhouette_score(X_audio, labels_audio)
    sil_lyrics = silhouette_score(X_lyrics, labels_lyrics)

    return {
        'ari': ari,
        'nmi': nmi,
        'fmi': fmi,
        'sil_audio': sil_audio,
        'sil_lyrics': sil_lyrics
    }


def compute_cross_correlations(df: pd.DataFrame):
    """Compute correlations between all audio and lyric features."""
    audio_cols = get_available_features(df, AUDIO_FEATURES)
    lyric_cols = get_available_features(df, LYRIC_FEATURES)

    correlations = []
    for audio_feat in audio_cols:
        for lyric_feat in lyric_cols:
            corr = df[audio_feat].corr(df[lyric_feat])
            if not np.isnan(corr):
                correlations.append({
                    'audio': audio_feat.replace('emb_', ''),
                    'lyric': lyric_feat.replace('emb_lyric_', ''),
                    'correlation': corr,
                    'abs_corr': abs(corr)
                })

    return pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)


def render_audio_vs_lyrics(df: pd.DataFrame):
    """Render Audio vs Lyrics comparison view."""
    st.header("🔀 Audio vs Lyrics Clustering Analysis")

    st.markdown("""
    This analysis compares how songs cluster based on **audio features only** vs. **lyric features only**.
    Low agreement indicates that songs which sound similar don't necessarily have similar lyrics,
    meaning the two feature spaces capture **complementary, not redundant** information.
    """)

    # Check if required features exist
    audio_cols = get_available_features(df, AUDIO_FEATURES)
    lyric_cols = get_available_features(df, LYRIC_FEATURES)

    if len(audio_cols) < 3:
        st.error(f"Insufficient audio features. Found: {len(audio_cols)}, need at least 3.")
        st.info("Available audio columns: " + ", ".join(audio_cols) if audio_cols else "None")
        return

    if len(lyric_cols) < 3:
        st.error(f"Insufficient lyric features. Found: {len(lyric_cols)}, need at least 3.")
        st.info("Available lyric columns: " + ", ".join(lyric_cols) if lyric_cols else "None")
        return

    st.success(f"Found {len(audio_cols)} audio features and {len(lyric_cols)} lyric features")

    # Cluster count selector
    n_clusters = st.slider(
        "Number of clusters for comparison",
        min_value=2,
        max_value=10,
        value=5,
        help="Both audio-only and lyric-only clusterings will use this many clusters"
    )

    # Compute clusterings
    with st.spinner("Computing audio-only and lyric-only clusterings..."):
        labels_audio, labels_lyrics, X_audio, X_lyrics = compute_audio_lyric_clusters(df, n_clusters)

    if labels_audio is None:
        st.error("Failed to compute clusterings")
        return

    # Store cluster labels in dataframe for analysis
    df_analysis = df.copy()
    df_analysis['audio_cluster'] = labels_audio
    df_analysis['lyric_cluster'] = labels_lyrics
    df_analysis['cluster_agreement'] = (labels_audio == labels_lyrics)

    # ====================
    # SECTION 1: Agreement Metrics
    # ====================
    st.markdown("---")
    st.subheader("1. Clustering Agreement Metrics")

    metrics = compute_agreement_metrics(labels_audio, labels_lyrics, X_audio, X_lyrics)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Adjusted Rand Index (ARI)",
            f"{metrics['ari']:.3f}",
            help="0=random, 1=perfect agreement, <0=worse than random"
        )

    with col2:
        st.metric(
            "Normalized Mutual Info (NMI)",
            f"{metrics['nmi']:.3f}",
            help="0=no mutual information, 1=perfect agreement"
        )

    with col3:
        st.metric(
            "Fowlkes-Mallows Index (FMI)",
            f"{metrics['fmi']:.3f}",
            help="0=no similarity, 1=identical clusterings"
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            "Audio Silhouette Score",
            f"{metrics['sil_audio']:.3f}",
            help="Clustering quality on audio features (-1 to 1, higher=better)"
        )

    with col5:
        st.metric(
            "Lyrics Silhouette Score",
            f"{metrics['sil_lyrics']:.3f}",
            help="Clustering quality on lyric features (-1 to 1, higher=better)"
        )

    with col6:
        agreement_rate = df_analysis['cluster_agreement'].mean()
        st.metric(
            "Same Cluster Rate",
            f"{agreement_rate*100:.1f}%",
            help="Percentage of tracks assigned to same cluster number in both spaces"
        )

    # Interpretation
    if metrics['ari'] < 0.15:
        interpretation = "LOW agreement - audio and lyrics capture fundamentally different aspects"
    elif metrics['ari'] < 0.35:
        interpretation = "MODERATE agreement - some shared structure between audio and lyrics"
    else:
        interpretation = "HIGH agreement - audio and lyrics capture similar groupings"

    st.info(f"**Interpretation:** {interpretation}")

    # ====================
    # SECTION 2: Contingency Matrix
    # ====================
    st.markdown("---")
    st.subheader("2. Contingency Matrix")

    st.markdown("""
    The contingency matrix shows how tracks from each **audio cluster** (rows)
    distribute across **lyric clusters** (columns). Perfect agreement would show
    all counts on the diagonal.
    """)

    cont_matrix = contingency_matrix(labels_audio, labels_lyrics)

    # Create heatmap (styled to match dissimilarity matrix - square cells)
    st.markdown(f"**Contingency Matrix ({n_clusters} clusters)**")
    st.caption("How tracks from each audio cluster distribute across lyric clusters")

    x_labels = [f"Lyric {i}" for i in range(n_clusters)]
    y_labels = [f"Audio {i}" for i in range(n_clusters)]

    fig = px.imshow(
        cont_matrix,
        x=x_labels,
        y=y_labels,
        labels=dict(color="Count"),
        color_continuous_scale="Blues",
        aspect="equal",
        text_auto="d",
    )

    fig.update_traces(textfont_size=12)
    fig.update_layout(
        height=600,
        margin=dict(t=0, l=0, r=0, b=0),
        xaxis_title="",
        yaxis_title="",
    )

    render_chart_with_export(fig, "contingency_matrix", "Audio vs Lyric Contingency Matrix", "audio_vs_lyrics")

    # Show raw contingency table
    with st.expander("View Raw Contingency Table"):
        cont_df = pd.DataFrame(
            cont_matrix,
            index=[f"Audio {i}" for i in range(n_clusters)],
            columns=[f"Lyric {i}" for i in range(n_clusters)]
        )
        cont_df['Total'] = cont_df.sum(axis=1)
        cont_df.loc['Total'] = cont_df.sum()
        st.dataframe(cont_df, use_container_width=True)

    # ====================
    # SECTION 3: Cluster Flow Analysis
    # ====================
    st.markdown("---")
    st.subheader("3. Cluster Flow Analysis")

    tab_audio_to_lyric, tab_lyric_to_audio = st.tabs([
        "Audio → Lyric (Where audio clusters go)",
        "Lyric ← Audio (Where lyric clusters come from)"
    ])

    with tab_audio_to_lyric:
        st.markdown("**Where Audio Clusters End Up in Lyric Space:**")

        flow_data = []
        for i in range(n_clusters):
            audio_cluster_size = cont_matrix[i].sum()
            for j in range(n_clusters):
                pct = cont_matrix[i, j] / audio_cluster_size * 100 if audio_cluster_size > 0 else 0
                if pct > 5:  # Only show significant flows
                    flow_data.append({
                        'Audio Cluster': f"Audio {i}",
                        'Lyric Cluster': f"Lyric {j}",
                        'Count': cont_matrix[i, j],
                        'Percentage': f"{pct:.1f}%"
                    })

        if flow_data:
            flow_df = pd.DataFrame(flow_data)

            # Create Sankey diagram
            labels = [f"Audio {i}" for i in range(n_clusters)] + [f"Lyric {i}" for i in range(n_clusters)]
            source = []
            target = []
            values = []

            for i in range(n_clusters):
                for j in range(n_clusters):
                    if cont_matrix[i, j] > 0:
                        source.append(i)
                        target.append(n_clusters + j)
                        values.append(cont_matrix[i, j])

            st.markdown("**Audio → Lyric Cluster Flow**")
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=CLUSTER_COLORS[:n_clusters] + CLUSTER_COLORS[:n_clusters]
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=values
                )
            )])

            fig_sankey.update_layout(height=600, margin=dict(t=0, l=0, r=0, b=0))

            render_chart_with_export(fig_sankey, "audio_lyric_sankey", "Audio → Lyric Cluster Flow", "audio_vs_lyrics")

            st.dataframe(flow_df, use_container_width=True, hide_index=True)

    with tab_lyric_to_audio:
        st.markdown("**Where Lyric Clusters Come From in Audio Space:**")

        flow_data = []
        for j in range(n_clusters):
            lyric_cluster_size = cont_matrix[:, j].sum()
            for i in range(n_clusters):
                pct = cont_matrix[i, j] / lyric_cluster_size * 100 if lyric_cluster_size > 0 else 0
                if pct > 5:  # Only show significant flows
                    flow_data.append({
                        'Lyric Cluster': f"Lyric {j}",
                        'Audio Cluster': f"Audio {i}",
                        'Count': cont_matrix[i, j],
                        'Percentage': f"{pct:.1f}%"
                    })

        if flow_data:
            flow_df = pd.DataFrame(flow_data)
            st.dataframe(flow_df, use_container_width=True, hide_index=True)

    # ====================
    # SECTION 4: Cross-Domain Feature Correlations
    # ====================
    st.markdown("---")
    st.subheader("4. Cross-Domain Feature Correlations")

    st.markdown("""
    These correlations show which audio features are most predictive of lyric features (and vice versa).
    Strong correlations indicate shared semantic meaning across modalities.
    """)

    with st.spinner("Computing cross-domain correlations..."):
        corr_df = compute_cross_correlations(df)

    if len(corr_df) > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 10 Positive Correlations:**")
            positive = corr_df[corr_df['correlation'] > 0].head(10)
            if len(positive) > 0:
                fig_pos = px.bar(
                    positive,
                    x='correlation',
                    y=positive.apply(lambda r: f"{r['audio']} ↔ {r['lyric']}", axis=1),
                    orientation='h',
                    color='correlation',
                    color_continuous_scale='Greens',
                )
                fig_pos.update_layout(
                    height=500,
                    yaxis_title="",
                    xaxis_title="Correlation",
                    showlegend=False,
                    margin=dict(t=0, l=0, r=0, b=0),
                )
                st.plotly_chart(fig_pos, use_container_width=True)

        with col2:
            st.markdown("**Top 10 Negative Correlations:**")
            negative = corr_df[corr_df['correlation'] < 0].head(10)
            if len(negative) > 0:
                fig_neg = px.bar(
                    negative,
                    x='correlation',
                    y=negative.apply(lambda r: f"{r['audio']} ↔ {r['lyric']}", axis=1),
                    orientation='h',
                    color='correlation',
                    color_continuous_scale='Reds_r',
                )
                fig_neg.update_layout(
                    height=500,
                    yaxis_title="",
                    xaxis_title="Correlation",
                    showlegend=False,
                    margin=dict(t=0, l=0, r=0, b=0),
                )
                st.plotly_chart(fig_neg, use_container_width=True)

        # Full correlation matrix heatmap
        with st.expander("View Full Correlation Matrix"):
            # Pivot to matrix form
            corr_matrix = corr_df.pivot(index='audio', columns='lyric', values='correlation')

            st.markdown("**Audio ↔ Lyric Feature Correlation Matrix**")
            st.caption("Red = negative correlation, Blue = positive correlation")

            fig_corr = px.imshow(
                corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.index.tolist(),
                labels=dict(color="Correlation"),
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                aspect="auto",
                text_auto=".2f",
            )

            fig_corr.update_traces(textfont_size=9)
            fig_corr.update_layout(
                height=600,
                margin=dict(t=0, l=0, r=0, b=0),
                xaxis_title="",
                yaxis_title="",
            )

            render_chart_with_export(fig_corr, "audio_lyric_correlation_matrix", "Audio-Lyric Correlation Matrix", "audio_vs_lyrics")

    # ====================
    # SECTION 5: Example Tracks
    # ====================
    st.markdown("---")
    st.subheader("5. Example Tracks")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Stable Tracks (same cluster in both spaces):**")
        stable_tracks = df_analysis[df_analysis['cluster_agreement']].head(10)
        if len(stable_tracks) > 0:
            stable_display = stable_tracks[['artist', 'track_name', 'audio_cluster']].copy()
            stable_display.columns = ['Artist', 'Track', 'Cluster']
            st.dataframe(stable_display, use_container_width=True, hide_index=True)
        else:
            st.info("No stable tracks found")

    with col2:
        st.markdown("**High Movement Tracks (large cluster shift):**")
        df_analysis['cluster_distance'] = abs(df_analysis['audio_cluster'] - df_analysis['lyric_cluster'])
        high_movement = df_analysis.nlargest(10, 'cluster_distance')
        if len(high_movement) > 0:
            movement_display = high_movement[['artist', 'track_name', 'audio_cluster', 'lyric_cluster']].copy()
            movement_display['Movement'] = movement_display.apply(
                lambda r: f"Audio {int(r['audio_cluster'])} → Lyric {int(r['lyric_cluster'])}", axis=1
            )
            # Select and rename columns properly
            movement_display = movement_display[['artist', 'track_name', 'Movement']]
            movement_display.columns = ['Artist', 'Track', 'Movement']
            st.dataframe(movement_display, use_container_width=True, hide_index=True)

    # ====================
    # SECTION 6: Instrumental Track Analysis
    # ====================
    if 'instrumentalness' in df.columns:
        st.markdown("---")
        st.subheader("6. Instrumental Track Analysis")

        instrumental_mask = df_analysis['instrumentalness'] > 0.5
        n_instrumental = instrumental_mask.sum()

        if n_instrumental > 0:
            st.write(f"**{n_instrumental} instrumental tracks** (instrumentalness > 0.5)")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Audio Cluster Distribution:**")
                audio_dist = df_analysis[instrumental_mask]['audio_cluster'].value_counts().sort_index()
                fig_audio = px.bar(
                    x=[f"Cluster {i}" for i in audio_dist.index],
                    y=audio_dist.values,
                    labels={'x': 'Cluster', 'y': 'Count'},
                    color_discrete_sequence=['#1DB954']
                )
                fig_audio.update_layout(height=400, showlegend=False, margin=dict(t=0, l=0, r=0, b=0))
                st.plotly_chart(fig_audio, use_container_width=True)

            with col2:
                st.markdown("**Lyric Cluster Distribution (should be concentrated):**")
                lyric_dist = df_analysis[instrumental_mask]['lyric_cluster'].value_counts().sort_index()
                fig_lyric = px.bar(
                    x=[f"Cluster {i}" for i in lyric_dist.index],
                    y=lyric_dist.values,
                    labels={'x': 'Cluster', 'y': 'Count'},
                    color_discrete_sequence=['#FF6B6B']
                )
                fig_lyric.update_layout(height=400, showlegend=False, margin=dict(t=0, l=0, r=0, b=0))
                st.plotly_chart(fig_lyric, use_container_width=True)

            st.caption(
                "Instrumental tracks have neutral lyric features (valence/arousal=0.5, moods=0), "
                "so they should cluster together in lyric space regardless of audio characteristics."
            )
        else:
            st.info("No instrumental tracks found (instrumentalness > 0.5)")

    # ====================
    # SECTION 7: Genre-Specific Movement
    # ====================
    if 'top_genre' in df.columns:
        st.markdown("---")
        st.subheader("7. Genre-Specific Cluster Movement")

        # Get top genres
        top_genres = df_analysis['top_genre'].value_counts().head(10).index.tolist()

        selected_genre = st.selectbox(
            "Select genre to analyze",
            options=top_genres,
            index=0
        )

        genre_mask = df_analysis['top_genre'] == selected_genre
        genre_df = df_analysis[genre_mask]

        if len(genre_df) > 10:
            st.write(f"**{selected_genre}** ({len(genre_df)} tracks)")

            col1, col2 = st.columns(2)

            with col1:
                genre_agreement = genre_df['cluster_agreement'].mean()
                st.metric("Agreement Rate", f"{genre_agreement*100:.1f}%")

            with col2:
                avg_movement = genre_df['cluster_distance'].mean()
                st.metric("Avg Cluster Distance", f"{avg_movement:.2f}")

            # Movement matrix for this genre (styled to match main contingency matrix)
            genre_cont = pd.crosstab(genre_df['audio_cluster'], genre_df['lyric_cluster'])

            st.markdown(f"**{selected_genre}: Audio → Lyric Movement**")

            # Use consistent labels (Audio/Lyric prefix)
            x_labels = [f"Lyric {i}" for i in genre_cont.columns]
            y_labels = [f"Audio {i}" for i in genre_cont.index]

            fig_genre = px.imshow(
                genre_cont.values,
                x=x_labels,
                y=y_labels,
                labels=dict(color="Count"),
                color_continuous_scale="Blues",
                aspect="equal",
                text_auto="d",
            )

            fig_genre.update_traces(textfont_size=12)
            fig_genre.update_layout(
                height=500,
                margin=dict(t=0, l=0, r=0, b=0),
                xaxis_title="",
                yaxis_title="",
            )

            st.plotly_chart(fig_genre, use_container_width=True)
        else:
            st.warning(f"Not enough tracks for {selected_genre} (need > 10)")

    # ====================
    # SECTION 8: Feature Weight Experiments
    # ====================
    st.markdown("---")
    st.subheader("8. Feature Combination Quality")

    st.markdown("""
    This experiment shows clustering quality (silhouette score) when combining
    audio and lyric features with different weights.
    """)

    audio_cols = get_available_features(df, AUDIO_FEATURES)
    lyric_cols = get_available_features(df, LYRIC_FEATURES)

    X_audio_full = np.nan_to_num(df[audio_cols].values, nan=0.0)
    X_lyrics_full = np.nan_to_num(df[lyric_cols].values, nan=0.0)

    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    with st.spinner("Testing different audio/lyric weight combinations..."):
        for w in weights:
            X_combined = np.hstack([X_audio_full * w, X_lyrics_full * (1 - w)])
            labels = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(X_combined)
            sil = silhouette_score(X_combined, labels)

            if w == 0:
                desc = "Lyrics only"
            elif w == 1:
                desc = "Audio only"
            else:
                desc = f"{w*100:.0f}% audio, {(1-w)*100:.0f}% lyrics"

            results.append({
                'Weight': w,
                'Description': desc,
                'Silhouette': sil
            })

    results_df = pd.DataFrame(results)

    st.markdown("**Silhouette Score by Feature Combination**")
    fig_weights = px.bar(
        results_df,
        x='Description',
        y='Silhouette',
        color='Silhouette',
        color_continuous_scale='Viridis',
        text=results_df['Silhouette'].round(3)
    )

    fig_weights.update_layout(
        height=500,
        xaxis_title="Feature Combination",
        yaxis_title="Silhouette Score",
        showlegend=False,
        margin=dict(t=0, l=0, r=0, b=0),
    )
    fig_weights.update_traces(textposition='outside')

    st.plotly_chart(fig_weights, use_container_width=True)

    best_combo = results_df.loc[results_df['Silhouette'].idxmax()]
    st.success(f"**Best combination:** {best_combo['Description']} (silhouette = {best_combo['Silhouette']:.3f})")

    # ====================
    # SECTION 9: Summary
    # ====================
    st.markdown("---")
    st.subheader("9. Summary")

    st.markdown(f"""
    ### Key Findings for {n_clusters} Clusters:

    1. **Agreement Level:** Audio and lyric clusters show {"LOW" if metrics['ari'] < 0.15 else "MODERATE" if metrics['ari'] < 0.35 else "HIGH"} agreement (ARI = {metrics['ari']:.3f})

    2. **Overlap Rate:** Only **{agreement_rate*100:.1f}%** of tracks stay in the same cluster number

    3. **Silhouette Comparison:**
       - Lyrics-only: **{metrics['sil_lyrics']:.3f}**
       - Audio-only: **{metrics['sil_audio']:.3f}**
       - {"Lyrics produce tighter clusters" if metrics['sil_lyrics'] > metrics['sil_audio'] else "Audio produces tighter clusters"}

    4. **Best Feature Mix:** {best_combo['Description']} achieves highest silhouette ({best_combo['Silhouette']:.3f})

    ### Interpretation:

    {"The low covariance between audio and lyric clusterings suggests that:" if metrics['ari'] < 0.15 else "The moderate/high covariance suggests some shared structure, but:"}

    - Songs that **sound similar** don't necessarily have **similar lyrics**
    - Lyrical themes **cross audio genre boundaries**
    - The 33-dim combined space captures **complementary, not redundant** information
    - Both feature types contribute unique signal to the final clustering
    """)

    # Download analysis data
    with st.expander("Download Analysis Data"):
        # Prepare downloadable data
        analysis_summary = {
            'n_clusters': n_clusters,
            'ari': metrics['ari'],
            'nmi': metrics['nmi'],
            'fmi': metrics['fmi'],
            'sil_audio': metrics['sil_audio'],
            'sil_lyrics': metrics['sil_lyrics'],
            'agreement_rate': agreement_rate,
            'n_tracks': len(df),
            'n_audio_features': len(audio_cols),
            'n_lyric_features': len(lyric_cols)
        }

        col1, col2 = st.columns(2)

        with col1:
            csv_contingency = pd.DataFrame(
                cont_matrix,
                index=[f"Audio_{i}" for i in range(n_clusters)],
                columns=[f"Lyric_{i}" for i in range(n_clusters)]
            ).to_csv()

            st.download_button(
                "Download Contingency Matrix (CSV)",
                csv_contingency,
                file_name=f"contingency_matrix_{n_clusters}clusters.csv",
                mime="text/csv"
            )

        with col2:
            csv_correlations = corr_df.to_csv(index=False)
            st.download_button(
                "Download Cross-Correlations (CSV)",
                csv_correlations,
                file_name="audio_lyric_correlations.csv",
                mime="text/csv"
            )

    # Chart export section for HTML exports
    render_export_section(default_dir="export/dimensions-of-taste-viz", section_key="audio_vs_lyrics")
