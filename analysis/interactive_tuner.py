import streamlit as st
import pickle
import numpy as np
import pandas as pd
import umap
import plotly.graph_objects as go
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, Birch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(layout="wide", page_title="Music Clustering Tuner")


@st.cache_data
def load_data():
    with open("../cache/audio_features.pkl", "rb") as f:
        audio_features = pickle.load(f)
    with open("../cache/lyric_features.pkl", "rb") as f:
        lyric_features = pickle.load(f)

    # Align features by track_id
    audio_by_id = {f["track_id"]: f for f in audio_features}
    lyric_by_id = {f["track_id"]: f for f in lyric_features}

    common_ids = set(audio_by_id.keys()) & set(lyric_by_id.keys())
    aligned_audio = [audio_by_id[tid] for tid in sorted(common_ids)]
    aligned_lyrics = [lyric_by_id[tid] for tid in sorted(common_ids)]

    return aligned_audio, aligned_lyrics


def prepare_features_for_mode(audio_features, lyric_features, mode, n_pca_components):
    """Prepare PCA-reduced features for clustering"""
    # For combined mode, filter out vocal songs without lyrics
    if mode == "combined":
        # Filter tracks where instrumentalness < 0.5 (vocal) but has_lyrics is False
        valid_mask = [
            not (audio.get("instrumentalness", 0.5) < 0.5 and not lyric.get("has_lyrics", False))
            for audio, lyric in zip(audio_features, lyric_features)
        ]
        audio_features = [f for f, valid in zip(audio_features, valid_mask) if valid]
        lyric_features = [f for f, valid in zip(lyric_features, valid_mask) if valid]

        filtered_count = sum(1 for v in valid_mask if not v)
        if filtered_count > 0:
            st.sidebar.info(f"ℹ️ Filtered out {filtered_count} vocal songs without lyrics in combined mode")

    audio_emb = np.vstack([f["embedding"] for f in audio_features])
    lyric_emb = np.vstack([f["embedding"] for f in lyric_features])

    if mode == "audio":
        # Standardize then PCA
        audio_norm = StandardScaler().fit_transform(audio_emb)
        n_components = min(audio_norm.shape[0], audio_norm.shape[1], n_pca_components)

        pca = PCA(n_components=n_components, random_state=42)
        features_reduced = pca.fit_transform(audio_norm)
        valid_indices = list(range(len(audio_features)))
        explained_var = np.sum(pca.explained_variance_ratio_)

    elif mode == "lyrics":
        has_lyrics = np.array([f["has_lyrics"] for f in lyric_features])
        valid_indices = np.where(has_lyrics)[0].tolist()

        # Standardize then PCA
        lyric_norm = StandardScaler().fit_transform(lyric_emb[has_lyrics])
        n_components = min(lyric_norm.shape[0], lyric_norm.shape[1], n_pca_components)

        pca = PCA(n_components=n_components, random_state=42)
        features_reduced = pca.fit_transform(lyric_norm)
        explained_var = np.sum(pca.explained_variance_ratio_)

    else:  # combined
        # Standardize first
        audio_norm = StandardScaler().fit_transform(audio_emb)
        lyric_norm = StandardScaler().fit_transform(lyric_emb)

        # PCA Reduction to balance modalities
        n_components_audio = min(
            audio_norm.shape[0], audio_norm.shape[1], n_pca_components
        )
        n_components_lyric = min(
            lyric_norm.shape[0], lyric_norm.shape[1], n_pca_components
        )

        pca_audio = PCA(n_components=n_components_audio, random_state=42)
        audio_reduced = pca_audio.fit_transform(audio_norm)

        pca_lyric = PCA(n_components=n_components_lyric, random_state=42)
        lyric_reduced = pca_lyric.fit_transform(lyric_norm)

        explained_var = (
            np.sum(pca_audio.explained_variance_ratio_)
            + np.sum(pca_lyric.explained_variance_ratio_)
        ) / 2

        # Combine the balanced, reduced vectors
        features_reduced = np.hstack([audio_reduced, lyric_reduced])
        valid_indices = list(range(len(audio_features)))

    return features_reduced, valid_indices, explained_var


def main():
    st.title("🎵 Music Clustering Tuner")
    st.markdown("""
    **Pipeline Architecture:**
    1. **Features → PCA** (dimensionality reduction, tunable below)
    2. **PCA Features → HAC/Birch/Spectral** (clustering, determines colors)
    3. **PCA Features → UMAP** (visualization only, determines 3D positions)
    """)

    # Load Data
    with st.spinner("Loading cached features..."):
        audio_features, lyric_features = load_data()

    st.sidebar.header("⚙️ Feature Preparation (PCA)")

    # Mode Selection
    mode = st.sidebar.selectbox("Feature Mode", ["combined", "audio", "lyrics"])

    # PCA Parameters - use mode-specific defaults for 75% variance
    pca_defaults = {
        "audio": 118,
        "lyrics": 162,
        "combined": 142
    }
    default_pca = pca_defaults.get(mode, 140)

    n_pca_components = st.sidebar.slider(
        "PCA Components",
        5,
        200,
        default_pca,
        step=5,
        help=f"Number of PCA components for dimensionality reduction before clustering. Default ({default_pca}) achieves ~75% variance for {mode} mode.",
    )

    # Prepare features
    with st.spinner("Preparing PCA-reduced features..."):
        pca_features, valid_indices, explained_var = prepare_features_for_mode(
            audio_features, lyric_features, mode, n_pca_components
        )

    st.sidebar.info(
        f"**PCA Features Shape:** {pca_features.shape}\n\n"
        f"**Explained Variance:** {explained_var:.2%}"
    )

    st.sidebar.header("🎯 Clustering Algorithm")

    # Clustering Algorithm Selection
    clustering_algorithm = st.sidebar.selectbox(
        "Algorithm",
        [
            "HAC (Hierarchical Agglomerative)",
            "Birch",
            "Spectral Clustering",
        ],
        help="Choose the clustering algorithm to use on PCA-reduced features.",
    )

    # Algorithm-specific parameters
    if clustering_algorithm == "HAC (Hierarchical Agglomerative)":
        st.sidebar.subheader("HAC Parameters")
        n_clusters_hac = st.sidebar.slider(
            "Number of Clusters",
            2,
            50,
            20,
            help="Fixed number of clusters to create.",
        )
        linkage_method = st.sidebar.selectbox(
            "Linkage Method",
            ["ward", "complete", "average", "single"],
            index=0,
            help="'ward' minimizes variance within clusters. 'complete' uses max distance between clusters. 'average' uses average distance. 'single' uses min distance.",
        )

        # Run HAC
        with st.spinner("Running Hierarchical Agglomerative Clustering..."):
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters_hac, linkage=linkage_method
            )
            labels = clusterer.fit_predict(pca_features)

    elif clustering_algorithm == "Birch":
        st.sidebar.subheader("Birch Parameters")
        st.sidebar.info(
            "⚡ Fast hierarchical clustering, similar to HAC but more efficient"
        )
        n_clusters_birch = st.sidebar.slider(
            "Number of Clusters",
            2,
            50,
            20,
            help="Final number of clusters to create.",
        )
        threshold = st.sidebar.slider(
            "Threshold",
            0.1,
            2.0,
            0.5,
            step=0.1,
            help="Radius of subcluster. Lower = more granular subclusters, might be slower.",
        )
        branching_factor = st.sidebar.slider(
            "Branching Factor",
            10,
            100,
            50,
            step=10,
            help="Max subclusters per node. Higher = more memory but better clustering.",
        )

        # Run Birch
        with st.spinner("Running Birch Clustering..."):
            clusterer = Birch(
                n_clusters=n_clusters_birch,
                threshold=threshold,
                branching_factor=branching_factor,
            )
            labels = clusterer.fit_predict(pca_features)

    elif clustering_algorithm == "Spectral Clustering":
        st.sidebar.subheader("Spectral Clustering Parameters")
        st.sidebar.info("🌐 Graph-based clustering, great for non-convex shapes")
        n_clusters_spectral = st.sidebar.slider(
            "Number of Clusters",
            2,
            50,
            20,
            help="Number of clusters to find.",
        )
        affinity = st.sidebar.selectbox(
            "Affinity",
            ["nearest_neighbors", "rbf"],
            index=0,
            help="'nearest_neighbors' uses k-NN graph. 'rbf' uses RBF kernel (slower).",
        )
        n_neighbors = st.sidebar.slider(
            "N Neighbors",
            5,
            50,
            15,
            help="Number of neighbors for k-NN graph (only used if affinity=nearest_neighbors).",
        )
        assign_labels = st.sidebar.selectbox(
            "Label Assignment",
            ["kmeans", "discretize"],
            index=0,
            help="'kmeans' is faster and usually better. 'discretize' is an alternative method.",
        )

        # Run Spectral Clustering
        with st.spinner("Running Spectral Clustering..."):
            clusterer = SpectralClustering(
                n_clusters=n_clusters_spectral,
                affinity=affinity,
                n_neighbors=n_neighbors if affinity == "nearest_neighbors" else 10,
                assign_labels=assign_labels,
                random_state=42,
            )
            labels = clusterer.fit_predict(pca_features)

    # Calculate metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = (labels == -1).sum()
    pct_outliers = (n_outliers / len(labels)) * 100

    # Silhouette score
    if len(set(labels)) > 1:
        sil_score = silhouette_score(pca_features, labels)
    else:
        sil_score = 0.0

    # Metrics Display
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clusters Found", n_clusters)
    col2.metric("Outliers", f"{n_outliers} ({pct_outliers:.1f}%)")
    col3.metric("Total Songs", len(labels))
    col4.metric("Silhouette Score", f"{sil_score:.3f}")

    st.sidebar.header("📊 UMAP Visualization")
    st.sidebar.markdown(
        "*These parameters only affect the plot layout, NOT clustering!*"
    )

    n_neighbors_viz = st.sidebar.slider(
        "n_neighbors (Viz)",
        5,
        100,
        20,
        help="Controls balance between local and global structure for the plot. Lower = more local detail, higher = more global overview.",
    )
    min_dist_viz = st.sidebar.slider(
        "min_dist (Viz)",
        0.0,
        1.0,
        0.2,
        step=0.01,
        help="How tightly points are packed in the visualization. Lower = tighter packing, clearer separation between visual clusters.",
    )

    # Run UMAP for visualization ONLY
    with st.spinner("Running UMAP for visualization..."):
        reducer_viz = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors_viz,
            min_dist=min_dist_viz,
            metric="cosine",
            random_state=42,
        )
        umap_coords = reducer_viz.fit_transform(pca_features)

    # Create DataFrame for Plotting
    plot_data = []
    for i, idx in enumerate(valid_indices):
        track = audio_features[idx]
        row_data = {
            "x": umap_coords[i, 0],
            "y": umap_coords[i, 1],
            "z": umap_coords[i, 2],
            "label": labels[i],
            "track_name": track["track_name"],
            "artist": track["artist"],
            "genre": track["top_3_genres"][0][0]
            if track["top_3_genres"]
            else "unknown",
            "mood_happy": track.get("mood_happy", 0.5),
            "mood_sad": track.get("mood_sad", 0.5),
            "mood_aggressive": track.get("mood_aggressive", 0.5),
            "mood_relaxed": track.get("mood_relaxed", 0.5),
            "mood_party": track.get("mood_party", 0.5),
            "valence": track.get("valence", 0.5),
            "arousal": track.get("arousal", 0.5),
            "engagement_score": track.get("engagement_score", 0.0),
            "approachability_score": track.get("approachability_score", 0.0),
            "danceability": track.get("danceability", 0.5),
            "instrumentalness": track.get("instrumentalness", 0.5),
            "bpm": track["bpm"],
            "key": track["key"],
        }

        plot_data.append(row_data)

    df = pd.DataFrame(plot_data)

    # Visualization
    fig = go.Figure()

    # Add clusters
    unique_labels = sorted(df["label"].unique())
    for label in unique_labels:
        cluster_points = df[df["label"] == label]
        if cluster_points.empty:
            continue

        if label == -1:
            name = f"Outliers ({len(cluster_points)})"
            color_val = "lightgrey"
            size = 3
            opacity = 0.3
        else:
            name = f"Cluster {label} ({len(cluster_points)})"
            color_val = label
            size = 4
            opacity = 0.8

        # Build hover text
        def build_hover_text(r):
            text = (
                f"<b>{r['track_name']}</b><br>"
                f"Artist: {r['artist']}<br>"
                f"Cluster: {r['label']}<br>"
                f"Genre: {r['genre']}<br>"
                f"BPM: {r['bpm']:.0f} | Key: {r['key']}<br>"
                f"Danceability: {r['danceability']:.2f}<br>"
                f"Instrumentalness: {r['instrumentalness']:.2f}<br>"
                f"Valence: {r['valence']:.2f} | Arousal: {r['arousal']:.2f}<br>"
                f"Engagement: {r['engagement_score']:.2f} | Approachability: {r['approachability_score']:.2f}<br>"
                f"Moods:<br>"
                f"- Happy: {r['mood_happy']:.2f}<br>"
                f"- Sad: {r['mood_sad']:.2f}<br>"
                f"- Aggressive: {r['mood_aggressive']:.2f}<br>"
                f"- Relaxed: {r['mood_relaxed']:.2f}<br>"
                f"- Party: {r['mood_party']:.2f}<br>"
            )
            return text

        fig.add_trace(
            go.Scatter3d(
                x=cluster_points["x"],
                y=cluster_points["y"],
                z=cluster_points["z"],
                mode="markers",
                name=name,
                marker=dict(
                    size=size,
                    color=color_val,
                    colorscale="Viridis",
                    opacity=opacity,
                ),
                text=cluster_points.apply(build_hover_text, axis=1),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        height=800,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        title=f"{clustering_algorithm} Clustering ({mode} mode) - UMAP Visualization",
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
