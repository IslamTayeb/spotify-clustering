"""Music Cluster Interpretability Dashboard - Unified Analysis Interface

This is the main entry point for interactive music taste analysis.
All features consolidated from interactive_tuner.py into this single dashboard.

NEW FEATURES:
- DBSCAN clustering algorithm
- Real-time metrics (silhouette score, outliers %)
- Cluster Inspector tab (filterable table)
- All static report analyses integrated interactively

REMOVED:
- Spotify player (not needed)
- Static HTML/markdown generation (use dashboard instead)
"""

import sys
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import streamlit as st
import pandas as pd

# Import component modules
from analysis.components.data import loaders, feature_prep, dataframe_builder
from analysis.components.clustering import algorithms, controls, metrics
from analysis.components.visualization import umap_3d
from analysis.components.widgets import feature_selectors, cluster_inspector
from analysis.components.export import spotify_export
from analysis.components.tabs import simplified_tabs

# Page configuration
st.set_page_config(
    page_title="Music Cluster Interpretability Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


def main():
    # Header
    st.markdown(
        '<div class="main-header">🎵 Music Cluster Interpretability Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    **Unified Analysis Interface** - Dynamic clustering with interpretable features

    - 🎯 **5 Clustering Algorithms**: HAC, Birch, Spectral, K-Means, DBSCAN
    - 🎨 **3 Feature Backends**: Essentia, MERT, Interpretable (29-dim)
    - 📊 **Real-time Metrics**: Silhouette score, outlier detection
    - 🗂️ **Interactive Tables**: Browse and filter tracks by cluster
    - 📤 **Spotify Export**: Create playlists from clustering results
    """)

    # Sidebar: Settings
    with st.sidebar:
        st.header("⚙️ Settings")

        # Data Source Selection
        data_source = st.radio(
            "Data Source",
            ["Static File (Pre-computed)", "Dynamic Tuning (Live)"],
            help="Load pre-computed results or run clustering live with parameter tuning",
        )

        df = None
        mode = "combined"

        if data_source == "Static File (Pre-computed)":
            # Load static file
            output_dir = Path("analysis/outputs")
            available_files = list(output_dir.glob("*.pkl"))
            file_options = [str(f) for f in available_files]

            if not file_options:
                st.error("No analysis data files found in analysis/outputs/")
                st.info("Run `python run_analysis.py` first to generate data")
                st.stop()

            default_file = "analysis/outputs/analysis_data.pkl"
            default_index = (
                file_options.index(default_file)
                if default_file in file_options
                else 0
            )

            data_file = st.selectbox(
                "Select Analysis Data",
                options=file_options,
                index=default_index,
                help="Choose the analysis results file to explore",
            )

            if data_file:
                try:
                    all_data = loaders.load_analysis_data(data_file)
                    st.success("✓ Data loaded successfully")

                    if "metadata" in all_data:
                        meta = all_data["metadata"]
                        st.info(f"Backend: {meta.get('audio_backend', 'unknown')}")

                    mode = st.selectbox("Clustering mode", ["combined", "audio", "lyrics"])
                    df = all_data[mode]["dataframe"].copy()

                except Exception as e:
                    st.error(f"Error loading data: {e}")

        else:  # Dynamic Tuning
            st.info("⚡ Live Tuning Mode")

            with st.spinner("Loading raw features..."):
                audio_features, lyric_features, mert_features = loaders.load_cached_features()

            if not audio_features:
                st.error("No cached features found! Run `python run_analysis.py` first.")
                st.stop()

            # Backend Selection
            has_mert = any(x is not None for x in mert_features)
            backend = feature_selectors.render_backend_selector(has_mert=has_mert)

            # Feature weights (only for interpretable mode)
            feature_weights = None
            if "Interpretable" in backend:
                st.sidebar.caption(
                    "Note: Lyric embeddings are NOT used. Only interpretable lyric features (valence, moods, themes, etc.)"
                )

                # Render weight sliders
                audio_weights = feature_selectors.render_audio_feature_weights()
                lyric_weights = feature_selectors.render_lyric_feature_weights()
                feature_weights = {**audio_weights, **lyric_weights}

            # Apply backend
            audio_features = feature_prep.apply_backend_overrides(
                audio_features,
                lyric_features,
                backend,
                mert_features=mert_features,
                feature_weights=feature_weights,
            )

            # Mode Selection
            mode = st.sidebar.selectbox("Feature Mode", ["combined", "audio", "lyrics"])

            # PCA Controls
            pca_config = feature_selectors.render_pca_controls(mode)

            # Clustering Controls
            st.sidebar.markdown("---")
            st.sidebar.header("🎯 Clustering")
            algo_name, algo_params = controls.render_clustering_controls()

            # UMAP Controls
            umap_config = feature_selectors.render_umap_controls()

            # Run Clustering Button
            if st.sidebar.button("▶️ Run Clustering", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    # 1. Prepare features
                    pca_feats, valid_idx, explained_var = feature_prep.prepare_features_for_mode(
                        audio_features,
                        lyric_features,
                        mode,
                        pca_config["n_pca_components"],
                        skip_pca=pca_config["skip_pca"],
                        interpretable_mode=("Interpretable" in backend),
                    )

                    st.sidebar.info(
                        f"**PCA Shape:** {pca_feats.shape}\\n\\n"
                        f"**Explained Variance:** {explained_var:.2%}"
                    )

                    # 2. Run clustering
                    labels = algorithms.run_clustering(algo_name, pca_feats, algo_params)

                    # 3. Display metrics
                    metrics.render_clustering_metrics(labels, pca_feats)

                    # 4. Compute UMAP for visualization
                    with st.spinner("Computing UMAP embedding..."):
                        coords = umap_3d.compute_umap_embedding(
                            pca_feats,
                            n_neighbors=umap_config["n_neighbors"],
                            min_dist=umap_config["min_dist"],
                        )

                    # 5. Build DataFrame
                    df = dataframe_builder.create_dataframe_from_clustering(
                        audio_features, valid_idx, labels, coords, lyric_features
                    )

                    # Store in session state
                    st.session_state["dynamic_df"] = df
                    st.session_state["algo_name"] = algo_name
                    st.session_state["mode"] = mode

                    st.success("✅ Clustering Complete!")

            # Retrieve from session state
            if "dynamic_df" in st.session_state:
                df = st.session_state["dynamic_df"]
                algo_name = st.session_state.get("algo_name", "Unknown")
                mode = st.session_state.get("mode", "combined")

        # Display Stats
        if df is not None:
            st.sidebar.markdown("---")
            st.sidebar.metric("Total Songs", len(df))
            st.sidebar.metric("Clusters", df["cluster"].nunique())
            st.sidebar.caption(f"Mode: {mode}")

            # Export Controls
            spotify_export.render_export_controls(df, mode)

    # Main Content
    if df is None:
        st.info("👈 Select a data source or run dynamic clustering to begin.")
        st.stop()

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 EDA Explorer",
        "🎯 Feature Importance",
        "⚖️ Cluster Comparison",
        "📝 Lyric Themes",
        "🔍 Overview",
        "🗂️ Cluster Inspector",  # NEW TAB
    ])

    with tab1:
        simplified_tabs.render_eda_explorer(df)

    with tab2:
        simplified_tabs.render_feature_importance(df)

    with tab3:
        simplified_tabs.render_cluster_comparison(df)

    with tab4:
        simplified_tabs.render_lyric_themes(df)

    with tab5:
        simplified_tabs.render_overview(df)

    with tab6:  # NEW CLUSTER INSPECTOR TAB
        st.header("🗂️ Cluster Inspector")
        st.write("Browse and filter tracks by cluster. Click a row to view details.")

        col1, col2 = st.columns([1, 3])

        with col1:
            # Filter control
            selected_cluster = cluster_inspector.render_cluster_filter(df)

        with col2:
            st.caption(f"Showing: {'All clusters' if selected_cluster is None else f'Cluster {selected_cluster}'}")

        # Table
        selected_track = cluster_inspector.render_cluster_table(df, selected_cluster)

        # Display selected track details
        if selected_track:
            cluster_inspector.render_track_details(selected_track)


if __name__ == "__main__":
    main()
