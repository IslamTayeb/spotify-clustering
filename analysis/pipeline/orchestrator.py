#!/usr/bin/env python3
"""Pipeline orchestration for music taste analysis.

This module provides high-level orchestration functions that coordinate all steps
of the analysis pipeline: feature extraction, clustering, visualization, and reporting.

Used by run_analysis.py CLI for batch processing.
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from analysis.pipeline import config, feature_cache
from analysis.pipeline.interpretable_features import build_interpretable_features
from analysis.pipeline.clustering import run_clustering_pipeline

logger = logging.getLogger(__name__)


def run_full_pipeline(
    backend: str = "interpretable",
    fresh: bool = False,
    audio_only: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run the complete music analysis pipeline with smart caching.

    Args:
        backend: Feature backend ("essentia", "mert", "interpretable")
        fresh: If True, force re-extract all features (ignore cache)
        audio_only: If True, extract audio features only and exit early

    Returns:
        all_results dict with keys for each mode + metadata, or None if audio_only
    """
    logger.info(f"Starting pipeline: backend={backend}, fresh={fresh}, audio_only={audio_only}")

    # =========================================================================
    # STEP 1: Load/extract audio features
    # =========================================================================
    logger.info("[1/5] Loading audio features...")
    print("\n[1/5] Extracting audio features...")

    audio_features = feature_cache.load_audio_features(fresh=fresh)
    print(f"  ✓ Processed {len(audio_features)} songs (Essentia)")

    # Early exit if audio-only mode
    if audio_only:
        logger.info("Audio-only mode: exiting early")
        return None

    # =========================================================================
    # STEP 1.5: Load/extract embeddings for clustering (MERT or Interpretable)
    # =========================================================================
    audio_embeddings_for_clustering = None

    if backend == "mert":
        logger.info("[1.5/5] Loading MERT embeddings...")
        print("\n[1.5/5] Loading MERT audio embeddings...")

        audio_embeddings_for_clustering = feature_cache.load_mert_embeddings(
            cache_path=config.CACHE_PATHS["mert"],
            fresh=fresh
        )
        print(f"  ✓ Processed {len(audio_embeddings_for_clustering)} songs (MERT)")

    elif backend == "interpretable":
        logger.info("[1.5/5] Constructing interpretable features...")
        print("\n[1.5/5] Constructing interpretable features (Audio + Lyrics)...")

        # Load lyrics early for interpretable mode
        # Note: Always use E5 backend (best quality) for lyrics
        lyric_features_for_interp = feature_cache.load_lyric_features(
            backend="e5",
            fresh=fresh
        )

        # Build interpretable features (uses shared module!)
        audio_embeddings_for_clustering = build_interpretable_features(
            audio_features, lyric_features_for_interp
        )
        print(f"  ✓ Constructed interpretable vectors for {len(audio_embeddings_for_clustering)} songs (29 dims)")

    else:  # backend == "essentia"
        logger.info("Using Essentia embeddings for clustering (default)")

    # =========================================================================
    # STEP 2: Load/extract lyric features
    # =========================================================================
    logger.info("[2/5] Loading lyric features...")
    print("\n[2/5] Extracting lyric features...")

    # Always use E5 backend (best quality, as per simplified design)
    lyric_features = feature_cache.load_lyric_features(
        backend="e5",
        fresh=fresh
    )
    print(f"  ✓ Processed {len(lyric_features)} songs (E5)")

    # =========================================================================
    # STEP 3: Run clustering for all 3 modes (audio, lyrics, combined)
    # =========================================================================
    logger.info("[3/5] Running clustering pipeline...")
    print("\n[3/5] Running clustering pipeline...")

    # Always run all 3 modes (fast, allows comparison)
    modes = ["audio", "lyrics", "combined"]
    all_results = {}

    for mode in modes:
        print(f"\n  Running {mode} mode clustering...")
        logger.info(f"Clustering in {mode} mode")

        # Get PCA components for this mode from config
        n_pca = config.PCA_COMPONENTS_MAP.get(mode, 118)

        # For interpretable backend, skip PCA (already low-dim: 29)
        # For audio mode only, we may want to skip PCA to preserve interpretability
        skip_pca = (backend == "interpretable" and mode != "lyrics")

        # Pass embedding overrides if MERT/Interpretable selected
        lyric_embeddings_override = lyric_features if backend != "essentia" else None

        mode_results = run_clustering_pipeline(
            audio_features,
            lyric_features,
            mode=mode,
            audio_embeddings_override=audio_embeddings_for_clustering,
            lyric_embeddings_override=lyric_embeddings_override,
            n_pca_components=n_pca if not skip_pca else None,
            **config.DEFAULT_CLUSTERING_PARAMS,
            **config.DEFAULT_UMAP_PARAMS,
        )

        all_results[mode] = mode_results

        # Print summary stats
        n_outliers = mode_results['n_outliers']
        total_songs = len(mode_results['dataframe'])
        pct_outliers = (n_outliers / total_songs * 100) if total_songs > 0 else 0

        print(f"    ✓ Found {mode_results['n_clusters']} clusters")
        print(f"    ✓ Outliers: {n_outliers} ({pct_outliers:.1f}%)")
        print(f"    ✓ Silhouette: {mode_results['silhouette_score']:.3f}")

    # Add metadata
    all_results["metadata"] = {
        "audio_backend": backend,
        "lyrics_backend": "e5",  # Always E5 (as per simplified design)
        "timestamp": datetime.now().isoformat(),
        "mode": "combined",  # Always run all 3 modes
    }

    logger.info("Clustering complete for all modes")
    return all_results


# NOTE: generate_visualizations() and generate_reports() functions removed
# All visualization and analysis is now done through the interactive Streamlit dashboard:
#   streamlit run analysis/interactive_interpretability.py
#
# The CLI pipeline now only:
#   1. Extracts features (cached to cache/ directory)
#   2. Runs clustering
#   3. Saves analysis_data.pkl for dashboard loading
#
# No static HTML or markdown files are generated.
#
# NOTE: generate_reports() function also removed - static markdown reports are deprecated
# Use the interactive Streamlit dashboard for analysis instead:
#   streamlit run analysis/interactive_interpretability.py


def save_analysis_data(
    all_results: Dict[str, Any],
    output_path: str = "analysis/outputs/analysis_data.pkl"
):
    """Save serialized analysis results for dashboard loading.

    Args:
        all_results: Results dict from run_full_pipeline()
        output_path: Path to save pickle file
    """
    logger.info("Saving analysis data...")
    print("\nSaving analysis data...")

    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(all_results, f)

    print(f"  ✓ Saved to {output_path}")
    logger.info(f"Analysis data saved to {output_path}")
