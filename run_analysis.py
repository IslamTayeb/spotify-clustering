#!/usr/bin/env python3
"""
Music Taste Analysis Pipeline
Run: python run_analysis.py [--use-cache]
"""

import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path

from analysis.pipeline.audio_analysis import (
    extract_audio_features,
    update_cached_features,
)
from analysis.pipeline.clustering import run_clustering_pipeline
from analysis.pipeline.lyric_analysis import extract_lyric_features
from analysis.pipeline.visualization import (
    create_interactive_map,
    create_combined_map,
    generate_report,
)


def setup_logging():
    log_dir = Path("logging")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"analysis_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


def main():
    parser = argparse.ArgumentParser(
        description="Analyze music taste using audio and lyric features"
    )
    parser.add_argument(
        "--use-cache", action="store_true", help="Use cached features if available"
    )
    parser.add_argument(
        "--re-embed-lyrics",
        action="store_true",
        help="Re-generate lyric embeddings while reusing cached audio features",
    )
    parser.add_argument(
        "--re-classify-audio",
        action="store_true",
        help="Run missing audio classifiers (valence, arousal, etc) on cached files",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Only extract/update audio features, then exit (skip lyrics, clustering, visualization)",
    )
    parser.add_argument(
        "--mode",
        choices=["audio", "lyrics", "combined"],
        default="combined",
        help="Clustering mode (default: combined)",
    )
    parser.add_argument(
        "--audio-embedding-backend",
        choices=["essentia", "mert"],
        default="essentia",
        help="Audio embedding backend for clustering (default: essentia). "
             "Essentia always runs for interpretation fields (genre/mood/BPM). "
             "MERT creates separate cache for clustering-optimized embeddings.",
    )
    parser.add_argument(
        "--lyrics-embedding-backend",
        choices=["bge-m3", "e5"],
        default="bge-m3",
        help="Lyrics embedding backend (default: bge-m3 for backward compatibility). "
             "E5 provides higher quality embeddings with separate cache.",
    )
    parser.add_argument(
        "--mert-cache-path",
        default="cache/mert_embeddings_24khz_30s_cls.pkl",
        help="Path to MERT embeddings cache (default: cache/mert_embeddings_24khz_30s_cls.pkl)",
    )
    args = parser.parse_args()

    logger = setup_logging()

    Path("analysis/outputs").mkdir(parents=True, exist_ok=True)
    Path("analysis/outputs/eda").mkdir(parents=True, exist_ok=True)
    Path("cache").mkdir(exist_ok=True)

    print("=" * 60)
    print("MUSIC TASTE ANALYSIS PIPELINE")
    print("=" * 60)

    logger.info(f"Starting analysis with mode: {args.mode}")
    logger.info(f"Use cache: {args.use_cache}")
    logger.info(f"Re-embed lyrics: {args.re_embed_lyrics}")
    logger.info(f"Re-classify audio: {args.re_classify_audio}")
    start_time = datetime.now()

    print("\n[1/5] Extracting audio features...")
    logger.info("Step 1/5: Audio feature extraction")

    if args.re_classify_audio and Path("cache/audio_features.pkl").exists():
        print("  Updating cached audio features with new classifiers...")
        logger.info("Updating cached audio features")
        audio_features = update_cached_features()
    elif (args.use_cache or args.re_embed_lyrics) and Path(
        "cache/audio_features.pkl"
    ).exists():
        print("  Loading from cache...")
        logger.info("Loading audio features from cache")
        with open("cache/audio_features.pkl", "rb") as f:
            audio_features = pickle.load(f)
    else:
        logger.info("Extracting audio features from MP3 files")
        audio_features = extract_audio_features()

    print(f"  ✓ Processed {len(audio_features)} songs (Essentia)")
    logger.info(f"Audio features extracted: {len(audio_features)} songs")

    # Step 1.5: Extract MERT embeddings if selected (separate cache for clustering)
    audio_embeddings_for_clustering = None
    if args.audio_embedding_backend == "mert":
        print("\n[1.5/5] Extracting MERT audio embeddings for clustering...")
        logger.info("Step 1.5/5: MERT audio embedding extraction")

        from analysis.pipeline.mert_embedding import extract_mert_embeddings

        if args.use_cache and Path(args.mert_cache_path).exists():
            print(f"  Loading MERT embeddings from cache...")
            with open(args.mert_cache_path, "rb") as f:
                audio_embeddings_for_clustering = pickle.load(f)
        else:
            print(f"  Extracting MERT embeddings...")
            audio_embeddings_for_clustering = extract_mert_embeddings(
                cache_path=args.mert_cache_path,
                use_cache=args.use_cache
            )

        print(f"  ✓ Processed {len(audio_embeddings_for_clustering)} songs (MERT)")
        logger.info(f"MERT embeddings extracted: {len(audio_embeddings_for_clustering)} songs")
    else:
        logger.info("Using Essentia embeddings for clustering (default)")

    # Early exit if only audio processing was requested
    if args.audio_only:
        print("\n" + "=" * 60)
        print("AUDIO EXTRACTION COMPLETE!")
        print("=" * 60)
        print(f"\nProcessed {len(audio_features)} songs")
        print("Cache updated: cache/audio_features.pkl")
        print("\nRun 'python tools/verify_cache.py' to verify the features.")
        elapsed_time = datetime.now() - start_time
        print(f"\nTotal time: {elapsed_time}")
        logger.info(f"Audio-only mode complete. Total time: {elapsed_time}")
        return

    print("\n[2/5] Extracting lyric features...")
    logger.info("Step 2/5: Lyric feature extraction")

    # Determine lyric cache path based on backend
    lyric_backend = args.lyrics_embedding_backend
    if lyric_backend == "bge-m3":
        lyric_cache = "cache/lyric_features.pkl"  # Preserve existing name for default
    elif lyric_backend == "e5":
        lyric_cache = "cache/lyric_features_e5.pkl"
    else:
        lyric_cache = f"cache/lyric_features_{lyric_backend}.pkl"

    if (
        args.use_cache
        and not args.re_embed_lyrics
        and Path(lyric_cache).exists()
    ):
        print(f"  Loading from cache ({lyric_backend})...")
        logger.info(f"Loading lyric features from cache ({lyric_backend})")
        with open(lyric_cache, "rb") as f:
            lyric_features = pickle.load(f)
    else:
        logger.info(f"Extracting lyric features using {lyric_backend}")
        lyric_features = extract_lyric_features(
            backend=lyric_backend,
            cache_path=lyric_cache
        )

    print(f"  ✓ Processed {len(lyric_features)} songs ({lyric_backend})")
    logger.info(f"Lyric features extracted: {len(lyric_features)} songs")

    print("\n[3/5] Running clustering pipeline...")
    logger.info("Step 3/5: Clustering")

    # Run clustering for all 3 modes to create separate visualizations
    modes = ["audio", "lyrics", "combined"] if args.mode == "combined" else [args.mode]
    all_results = {}

    for mode in modes:
        print(f"\n  Running {mode} mode clustering...")
        logger.info(f"Running clustering in {mode} mode")

        # Use mode-specific PCA components for 75% cumulative variance
        # Determined via tools/find_optimal_pca.py
        pca_components_map = {
            "audio": 118,    # 75.01% variance
            "lyrics": 162,   # 75.02% variance
            "combined": 142  # 75.04% variance (audio) + 75.04% variance (lyrics)
        }
        n_pca = pca_components_map[mode]

        # Pass embedding overrides if MERT/E5 selected
        lyric_embeddings_for_clustering = None
        if lyric_backend == "e5":
            lyric_embeddings_for_clustering = lyric_features

        mode_results = run_clustering_pipeline(
            audio_features,  # Always Essentia (for interpretation)
            lyric_features,
            mode=mode,
            audio_embeddings_override=audio_embeddings_for_clustering,  # MERT if selected
            lyric_embeddings_override=lyric_embeddings_for_clustering,  # E5 if selected
            n_pca_components=n_pca,
            clustering_algorithm="hac",
            n_clusters_hac=5,
            linkage_method="ward",
            umap_n_neighbors=20,
            umap_min_dist=0.2,
            umap_n_components=3,
        )

        all_results[mode] = mode_results

        print(f"    ✓ Found {mode_results['n_clusters']} clusters")
        print(
            f"    ✓ Outliers: {mode_results['n_outliers']} songs ({mode_results['n_outliers'] / len(mode_results['dataframe']) * 100:.1f}%)"
        )
        print(f"    ✓ Silhouette score: {mode_results['silhouette_score']:.3f}")
        logger.info(
            f"{mode} mode: {mode_results['n_clusters']} clusters, {mode_results['n_outliers']} outliers, silhouette={mode_results['silhouette_score']:.3f}"
        )

    # Use combined mode for the main report (or the only mode if not combined)
    results = all_results.get("combined", all_results[args.mode])

    print("\n[4/5] Generating interactive visualizations...")
    logger.info("Step 4/5: Generating visualizations")

    # Determine output file suffix based on backends
    backend_suffix = ""
    if args.audio_embedding_backend == "mert" or args.lyrics_embedding_backend == "e5":
        audio_suffix = "_mert" if args.audio_embedding_backend == "mert" else ""
        lyrics_suffix = "_e5" if args.lyrics_embedding_backend == "e5" else ""
        backend_suffix = f"{audio_suffix}{lyrics_suffix}"

    # Create combined visualization with all 3 modes
    if len(all_results) == 3:
        print("  Creating combined visualization with all 3 modes...")
        combined_fig = create_combined_map(all_results)
        combined_output = f"analysis/outputs/music_taste_map_combined_comparison{backend_suffix}.html"
        combined_fig.write_html(
            combined_output,
            config={"displayModeBar": True, "displaylogo": False},
            include_plotlyjs="cdn",
        )
        print(f"  ✓ Saved combined comparison to {combined_output}")
        logger.info("Combined visualization saved")

    # Also create individual HTMLs for detailed exploration
    for mode, mode_results in all_results.items():
        output_file = f"analysis/outputs/music_taste_map_{mode}{backend_suffix}.html"
        fig = create_interactive_map(mode_results["dataframe"], mode_results)
        fig.write_html(
            output_file,
            config={"displayModeBar": True, "displaylogo": False},
            include_plotlyjs="cdn",
        )
        print(f"  ✓ Saved {mode} mode to {output_file}")
        logger.info(f"Visualization saved to {output_file}")

    print("\n[5/5] Generating report...")
    logger.info("Step 5/5: Generating report")
    generate_report(results, output_dir="analysis/outputs")
    print("  ✓ Report saved to analysis/outputs/music_taste_report.md")
    print("  ✓ Outliers saved to analysis/outputs/outliers.txt")
    logger.info("Report saved to outputs/music_taste_report.md")

    with open("analysis/outputs/analysis_data.pkl", "wb") as f:
        pickle.dump(all_results, f)
    logger.info("Analysis data saved to analysis/outputs/analysis_data.pkl")

    elapsed_time = datetime.now() - start_time

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nOutputs:")
    if args.mode == "combined":
        print("  📊 COMBINED VISUALIZATION (recommended):")
        print("     - analysis/outputs/music_taste_map_combined_comparison.html")
        print("       (side-by-side comparison of all 3 modes)")
        print("\n  📈 INDIVIDUAL VISUALIZATIONS (detailed exploration):")
        print("     - analysis/outputs/music_taste_map_audio.html")
        print("     - analysis/outputs/music_taste_map_lyrics.html")
        print("     - analysis/outputs/music_taste_map_combined.html")
    else:
        print(
            f"  - analysis/outputs/music_taste_map_{args.mode}.html (interactive visualization)"
        )
    print("\n  📝 REPORTS:")
    print("     - analysis/outputs/music_taste_report.md (detailed cluster analysis)")
    print("     - analysis/outputs/outliers.txt (unclustered songs)")
    print("     - analysis/outputs/analysis_data.pkl (serialized results)")
    print(f"\nTotal time: {elapsed_time}")
    print("\nNext steps:")
    if args.mode == "combined":
        print("  1. Open analysis/outputs/music_taste_map_combined_comparison.html")
        print("     to see all 3 modes side-by-side")
        print("  2. Read analysis/outputs/music_taste_report.md for insights")
        print("  3. Explore individual mode HTMLs for detailed views")
    else:
        print("  1. Open the HTML visualization in your browser")
        print("  2. Read analysis/outputs/music_taste_report.md for insights")
        print("  3. Check analysis/outputs/outliers.txt for unique songs")

    logger.info(f"Analysis complete! Total time: {elapsed_time}")
    logger.info("All outputs saved successfully")


if __name__ == "__main__":
    main()
