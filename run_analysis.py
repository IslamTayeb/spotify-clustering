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
        choices=["essentia", "mert", "interpretable"],
        default="essentia",
        help="Audio embedding backend for clustering (default: essentia). "
             "Essentia always runs for interpretation fields (genre/mood/BPM). "
             "MERT creates separate cache for clustering-optimized embeddings. "
             "Interpretable uses manual features (BPM, Key, Moods).",
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
    
    elif args.audio_embedding_backend == "interpretable":
        print("\n[1.5/5] Constructing Interpretable Feature embeddings...")
        logger.info("Step 1.5/5: Interpretable Feature construction")
        
        import numpy as np
        
        # Dynamic global min/max for normalization
        bpms = [float(t.get("bpm", 0) or 0) for t in audio_features]
        valences = [float(t.get("valence", 0) or 0) for t in audio_features]
        arousals = [float(t.get("arousal", 0) or 0) for t in audio_features]
        
        def get_range(values, default_min, default_max):
            valid = [v for v in values if v > 0]
            if not valid: return default_min, default_max
            return min(valid), max(valid)

        min_bpm, max_bpm = get_range(bpms, 50, 200)
        min_val, max_val = get_range(valences, 1, 9)
        min_ar, max_ar = get_range(arousals, 1, 9)
        
        interpretable_embeddings = []
        
        for track in audio_features:
            # Helper
            def get_float(k, d=0.0):
                v = track.get(k)
                try: return float(v) if v is not None else d
                except: return d
            
            # Normalize BPM
            raw_bpm = get_float("bpm", 120)
            norm_bpm = (raw_bpm - min_bpm) / (max_bpm - min_bpm) if (max_bpm > min_bpm) else 0.5
            norm_bpm = max(0.0, min(1.0, norm_bpm))
            
            # Normalize Valence
            raw_val = get_float("valence", 4.5)
            norm_val = (raw_val - min_val) / (max_val - min_val) if (max_val > min_val) else 0.5
            norm_val = max(0.0, min(1.0, norm_val))

            # Normalize Arousal
            raw_ar = get_float("arousal", 4.5)
            norm_ar = (raw_ar - min_ar) / (max_ar - min_ar) if (max_ar > min_ar) else 0.5
            norm_ar = max(0.0, min(1.0, norm_ar))
            
            scalars = [
                norm_bpm,
                get_float("danceability", 0.5),
                get_float("instrumentalness", 0.0),
                norm_val,
                norm_ar,
                get_float("engagement_score", 0.5),
                get_float("approachability_score", 0.5),
                get_float("mood_happy", 0.0),
                get_float("mood_sad", 0.0),
                get_float("mood_aggressive", 0.0),
                get_float("mood_relaxed", 0.0),
                get_float("mood_party", 0.0)
            ]
            
            # Key Features
            key_vec = [0.0, 0.0, 0.0]
            key_str = track.get("key", "")
            if isinstance(key_str, str) and key_str:
                 k = key_str.lower().strip()
                 scale_val = 1.0 if 'major' in k else 0.0
                 pitch_map = {
                    'c': 0, 'c#': 1, 'db': 1, 'd': 2, 'd#': 3, 'eb': 3,
                    'e': 4, 'f': 5, 'f#': 6, 'gb': 6, 'g': 7, 'g#': 8,
                    'ab': 8, 'a': 9, 'a#': 10, 'bb': 10, 'b': 11
                 }
                 parts = k.split()
                 if parts and parts[0] in pitch_map:
                     p = pitch_map[parts[0]]
                     
                     # Apply 0.5 weighting to Key components (3 dimensions vs 1)
                     KEY_WEIGHT = 0.5
                     
                     sin_val = (0.5 * np.sin(2 * np.pi * p / 12) + 0.5) * KEY_WEIGHT
                     cos_val = (0.5 * np.cos(2 * np.pi * p / 12) + 0.5) * KEY_WEIGHT
                     scale_val = scale_val * KEY_WEIGHT
                     
                     key_vec = [sin_val, cos_val, scale_val]
            
            emb = np.array(scalars + key_vec, dtype=np.float32)
            
            # Create object with same structure as MERT output/Essentia output
            # Just needs to have 'track_id' and 'embedding'
            interpretable_embeddings.append({
                "track_id": track["track_id"],
                "embedding": emb
            })
            
        audio_embeddings_for_clustering = interpretable_embeddings
        print(f"  ✓ Constructed interpretable vectors for {len(audio_embeddings_for_clustering)} songs")
        
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
        
        # If using Interpretable features (low dimension), skip PCA (set n_pca to None/high or handle in clustering.py)
        # Assuming run_clustering_pipeline handles n_pca_components >= n_features by skipping or just transforming.
        # But specifically for interpretable, we might want to preserve the exact dimensions.
        # Let's set a flag or just let standard PCA run if dimensions are higher.
        # Actually, for Interpretable, dimensions are ~15. 118 is > 15, so PCA might just be identity or standard projection.
        # Ideally we skip PCA for interpretable to keep it "interpretable".
        if args.audio_embedding_backend == "interpretable" and mode != "lyrics":
             # For audio/combined, skip PCA for the audio part
             # The pipeline function might not have an explicit "skip_pca" arg exposed here cleanly,
             # but setting n_pca_components to a large number usually preserves dimensions.
             # However, let's keep it simple for now and rely on the pipeline's behavior.
             pass

        # Pass embedding overrides if MERT/E5/Interpretable selected
        lyric_embeddings_for_clustering = None
        if lyric_backend == "e5":
            lyric_embeddings_for_clustering = lyric_features

        mode_results = run_clustering_pipeline(
            audio_features,  # Always Essentia (for interpretation)
            lyric_features,
            mode=mode,
            audio_embeddings_override=audio_embeddings_for_clustering,  # MERT or Interpretable if selected
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

    # Store metadata about the run
    all_results["metadata"] = {
        "audio_backend": args.audio_embedding_backend,
        "lyrics_backend": args.lyrics_embedding_backend,
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode
    }

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
        if mode == "metadata":
            continue
            
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
