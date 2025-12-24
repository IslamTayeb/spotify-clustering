#!/usr/bin/env python3
"""Music Taste Analysis Pipeline - Ultra-Simplified Orchestrator

This script analyzes your music library by extracting audio and lyric features,
clustering similar songs, and generating interactive visualizations.

Simplified Design:
- 3 flags instead of 8 (--backend, --fresh, --audio-only)
- Smart caching (automatic - if cache exists, use it)
- Always uses best backends (E5 for lyrics)
- Always generates all 3 modes (audio, lyrics, combined) for comparison

Usage:
    python run_analysis.py                    # Smart cache, interpretable backend
    python run_analysis.py --backend mert     # Use MERT embeddings
    python run_analysis.py --fresh            # Force re-extract everything
    python run_analysis.py --audio-only       # Extract audio only (batch job)
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from analysis.pipeline import orchestrator

# Load environment variables
load_dotenv()


def setup_logging():
    """Configure logging to file and console."""
    log_dir = Path("logging")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"analysis_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


def main():
    """Run the music taste analysis pipeline."""

    parser = argparse.ArgumentParser(
        description="Analyze music taste using audio and lyric features",
        epilog="""
Examples:
  python run_analysis.py                    # Smart cache, interpretable backend
  python run_analysis.py --backend mert     # Use MERT embeddings
  python run_analysis.py --fresh            # Force re-extract everything
  python run_analysis.py --audio-only       # Extract audio only (batch job)

Simplified Design:
  - Smart caching: Automatically uses cache if it exists (no flag needed)
  - Best quality: Always uses E5 for lyric embeddings (highest quality)
  - Complete output: Always generates all 3 modes (audio, lyrics, combined)
  - For interactive exploration: Run streamlit run analysis/interactive_interpretability.py
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--backend",
        choices=["essentia", "mert", "interpretable"],
        default="interpretable",
        help="Feature backend (default: interpretable for best interpretability)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force re-extract all features (ignore cache)",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Extract audio features only, then exit (for batch jobs)",
    )

    args = parser.parse_args()
    logger = setup_logging()

    # Create output directories
    Path("analysis/outputs").mkdir(parents=True, exist_ok=True)
    Path("cache").mkdir(exist_ok=True)

    print("=" * 60)
    print("MUSIC TASTE ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Backend: {args.backend}")
    print(f"Mode: {'Fresh extraction' if args.fresh else 'Smart caching'}")
    print("=" * 60)

    logger.info(f"Starting analysis: backend={args.backend}, fresh={args.fresh}")
    start_time = datetime.now()

    # =========================================================================
    # Run pipeline (smart caching built-in)
    # =========================================================================
    all_results = orchestrator.run_full_pipeline(
        backend=args.backend,
        fresh=args.fresh,
        audio_only=args.audio_only,
    )

    # Early exit if audio-only mode
    if args.audio_only:
        print("\n" + "=" * 60)
        print("AUDIO EXTRACTION COMPLETE!")
        print("=" * 60)
        elapsed = datetime.now() - start_time
        print(f"\nTotal time: {elapsed}")
        print("\nNext: Run without --audio-only to cluster and visualize")
        logger.info(f"Audio-only mode complete. Time: {elapsed}")
        return

    # =========================================================================
    # Save analysis data for dashboard
    # =========================================================================
    orchestrator.save_analysis_data(all_results)

    # =========================================================================
    # Print summary
    # =========================================================================
    elapsed = datetime.now() - start_time
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nOutput:")
    print("  💾 analysis_data.pkl (saved for dashboard)")
    print(f"\nTotal time: {elapsed}")
    print("\nNext step:")
    print("  Run: streamlit run analysis/interactive_interpretability.py")
    print("\n  All visualizations, reports, and analysis are in the interactive dashboard.")

    logger.info(f"Analysis complete! Total time: {elapsed}")


if __name__ == "__main__":
    main()
