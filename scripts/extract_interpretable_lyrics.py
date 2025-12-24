#!/usr/bin/env python3
"""Extract interpretable lyric features using GPT-5-mini.

This script analyzes lyrics using an LLM to extract interpretable features like:
- Emotional valence and arousal
- Mood classification (happy, sad, aggressive, relaxed)
- Content analysis (explicit, narrative style)
- Theme classification (party, love, introspection, etc.)
- Language detection
- Vocabulary richness and repetition metrics

Requires OPENAI_API_KEY environment variable to be set.
Cost: ~$0.01-0.05 depending on library size.

Usage:
    python scripts/extract_interpretable_lyrics.py

The script will:
1. Load existing lyric features from cache/lyric_features.pkl
2. Load lyrics text files from lyrics/temp/
3. Call GPT-5-mini API to extract interpretable features
4. Merge features back into cache/lyric_features.pkl
"""

import os
import sys
import logging
import pickle
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from analysis.pipeline.lyric_features import batch_extract_interpretable_features

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Extract interpretable lyric features using GPT-5-mini."""

    print("=" * 60)
    print("INTERPRETABLE LYRICS EXTRACTION")
    print("=" * 60)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='sk-your-key'")
        return 1

    # Check for lyric cache
    lyric_cache = Path("cache/lyric_features.pkl")
    if not lyric_cache.exists():
        print(f"\n❌ ERROR: {lyric_cache} not found!")
        print("Run the main pipeline first: python run_analysis.py")
        return 1

    logger.info("Starting interpretable lyrics extraction")
    start_time = datetime.now()

    # =========================================================================
    # STEP 1: Load existing lyric features
    # =========================================================================
    print("\n[1/3] Loading existing lyric features...")
    with open(lyric_cache, "rb") as f:
        lyric_features = pickle.load(f)

    tracks_with_lyrics = [f for f in lyric_features if f.get("has_lyrics", False)]
    print(f"  Found {len(tracks_with_lyrics)} tracks with lyrics")
    logger.info(f"Loaded {len(tracks_with_lyrics)} tracks with lyrics")

    # =========================================================================
    # STEP 2: Load lyrics text files
    # =========================================================================
    print("\n[2/3] Loading lyrics text...")
    lyrics_dir = Path("lyrics/temp")

    if not lyrics_dir.exists():
        print(f"\n❌ ERROR: Lyrics directory not found: {lyrics_dir}")
        return 1

    tracks_for_extraction = []
    lyrics_texts = []

    for track in tracks_with_lyrics:
        filename = track.get("filename", "")
        if not filename:
            continue

        # Try multiple filename patterns
        lyric_file = lyrics_dir / filename.replace(".mp3", ".txt")
        if not lyric_file.exists():
            lyric_file = lyrics_dir / f"{track['track_name']}.txt"

        if lyric_file.exists():
            try:
                with open(lyric_file, "r", encoding="utf-8") as f:
                    lyrics_text = f.read().strip()

                # Only include if lyrics have substance
                if lyrics_text and len(lyrics_text) > 10:
                    tracks_for_extraction.append(track)
                    lyrics_texts.append(lyrics_text)
            except Exception as e:
                logger.warning(f"Could not read {lyric_file}: {e}")

    print(f"  Loaded {len(lyrics_texts)} lyrics files")
    logger.info(f"Loaded {len(lyrics_texts)} lyrics text files")

    if not lyrics_texts:
        print("\n❌ ERROR: No lyrics found!")
        print(f"Expected lyrics in: {lyrics_dir}")
        return 1

    # =========================================================================
    # STEP 3: Extract interpretable features via GPT-5-mini
    # =========================================================================
    print("\n[3/3] Extracting interpretable features via GPT-5-mini...")
    print("  (This will make API calls - costs ~$0.01-0.05 depending on library size)")
    print(f"  Processing {len(lyrics_texts)} tracks...")

    interpretable_features = batch_extract_interpretable_features(
        tracks_for_extraction,
        lyrics_texts,
        cache_path="cache/lyric_interpretable_features.pkl",
        use_cache=False,  # Force re-extraction (script is run manually)
        batch_delay=0.1,  # Small delay between API calls
    )

    print(f"\n✅ Extracted features for {len(interpretable_features)} tracks")
    logger.info(f"Extracted interpretable features for {len(interpretable_features)} tracks")

    # =========================================================================
    # STEP 4: Merge into lyric_features cache
    # =========================================================================
    print("\nMerging into lyric_features.pkl...")
    interpretable_by_id = {f["track_id"]: f for f in interpretable_features}

    # List of features to merge
    feature_keys = [
        "lyric_valence",
        "lyric_arousal",
        "lyric_mood_happy",
        "lyric_mood_sad",
        "lyric_mood_aggressive",
        "lyric_mood_relaxed",
        "lyric_explicit",
        "lyric_narrative",
        "lyric_theme",
        "lyric_language",
        "lyric_vocabulary_richness",
        "lyric_repetition",
    ]

    updated_count = 0
    for feature in lyric_features:
        track_id = feature["track_id"]

        if track_id in interpretable_by_id:
            # Merge extracted features
            interp = interpretable_by_id[track_id]
            for key in feature_keys:
                if key in interp:
                    feature[key] = interp[key]
            updated_count += 1
        else:
            # Set defaults for tracks without interpretable features
            feature.setdefault("lyric_valence", 0.5)
            feature.setdefault("lyric_arousal", 0.5)
            feature.setdefault("lyric_mood_happy", 0.0)
            feature.setdefault("lyric_mood_sad", 0.0)
            feature.setdefault("lyric_mood_aggressive", 0.0)
            feature.setdefault("lyric_mood_relaxed", 0.0)
            feature.setdefault("lyric_explicit", 0.0)
            feature.setdefault("lyric_narrative", 0.0)
            feature.setdefault("lyric_theme", "none")
            feature.setdefault("lyric_language", "none")
            feature.setdefault("lyric_vocabulary_richness", 0.0)
            feature.setdefault("lyric_repetition", 0.0)

    # Save updated cache
    with open(lyric_cache, "wb") as f:
        pickle.dump(lyric_features, f)

    elapsed_time = datetime.now() - start_time

    print(f"  Updated {updated_count} tracks")
    print(f"  Saved to: {lyric_cache}")
    logger.info(f"Updated {updated_count} tracks with interpretable features")

    print("\n" + "=" * 60)
    print("INTERPRETABLE LYRICS EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"\nTotal time: {elapsed_time}")
    print(f"Updated: {updated_count} tracks")
    print(f"Cache: {lyric_cache}")
    print("\nNext steps:")
    print("  1. Run: python run_analysis.py --backend interpretable")
    print("  2. Or run: streamlit run analysis/interactive_interpretability.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
