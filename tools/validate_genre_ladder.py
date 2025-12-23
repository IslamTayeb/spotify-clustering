#!/usr/bin/env python3
"""
Validate Genre Ladder Implementation (Entropy-based)

This script validates the genre_ladder feature by:
1. Loading audio features from cache
2. Checking that genre_ladder exists for all tracks
3. Visualizing distribution of genre_ladder values
4. Showing most genre-pure and genre-fluid songs
5. Comparing with other audio features
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def load_audio_features(cache_path="cache/audio_features.pkl"):
    """Load audio features from cache."""
    if not Path(cache_path).exists():
        print(f"Error: Cache file not found: {cache_path}")
        return None

    with open(cache_path, "rb") as f:
        features = pickle.load(f)

    return features


def validate_genre_ladder(features):
    """Validate genre_ladder feature."""
    print("=" * 70)
    print("GENRE LADDER VALIDATION (Entropy-based)")
    print("=" * 70)

    # Check version
    version = features[0].get("genre_ladder_version", "unknown")
    print(f"\nVersion: {version}")

    # Check if genre_ladder exists
    has_ladder = [f.get("genre_ladder") is not None for f in features]
    coverage = sum(has_ladder) / len(features) * 100

    print(f"\n1. Coverage Check:")
    print(f"   Total tracks: {len(features)}")
    print(f"   Tracks with genre_ladder: {sum(has_ladder)} ({coverage:.1f}%)")

    if coverage < 100:
        print(
            f"   ⚠️  Warning: {len(features) - sum(has_ladder)} tracks missing genre_ladder"
        )

    # Extract genre_ladder values
    ladder_values = [
        f.get("genre_ladder") for f in features if f.get("genre_ladder") is not None
    ]

    if len(ladder_values) == 0:
        print("\n⚠️  No genre_ladder values found!")
        print(
            "   Run: python run_analysis.py --audio-embedding-backend interpretable --use-cache"
        )
        return None

    ladder_values = np.array(ladder_values)

    print(f"\n2. Distribution Statistics:")
    print(f"   Mean:   {np.mean(ladder_values):.3f}")
    print(f"   Median: {np.median(ladder_values):.3f}")
    print(f"   Std:    {np.std(ladder_values):.3f}")
    print(f"   Min:    {np.min(ladder_values):.3f}")
    print(f"   Max:    {np.max(ladder_values):.3f}")

    # Build DataFrame
    df = pd.DataFrame(
        [
            {
                "track_name": f.get("track_name", "Unknown")[:40],
                "artist": f.get("artist", "Unknown")[:25],
                "top_genre": (
                    f.get("top_3_genres", [("Unknown", 0)])[0][0][:35]
                    if f.get("top_3_genres")
                    else "Unknown"
                ),
                "genre_ladder": f.get("genre_ladder", 0.5),
            }
            for f in features
        ]
    )

    print("\n3. Most Genre-PURE Songs (Low Entropy → 0.0):")
    print("   These songs are clearly one genre - AI is confident.\n")
    pure = df.nsmallest(15, "genre_ladder")[
        ["track_name", "artist", "top_genre", "genre_ladder"]
    ]
    for _, row in pure.iterrows():
        print(
            f"   {row['track_name']:40s} | {row['top_genre']:35s} | {row['genre_ladder']:.3f}"
        )

    print("\n4. Most Genre-FLUID Songs (High Entropy → 1.0):")
    print("   These songs cross genres - AI is uncertain.\n")
    fluid = df.nlargest(15, "genre_ladder")[
        ["track_name", "artist", "top_genre", "genre_ladder"]
    ]
    for _, row in fluid.iterrows():
        print(
            f"   {row['track_name']:40s} | {row['top_genre']:35s} | {row['genre_ladder']:.3f}"
        )

    print("\n5. Distribution by Bucket:")
    buckets = [
        (0.0, 0.2, "Very Pure"),
        (0.2, 0.4, "Pure"),
        (0.4, 0.6, "Mixed"),
        (0.6, 0.8, "Fusion"),
        (0.8, 1.0, "Very Fusion"),
    ]
    for lo, hi, label in buckets:
        count = sum(1 for v in ladder_values if lo <= v < hi)
        pct = count / len(ladder_values) * 100
        bar = "█" * int(pct / 2)
        print(f"   {label:12s} ({lo:.1f}-{hi:.1f}): {count:4d} ({pct:5.1f}%) {bar}")

    print("\n6. Interpretation Guide:")
    print("   0.0 = Pure genre (e.g., pure Trap, clear Cloud Rap)")
    print("   0.5 = Mixed (could be a few genres)")
    print("   1.0 = Genre fusion (crosses many genres, experimental)")

    return df


def main():
    print("\nLoading audio features...")
    features = load_audio_features()

    if features is None:
        return

    df = validate_genre_ladder(features)

    if df is not None:
        # Save summary
        output_path = Path("analysis/outputs/genre_ladder_validation.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("Genre Ladder Validation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Version: entropy-based\n")
            f.write(f"Total tracks: {len(features)}\n")
            ladder_values = [feat.get("genre_ladder", 0.5) for feat in features]
            f.write(f"Mean: {np.mean(ladder_values):.3f}\n")
            f.write(f"Std: {np.std(ladder_values):.3f}\n")
            f.write(f"\nInterpretation:\n")
            f.write("  0.0 = Pure genre (clearly one genre)\n")
            f.write("  1.0 = Genre fusion (crosses many genres)\n")

        print(f"\n✓ Summary saved to: {output_path}")


if __name__ == "__main__":
    main()
