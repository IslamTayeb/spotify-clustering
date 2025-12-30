#!/usr/bin/env python3
"""
Validate Genre Fusion Implementation (Entropy-based)

This script validates the genre_fusion feature by:
1. Loading audio features from cache
2. Checking that genre_fusion exists for all tracks
3. Visualizing distribution of genre_fusion values
4. Showing most genre-pure and genre-fluid songs
5. Comparing with other audio features
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def load_audio_features(cache_path="analysis/cache/audio_features.pkl"):
    """Load audio features from cache."""
    if not Path(cache_path).exists():
        print(f"Error: Cache file not found: {cache_path}")
        return None

    with open(cache_path, "rb") as f:
        features = pickle.load(f)

    return features


def validate_genre_fusion(features):
    """Validate genre_fusion feature."""
    print("=" * 70)
    print("GENRE FUSION VALIDATION (Entropy-based)")
    print("=" * 70)

    # Check version
    version = features[0].get("genre_fusion_version", "unknown")
    print(f"\nVersion: {version}")

    # Check if genre_fusion exists
    has_fusion = [f.get("genre_fusion") is not None for f in features]
    coverage = sum(has_fusion) / len(features) * 100

    print(f"\n1. Coverage Check:")
    print(f"   Total tracks: {len(features)}")
    print(f"   Tracks with genre_fusion: {sum(has_fusion)} ({coverage:.1f}%)")

    if coverage < 100:
        print(
            f"   ⚠️  Warning: {len(features) - sum(has_fusion)} tracks missing genre_fusion"
        )

    # Extract genre_fusion values
    fusion_values = [
        f.get("genre_fusion") for f in features if f.get("genre_fusion") is not None
    ]

    if len(fusion_values) == 0:
        print("\n⚠️  No genre_fusion values found!")
        print(
            "   Run: python analysis/run_analysis.py --audio-embedding-backend interpretable --use-cache"
        )
        return None

    fusion_values = np.array(fusion_values)

    print(f"\n2. Distribution Statistics:")
    print(f"   Mean:   {np.mean(fusion_values):.3f}")
    print(f"   Median: {np.median(fusion_values):.3f}")
    print(f"   Std:    {np.std(fusion_values):.3f}")
    print(f"   Min:    {np.min(fusion_values):.3f}")
    print(f"   Max:    {np.max(fusion_values):.3f}")

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
                "genre_fusion": f.get("genre_fusion", 0.5),
            }
            for f in features
        ]
    )

    print("\n3. Most Genre-PURE Songs (Low Entropy → 0.0):")
    print("   These songs are clearly one genre - AI is confident.\n")
    pure = df.nsmallest(15, "genre_fusion")[
        ["track_name", "artist", "top_genre", "genre_fusion"]
    ]
    for _, row in pure.iterrows():
        print(
            f"   {row['track_name']:40s} | {row['top_genre']:35s} | {row['genre_fusion']:.3f}"
        )

    print("\n4. Most Genre-FLUID Songs (High Entropy → 1.0):")
    print("   These songs cross genres - AI is uncertain.\n")
    fluid = df.nlargest(15, "genre_fusion")[
        ["track_name", "artist", "top_genre", "genre_fusion"]
    ]
    for _, row in fluid.iterrows():
        print(
            f"   {row['track_name']:40s} | {row['top_genre']:35s} | {row['genre_fusion']:.3f}"
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
        count = sum(1 for v in fusion_values if lo <= v < hi)
        pct = count / len(fusion_values) * 100
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

    df = validate_genre_fusion(features)

    if df is not None:
        # Save summary
        output_path = Path("analysis/outputs/genre_fusion_validation.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("Genre Fusion Validation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Version: entropy-based\n")
            f.write(f"Total tracks: {len(features)}\n")
            fusion_values = [feat.get("genre_fusion", 0.5) for feat in features]
            f.write(f"Mean: {np.mean(fusion_values):.3f}\n")
            f.write(f"Std: {np.std(fusion_values):.3f}\n")
            f.write(f"\nInterpretation:\n")
            f.write("  0.0 = Pure genre (clearly one genre)\n")
            f.write("  1.0 = Genre fusion (crosses many genres)\n")

        print(f"\n✓ Summary saved to: {output_path}")


if __name__ == "__main__":
    main()
