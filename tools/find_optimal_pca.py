#!/usr/bin/env python3
"""
Find optimal PCA dimensions for 75% cumulative variance
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def find_pca_for_variance(features, target_variance=0.75, mode_name=""):
    """Find number of PCA components needed for target cumulative variance"""

    # Standardize features first
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    # Fit PCA with all possible components
    max_components = min(features_norm.shape[0], features_norm.shape[1])
    pca = PCA(n_components=max_components, random_state=42)
    pca.fit(features_norm)

    # Find cumulative variance
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find first component where we exceed target variance
    n_components_needed = np.argmax(cumsum_variance >= target_variance) + 1
    actual_variance = cumsum_variance[n_components_needed - 1]

    print(f"\n{mode_name}:")
    print(f"  Original shape: {features.shape}")
    print(f"  Max possible components: {max_components}")
    print(f"  Components for {target_variance*100}% variance: {n_components_needed}")
    print(f"  Actual variance at {n_components_needed} components: {actual_variance:.4f} ({actual_variance*100:.2f}%)")

    # Show variance at nearby component counts for context
    print(f"\n  Variance progression:")
    checkpoints = [50, 75, 100, 120, n_components_needed, 150, 165, 200, 250]
    checkpoints = sorted(set([c for c in checkpoints if c <= max_components]))

    for n in checkpoints:
        var = cumsum_variance[n-1]
        marker = " ← Current" if n == 120 else (" ← Target" if n == n_components_needed else "")
        print(f"    {n:3d} components: {var:.4f} ({var*100:.2f}%){marker}")

    return n_components_needed, actual_variance


def main():
    # Load cached features
    audio_path = Path("cache/audio_features.pkl")
    lyric_path = Path("cache/lyric_features.pkl")

    if not audio_path.exists():
        print("Error: cache/audio_features.pkl not found")
        print("Run: python run_analysis.py first to generate cache")
        return

    if not lyric_path.exists():
        print("Error: cache/lyric_features.pkl not found")
        print("Run: python run_analysis.py first to generate cache")
        return

    print("=" * 70)
    print("FINDING OPTIMAL PCA DIMENSIONS FOR 75% CUMULATIVE VARIANCE")
    print("=" * 70)

    with open(audio_path, 'rb') as f:
        audio_features = pickle.load(f)

    with open(lyric_path, 'rb') as f:
        lyric_features = pickle.load(f)

    # Extract embeddings
    audio_emb = np.vstack([f['embedding'] for f in audio_features])
    lyric_emb = np.vstack([f['embedding'] for f in lyric_features])

    # Filter lyrics - only include tracks with lyrics
    has_lyrics = np.array([f['has_lyrics'] for f in lyric_features])
    lyric_emb_with_lyrics = lyric_emb[has_lyrics]

    print(f"\nLoaded features:")
    print(f"  Audio: {len(audio_features)} tracks, embedding dim: {audio_emb.shape[1]}")
    print(f"  Lyrics: {len(lyric_features)} total ({has_lyrics.sum()} with lyrics), embedding dim: {lyric_emb.shape[1]}")

    # Find optimal components for each mode
    target_variance = 0.75

    # Audio-only mode
    audio_components, audio_var = find_pca_for_variance(
        audio_emb,
        target_variance,
        "AUDIO-ONLY MODE"
    )

    # Lyrics-only mode (only tracks with lyrics)
    lyric_components, lyric_var = find_pca_for_variance(
        lyric_emb_with_lyrics,
        target_variance,
        "LYRICS-ONLY MODE"
    )

    # Combined mode - analyze both separately
    print("\n" + "=" * 70)
    print("COMBINED MODE (both modalities analyzed separately)")
    print("=" * 70)

    print("\nAudio component of combined mode:")
    audio_combined_components, audio_combined_var = find_pca_for_variance(
        audio_emb,
        target_variance,
        "  Audio embeddings"
    )

    print("\nLyrics component of combined mode:")
    lyric_combined_components, lyric_combined_var = find_pca_for_variance(
        lyric_emb,
        target_variance,
        "  Lyric embeddings (all tracks)"
    )

    # Summary and recommendations
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    print(f"\nTo achieve {target_variance*100}% cumulative variance, use these n_pca_components values:")
    print(f"\n  Audio-only mode:    {audio_components} components")
    print(f"  Lyrics-only mode:   {lyric_components} components")
    print(f"  Combined mode:      {audio_combined_components} components (used for BOTH modalities)")
    print(f"                      (In combined mode, same n_pca is applied to audio & lyrics separately,")
    print(f"                       then concatenated. Audio will explain {audio_combined_var*100:.1f}%,")
    print(f"                       lyrics will explain {lyric_combined_var*100:.1f}%)")

    print("\nUpdate run_analysis.py line 157 with appropriate value based on mode.")
    print("For combined mode, use the higher of the two individual requirements.")

    # Suggest specific value for combined mode
    combined_suggestion = max(audio_combined_components, lyric_combined_components)
    print(f"\nSuggested value for combined mode: {combined_suggestion}")


if __name__ == "__main__":
    main()
