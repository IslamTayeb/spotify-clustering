#!/usr/bin/env python3
"""
Quick script to analyze cached feature vectors for clustering algorithm recommendations
"""
import pickle
import numpy as np
from pathlib import Path

def load_and_analyze():
    cache_dir = Path("cache")

    # Load audio features
    with open(cache_dir / "audio_features.pkl", "rb") as f:
        audio_data = pickle.load(f)

    # Load lyric features
    with open(cache_dir / "lyric_features.pkl", "rb") as f:
        lyric_data = pickle.load(f)

    print("=" * 80)
    print("AUDIO FEATURES ANALYSIS")
    print("=" * 80)
    print(f"Type: {type(audio_data)}")
    if isinstance(audio_data, dict):
        print(f"Keys: {list(audio_data.keys())[:5]}...")
        print(f"Number of tracks: {len(audio_data)}")
        # Get sample feature vector
        sample_key = list(audio_data.keys())[0]
        sample_features = audio_data[sample_key]
        if isinstance(sample_features, dict):
            print(f"Feature dict keys: {sample_features.keys()}")
            for key, val in sample_features.items():
                if isinstance(val, np.ndarray):
                    print(f"  {key}: shape {val.shape}, dtype {val.dtype}")
        else:
            print(f"Sample feature shape: {sample_features.shape}")
            print(f"Sample feature dtype: {sample_features.dtype}")
    elif isinstance(audio_data, np.ndarray):
        print(f"Shape: {audio_data.shape}")
        print(f"Dtype: {audio_data.dtype}")
    elif isinstance(audio_data, list):
        print(f"Number of tracks: {len(audio_data)}")
        if len(audio_data) > 0:
            sample = audio_data[0]
            print(f"Sample type: {type(sample)}")
            if isinstance(sample, dict):
                print(f"Sample keys: {sample.keys()}")
            elif isinstance(sample, (list, np.ndarray)):
                print(f"Sample shape/length: {np.array(sample).shape}")

    print("\n" + "=" * 80)
    print("LYRIC FEATURES ANALYSIS")
    print("=" * 80)
    print(f"Type: {type(lyric_data)}")
    if isinstance(lyric_data, dict):
        print(f"Keys: {list(lyric_data.keys())[:5]}...")
        print(f"Number of tracks: {len(lyric_data)}")
        sample_key = list(lyric_data.keys())[0]
        sample_features = lyric_data[sample_key]
        if isinstance(sample_features, dict):
            print(f"Feature dict keys: {sample_features.keys()}")
        else:
            print(f"Sample feature shape: {sample_features.shape}")
            print(f"Sample feature dtype: {sample_features.dtype}")
    elif isinstance(lyric_data, np.ndarray):
        print(f"Shape: {lyric_data.shape}")
        print(f"Dtype: {lyric_data.dtype}")
    elif isinstance(lyric_data, list):
        print(f"Number of tracks: {len(lyric_data)}")
        if len(lyric_data) > 0:
            sample = lyric_data[0]
            print(f"Sample type: {type(sample)}")
            if isinstance(sample, dict):
                print(f"Sample keys: {sample.keys()}")
            elif isinstance(sample, (list, np.ndarray)):
                print(f"Sample shape/length: {np.array(sample).shape}")

    print("\n" + "=" * 80)
    print("COMBINED STATISTICS")
    print("=" * 80)

    # Try to extract feature matrices
    if isinstance(audio_data, dict):
        audio_matrix = np.array([v if isinstance(v, np.ndarray) else np.array(v)
                                 for v in audio_data.values()])
    elif isinstance(audio_data, list):
        # Extract embeddings from list of dicts
        if len(audio_data) > 0 and isinstance(audio_data[0], dict):
            audio_matrix = np.array([item['embedding'] for item in audio_data])
        else:
            audio_matrix = np.array(audio_data)
    else:
        audio_matrix = audio_data

    if isinstance(lyric_data, dict):
        lyric_matrix = np.array([v if isinstance(v, np.ndarray) else np.array(v)
                                for v in lyric_data.values()])
    elif isinstance(lyric_data, list):
        # Extract embeddings from list of dicts
        if len(lyric_data) > 0 and isinstance(lyric_data[0], dict):
            lyric_matrix = np.array([item['embedding'] for item in lyric_data])
        else:
            lyric_matrix = np.array(lyric_data)
    else:
        lyric_matrix = lyric_data

    print(f"Audio matrix shape: {audio_matrix.shape}")
    print(f"Lyric matrix shape: {lyric_matrix.shape}")

    print(f"\nAudio features:")
    print(f"  Mean: {np.mean(audio_matrix):.4f}")
    print(f"  Std: {np.std(audio_matrix):.4f}")
    print(f"  Min: {np.min(audio_matrix):.4f}")
    print(f"  Max: {np.max(audio_matrix):.4f}")

    print(f"\nLyric features:")
    print(f"  Mean: {np.mean(lyric_matrix):.4f}")
    print(f"  Std: {np.std(lyric_matrix):.4f}")
    print(f"  Min: {np.min(lyric_matrix):.4f}")
    print(f"  Max: {np.max(lyric_matrix):.4f}")

    # Distance analysis on a sample
    sample_size = min(100, len(audio_matrix))
    sample_audio = audio_matrix[:sample_size]

    from scipy.spatial.distance import pdist
    distances = pdist(sample_audio, metric='euclidean')

    print(f"\nPairwise distances (sample of {sample_size} tracks):")
    print(f"  Mean distance: {np.mean(distances):.4f}")
    print(f"  Std distance: {np.std(distances):.4f}")
    print(f"  Min distance: {np.min(distances):.4f}")
    print(f"  Max distance: {np.max(distances):.4f}")

if __name__ == "__main__":
    load_and_analyze()
