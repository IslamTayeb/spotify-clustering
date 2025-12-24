#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def load_data():
    print("Loading cached features...")
    cache_dirs = [Path("cache"), Path("../cache")]
    cache_dir = next((d for d in cache_dirs if d.exists()), None)
    
    if not cache_dir:
        raise FileNotFoundError("Cache directory not found!")

    with open(cache_dir / "audio_features.pkl", "rb") as f:
        audio_features = pickle.load(f)
        
    return audio_features

def extract_interpretable_features(audio_features):
    print("Extracting interpretable features (BPM, Key, Moods)...")
    feature_vectors = []
    
    for track in audio_features:
        # 1. Scalar Features
        def get_float(k, d=0.0):
            v = track.get(k)
            if v is None: return d
            try: return float(v)
            except: return d
        
        scalars = [
            get_float("bpm", 120),
            get_float("danceability", 0.5),
            get_float("instrumentalness", 0.0),
            get_float("valence", 0.5),
            get_float("arousal", 0.5),
            get_float("engagement_score", 0.5),
            get_float("approachability_score", 0.5),
            get_float("mood_happy", 0.0),
            get_float("mood_sad", 0.0),
            get_float("mood_aggressive", 0.0),
            get_float("mood_relaxed", 0.0),
            get_float("mood_party", 0.0)
        ]
        
        # 2. Key Features
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
             if parts:
                 note = parts[0]
                 if note in pitch_map:
                     p = pitch_map[note]
                     # Map 0-11 to 0-2pi
                     sin_val = np.sin(2 * np.pi * p / 12)
                     cos_val = np.cos(2 * np.pi * p / 12)
                     key_vec = [sin_val, cos_val, scale_val]
        
        feature_vectors.append(scalars + key_vec)
    
    X = np.array(feature_vectors, dtype=np.float32)
    
    # IMPORTANT: Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

def analyze_clusters(X):
    k_range = range(2, 41)
    
    results = {
        'hac': {'sil': [], 'ch': []},
        'kmeans': {'sil': [], 'ch': [], 'inertia': []}
    }
    
    print("\nStarting parameter sweep (k=2..40)...")
    print(f"{ 'k':<5} | { 'HAC Sil':<10} | { 'KM Sil':<10} | { 'KM Inertia':<10}")
    print("-" * 45)
    
    for k in k_range:
        # HAC
        hac = AgglomerativeClustering(n_clusters=k)
        hac_labels = hac.fit_predict(X)
        hac_sil = silhouette_score(X, hac_labels)
        hac_ch = calinski_harabasz_score(X, hac_labels)
        
        results['hac']['sil'].append(hac_sil)
        results['hac']['ch'].append(hac_ch)
        
        # KMeans
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km_labels = km.fit_predict(X)
        km_sil = silhouette_score(X, km_labels)
        km_ch = calinski_harabasz_score(X, km_labels)
        
        results['kmeans']['sil'].append(km_sil)
        results['kmeans']['ch'].append(km_ch)
        results['kmeans']['inertia'].append(km.inertia_)
        
        print(f"{k:<5} | {hac_sil:.4f}     | {km_sil:.4f}     | {km.inertia_:.1f}")
        
    return k_range, results

def plot_results(k_range, results):
    plt.figure(figsize=(15, 10))
    
    # Silhouette
    plt.subplot(2, 2, 1)
    plt.plot(k_range, results['hac']['sil'], label='HAC', marker='o')
    plt.plot(k_range, results['kmeans']['sil'], label='KMeans', marker='x')
    plt.title('Silhouette Score (Higher is better)')
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calinski-Harabasz
    plt.subplot(2, 2, 2)
    plt.plot(k_range, results['hac']['ch'], label='HAC', marker='o')
    plt.plot(k_range, results['kmeans']['ch'], label='KMeans', marker='x')
    plt.title('Calinski-Harabasz Score (Higher is better)')
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Inertia (Elbow)
    plt.subplot(2, 2, 3)
    plt.plot(k_range, results['kmeans']['inertia'], marker='o', color='purple')
    plt.title('KMeans Inertia (Elbow Method)')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.grid(True, alpha=0.3)
    
    output_path = 'outputs/optimal_clusters_interpretable.png'
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

def print_top_configs(k_range, results):
    print("\n" + "="*50)
    print("TOP CONFIGURATIONS")
    print("="*50)
    
    # HAC Best
    hac_sil = np.array(results['hac']['sil'])
    top_3_hac = np.argsort(hac_sil)[::-1][:3]
    print("\nHAC Best by Silhouette:")
    for idx in top_3_hac:
        print(f"  k={k_range[idx]}: {hac_sil[idx]:.4f}")
        
    # KMeans Best
    km_sil = np.array(results['kmeans']['sil'])
    top_3_km = np.argsort(km_sil)[::-1][:3]
    print("\nKMeans Best by Silhouette:")
    for idx in top_3_km:
        print(f"  k={k_range[idx]}: {km_sil[idx]:.4f}")

def main():
    audio_features = load_data()
    X = extract_interpretable_features(audio_features)
    print(f"Feature matrix shape: {X.shape}")

    k_range, results = analyze_clusters(X)
    # Removed plot_results - use Streamlit dashboard instead
    print_top_configs(k_range, results)

if __name__ == "__main__":
    main()
