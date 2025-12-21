#!/usr/bin/env python3
"""
Tool to find optimal number of clusters for hierarchical clustering.
Uses multiple metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz, and Elbow method.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import Tuple, List

try:
    from kneed import KneeLocator
    HAS_KNEED = True
except ImportError:
    HAS_KNEED = False
    print("Warning: kneed package not found. Install with: pip install kneed")

def prepare_features(
    audio_features: List,
    lyric_features: List,
    mode: str = 'combined',
    n_pca_components: int = 50
) -> np.ndarray:
    """Prepare PCA-reduced features for clustering"""
    audio_emb = np.vstack([f['embedding'] for f in audio_features])
    lyric_emb = np.vstack([f['embedding'] for f in lyric_features])

    if mode == 'audio':
        audio_norm = StandardScaler().fit_transform(audio_emb)
        n_components = min(audio_norm.shape[0], audio_norm.shape[1], n_pca_components)
        pca = PCA(n_components=n_components, random_state=42)
        return pca.fit_transform(audio_norm)

    elif mode == 'lyrics':
        has_lyrics = np.array([f['has_lyrics'] for f in lyric_features])
        lyric_norm = StandardScaler().fit_transform(lyric_emb[has_lyrics])
        n_components = min(lyric_norm.shape[0], lyric_norm.shape[1], n_pca_components)
        pca = PCA(n_components=n_components, random_state=42)
        return pca.fit_transform(lyric_norm)

    else:  # combined
        audio_norm = StandardScaler().fit_transform(audio_emb)
        lyric_norm = StandardScaler().fit_transform(lyric_emb)

        n_components_audio = min(audio_norm.shape[0], audio_norm.shape[1], n_pca_components)
        n_components_lyric = min(lyric_norm.shape[0], lyric_norm.shape[1], n_pca_components)

        pca_audio = PCA(n_components=n_components_audio, random_state=42)
        audio_reduced = pca_audio.fit_transform(audio_norm)

        pca_lyric = PCA(n_components=n_components_lyric, random_state=42)
        lyric_reduced = pca_lyric.fit_transform(lyric_norm)

        return np.hstack([audio_reduced, lyric_reduced])


def calculate_metrics(features: np.ndarray, k_range: range, linkage_method: str = 'ward') -> dict:
    """Calculate clustering quality metrics for different numbers of clusters"""
    print(f"\nCalculating metrics for k = {k_range.start} to {k_range.stop-1}...")
    print(f"Linkage method: {linkage_method}")
    print("=" * 80)

    metrics = {
        'k_values': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': [],
        'inertia': []  # Within-cluster sum of squares
    }

    for k in k_range:
        print(f"Testing k={k}...", end=' ')

        clusterer = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        labels = clusterer.fit_predict(features)

        # Silhouette Score (higher is better, range: -1 to 1)
        sil_score = silhouette_score(features, labels)

        # Davies-Bouldin Index (lower is better, range: 0 to inf)
        db_score = davies_bouldin_score(features, labels)

        # Calinski-Harabasz Score (higher is better, range: 0 to inf)
        ch_score = calinski_harabasz_score(features, labels)

        # Within-cluster sum of squares (for elbow method)
        inertia = 0
        for cluster_id in range(k):
            cluster_points = features[labels == cluster_id]
            if len(cluster_points) > 0:
                centroid = cluster_points.mean(axis=0)
                inertia += np.sum((cluster_points - centroid) ** 2)

        metrics['k_values'].append(k)
        metrics['silhouette'].append(sil_score)
        metrics['davies_bouldin'].append(db_score)
        metrics['calinski_harabasz'].append(ch_score)
        metrics['inertia'].append(inertia)

        print(f"Sil={sil_score:.3f}, DB={db_score:.3f}, CH={ch_score:.0f}")

    return metrics


def plot_metrics(metrics: dict, output_path: str = 'outputs/optimal_clusters_analysis.png'):
    """Create a 4-subplot figure showing all metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Optimal Number of Clusters Analysis', fontsize=16, fontweight='bold')

    k_values = metrics['k_values']

    # Subplot 1: Silhouette Score (higher is better)
    axes[0, 0].plot(k_values, metrics['silhouette'], marker='o', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].set_title('Silhouette Score (Higher is Better)')
    axes[0, 0].grid(True, alpha=0.3)
    best_k_sil = k_values[np.argmax(metrics['silhouette'])]
    axes[0, 0].axvline(best_k_sil, color='red', linestyle='--', alpha=0.5, label=f'Best k={best_k_sil}')
    axes[0, 0].legend()

    # Subplot 2: Davies-Bouldin Index (lower is better)
    axes[0, 1].plot(k_values, metrics['davies_bouldin'], marker='o', linewidth=2, markersize=6, color='orange')
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Davies-Bouldin Index')
    axes[0, 1].set_title('Davies-Bouldin Index (Lower is Better)')
    axes[0, 1].grid(True, alpha=0.3)
    best_k_db = k_values[np.argmin(metrics['davies_bouldin'])]
    axes[0, 1].axvline(best_k_db, color='red', linestyle='--', alpha=0.5, label=f'Best k={best_k_db}')
    axes[0, 1].legend()

    # Subplot 3: Calinski-Harabasz Score (higher is better)
    axes[1, 0].plot(k_values, metrics['calinski_harabasz'], marker='o', linewidth=2, markersize=6, color='green')
    axes[1, 0].set_xlabel('Number of Clusters (k)')
    axes[1, 0].set_ylabel('Calinski-Harabasz Score')
    axes[1, 0].set_title('Calinski-Harabasz Score (Higher is Better)')
    axes[1, 0].grid(True, alpha=0.3)
    best_k_ch = k_values[np.argmax(metrics['calinski_harabasz'])]
    axes[1, 0].axvline(best_k_ch, color='red', linestyle='--', alpha=0.5, label=f'Best k={best_k_ch}')
    axes[1, 0].legend()

    # Subplot 4: Elbow Method (looking for elbow)
    axes[1, 1].plot(k_values, metrics['inertia'], marker='o', linewidth=2, markersize=6, color='purple')
    axes[1, 1].set_xlabel('Number of Clusters (k)')
    axes[1, 1].set_ylabel('Within-Cluster Sum of Squares')
    axes[1, 1].set_title('Elbow Method (Look for Elbow)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Create outputs directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    return fig


def find_elbow_point(k_values: list, inertias: list) -> tuple:
    """
    Find elbow point using Kneedle algorithm (if available) or angle method.
    Returns (elbow_k, method_used)
    """
    if HAS_KNEED:
        # Use Kneedle algorithm (more robust)
        kl = KneeLocator(
            k_values,
            inertias,
            curve='convex',
            direction='decreasing',
            S=1.0  # Sensitivity parameter (1.0 is default)
        )
        if kl.elbow is not None:
            return kl.elbow, 'Kneedle'
        else:
            print("Warning: Kneedle couldn't find an elbow, falling back to angle method")

    # Fallback: angle method
    # Normalize the data
    k_norm = np.array(k_values, dtype=float)
    k_norm = (k_norm - k_norm.min()) / (k_norm.max() - k_norm.min())

    inertia_norm = np.array(inertias, dtype=float)
    inertia_norm = (inertia_norm - inertia_norm.min()) / (inertia_norm.max() - inertia_norm.min())

    # Calculate angles
    angles = []
    for i in range(1, len(k_values) - 1):
        # Vector from point i-1 to i
        v1 = np.array([k_norm[i] - k_norm[i-1], inertia_norm[i] - inertia_norm[i-1]])
        # Vector from point i to i+1
        v2 = np.array([k_norm[i+1] - k_norm[i], inertia_norm[i+1] - inertia_norm[i]])

        # Calculate angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        angles.append(angle)

    # The elbow is where the angle is largest (sharpest turn)
    elbow_idx = np.argmax(angles) + 1  # +1 because we started from index 1
    return k_values[elbow_idx], 'Angle'


def print_recommendations(metrics: dict):
    """Print recommendations based on all metrics"""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    k_values = metrics['k_values']

    # Best k for each metric
    best_sil = k_values[np.argmax(metrics['silhouette'])]
    best_db = k_values[np.argmin(metrics['davies_bouldin'])]
    best_ch = k_values[np.argmax(metrics['calinski_harabasz'])]
    best_elbow, elbow_method = find_elbow_point(k_values, metrics['inertia'])

    print(f"\nBest k by Silhouette Score:        {best_sil} (score: {max(metrics['silhouette']):.3f})")
    print(f"Best k by Davies-Bouldin Index:    {best_db} (score: {min(metrics['davies_bouldin']):.3f})")
    print(f"Best k by Calinski-Harabasz Score: {best_ch} (score: {max(metrics['calinski_harabasz']):.0f})")
    print(f"Best k by Elbow Method ({elbow_method}):     {best_elbow}")

    # Find consensus
    votes = [best_sil, best_db, best_ch, best_elbow]
    unique, counts = np.unique(votes, return_counts=True)
    consensus_idx = np.argmax(counts)
    consensus_k = unique[consensus_idx]
    consensus_votes = counts[consensus_idx]

    print(f"\n{'='*80}")
    print(f"CONSENSUS RECOMMENDATION: k = {consensus_k}")
    print(f"  (Agreed upon by {consensus_votes}/4 metrics)")
    print(f"{'='*80}")

    # Show top 3 candidates overall
    print("\nTop 3 candidates (by average ranking):")
    rankings = {}
    for k in k_values:
        rank = 0
        rank += k_values.index(k) if k == best_sil else abs(k - best_sil)
        rank += k_values.index(k) if k == best_db else abs(k - best_db)
        rank += k_values.index(k) if k == best_ch else abs(k - best_ch)
        rank += k_values.index(k) if k == best_elbow else abs(k - best_elbow)
        rankings[k] = rank

    top_3 = sorted(rankings.items(), key=lambda x: x[1])[:3]
    for i, (k, _) in enumerate(top_3, 1):
        print(f"  {i}. k = {k}")


def main():
    # Load cached features
    print("Loading cached features...")
    with open('cache/audio_features.pkl', 'rb') as f:
        audio_features = pickle.load(f)
    with open('cache/lyric_features.pkl', 'rb') as f:
        lyric_features = pickle.load(f)

    # Align features by track_id
    audio_by_id = {f['track_id']: f for f in audio_features}
    lyric_by_id = {f['track_id']: f for f in lyric_features}

    common_ids = set(audio_by_id.keys()) & set(lyric_by_id.keys())
    aligned_audio = [audio_by_id[tid] for tid in sorted(common_ids)]
    aligned_lyrics = [lyric_by_id[tid] for tid in sorted(common_ids)]

    print(f"Found {len(common_ids)} tracks with both audio and lyric features")

    # Configuration
    mode = 'combined'  # 'audio', 'lyrics', or 'combined'
    n_pca_components = 50
    k_range = range(5, 51, 2)  # Test k from 5 to 50, step 2
    linkage_method = 'ward'  # 'ward', 'complete', 'average', 'single'

    # Prepare features
    print(f"\nPreparing {mode} features with {n_pca_components} PCA components...")
    features = prepare_features(aligned_audio, aligned_lyrics, mode, n_pca_components)
    print(f"Feature matrix shape: {features.shape}")

    # Calculate metrics
    metrics = calculate_metrics(features, k_range, linkage_method)

    # Plot results
    plot_metrics(metrics)

    # Print recommendations
    print_recommendations(metrics)


if __name__ == '__main__':
    main()
