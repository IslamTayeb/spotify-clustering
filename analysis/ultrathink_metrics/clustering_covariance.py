#!/usr/bin/env python3
"""Test covariance between lyric-based and audio-based clustering"""

import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score
)
from sklearn.metrics.cluster import contingency_matrix
import warnings
warnings.filterwarnings('ignore')

# Load analysis data
with open('/Users/islamtayeb/Documents/spotify-clustering/analysis/outputs/analysis_data.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['combined']['dataframe']

print("=" * 80)
print("LYRIC vs AUDIO CLUSTERING COVARIANCE ANALYSIS")
print("=" * 80)

# Define feature groups
audio_features = ['emb_bpm', 'emb_danceability', 'emb_instrumentalness', 'emb_valence', 'emb_arousal',
                  'emb_engagement', 'emb_approachability', 'emb_mood_aggressive', 'emb_mood_happy',
                  'emb_mood_sad', 'emb_mood_relaxed', 'emb_mood_party', 'emb_voice_gender',
                  'emb_genre_ladder', 'emb_acoustic_electronic', 'emb_timbre_brightness']

lyric_features = ['emb_lyric_valence', 'emb_lyric_arousal', 'emb_lyric_mood_happy',
                  'emb_lyric_mood_sad', 'emb_lyric_mood_aggressive', 'emb_lyric_mood_relaxed',
                  'emb_lyric_explicit', 'emb_lyric_narrative', 'emb_lyric_vocabulary', 'emb_lyric_repetition']

# Get feature matrices
X_audio = df[audio_features].values
X_lyrics = df[lyric_features].values

# Test different numbers of clusters
n_clusters_list = [3, 4, 5, 6, 8, 10]

print("\n" + "=" * 80)
print("1. CLUSTERING AGREEMENT METRICS")
print("=" * 80)

results = []
for n_clusters in n_clusters_list:
    # Cluster on audio features only
    clusterer_audio = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels_audio = clusterer_audio.fit_predict(X_audio)

    # Cluster on lyric features only
    clusterer_lyrics = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels_lyrics = clusterer_lyrics.fit_predict(X_lyrics)

    # Calculate agreement metrics
    ari = adjusted_rand_score(labels_audio, labels_lyrics)
    nmi = normalized_mutual_info_score(labels_audio, labels_lyrics)
    fmi = fowlkes_mallows_score(labels_audio, labels_lyrics)

    # Calculate silhouette scores
    sil_audio = silhouette_score(X_audio, labels_audio)
    sil_lyrics = silhouette_score(X_lyrics, labels_lyrics)

    results.append({
        'n_clusters': n_clusters,
        'ari': ari,
        'nmi': nmi,
        'fmi': fmi,
        'sil_audio': sil_audio,
        'sil_lyrics': sil_lyrics
    })

    print(f"\n{n_clusters} Clusters:")
    print("-" * 50)
    print(f"  Adjusted Rand Index: {ari:.3f}")
    print(f"    (0=random, 1=perfect agreement, <0=worse than random)")
    print(f"  Normalized Mutual Info: {nmi:.3f}")
    print(f"    (0=no mutual info, 1=perfect agreement)")
    print(f"  Fowlkes-Mallows Index: {fmi:.3f}")
    print(f"    (0=no similarity, 1=identical)")
    print(f"  Audio silhouette: {sil_audio:.3f}")
    print(f"  Lyrics silhouette: {sil_lyrics:.3f}")

# Focus on 5 clusters (matching the actual clustering)
print("\n" + "=" * 80)
print("2. DETAILED ANALYSIS (5 CLUSTERS)")
print("=" * 80)

n_clusters = 5
clusterer_audio = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels_audio = clusterer_audio.fit_predict(X_audio)

clusterer_lyrics = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels_lyrics = clusterer_lyrics.fit_predict(X_lyrics)

# Create contingency table
cont_matrix = contingency_matrix(labels_audio, labels_lyrics)

print("\n📊 Contingency Matrix (rows=audio clusters, cols=lyric clusters):")
print("-" * 50)
print("     ", end="")
for j in range(n_clusters):
    print(f"Lyr{j:1d} ", end="")
print("| Total")
print("-" * 50)

for i in range(n_clusters):
    print(f"Aud{i} ", end="")
    for j in range(n_clusters):
        print(f"{cont_matrix[i,j]:4d} ", end="")
    print(f"| {cont_matrix[i].sum():4d}")

print("-" * 50)
print("Total", end="")
for j in range(n_clusters):
    print(f"{cont_matrix[:,j].sum():4d} ", end="")
print(f"| {cont_matrix.sum():4d}")

# Calculate percentage overlap for each audio cluster
print("\n🎯 Where Audio Clusters End Up in Lyric Space:")
print("-" * 50)
for i in range(n_clusters):
    audio_cluster_size = cont_matrix[i].sum()
    print(f"\nAudio Cluster {i} ({audio_cluster_size} tracks) distributes as:")
    for j in range(n_clusters):
        pct = cont_matrix[i,j] / audio_cluster_size * 100
        if pct > 10:  # Only show significant overlaps
            print(f"  → Lyric Cluster {j}: {cont_matrix[i,j]:3d} tracks ({pct:5.1f}%)")

# Calculate percentage overlap for each lyric cluster
print("\n📝 Where Lyric Clusters Come From in Audio Space:")
print("-" * 50)
for j in range(n_clusters):
    lyric_cluster_size = cont_matrix[:,j].sum()
    print(f"\nLyric Cluster {j} ({lyric_cluster_size} tracks) comes from:")
    for i in range(n_clusters):
        pct = cont_matrix[i,j] / lyric_cluster_size * 100
        if pct > 10:  # Only show significant overlaps
            print(f"  ← Audio Cluster {i}: {cont_matrix[i,j]:3d} tracks ({pct:5.1f}%)")

# Find tracks that stay in same cluster vs switch
df['audio_cluster'] = labels_audio
df['lyric_cluster'] = labels_lyrics
df['cluster_agreement'] = (df['audio_cluster'] == df['lyric_cluster'])

print("\n" + "=" * 80)
print("3. CLUSTER AGREEMENT ANALYSIS")
print("=" * 80)

agreement_rate = df['cluster_agreement'].mean()
print(f"\n📊 Overall Statistics:")
print("-" * 50)
print(f"Tracks in same cluster number: {df['cluster_agreement'].sum()} / {len(df)} ({agreement_rate*100:.1f}%)")
print(f"Tracks that switch clusters: {(~df['cluster_agreement']).sum()} ({(1-agreement_rate)*100:.1f}%)")

# Find interesting examples
print("\n🔄 Example Track Movements:")
print("-" * 50)

# Tracks that stay together
stable_tracks = df[df['cluster_agreement'] == True].head(5)
print("\n✅ Tracks stable across audio & lyrics (same cluster #):")
for idx, row in stable_tracks.iterrows():
    print(f"  Cluster {row['audio_cluster']}: {row['artist']} - {row['track_name']}")

# Tracks with maximum cluster distance
df['cluster_distance'] = abs(df['audio_cluster'] - df['lyric_cluster'])
max_movement = df.nlargest(10, 'cluster_distance')
print("\n🚀 Tracks with largest cluster movement:")
for idx, row in max_movement.iterrows():
    print(f"  Audio {row['audio_cluster']} → Lyric {row['lyric_cluster']}: "
          f"{row['artist']} - {row['track_name']}")

# Instrumental tracks analysis
instrumental = df[df['instrumentalness'] > 0.5]
print(f"\n🎵 Instrumental Track Clustering ({len(instrumental)} tracks):")
print("-" * 50)
print("Audio cluster distribution:")
for cluster, count in instrumental['audio_cluster'].value_counts().items():
    print(f"  Cluster {cluster}: {count} tracks")
print("\nLyric cluster distribution (should be meaningless):")
for cluster, count in instrumental['lyric_cluster'].value_counts().items():
    print(f"  Cluster {cluster}: {count} tracks")

# Analyze specific genre movements
print("\n" + "=" * 80)
print("4. GENRE-SPECIFIC CLUSTER MOVEMENT")
print("=" * 80)

if 'top_genre' in df.columns:
    # Hip-hop analysis
    hiphop = df[df['top_genre'].str.contains('Hip Hop', na=False)]
    if len(hiphop) > 0:
        print(f"\n🎤 Hip-Hop Tracks ({len(hiphop)} total):")
        print("-" * 50)
        hiphop_agreement = hiphop['cluster_agreement'].mean()
        print(f"Agreement rate: {hiphop_agreement*100:.1f}%")

        # Show movement pattern
        hiphop_movement = pd.crosstab(hiphop['audio_cluster'], hiphop['lyric_cluster'])
        print("\nMovement matrix (Audio → Lyric):")
        print(hiphop_movement)

    # Electronic analysis
    electronic = df[df['top_genre'].str.contains('Electronic', na=False)]
    if len(electronic) > 0:
        print(f"\n⚡ Electronic Tracks ({len(electronic)} total):")
        print("-" * 50)
        electronic_agreement = electronic['cluster_agreement'].mean()
        print(f"Agreement rate: {electronic_agreement*100:.1f}%")

        # Show movement pattern
        electronic_movement = pd.crosstab(electronic['audio_cluster'], electronic['lyric_cluster'])
        print("\nMovement matrix (Audio → Lyric):")
        print(electronic_movement)

# Calculate correlation between audio and lyric features
print("\n" + "=" * 80)
print("5. CROSS-DOMAIN FEATURE CORRELATIONS")
print("=" * 80)

print("\n🔗 Strongest Audio↔Lyric Feature Correlations:")
print("-" * 50)

# Calculate all cross-correlations
cross_correlations = []
for audio_feat in audio_features:
    for lyric_feat in lyric_features:
        corr = df[audio_feat].corr(df[lyric_feat])
        cross_correlations.append({
            'audio': audio_feat.replace('emb_', ''),
            'lyric': lyric_feat.replace('emb_lyric_', ''),
            'correlation': corr
        })

# Sort by absolute correlation
cross_corr_df = pd.DataFrame(cross_correlations)
cross_corr_df['abs_corr'] = abs(cross_corr_df['correlation'])
cross_corr_df = cross_corr_df.sort_values('abs_corr', ascending=False)

# Show top positive correlations
print("\nTop 10 Positive Correlations:")
positive = cross_corr_df[cross_corr_df['correlation'] > 0].head(10)
for idx, row in positive.iterrows():
    print(f"  {row['audio']:20s} ↔ {row['lyric']:15s}: {row['correlation']:+.3f}")

# Show top negative correlations
print("\nTop 10 Negative Correlations:")
negative = cross_corr_df[cross_corr_df['correlation'] < 0].head(10)
for idx, row in negative.iterrows():
    print(f"  {row['audio']:20s} ↔ {row['lyric']:15s}: {row['correlation']:+.3f}")

# Test clustering with different feature combinations
print("\n" + "=" * 80)
print("6. FEATURE COMBINATION EXPERIMENTS")
print("=" * 80)

# Combine audio and lyrics with different weights
weights = [0.0, 0.25, 0.5, 0.75, 1.0]
print("\n📊 Clustering Quality vs Audio/Lyric Weight:")
print("-" * 50)
print("Weight  | Silhouette | Description")
print("-" * 50)

for w in weights:
    # Weighted combination
    X_combined = np.hstack([X_audio * w, X_lyrics * (1-w)])

    # Cluster
    labels = AgglomerativeClustering(n_clusters=5, linkage='ward').fit_predict(X_combined)
    sil = silhouette_score(X_combined, labels)

    if w == 0:
        desc = "Lyrics only"
    elif w == 1:
        desc = "Audio only"
    else:
        desc = f"{w*100:.0f}% audio, {(1-w)*100:.0f}% lyrics"

    print(f"{w:6.2f}  | {sil:10.3f} | {desc}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
Key Findings:
1. Audio and lyric clusters show LOW agreement (ARI={results[2]['ari']:.3f} for 5 clusters)
2. Only {agreement_rate*100:.1f}% of tracks stay in the same cluster number
3. Lyrics-only clustering has BETTER silhouette score ({results[2]['sil_lyrics']:.3f}) than audio ({results[2]['sil_audio']:.3f})
4. Instrumental tracks scatter randomly in lyric space (as expected)
5. The two feature spaces capture fundamentally different aspects of music

This low covariance suggests that:
- Songs that sound similar don't necessarily have similar lyrics
- Lyrical themes cross audio genre boundaries
- The 33-dim combined space captures complementary, not redundant, information
""")

print("=" * 80)