#!/usr/bin/env python3
"""Comprehensive metrics analysis for ultrathink writing"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load analysis data
with open('/Users/islamtayeb/Documents/spotify-clustering/analysis/outputs/analysis_data.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['combined']['dataframe']

print("=" * 80)
print("COMPREHENSIVE SPOTIFY CLUSTERING ANALYSIS")
print("=" * 80)

# ============================================================================
# SECTION 1: DRAKE DEEP DIVE
# ============================================================================
print("\n" + "=" * 80)
print("1. DRAKE COMPREHENSIVE ANALYSIS")
print("=" * 80)

drake_mask = df['artist'].str.contains('Drake', case=False, na=False)
drake_tracks = df[drake_mask]

print(f"\nTotal Drake tracks in library: {len(drake_tracks)}")
print(f"Percentage of library: {len(drake_tracks)/len(df)*100:.1f}%")

# Drake's feature averages vs library
features_to_compare = ['danceability', 'mood_party', 'mood_aggressive', 'valence', 'arousal',
                       'engagement_score', 'approachability_score', 'instrumentalness']

print("\n📊 Drake vs. Library Averages:")
print("-" * 50)
for feat in features_to_compare:
    drake_avg = drake_tracks[feat].mean()
    lib_avg = df[feat].mean()
    diff_pct = (drake_avg - lib_avg) / lib_avg * 100
    print(f"{feat:20s}: Drake={drake_avg:6.3f}, Library={lib_avg:6.3f} ({diff_pct:+6.1f}%)")

# All Drake tracks with key features
print("\n📋 All Drake Tracks (sorted by danceability):")
print("-" * 50)
drake_sorted = drake_tracks.sort_values('danceability', ascending=False)
for idx, row in drake_sorted.iterrows():
    print(f"{row['track_name']:30s} | dance={row['danceability']:.3f} party={row['mood_party']:.3f} "
          f"aggr={row['mood_aggressive']:.3f} val={row['valence']:.2f}")

# Drake clustering analysis
if 'cluster' in drake_tracks.columns:
    print("\n🎯 Drake's Cluster Distribution:")
    print("-" * 50)
    drake_clusters = drake_tracks['cluster'].value_counts()
    for cluster, count in drake_clusters.items():
        cluster_total = len(df[df['cluster'] == cluster])
        print(f"Cluster {cluster}: {count} Drake tracks out of {cluster_total} total ({count/cluster_total*100:.1f}%)")

# ============================================================================
# SECTION 2: FEATURE CORRELATIONS MATRIX
# ============================================================================
print("\n" + "=" * 80)
print("2. COMPREHENSIVE CORRELATION ANALYSIS")
print("=" * 80)

# Key correlations
correlation_pairs = [
    ('engagement_score', 'mood_party'),
    ('engagement_score', 'danceability'),
    ('engagement_score', 'valence'),
    ('engagement_score', 'arousal'),
    ('mood_party', 'danceability'),
    ('mood_party', 'arousal'),
    ('mood_party', 'valence'),
    ('danceability', 'valence'),
    ('danceability', 'arousal'),
    ('valence', 'arousal'),
    ('mood_aggressive', 'mood_party'),
    ('mood_happy', 'valence'),
    ('mood_sad', 'valence'),
    ('instrumentalness', 'voice_gender')
]

print("\n🔗 Pearson Correlations:")
print("-" * 50)
for feat1, feat2 in correlation_pairs:
    if feat1 in df.columns and feat2 in df.columns:
        corr = df[feat1].corr(df[feat2])
        print(f"{feat1:20s} ↔ {feat2:20s}: {corr:+.3f}")

# ============================================================================
# SECTION 3: ENGAGEMENT/PARTY PARADOXES
# ============================================================================
print("\n" + "=" * 80)
print("3. ENGAGEMENT/PARTY PARADOXES & EDGE CASES")
print("=" * 80)

# High engagement, low party
high_eng_low_party = df[(df['engagement_score'] > 0.7) & (df['mood_party'] < 0.3)]
print(f"\n🎭 High Engagement (>0.7) + Low Party (<0.3): {len(high_eng_low_party)} tracks")
print("-" * 50)
for idx, row in high_eng_low_party.head(10).iterrows():
    print(f"{row['artist']:20s} - {row['track_name']:30s} | "
          f"eng={row['engagement_score']:.2f} party={row['mood_party']:.2f} dance={row['danceability']:.2f}")

# Low engagement, high party
low_eng_high_party = df[(df['engagement_score'] < 0.3) & (df['mood_party'] > 0.7)]
print(f"\n🎉 Low Engagement (<0.3) + High Party (>0.7): {len(low_eng_high_party)} tracks")
print("-" * 50)
for idx, row in low_eng_high_party.head(10).iterrows():
    print(f"{row['artist']:20s} - {row['track_name']:30s} | "
          f"eng={row['engagement_score']:.2f} party={row['mood_party']:.2f} dance={row['danceability']:.2f}")

# High danceability but low party
high_dance_low_party = df[(df['danceability'] > 0.7) & (df['mood_party'] < 0.3)]
print(f"\n💃 High Danceability (>0.7) + Low Party (<0.3): {len(high_dance_low_party)} tracks")
print("-" * 50)
for idx, row in high_dance_low_party.head(10).iterrows():
    print(f"{row['artist']:20s} - {row['track_name']:30s} | "
          f"dance={row['danceability']:.2f} party={row['mood_party']:.2f} val={row['valence']:.2f}")

# ============================================================================
# SECTION 4: CLUSTERING DIMENSION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("4. 33-DIMENSIONAL FEATURE SPACE ANALYSIS")
print("=" * 80)

# Feature groups with exact features
audio_features = ['emb_bpm', 'emb_danceability', 'emb_instrumentalness', 'emb_valence', 'emb_arousal',
                  'emb_engagement', 'emb_approachability', 'emb_mood_aggressive', 'emb_mood_happy',
                  'emb_mood_sad', 'emb_mood_relaxed', 'emb_mood_party', 'emb_voice_gender',
                  'emb_genre_ladder', 'emb_acoustic_electronic', 'emb_timbre_brightness']
key_features = ['emb_key_cos', 'emb_key_sin', 'emb_key_scale']
lyric_features = ['emb_lyric_valence', 'emb_lyric_arousal', 'emb_lyric_mood_happy',
                  'emb_lyric_mood_sad', 'emb_lyric_mood_aggressive', 'emb_lyric_mood_relaxed',
                  'emb_lyric_explicit', 'emb_lyric_narrative', 'emb_lyric_vocabulary', 'emb_lyric_repetition']
meta_features = ['emb_theme', 'emb_language', 'emb_popularity', 'emb_release_year']

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA

# Get feature matrix
feature_cols = audio_features + key_features + lyric_features + meta_features
X = df[feature_cols].values

# Already standardized (emb_ features), but let's verify
print("\n📐 Feature Space Statistics:")
print("-" * 50)
print(f"Total dimensions: {X.shape[1]}")
print(f"Audio dimensions: {len(audio_features)} ({len(audio_features)/X.shape[1]*100:.1f}%)")
print(f"Key dimensions: {len(key_features)} ({len(key_features)/X.shape[1]*100:.1f}%)")
print(f"Lyric dimensions: {len(lyric_features)} ({len(lyric_features)/X.shape[1]*100:.1f}%)")
print(f"Meta dimensions: {len(meta_features)} ({len(meta_features)/X.shape[1]*100:.1f}%)")

# Variance analysis
total_variance = X.var(axis=0).sum()
audio_var = X[:, :len(audio_features)].var(axis=0).sum()
key_var = X[:, len(audio_features):len(audio_features)+len(key_features)].var(axis=0).sum()
lyric_start = len(audio_features) + len(key_features)
lyric_var = X[:, lyric_start:lyric_start+len(lyric_features)].var(axis=0).sum()
meta_var = X[:, -len(meta_features):].var(axis=0).sum()

print("\n📊 Variance Contribution by Feature Group:")
print("-" * 50)
print(f"Audio features: {audio_var/total_variance*100:.1f}% of total variance")
print(f"Key features: {key_var/total_variance*100:.1f}%")
print(f"Lyric features: {lyric_var/total_variance*100:.1f}%")
print(f"Meta features: {meta_var/total_variance*100:.1f}%")

# Silhouette analysis by feature subset
if 'cluster' in df.columns:
    print("\n🎯 Clustering Quality by Feature Subset:")
    print("-" * 50)

    # Audio only
    sil_audio = silhouette_samples(X[:, :len(audio_features)], df['cluster'].values)
    print(f"Audio-only silhouette: {sil_audio.mean():.3f} (std: {sil_audio.std():.3f})")

    # Lyrics only
    sil_lyrics = silhouette_samples(X[:, lyric_start:lyric_start+len(lyric_features)], df['cluster'].values)
    print(f"Lyrics-only silhouette: {sil_lyrics.mean():.3f} (std: {sil_lyrics.std():.3f})")

    # Full features
    sil_full = silhouette_samples(X, df['cluster'].values)
    print(f"Full 33-dim silhouette: {sil_full.mean():.3f} (std: {sil_full.std():.3f})")

    # Audio + Lyrics (no meta)
    X_no_meta = X[:, :-len(meta_features)]
    sil_no_meta = silhouette_samples(X_no_meta, df['cluster'].values)
    print(f"Audio+Lyrics (no meta): {sil_no_meta.mean():.3f} (std: {sil_no_meta.std():.3f})")

# PCA analysis
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)

print("\n🔬 PCA Dimensionality Analysis:")
print("-" * 50)
print(f"Dimensions for 80% variance: {np.argmax(cumsum >= 0.8) + 1}")
print(f"Dimensions for 90% variance: {np.argmax(cumsum >= 0.9) + 1}")
print(f"Dimensions for 95% variance: {np.argmax(cumsum >= 0.95) + 1}")
print(f"First 10 components explain: {cumsum[9]:.1%} of variance")

# ============================================================================
# SECTION 5: TEMPORAL ANALYSIS (MONTHLY RESOLUTION)
# ============================================================================
print("\n" + "=" * 80)
print("5. COMPREHENSIVE TEMPORAL ANALYSIS")
print("=" * 80)

df['added_date'] = pd.to_datetime(df['added_at'])
df['year_month'] = df['added_date'].dt.to_period('M')

# Overall temporal stats
print("\n📅 Temporal Coverage:")
print("-" * 50)
print(f"Date range: {df['added_date'].min().date()} to {df['added_date'].max().date()}")
print(f"Total months: {df['year_month'].nunique()}")
print(f"Average tracks per month: {len(df) / df['year_month'].nunique():.1f}")

# Monthly aggregation for all key features
monthly_stats = df.groupby('year_month').agg({
    'mood_party': 'mean',
    'mood_happy': 'mean',
    'mood_sad': 'mean',
    'mood_aggressive': 'mean',
    'mood_relaxed': 'mean',
    'valence': 'mean',
    'arousal': 'mean',
    'danceability': 'mean',
    'engagement_score': 'mean',
    'track_id': 'count'  # count of tracks
}).rename(columns={'track_id': 'track_count'})

# Find significant changes
print("\n📈 Months with Extreme Values:")
print("-" * 50)

for feature in ['mood_party', 'valence', 'arousal', 'danceability']:
    max_month = monthly_stats[feature].idxmax()
    min_month = monthly_stats[feature].idxmin()
    print(f"\n{feature}:")
    print(f"  Highest: {max_month} ({monthly_stats.loc[max_month, feature]:.3f})")
    print(f"  Lowest: {min_month} ({monthly_stats.loc[min_month, feature]:.3f})")

# Detailed Spring 2024 analysis
print("\n🌸 Spring 2024 Deep Dive (April-June):")
print("-" * 50)
spring_2024 = df[(df['added_date'] >= '2024-04-01') & (df['added_date'] < '2024-07-01')]

for month in ['2024-04', '2024-05', '2024-06']:
    month_period = pd.Period(month, 'M')
    if month_period in monthly_stats.index:
        stats = monthly_stats.loc[month_period]
        print(f"\n{month}: {int(stats['track_count'])} tracks")
        print(f"  party={stats['mood_party']:.3f}, happy={stats['mood_happy']:.3f}, "
              f"sad={stats['mood_sad']:.3f}")
        print(f"  valence={stats['valence']:.3f}, arousal={stats['arousal']:.3f}, "
              f"dance={stats['danceability']:.3f}")

# October 2024 with Claire
print("\n🍂 October 2024 Analysis:")
print("-" * 50)
oct_2024 = df[(df['added_date'] >= '2024-10-01') & (df['added_date'] < '2024-11-01')]
if len(oct_2024) > 0:
    print(f"Tracks added: {len(oct_2024)}")
    print(f"Average mood_party: {oct_2024['mood_party'].mean():.3f}")
    print(f"Average valence: {oct_2024['valence'].mean():.3f}")
    print("\nTrack list:")
    for idx, row in oct_2024.iterrows():
        print(f"  • {row['artist']} - {row['track_name']}")

# ============================================================================
# SECTION 6: EXTREME TRACKS (OUTLIERS)
# ============================================================================
print("\n" + "=" * 80)
print("6. EXTREME TRACKS ANALYSIS")
print("=" * 80)

# Most danceable tracks
print("\n💃 Top 15 Most Danceable Tracks:")
print("-" * 50)
for idx, row in df.nlargest(15, 'danceability').iterrows():
    print(f"{row['danceability']:.3f} | {row['artist']:25s} - {row['track_name']:35s} | "
          f"party={row['mood_party']:.2f} val={row['valence']:.1f}")

# Least danceable (but high party)
print("\n🚫 Least Danceable but High Party (party > 0.7):")
print("-" * 50)
low_dance_high_party = df[df['mood_party'] > 0.7].nsmallest(10, 'danceability')
for idx, row in low_dance_high_party.iterrows():
    print(f"dance={row['danceability']:.3f} party={row['mood_party']:.2f} | "
          f"{row['artist']:20s} - {row['track_name']}")

# Most engaging tracks
print("\n⚡ Top 10 Most Engaging Tracks:")
print("-" * 50)
for idx, row in df.nlargest(10, 'engagement_score').iterrows():
    print(f"{row['engagement_score']:.3f} | {row['artist']:25s} - {row['track_name']:35s} | "
          f"party={row['mood_party']:.2f} dance={row['danceability']:.2f}")

# ============================================================================
# SECTION 7: GENRE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("7. GENRE DISTRIBUTION & FUSION")
print("=" * 80)

# Genre fusion distribution
print("\n🎵 Genre Fusion Score Distribution:")
print("-" * 50)
if 'emb_genre_ladder' in df.columns:
    print(f"Mean genre fusion: {df['emb_genre_ladder'].mean():.3f}")
    print(f"Median: {df['emb_genre_ladder'].median():.3f}")
    print(f"Std dev: {df['emb_genre_ladder'].std():.3f}")

    # Most genre-pure tracks (low fusion)
    print("\n🎯 Most Genre-Pure Tracks (lowest fusion):")
    for idx, row in df.nsmallest(5, 'emb_genre_ladder').iterrows():
        print(f"  fusion={row['emb_genre_ladder']:.3f} | {row['artist']} - {row['track_name']}")

    # Most genre-fused tracks
    print("\n🌈 Most Genre-Fused Tracks (highest fusion):")
    for idx, row in df.nlargest(5, 'emb_genre_ladder').iterrows():
        print(f"  fusion={row['emb_genre_ladder']:.3f} | {row['artist']} - {row['track_name']}")

# Top genres
if 'top_genre' in df.columns:
    print("\n📊 Top 10 Genres in Library:")
    print("-" * 50)
    genre_counts = df['top_genre'].value_counts().head(10)
    for genre, count in genre_counts.items():
        print(f"{genre:30s}: {count:4d} tracks ({count/len(df)*100:5.1f}%)")

# ============================================================================
# SECTION 8: CLUSTER CHARACTERISTICS
# ============================================================================
print("\n" + "=" * 80)
print("8. CLUSTER DEEP DIVE")
print("=" * 80)

if 'cluster' in df.columns:
    n_clusters = df['cluster'].nunique()
    print(f"\nTotal clusters: {n_clusters}")

    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        print(f"\n🎯 Cluster {cluster_id}: {len(cluster_df)} tracks ({len(cluster_df)/len(df)*100:.1f}%)")
        print("-" * 50)

        # Cluster averages
        print("Average values:")
        features = ['mood_party', 'mood_happy', 'mood_sad', 'valence', 'arousal',
                   'danceability', 'engagement_score', 'instrumentalness']
        for feat in features:
            cluster_mean = cluster_df[feat].mean()
            overall_mean = df[feat].mean()
            diff = (cluster_mean - overall_mean) / overall_mean * 100
            print(f"  {feat:20s}: {cluster_mean:.3f} ({diff:+5.1f}% vs overall)")

        # Top artists in cluster
        print(f"\nTop 5 artists:")
        top_artists = cluster_df['artist'].value_counts().head(5)
        for artist, count in top_artists.items():
            print(f"  • {artist}: {count} tracks")

        # Sample tracks
        print(f"\nSample tracks:")
        for idx, row in cluster_df.head(3).iterrows():
            print(f"  • {row['artist']} - {row['track_name']}")

# ============================================================================
# SECTION 9: ACOUSTIC VS ELECTRONIC ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("9. ACOUSTIC VS ELECTRONIC SPECTRUM")
print("=" * 80)

if 'emb_acoustic_electronic' in df.columns:
    print("\n🎸 Acoustic/Electronic Distribution:")
    print("-" * 50)
    print(f"Mean (0=electronic, 1=acoustic): {df['emb_acoustic_electronic'].mean():.3f}")

    # Most electronic
    print("\n⚡ Most Electronic Tracks:")
    for idx, row in df.nsmallest(10, 'emb_acoustic_electronic').iterrows():
        print(f"  {row['emb_acoustic_electronic']:.3f} | {row['artist']} - {row['track_name']}")

    # Most acoustic
    print("\n🪕 Most Acoustic Tracks:")
    for idx, row in df.nlargest(10, 'emb_acoustic_electronic').iterrows():
        print(f"  {row['emb_acoustic_electronic']:.3f} | {row['artist']} - {row['track_name']}")

# ============================================================================
# SECTION 10: LYRIC ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("10. LYRIC FEATURES ANALYSIS")
print("=" * 80)

# Instrumental vs vocal
print(f"\n🎤 Vocal vs Instrumental:")
print("-" * 50)
instrumental = df[df['instrumentalness'] > 0.5]
vocal = df[df['instrumentalness'] <= 0.5]
print(f"Instrumental tracks: {len(instrumental)} ({len(instrumental)/len(df)*100:.1f}%)")
print(f"Vocal tracks: {len(vocal)} ({len(vocal)/len(df)*100:.1f}%)")

# Lyric mood analysis
if 'lyric_valence' in df.columns:
    print("\n📝 Lyric Features (vocal tracks only):")
    print("-" * 50)
    vocal_with_lyrics = vocal[vocal['has_lyrics'] == True]
    print(f"Tracks with lyrics analyzed: {len(vocal_with_lyrics)}")

    if len(vocal_with_lyrics) > 0:
        print(f"\nLyric averages:")
        lyric_features_raw = ['lyric_valence', 'lyric_arousal', 'lyric_explicit',
                              'lyric_narrative', 'lyric_vocabulary_richness', 'lyric_repetition']
        for feat in lyric_features_raw:
            if feat in vocal_with_lyrics.columns:
                print(f"  {feat:25s}: {vocal_with_lyrics[feat].mean():.3f}")

# ============================================================================
# SECTION 11: SPECIAL COMPARISONS
# ============================================================================
print("\n" + "=" * 80)
print("11. SPECIAL COMPARISONS")
print("=" * 80)

# Jazz vs Electronic Game Music comparison
print("\n🎷 Jazz vs 🎮 Electronic/Game Music:")
print("-" * 50)

# Find jazz-like tracks (acoustic, instrumental, relaxed)
jazz_like = df[(df['emb_acoustic_electronic'] > 0.7) &
               (df['instrumentalness'] > 0.5) &
               (df['mood_relaxed'] > 0.5)]
print(f"Jazz-like tracks: {len(jazz_like)}")
if len(jazz_like) > 0:
    print(f"  Avg danceability: {jazz_like['danceability'].mean():.3f}")
    print(f"  Avg party mood: {jazz_like['mood_party'].mean():.3f}")
    print("  Examples:", jazz_like.head(3)['track_name'].tolist())

# Find electronic game music (electronic, instrumental, engaging)
game_like = df[(df['emb_acoustic_electronic'] < 0.3) &
                (df['instrumentalness'] > 0.5) &
                (df['engagement_score'] > 0.5)]
print(f"\nElectronic/Game-like tracks: {len(game_like)}")
if len(game_like) > 0:
    print(f"  Avg danceability: {game_like['danceability'].mean():.3f}")
    print(f"  Avg party mood: {game_like['mood_party'].mean():.3f}")
    print("  Examples:", game_like.head(3)['track_name'].tolist())

# ============================================================================
# SECTION 12: SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("12. LIBRARY SUMMARY STATISTICS")
print("=" * 80)

print(f"\n📊 Overall Library Stats:")
print("-" * 50)
print(f"Total tracks: {len(df)}")
print(f"Unique artists: {df['artist'].nunique()}")
print(f"Date range: {df['added_date'].min().date()} to {df['added_date'].max().date()}")
print(f"Average popularity: {df['popularity'].mean():.1f}")

# Feature ranges
print("\n📏 Feature Ranges (min/mean/max):")
print("-" * 50)
features_to_summarize = ['danceability', 'mood_party', 'valence', 'arousal',
                         'engagement_score', 'instrumentalness']
for feat in features_to_summarize:
    print(f"{feat:20s}: [{df[feat].min():.3f}, {df[feat].mean():.3f}, {df[feat].max():.3f}]")

print("\n" + "=" * 80)
print("END OF COMPREHENSIVE ANALYSIS")
print("=" * 80)