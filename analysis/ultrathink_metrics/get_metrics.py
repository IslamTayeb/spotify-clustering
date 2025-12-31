#!/usr/bin/env python3
"""Get specific metrics for ultrathink writing"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Load analysis data
with open('/Users/islamtayeb/Documents/spotify-clustering/analysis/outputs/analysis_data.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['combined']['dataframe']

print("=" * 70)
print("1. DRAKE'S DANCEABILITY ANALYSIS")
print("=" * 70)

# Filter for Drake tracks
drake_mask = df['artist'].str.contains('Drake', case=False, na=False)
drake_tracks = df[drake_mask]

if len(drake_tracks) > 0:
    avg_dance = drake_tracks['danceability'].mean()
    print(f"\nDrake's average danceability: {avg_dance:.3f}")
    print(f"Your library average: {df['danceability'].mean():.3f}")
    print(f"Drake is {(avg_dance - df['danceability'].mean()) / df['danceability'].mean() * 100:.1f}% more danceable than your average\n")

    print("Top 5 most danceable Drake tracks in your library:")
    top_drake = drake_tracks.nlargest(5, 'danceability')[['track_name', 'danceability', 'mood_party', 'valence']]
    for idx, row in top_drake.iterrows():
        print(f"  • {row['track_name']}: dance={row['danceability']:.3f}, party={row['mood_party']:.3f}, valence={row['valence']:.3f}")
else:
    print("No Drake tracks found in your library")

print("\n" + "=" * 70)
print("2. ENGAGEMENT VS. MOOD_PARTY CORRELATION")
print("=" * 70)

# Calculate correlation
pearson_r = df['engagement_score'].corr(df['mood_party'])
print(f"\nPearson correlation (engagement ↔ mood_party): {pearson_r:.3f}")

# Additional correlations for context
print("\nOther relevant correlations:")
print(f"  • engagement ↔ danceability: {df['engagement_score'].corr(df['danceability']):.3f}")
print(f"  • engagement ↔ valence: {df['engagement_score'].corr(df['valence']):.3f}")
print(f"  • mood_party ↔ danceability: {df['mood_party'].corr(df['danceability']):.3f}")
print(f"  • mood_party ↔ arousal: {df['mood_party'].corr(df['arousal']):.3f}")

# Find examples of high engagement but low party
high_engage_low_party = df[(df['engagement_score'] > 0.7) & (df['mood_party'] < 0.3)]
print(f"\nHigh engagement (>0.7) but low party (<0.3): {len(high_engage_low_party)} tracks")
if len(high_engage_low_party) > 0:
    print("Examples:")
    for idx, row in high_engage_low_party.head(3).iterrows():
        print(f"  • {row['artist']} - {row['track_name']}: engage={row['engagement_score']:.2f}, party={row['mood_party']:.2f}")

print("\n" + "=" * 70)
print("3. LYRIC VS. AUDIO DIMENSION CLUSTERING CONTRIBUTION")
print("=" * 70)

# Define feature groups (using emb_ versions which are the standardized features)
audio_features = ['emb_bpm', 'emb_danceability', 'emb_instrumentalness', 'emb_valence', 'emb_arousal',
                  'emb_engagement', 'emb_approachability', 'emb_mood_aggressive', 'emb_mood_happy',
                  'emb_mood_sad', 'emb_mood_relaxed', 'emb_mood_party', 'emb_voice_gender',
                  'emb_genre_ladder', 'emb_acoustic_electronic', 'emb_timbre_brightness']
key_features = ['emb_key_cos', 'emb_key_sin', 'emb_key_scale']
lyric_features = ['emb_lyric_valence', 'emb_lyric_arousal', 'emb_lyric_mood_happy',
                  'emb_lyric_mood_sad', 'emb_lyric_mood_aggressive', 'emb_lyric_mood_relaxed',
                  'emb_lyric_explicit', 'emb_lyric_narrative', 'emb_lyric_vocabulary', 'emb_lyric_repetition']
meta_features = ['emb_theme', 'emb_language', 'emb_popularity', 'emb_release_year']

# Calculate variance explained by each feature group
from sklearn.preprocessing import StandardScaler

# Get the features used in clustering
feature_cols = audio_features + key_features + lyric_features + meta_features
X = df[feature_cols].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate variance contribution by feature group
total_variance = X_scaled.var(axis=0).sum()

audio_var = X_scaled[:, :len(audio_features)].var(axis=0).sum()
key_var = X_scaled[:, len(audio_features):len(audio_features)+len(key_features)].var(axis=0).sum()
lyric_start = len(audio_features) + len(key_features)
lyric_var = X_scaled[:, lyric_start:lyric_start+len(lyric_features)].var(axis=0).sum()
meta_var = X_scaled[:, -len(meta_features):].var(axis=0).sum()

print(f"\nVariance contribution to clustering space:")
print(f"  • Audio features: {audio_var/total_variance*100:.1f}%")
print(f"  • Key features: {key_var/total_variance*100:.1f}%")
print(f"  • Lyric features: {lyric_var/total_variance*100:.1f}%")
print(f"  • Meta features: {meta_var/total_variance*100:.1f}%")

# Cluster separation analysis
if 'cluster' in df.columns:
    from sklearn.metrics import silhouette_samples

    # Calculate silhouette scores for different feature subsets
    silhouette_audio = silhouette_samples(X_scaled[:, :len(audio_features)], df['cluster'].values)
    silhouette_lyrics = silhouette_samples(X_scaled[:, lyric_start:lyric_start+len(lyric_features)], df['cluster'].values)
    silhouette_full = silhouette_samples(X_scaled, df['cluster'].values)

    print(f"\nAverage silhouette scores by feature subset:")
    print(f"  • Audio-only clustering quality: {silhouette_audio.mean():.3f}")
    print(f"  • Lyrics-only clustering quality: {silhouette_lyrics.mean():.3f}")
    print(f"  • Full 33-dim clustering quality: {silhouette_full.mean():.3f}")

print("\n" + "=" * 70)
print("4. MONTHLY MOOD PATTERNS (APRIL-JUNE 2024)")
print("=" * 70)

# Parse dates and create monthly aggregation
df['added_date'] = pd.to_datetime(df['added_at'])
df['year_month'] = df['added_date'].dt.to_period('M')

# Filter for April-June 2024
spring_2024 = df[(df['added_date'] >= '2024-04-01') & (df['added_date'] < '2024-07-01')]

if len(spring_2024) > 0:
    monthly_moods = spring_2024.groupby('year_month')[['mood_party', 'mood_happy', 'mood_sad', 'mood_relaxed', 'mood_aggressive', 'valence', 'arousal']].mean()

    print(f"\nTracks added April-June 2024: {len(spring_2024)}")
    print("\nMonthly mood averages:")
    for month, row in monthly_moods.iterrows():
        print(f"\n{month}:")
        print(f"  • mood_party: {row['mood_party']:.3f}")
        print(f"  • mood_happy: {row['mood_happy']:.3f}")
        print(f"  • mood_sad: {row['mood_sad']:.3f}")
        print(f"  • valence: {row['valence']:.3f}")
        print(f"  • arousal: {row['arousal']:.3f}")

    # Show the dip if present
    if len(monthly_moods) > 1:
        party_change = monthly_moods['mood_party'].iloc[-1] - monthly_moods['mood_party'].iloc[0]
        print(f"\nParty mood change Apr→Jun: {party_change:+.3f}")
else:
    print("\nNo tracks found for April-June 2024")

print("\n" + "=" * 70)
print("5. OCTOBER 2024 DATA CHECK")
print("=" * 70)

oct_2024 = df[(df['added_date'] >= '2024-10-01') & (df['added_date'] < '2024-11-01')]
print(f"\nTracks added in October 2024: {len(oct_2024)}")

if len(oct_2024) > 0:
    print("\nOctober 2024 averages:")
    print(f"  • mood_party: {oct_2024['mood_party'].mean():.3f}")
    print(f"  • mood_happy: {oct_2024['mood_happy'].mean():.3f}")
    print(f"  • mood_relaxed: {oct_2024['mood_relaxed'].mean():.3f}")
    print(f"  • valence: {oct_2024['valence'].mean():.3f}")

    # Show some example tracks
    print("\nSample tracks from October 2024:")
    for idx, row in oct_2024.head(5).iterrows():
        print(f"  • {row['artist']} - {row['track_name']}")

# Bonus: Show monthly resolution availability
print("\n" + "=" * 70)
print("BONUS: DATA RESOLUTION CHECK")
print("=" * 70)

monthly_counts = df.groupby('year_month').size()
print(f"\nYou have monthly resolution data for {len(monthly_counts)} months")
print(f"Date range: {monthly_counts.index.min()} to {monthly_counts.index.max()}")
print(f"\nMonths with most additions:")
for month, count in monthly_counts.nlargest(5).items():
    print(f"  • {month}: {count} tracks")