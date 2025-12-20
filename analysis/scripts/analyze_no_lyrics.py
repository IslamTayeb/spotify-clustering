#!/usr/bin/env python3
"""
Analyze tracks without lyrics to determine how many are truly instrumental
vs vocal tracks where lyrics weren't found.
"""

import pickle
import pandas as pd

# Load cached features
print("Loading audio features...")
with open('cache/audio_features.pkl', 'rb') as f:
    audio_features = pickle.load(f)

print("Loading lyric features...")
with open('cache/lyric_features.pkl', 'rb') as f:
    lyric_features = pickle.load(f)

# Create dataframes for easier analysis
audio_df = pd.DataFrame(audio_features)
lyric_df = pd.DataFrame(lyric_features)

# Merge on track_id
merged = pd.merge(
    lyric_df[['track_id', 'track_name', 'artist', 'has_lyrics', 'word_count']],
    audio_df[['track_id', 'instrumentalness']],
    on='track_id'
)

# Filter tracks without lyrics
no_lyrics = merged[merged['has_lyrics'] == False].copy()

print(f"\n{'='*70}")
print(f"TRACKS WITHOUT LYRICS ANALYSIS")
print(f"{'='*70}")
print(f"\nTotal tracks without lyrics: {len(no_lyrics)}")

# Instrumentalness is a probability: 0 = vocal, 1 = instrumental
# Let's use 0.5 as threshold (common practice)
instrumental_threshold = 0.5

no_lyrics['category'] = no_lyrics['instrumentalness'].apply(
    lambda x: 'Instrumental' if x >= instrumental_threshold else 'Vocal (lyrics missing)'
)

instrumental = no_lyrics[no_lyrics['category'] == 'Instrumental']
vocal_missing_lyrics = no_lyrics[no_lyrics['category'] == 'Vocal (lyrics missing)']

print(f"\nInstrumental tracks: {len(instrumental)} ({len(instrumental)/len(no_lyrics)*100:.1f}%)")
print(f"Vocal tracks (lyrics missing): {len(vocal_missing_lyrics)} ({len(vocal_missing_lyrics)/len(no_lyrics)*100:.1f}%)")

# Show statistics
print(f"\n{'='*70}")
print("INSTRUMENTALNESS DISTRIBUTION FOR TRACKS WITHOUT LYRICS")
print(f"{'='*70}")
print(no_lyrics['instrumentalness'].describe())

# Show most instrumental tracks (likely true instrumentals)
print(f"\n{'='*70}")
print("TOP 20 MOST INSTRUMENTAL TRACKS (Highest Confidence)")
print(f"{'='*70}")
most_instrumental = no_lyrics.nlargest(20, 'instrumentalness')[['artist', 'track_name', 'instrumentalness']]
for idx, row in most_instrumental.iterrows():
    print(f"{row['instrumentalness']:.3f} | {row['artist']} - {row['track_name']}")

# Show most vocal tracks without lyrics (lyrics probably missing)
print(f"\n{'='*70}")
print("TOP 20 MOST VOCAL TRACKS WITHOUT LYRICS (Lyrics Likely Missing)")
print(f"{'='*70}")
most_vocal = no_lyrics.nsmallest(20, 'instrumentalness')[['artist', 'track_name', 'instrumentalness']]
for idx, row in most_vocal.iterrows():
    print(f"{row['instrumentalness']:.3f} | {row['artist']} - {row['track_name']}")

# Save detailed results
output_file = 'outputs/no_lyrics_analysis.csv'
no_lyrics_sorted = no_lyrics.sort_values('instrumentalness', ascending=False)
no_lyrics_sorted.to_csv(output_file, index=False)
print(f"\n{'='*70}")
print(f"Full results saved to: {output_file}")
print(f"{'='*70}")
