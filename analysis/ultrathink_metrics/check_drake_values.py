import pickle
import pandas as pd

# Load analysis data
with open('/Users/islamtayeb/Documents/spotify-clustering/analysis/outputs/analysis_data.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['combined']['dataframe']

# Check if we have Drake tracks
drake_mask = df['artist'].str.contains('Drake', case=False, na=False)
drake_tracks = df[drake_mask]

print(f"Drake tracks found: {len(drake_tracks)}")
print("\nDrake track danceability values (raw):")
print(drake_tracks[['track_name', 'danceability', 'emb_danceability']].head(10))

print("\n\nGeneral danceability stats:")
print(f"Min: {df['danceability'].min():.3f}")
print(f"Max: {df['danceability'].max():.3f}")
print(f"Mean: {df['danceability'].mean():.3f}")
print(f"Median: {df['danceability'].median():.3f}")

# Let's check the top 10 most danceable tracks overall
print("\n\nTop 10 most danceable tracks in library:")
top_dance = df.nlargest(10, 'danceability')[['artist', 'track_name', 'danceability']]
for idx, row in top_dance.iterrows():
    print(f"  • {row['artist']} - {row['track_name']}: {row['danceability']:.3f}")