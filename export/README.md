# Playlist Export Tool

Export your clustering analysis results to Spotify playlists automatically.

## Overview

This tool creates Spotify playlists from your music clustering analysis:
- **5 playlists** for audio-based clusters (based on musical features like genre, mood, tempo)
- **5 playlists** for lyric-based clusters (based on lyrical themes and sentiment)

Each playlist is automatically named with a descriptive title based on the cluster's characteristics.

## Prerequisites

1. **Spotify Developer Credentials**: You need a Spotify app with playlist modification permissions
2. **Clustering Analysis**: You must have run the analysis pipeline first (`run_analysis.py`)
3. **Python Dependencies**: The script uses `spotipy` for Spotify API access

## Setup

### 1. Update Spotify App Permissions

The export script needs additional permissions beyond the basic `user-library-read` scope. No changes to your `.env` file are needed, but make sure your Spotify app settings are correct:

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Select your app
3. Verify the Redirect URI is set to: `http://127.0.0.1:3000/callback`

The script will automatically request the necessary permissions (`playlist-modify-public` and `playlist-modify-private`) when you run it for the first time.

### 2. Install Dependencies

The export tool uses `spotipy` and `python-dotenv`, which are already included in the main requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

### Preview Mode (Recommended First Step)

Preview what playlists would be created without actually creating them:

```bash
python export/create_playlists.py --dry-run
```

This shows you:
- All 10 playlists that would be created (5 audio + 5 lyrics)
- Track counts for each cluster
- Sample tracks from each playlist

### Create Playlists

After previewing, create all playlists:

```bash
python export/create_playlists.py
```

### Options

**Preview before creating** (recommended):
```bash
python export/create_playlists.py --dry-run
```

**Create private playlists** (instead of public):
```bash
python export/create_playlists.py --private
```

**Add a custom prefix** to playlist names:
```bash
python export/create_playlists.py --prefix "My Music"
# Results in: "My Music - Audio Cluster 0: genre_185 - aggressive"
```

**Create only audio cluster playlists**:
```bash
python export/create_playlists.py --audio-only
```

**Create only lyric cluster playlists**:
```bash
python export/create_playlists.py --lyrics-only
```

**Combine multiple options**:
```bash
python export/create_playlists.py --private --prefix "Clustering Analysis" --audio-only

# Preview with options
python export/create_playlists.py --dry-run --prefix "My Music" --audio-only
```

### Custom Data Path

If your analysis data is in a different location:

```bash
python export/create_playlists.py --data-path path/to/analysis_data.pkl
```

## Example Output

```
============================================================
Creating AUDIO cluster playlists for user: your_username
============================================================

Creating playlist: Audio Cluster 0: genre_rock - relaxed
  Tracks: 357
  Description: genre_rock - relaxed
  ✓ Created: https://open.spotify.com/playlist/xxxxx

Creating playlist: Audio Cluster 1: genre_pop - happy
  Tracks: 141
  Description: genre_pop - happy
  ✓ Created: https://open.spotify.com/playlist/yyyyy

[...]

============================================================
Creating LYRICS cluster playlists for user: your_username
============================================================

[...]

============================================================
SUMMARY
============================================================
Total playlists created: 10

  • Audio Cluster 0: genre_rock - relaxed
    Tracks: 357
    URL: https://open.spotify.com/playlist/xxxxx

  [...]

✓ All playlists created successfully!
```

## Playlist Details

### Audio Cluster Playlists

Named based on the dominant musical characteristics:
- **Top Genre**: The most common genre in the cluster
- **Dominant Mood**: The prevailing mood (happy, sad, aggressive, relaxed, party)

Example: `Audio Cluster 2: genre_185 - aggressive`

### Lyric Cluster Playlists

Named based on lyrical grouping:
- **Primary Language**: The most common language in the cluster

Example: `Lyrics Cluster 1: Primarily PT` (Portuguese)

## Technical Details

- **Batch Processing**: Tracks are added to playlists in batches of 100 (Spotify API limit)
- **Authentication**: Uses OAuth 2.0 with automatic token caching (stored in `.cache` file)
- **Cluster Information**: Reads from `analysis/outputs/analysis_data.pkl` by default
- **Track URIs**: Constructed from Spotify track IDs in the format `spotify:track:{track_id}`

## Troubleshooting

### Authentication Error

If you get an authentication error:
1. Delete the `.cache` file in the project root
2. Run the script again
3. Complete the authorization flow in your browser

### Missing Tracks

Some tracks might not be added if:
- The track has been removed from Spotify
- The track is not available in your region
- The track ID is invalid

The script will continue and create playlists with available tracks.

### Playlist Limit

Spotify has a limit on the number of playlists you can create. If you hit this limit, you'll need to delete some existing playlists first.

## Re-running the Script

**Important**: Running the script multiple times will create duplicate playlists. The script does not check for existing playlists with the same name.

To avoid duplicates:
1. Delete the old playlists manually from Spotify
2. Run the script again with the same parameters

## Next Steps

After creating your playlists:
1. Open Spotify and find your new playlists in your library
2. Explore each cluster to understand the musical/lyrical patterns
3. Share playlists with friends or make them collaborative
4. Use them as a foundation for discovering new music within each cluster
