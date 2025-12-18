# Spotify Saved Tracks Metadata Fetcher

Fetch comprehensive metadata for all your saved/liked songs from Spotify.

## What It Does

This script:
- Fetches **all** your saved tracks (handles pagination automatically)
- Extracts comprehensive metadata for each track
- Displays summary statistics
- Saves all data to `data/saved_tracks.json` for analysis

## Metadata Collected

For each track, the script collects:
- Track ID, name, and artists
- Album information (name, ID, type, release date)
- Duration (ms and minutes)
- Popularity score (0-100)
- Explicit flag
- Track/disc numbers
- Date added to your library
- Preview URL and external Spotify URL
- ISRC code

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a Spotify App:
   - Go to [Spotify Dashboard](https://developer.spotify.com/dashboard)
   - Click "Create app"
   - Fill in app details (any name/description)
   - **Important:** Set Redirect URI to `http://127.0.0.1:3000/callback`
   - Save and go to Settings to view your credentials

3. Create a `.env` file in the project root:
   ```bash
   cp ../.env.example ../.env
   ```

4. Add your credentials to `.env`:
   ```
   SPOTIFY_CLIENT_ID=your_client_id_here
   SPOTIFY_CLIENT_SECRET=your_client_secret_here
   ```

## Usage

```bash
python fetch_audio_data.py
```

On first run:
1. Opens browser for Spotify authorization
2. You approve access to your saved tracks
3. Token cached in `.cache` for future use

The script will:
1. Fetch all your saved tracks with pagination
2. Display a summary with statistics
3. Save the full data to `data/saved_tracks.json`

## Output

**Console Summary:**
- Total tracks count
- Unique artists and albums
- Total listening duration
- Average popularity
- Explicit content percentage
- Sample of first 5 tracks

**JSON File:**
- Saved to `data/saved_tracks.json`
- Contains complete metadata for all tracks
- Ready for analysis, clustering, or visualization

## Notes

- The script automatically paginates through all your saved tracks
- Tokens refresh automatically via spotipy
- Data folder is gitignored to keep your listening data private
- Only requires `user-library-read` scope (no deprecated endpoints)
