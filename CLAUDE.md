# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based Spotify clustering project that analyzes audio features and characteristics of songs from user's Spotify library.

### Key Components

- `api/` - Spotify Web API integration for fetching track metadata
  - `fetch_audio_data.py` - Fetches all saved tracks with comprehensive metadata
  - Uses OAuth 2.0 Authorization Code Flow with spotipy library
  - Requires Client ID and Client Secret with `user-library-read` scope
  - Saves data to `api/data/saved_tracks.json` for analysis

- `download/` - Scripts for downloading audio files of saved tracks
  - `check_matches.py` - Analyzes which tracks are already downloaded vs missing
  - `download_missing.py` - Downloads missing tracks using spotdl (batch download)
  - `download_ytdlp.py` - Downloads missing tracks using yt-dlp (YouTube search)
  - All scripts use aggressive filename normalization to match tracks across systems
  - Downloads saved to `songs/` directory

### Data Flow

1. Fetch track metadata from Spotify API → `api/data/saved_tracks.json`
2. Check which tracks are already downloaded → `check_matches.py`
3. Download missing tracks → `download_missing.py` (spotdl) or `download_ytdlp.py` (yt-dlp)
4. MP3 files stored in `songs/` directory

### Filename Normalization Strategy

All download scripts share a common normalization approach to match Spotify track names with downloaded filenames:
- Unicode normalization (Japanese wave dashes, quotes, etc.)
- Special character removal/replacement (brackets, punctuation, symbols)
- Colon → dash conversion (filesystem compatibility)
- Ampersand → "and" conversion
- Aggressive whitespace and dash normalization
- Case-insensitive matching

This ensures `"Artist: Song (Remix)"` matches `artist-song-remix.mp3`.

## Development Setup

The project uses standard Python .gitignore patterns, supporting various package managers (pip, poetry, pipenv, uv, pdm, pixi) and development tools.

### Environment Setup

Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

### Dependencies

Install API dependencies:
```bash
pip install -r api/requirements.txt
```

For download scripts (optional):
```bash
pip install spotdl  # For download_missing.py
pip install yt-dlp  # For download_ytdlp.py
```

### Spotify API Setup

1. Create a Spotify app at [Spotify Dashboard](https://developer.spotify.com/dashboard)
   - Click "Create app"
   - Set Redirect URI to `http://127.0.0.1:3000/callback`
   - Copy Client ID and Client Secret from Settings

2. Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```

3. Add credentials to `.env`:
   ```
   SPOTIFY_CLIENT_ID=your_client_id
   SPOTIFY_CLIENT_SECRET=your_client_secret
   ```

## Common Commands

### Fetch Spotify Track Metadata
```bash
python api/fetch_audio_data.py
```
First run opens browser for authorization. Token is cached in `.cache` file for subsequent runs.

### Check Download Status
```bash
python download/check_matches.py
```
Shows statistics on matched vs missing tracks, duplicates, and sample outputs.

### Download Missing Tracks
```bash
# Option 1: Using spotdl (recommended, faster batch downloads)
python download/download_missing.py

# Option 2: Using yt-dlp (fallback, YouTube search)
python download/download_ytdlp.py
```

Both scripts:
- Compare `api/data/saved_tracks.json` with `songs/*.mp3`
- Show statistics before downloading
- Require confirmation before starting
- Save failed downloads to `failed_*.txt` files

## Spotify API Endpoints Used

- **Get User's Saved Tracks**: `GET /v1/me/tracks` - Fetch all saved/liked songs with pagination

## Data Management

- Track metadata: `api/data/saved_tracks.json`
- Downloaded MP3s: `songs/` directory
- Both `data/` and `songs/` folders are gitignored to keep listening data private
- OAuth tokens cached in `.cache` file (also gitignored)

## Testing

Once tests are added, they can be run with:
```bash
pytest  # Standard pytest execution
pytest -v  # Verbose output
pytest tests/test_specific.py  # Single test file
pytest -k test_function_name  # Single test function
```

## Code Quality

The .gitignore includes ruff cache, suggesting potential use of ruff for linting:
```bash
ruff check .  # Lint code
ruff format .  # Format code
```

## Jupyter Notebooks

The project supports Jupyter notebooks (.ipynb_checkpoints are ignored). Run notebooks with:
```bash
jupyter notebook
# or
jupyter lab
```
