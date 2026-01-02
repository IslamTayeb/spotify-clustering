# Spotify Music Taste Clustering

Cluster your Spotify library using interpretable audio features and GPT-powered lyric analysis. Produces a **33-dimensional interpretable feature vector** where every dimension has human-readable meaning.

**For the full methodology, design decisions, and technical deep-dives, see [the accompanying essay](YOUR_ESSAY_URL_HERE).**

**Time estimate:** ~3-4 hours for 1,500 songs (mostly waiting: downloads, audio extraction, GPT API calls). All steps cache progress, so you can stop/resume.

---

## What It Does

1. Extracts your Spotify library w/ Metadata
2. Downloads your Spotify saved songs as MP3s
3. Fetches lyrics from Genius + MusixMatch
4. Extracts audio features via Essentia (genre, mood, energy, etc.)
5. Classifies lyrics via GPT (valence, themes, explicit content, etc.)
6. Clusters songs into meaningful groups
7. Visualizes with interactive 3D UMAP

---

## Quickstart

### Prerequisites

- Python 3.9+
- FFmpeg (`brew install ffmpeg` / `apt install ffmpeg`)
- ~2GB disk for Essentia models

### 1. Setup

Clone and install dependencies.

```bash
git clone https://github.com/yourusername/spotify-clustering.git
cd spotify-clustering
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. API Keys

You'll need Spotify (to fetch your library), Genius (for lyrics), and OpenAI (for lyric classification).

```bash
# .env file

# Spotify - https://developer.spotify.com/dashboard
# Set redirect URI to http://127.0.0.1:3000/callback
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...

# Genius - https://genius.com/api-clients
GENIUS_ACCESS_TOKEN=...

# OpenAI - https://platform.openai.com/api-keys
OPENAI_API_KEY=...
```

### 3. Fetch Your Library

Pulls your saved tracks metadata from Spotify. First run opens browser for OAuth.

```bash
python spotify/fetch_spotify_saved_songs.py
```

### 4. Download Songs

Downloads MP3s for local audio analysis. Safe to stop/resume.

```bash
python songs/download_via_spotdl.py   # or download_via_ytdlp.py
```

### 5. Fetch Lyrics

Fetches lyrics from Genius. Also safe to stop/resume.

```bash
python lyrics/fetch_lyrics.py
```

### 6. Run Analysis

Extracts audio features (Essentia) and classifies lyrics (GPT). First run is slow (~2-3 hours for 1,500 songs: ~90 min audio extraction + ~60 min GPT API calls). Uses cache afterward.

```bash
python analysis/run_analysis.py --songs songs/data/ --lyrics lyrics/data/
```

### 7. Interactive Dashboard

Explore clusters, tune parameters, and visualize results.

```bash
streamlit run analysis/interactive_interpretability.py
```

### 8. Export to Spotify Playlists (Optional)

Creates Spotify playlists from your clusters.

```bash
python export/export_clusters_as_playlists.py --dry-run  # preview
python export/export_clusters_as_playlists.py            # create
```

---

## Key Files

| File | Purpose |
|------|---------|
| `analysis/run_analysis.py` | Main entry point |
| `analysis/interactive_interpretability.py` | Streamlit dashboard |
| `analysis/pipeline/interpretable_features.py` | 33-dim vector construction |
| `analysis/pipeline/audio_analysis.py` | Essentia feature extraction |
| `analysis/pipeline/lyric_features.py` | GPT lyric classification |
| `analysis/pipeline/config.py` | Configuration & scales |
