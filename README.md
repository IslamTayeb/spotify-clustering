# Music Taste Analysis

A comprehensive end-to-end pipeline that fetches your Spotify library, downloads audio files, analyzes them using deep learning, discovers clusters in your musical taste, and exports results back to Spotify playlists.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Clustering Modes](#clustering-modes)
- [Outputs](#outputs)
- [Project Structure](#project-structure)
- [Complete Data Flow](#complete-data-flow)
- [How It Works](#how-it-works)
- [Available Commands & Scripts](#available-commands--scripts)
- [Filename Normalization Strategy](#filename-normalization-strategy)
- [Spotify API Endpoints Used](#spotify-api-endpoints-used)
- [Requirements](#requirements)
- [Data Management](#data-management)
- [Troubleshooting](#troubleshooting)
- [Model Information](#model-information)
- [Performance](#performance)
- [Development](#development)
- [Advanced Usage](#advanced-usage)
- [Use Cases](#use-cases)
- [Performance Benchmarks](#performance-benchmarks)
- [Privacy and Data Security](#privacy-and-data-security)
- [License](#license)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [Related Projects](#related-projects)
- [FAQs](#faqs)

## Features

- **Spotify API Integration**: Fetch your entire saved library with metadata
- **Audio Download**: Automated download of missing tracks using spotdl or yt-dlp
- **Lyrics Fetching**: Fetch and cache lyrics from Genius API
- **Triple Audio Analysis**:
  - **Essentia Models**: 1280-dim embeddings + genre/mood/BPM interpretation (always runs)
  - **MERT Embeddings**: Optional 768-dim music-understanding transformer for clustering
  - **Interpretable Audio Features**: 17-dim human-readable vector (voice gender, genre purity, moods)
- **Triple Lyric Analysis**:
  - **BGE-M3**: 1024-dim multilingual embeddings (default, max 8192 tokens)
  - **E5-Large**: Optional 1024-dim high-quality multilingual embeddings
  - **GPT Interpretable**: 12-dim human-readable features (valence, arousal, theme, language, moods)
- **Combined Interpretable Mode**: 29-dim unified audio+lyrics vector for maximum interpretability
- **Clustering**: PCA + HAC (Hierarchical Agglomerative Clustering)
- **Lyric Themes**: TF-IDF keyword extraction, sentiment analysis, vocabulary richness
- **Visualization**: Interactive Plotly-based HTML visualization with UMAP
- **Reporting**: Detailed markdown reports with cluster statistics and lyric themes
- **Playlist Export**: Create Spotify playlists from discovered clusters
- **Interactive Apps**:
  - **Tuner**: Streamlit app for experimenting with clustering parameters
  - **Interpretability**: Streamlit app for interpretable combined audio+lyrics clustering with weight sliders

## Quick Start

### 1. Setup Environment

Create and activate a virtual environment:

Using venv:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
```

Or using pyenv:

```bash
pyenv activate spotify-clustering
```

Install core dependencies:

```bash
pip install -r requirements.txt
```

Install optional dependencies:

```bash
pip install -r api/requirements.txt  # For Spotify API integration
pip install -r requirements-lyrics.txt  # For lyrics fetching
pip install spotdl  # For audio downloads (Option 1)
pip install yt-dlp  # For audio downloads (Option 2)
```

### 2. Configure API Credentials

#### Spotify API Setup

1. Create a Spotify app at [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)
   - Click "Create app"
   - Set Redirect URI to `http://127.0.0.1:3000/callback`
   - Copy Client ID and Client Secret from Settings

2. Create `.env` file from example:

   ```bash
   cp .env.example .env
   ```

3. Add credentials to `.env`:

   ```
   SPOTIFY_CLIENT_ID=your_client_id
   SPOTIFY_CLIENT_SECRET=your_client_secret
   ```

#### Genius API Setup (Optional - for lyrics)

1. Create a Genius API client at [genius.com/api-clients](https://genius.com/api-clients)
   - Click "New API Client"
   - Fill in app details (name, website URL, redirect URI can all be `http://localhost`)
   - Generate a "Client Access Token"

2. Add token to `.env`:

   ```
   GENIUS_ACCESS_TOKEN=your_genius_access_token
   ```

#### OpenAI API Setup (Optional - for interpretable lyrics)

1. Get an API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

2. Add to `.env`:

   ```
   OPENAI_API_KEY=sk-your-api-key
   ```

3. Cost: ~$0.01-0.05 for a 1,500 song library (uses GPT-4o-mini)

### 3. Fetch Your Spotify Library

Fetch all your saved tracks from Spotify:

```bash
python api/fetch_audio_data.py
```

First run opens browser for authorization. Token is cached in `.cache` file for subsequent runs.

### 4. Download Audio Files

Check which tracks need to be downloaded:

```bash
python download/check_matches.py
```

Download missing tracks (choose one method):

```bash
# Option 1: Using spotdl (recommended, faster batch downloads)
python download/download_missing.py

# Option 2: Using yt-dlp (fallback, YouTube search)
python download/download_ytdlp.py
```

### 5. Fetch Lyrics (Optional)

Fetch lyrics for all tracks:

```bash
python lyrics/fetch_lyrics.py
```

Uses smart caching - safe to stop and resume.

### 6. Build Master Index

Create unified track mapping:

```bash
python tools/build_master_index.py
```

### 7. Download Essentia Models

Download required TensorFlow models (~22MB):

```bash
python analysis/models/download_models.py
```

Downloads to `~/.essentia/models/` (one-time setup).

### 8. Run Analysis

**Default behavior** (uses Essentia + BGE-M3):

```bash
# First run (extracts all features, ~90 minutes for 1,500 songs)
python run_analysis.py

# Subsequent runs (uses cache, <5 minutes)
python run_analysis.py --use-cache
```

**Advanced: Using MERT for clustering** (optional, higher quality):

```bash
# First run: Extract MERT embeddings (~45 min on GPU for 1,500 songs)
python run_analysis.py --audio-embedding-backend mert

# Subsequent runs: Use cached MERT embeddings
python run_analysis.py --use-cache --audio-embedding-backend mert
```

**Advanced: Using E5 for lyrics** (optional, higher quality):

```bash
# Extract E5 embeddings (~20 min on GPU for 1,500 songs)
python run_analysis.py --use-cache --lyrics-embedding-backend e5
```

**Full upgrade: MERT + E5** (best quality):

```bash
python run_analysis.py --use-cache --audio-embedding-backend mert --lyrics-embedding-backend e5 --mode combined
```

**Note**: Essentia ALWAYS runs for interpretation (genre/mood/BPM), even when using MERT for clustering. MERT/E5 create separate caches without modifying existing files.

### 9. Export to Spotify Playlists

Preview playlists without creating them:

```bash
python export/create_playlists.py --dry-run
```

Create playlists in your Spotify account:

```bash
python export/create_playlists.py
```

## Clustering Modes

- `--mode combined` (default): Uses both audio and lyric features
- `--mode audio`: Audio features only
- `--mode lyrics`: Lyric features only

Example:

```bash
python run_analysis.py --use-cache --mode audio
```

## Outputs

After running, check the `outputs/` directory:

- **music_taste_map.html** - Interactive visualization (double-click to open)
- **music_taste_report.md** - Detailed cluster analysis with statistics
- **outliers.txt** - Songs that don't fit any cluster
- **analysis_data.pkl** - Serialized results for future use

Logs are saved to `logging/analysis_YYYYMMDD_HHMMSS.log` for debugging and monitoring progress.

## Project Structure

```
spotify-clustering/
├── api/                            # Spotify Web API integration
│   ├── fetch_audio_data.py         # Fetch saved tracks with metadata
│   ├── requirements.txt            # API dependencies (spotipy, etc.)
│   └── data/
│       └── saved_tracks.json       # Raw Spotify API data
├── download/                       # Audio download scripts
│   ├── check_matches.py            # Analyze downloaded vs missing tracks
│   ├── download_missing.py         # Batch download using spotdl
│   └── download_ytdlp.py           # Alternative download using yt-dlp
├── lyrics/                         # Lyrics fetching and storage
│   ├── fetch_lyrics.py             # Fetch from Genius API
│   ├── requirements-lyrics.txt     # Lyrics dependencies
│   └── data/                       # Lyrics storage
│       ├── tracks_with_lyrics.json # Complete data with lyrics
│       ├── lyrics_only.json        # Simplified lyrics format
│       ├── lyrics_cache.json       # Cache to avoid re-fetching
│       └── individual/             # Individual text files (Artist - Song.txt)
├── analysis/                       # Music analysis pipeline
│   ├── pipeline/
│   │   ├── audio_analysis.py       # Essentia audio feature extraction
│   │   ├── mert_embedding.py       # MERT audio embeddings (optional)
│   │   ├── lyric_analysis.py       # BGE-M3/E5 lyric embeddings
│   │   ├── lyric_features.py       # GPT-based interpretable lyric features
│   │   ├── clustering.py           # PCA + HAC/Birch/Spectral with embedding overrides
│   │   ├── visualization.py        # Plotly visualizations and reports
│   │   └── genre_ladder.py         # Genre purity/fusion (entropy-based)
│   ├── interpretability/
│   │   └── lyric_themes.py         # TF-IDF keywords, sentiment analysis
│   ├── models/
│   │   ├── download_models.py      # Download Essentia TensorFlow models
│   │   └── list_models.py          # List available models
│   ├── scripts/                    # Utility analysis scripts
│   ├── outputs/                    # Analysis results
│   │   ├── music_taste_map.html    # Interactive visualization
│   │   ├── music_taste_report.md   # Detailed cluster analysis
│   │   ├── outliers.txt            # Unclustered songs
│   │   └── analysis_data.pkl       # Serialized results
│   ├── interactive_tuner.py        # Streamlit app for parameter tuning
│   └── interactive_interpretability.py  # Streamlit app for interpretable clustering
├── export/                         # Spotify playlist export
│   ├── create_playlists.py         # Create playlists from clusters
│   ├── README.md                   # Export documentation
│   └── requirements.txt            # Export dependencies
├── spotify/                        # Spotify data and indexing
│   ├── fetch_audio_data.py         # Legacy script (use api/ instead)
│   └── master_index.json           # Unified track mapping with file paths
├── tools/                          # General utilities
│   ├── build_master_index.py       # Build unified track index
│   ├── clean_lyrics.py             # Clean lyric data
│   ├── deduplicate_songs.py        # Remove duplicate MP3s
│   ├── verify_cache.py             # Verify cached features
│   └── validate_genre_ladder.py    # Validate genre purity/fusion scores
├── songs/data/                     # MP3 files (gitignored)
├── cache/                          # Feature cache (speeds up re-runs)
│   ├── audio_features.pkl          # Cached audio features
│   └── lyric_features.pkl          # Cached lyric features
├── logging/                        # Timestamped log files
├── run_analysis.py                 # Main pipeline orchestration script
├── requirements.txt                # Core analysis dependencies
├── .env                            # API credentials (gitignored)
├── .cache                          # Spotify OAuth token (gitignored)
├── CLAUDE.md                       # Project instructions for Claude Code
└── README.md                       # This file
```

## Complete Data Flow

### Phase 1: Data Acquisition

1. **Fetch Spotify Library** (`api/fetch_audio_data.py`)
   - Connects to Spotify API using OAuth 2.0
   - Fetches all saved tracks with comprehensive metadata
   - Saves to `api/data/saved_tracks.json`

2. **Check Download Status** (`download/check_matches.py`)
   - Compares Spotify library with existing MP3 files
   - Uses aggressive filename normalization for matching
   - Reports matched, missing, and duplicate tracks

3. **Download Missing Audio** (`download/download_missing.py` or `download/download_ytdlp.py`)
   - Downloads missing tracks from YouTube
   - Applies same normalization strategy for consistency
   - Saves MP3s to `songs/data/`
   - Logs failed downloads for manual review

4. **Fetch Lyrics** (`lyrics/fetch_lyrics.py`) - Optional
   - Fetches lyrics from Genius API
   - Uses smart caching to avoid re-fetching
   - Saves multiple formats to `lyrics/data/`
   - Resume-safe with progress tracking

5. **Build Master Index** (`tools/build_master_index.py`)
   - Creates unified track mapping
   - Links Spotify metadata with local MP3 and lyric files
   - Saves to `spotify/master_index.json`

### Phase 2: Feature Extraction

1. **Download ML Models** (`analysis/models/download_models.py`)
   - One-time download of Essentia TensorFlow models (~22MB)
   - Saves to `~/.essentia/models/`

2. **Extract Audio Features** (`analysis/pipeline/audio_analysis.py`)
   - Processes all MP3 files using Essentia
   - Extracts 1280-dim embeddings, 400 genre probabilities, moods, BPM, key
   - Extracts voice gender (male/female vocals) and production features
   - Computes genre ladder (entropy-based purity/fusion score)
   - Caches results to `cache/audio_features.pkl`

3. **Extract Lyric Features** (`analysis/pipeline/lyric_analysis.py`)
   - Processes lyric text files using sentence-transformers
   - Generates 384-dim multilingual embeddings
   - Detects language and counts words
   - Caches results to `cache/lyric_features.pkl`

### Phase 3: Clustering & Visualization

1. **Dimensionality Reduction** (`analysis/pipeline/clustering.py`)
   - PCA for clustering (reduces to optimal dimensions)
   - UMAP for 2D visualization

2. **Clustering** (`analysis/pipeline/clustering.py`)
    - HAC (Hierarchical Agglomerative Clustering)
    - Alternative algorithms: Birch, Spectral (configurable)
    - Identifies natural groupings in musical taste

3. **Generate Outputs** (`analysis/pipeline/visualization.py`)
    - Interactive HTML visualization with Plotly
    - Detailed markdown report with cluster statistics
    - Outliers file (if any)
    - Serialized results for future use

### Phase 4: Export to Spotify

1. **Create Playlists** (`export/create_playlists.py`)
    - Reads clustering results from `analysis/outputs/analysis_data.pkl`
    - Creates up to 10 playlists (5 audio clusters + 5 lyric clusters)
    - Names playlists based on cluster characteristics
    - Supports dry-run mode for preview

## How It Works

### Dual-Path Audio Architecture

The pipeline uses a **dual-path architecture** for audio analysis:

**Path 1: Essentia (Interpretation) - ALWAYS RUNS**

- Extracts human-readable musical attributes
- Genre probabilities, mood scores, BPM, key, danceability
- Used for cluster interpretation and reporting
- Cache: `cache/audio_features.pkl` (never modified by MERT)

**Path 2: MERT (Clustering) - OPTIONAL**

- Semantic audio embeddings optimized for music understanding
- Used for clustering when `--audio-embedding-backend mert`
- Separate cache: `cache/mert_embeddings_*.pkl`
- Passed as in-memory override without modifying Essentia cache

**Why both?**

- **Essentia**: Fast, interpretable, provides genre/mood/BPM tags
- **MERT**: Deep semantic understanding, better clustering quality
- **Together**: Best of both worlds - accurate clustering + interpretable results

### Audio Feature Extraction

**Essentia (Default for clustering)**

- **Embeddings**: 1280-dimensional audio representations (discogs-effnet-bs64-1)
- **Genre**: 400 genre probabilities (genre_discogs400)
- **Moods**: happy, sad, aggressive, relaxed, party (5 separate models)
- **Musical Attributes**: BPM, key, danceability, instrumentalness, valence, arousal
- **Voice Gender**: Male/female vocal probability (gender-discogs-effnet)
- **Production**: Acoustic/electronic, timbre (bright/dark)

**MERT (Optional for clustering)**

- **Embeddings**: 768-dimensional transformer representations
- **Preprocessing**: 24kHz mono, 30s excerpts, L2-normalized
- **Optimized for**: Music similarity, semantic understanding
- **Use case**: Higher quality clustering of similar-sounding tracks

**Interpretable Audio Feature Vector (17 dimensions)**

When using `--audio-embedding-backend interpretable`, the pipeline constructs a human-readable feature vector instead of raw embeddings:

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | BPM (normalized) | 0-1 | Tempo relative to library min/max |
| 1 | Danceability | 0-1 | How suitable for dancing |
| 2 | Instrumentalness | 0-1 | Vocals (0) vs instrumental (1) |
| 3 | Valence | 0-1 | Musical positivity/happiness |
| 4 | Arousal | 0-1 | Energy/intensity level |
| 5 | Engagement | 0-1 | How attention-holding |
| 6 | Approachability | 0-1 | Mainstream accessibility |
| 7-11 | Moods | 0-1 | Happy, sad, aggressive, relaxed, party |
| 12 | Voice Gender | 0-1 | Female (0) ↔ Male (1) vocals |
| 13 | Genre Ladder | 0-1 | Pure genre (0) ↔ Genre fusion (1) |
| 14-16 | Key Features | 0-1 | Musical key encoding (3D) |

**Why Interpretable?**

- Each dimension has clear meaning (vs opaque 1280-dim embeddings)
- Weights can be adjusted via interactive sliders
- Easier to understand why songs cluster together
- Combines multiple Essentia models into unified representation

**Combined Interpretable Feature Vector (29 dimensions)**

The Interactive Interpretability App constructs a 29-dimensional vector that combines audio AND lyric features for maximum interpretability:

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0-6 | Core Audio | 0-1 | BPM, danceability, instrumentalness, valence, arousal, engagement, approachability |
| 7-11 | Audio Moods | 0-1 | Happy, sad, aggressive, relaxed, party |
| 12 | Voice Gender | 0-1 | Female (0) ↔ Male (1) vocals |
| 13 | Genre Ladder | 0-1 | Pure genre (0) ↔ Genre fusion (1) |
| 14-16 | Key Features | 0-1 | Musical key encoding (3D) |
| 17-18 | Lyric Emotion | 0-1 | Valence, arousal from lyrics |
| 19-22 | Lyric Moods | 0-1 | Happy, sad, aggressive, relaxed from lyrics |
| 23-26 | Lyric Content | 0-1 | Explicit, narrative, vocabulary richness, repetition |
| 27 | Theme | 0-1 | Semantic scale (party=1.0 → none=0.0) |
| 28 | Language | 0-1 | Ordinal scale (english=1.0 → none=0.0) |

**Theme Scale (Semantic Encoding)**

Themes are encoded on a 0-1 scale based on energy/positivity:

| Theme | Value | Description |
|-------|-------|-------------|
| party | 1.0 | Highest energy |
| flex | 0.9 | Confident, boastful |
| love | 0.8 | Positive emotion |
| social | 0.7 | Community focused |
| spirituality | 0.6 | Contemplative but uplifting |
| introspection | 0.5 | Neutral, internal |
| street | 0.4 | Raw, realistic |
| heartbreak | 0.3 | Sad |
| struggle | 0.2 | Difficult |
| other | 0.1 | Has lyrics, unknown theme |
| none | 0.0 | No lyrics/instrumental |

**Language Scale (Ordinal Encoding)**

Languages are encoded on a 0-1 scale with even spacing:

| Language | Value |
|----------|-------|
| english | 1.0 |
| spanish | 0.86 |
| french | 0.71 |
| arabic | 0.57 |
| korean | 0.43 |
| japanese | 0.29 |
| unknown | 0.14 |
| none | 0.0 |

### Genre Ladder (Entropy-based)

The genre ladder measures **how categorizable** a song is - not what genre it is, but how confident the AI is about the classification:

```
0.0 ─────────────────────────────────────────────────────────────── 1.0
PURE                            MIXED                          FUSION
│                                 │                                 │
│  "This is clearly Trap"         │  "Probably Hip Hop"             │  "Could be anything"
│  AI confidence: 95%+            │  AI confidence: 50-70%          │  AI confidence: <30%
```

**How it works:**

1. Essentia's discogs400 model outputs a 400-dimensional probability vector
2. We compute **Shannon entropy**: `H(X) = -Σ p(x) × log(p(x))`
3. Low entropy = one genre dominates = pure
4. High entropy = probabilities spread across genres = fusion
5. Normalize to [0, 1] across your library

**Examples from a typical library:**

| Song | Genre | Entropy | Interpretation |
|------|-------|---------|----------------|
| BANG THAT | Cloud Rap | 0.00 | AI is 100% confident |
| New Tank | Trap | 0.06 | Very pure Hip Hop |
| Jazz track | Swing | 0.62 | Could be Jazz, Soul, or Funk |
| Collage | Soundtrack | 1.00 | AI has no idea |

**Why entropy instead of acoustic↔electronic?**

- Acoustic/electronic is **redundant** with existing features (0.71 correlation with danceability)
- Entropy is **0.54 unique** - captures something new
- Measures artistic intent: traditionalist vs genre-bender
- Won't overlap with future lyrics analysis

### Voice Gender

Captures the **vocal character** of a song using Essentia's gender classifier:

| Value | Meaning |
|-------|---------|
| 0.0 | Female vocals |
| 0.5 | Mixed/unclear |
| 1.0 | Male vocals |

**Why include this?**

- **0.67 uniqueness** - highest of any candidate feature
- Won't be captured by lyrics (text doesn't reveal voice timbre)
- Helps separate songs by vocal character
- Already computed by Essentia, zero extra cost

### Lyric Feature Extraction

**BGE-M3 (Default)**

- **Embeddings**: 1024-dimensional semantic representations
- **Context**: 8192 tokens (very long lyrics supported)
- **Languages**: 100+ multilingual support
- **Language Detection**: Automatic via langdetect

**E5-Large (Optional)**

- **Embeddings**: 1024-dimensional high-quality representations
- **Context**: 512 tokens
- **Instruction**: Uses "passage: " prefix for better encoding
- **Use case**: Higher quality lyric clustering

**GPT Interpretable Lyric Features (12 dimensions)**

When using `--extract-interpretable-lyrics`, GPT-4o-mini analyzes lyrics to extract human-readable features:

| Feature | Range | Description |
|---------|-------|-------------|
| lyric_valence | 0-1 | Emotional positivity from lyrics |
| lyric_arousal | 0-1 | Energy/intensity from lyrics |
| lyric_mood_happy | 0-1 | Happiness detected in lyrics |
| lyric_mood_sad | 0-1 | Sadness detected in lyrics |
| lyric_mood_aggressive | 0-1 | Aggression detected in lyrics |
| lyric_mood_relaxed | 0-1 | Calmness detected in lyrics |
| lyric_explicit | 0-1 | Explicit content level |
| lyric_narrative | 0-1 | Storytelling vs abstract |
| lyric_vocabulary_richness | 0-1 | Lexical diversity |
| lyric_repetition | 0-1 | How repetitive the lyrics are |
| lyric_theme | string | Primary theme (love, party, struggle, etc.) |
| lyric_language | string | Detected language |

**Default Values for Missing Lyrics**

The pipeline uses different defaults depending on the scenario:

| Scenario | Valence/Arousal | Moods | Theme | Language | Other Features |
|----------|-----------------|-------|-------|----------|----------------|
| **No lyrics** (instrumental or missing) | 0.5 (neutral) | 0.0 | `"none"` | `"none"` | All 0.0 |
| **GPT error** (lyrics exist but analysis failed) | 0.5 (neutral) | 0.0 | `"other"` | `"unknown"` | All 0.0 |

The rationale:

- **No lyrics**: Uses neutral midpoint (0.5) for valence/arousal to avoid biasing clustering
- **GPT error**: Same neutral values since we can't determine the actual emotion

**Why GPT Interpretable Lyrics?**

- Captures semantic meaning, not just embedding similarity
- Theme and language provide categorical insights
- Emotion features complement audio mood detection
- Can weight lyrics vs audio independently in clustering

### Dimensionality Reduction

- **PCA**: Reduces features to optimal dimensions for clustering (preserves variance)
- **UMAP**: Creates 2D projection for visualization (preserves local structure)

### Clustering

- **HAC (Hierarchical Agglomerative Clustering)**: Default algorithm
  - Automatically finds optimal number of clusters
  - Ward linkage for balanced cluster sizes
  - No outliers (all songs assigned to clusters)
- **Alternative Algorithms**: Birch, Spectral (experimental)

### Visualization & Reporting

- **Interactive HTML**: Plotly-based scatter plot with hover info, color-coded clusters
- **Markdown Report**: Detailed statistics, top tracks per cluster, characteristics
- **Serialized Data**: PKL format for programmatic access

## Available Commands & Scripts

### Spotify API

```bash
# Fetch all saved tracks from your Spotify library
python api/fetch_audio_data.py

# Uses OAuth 2.0 Authorization Code Flow
# First run: Opens browser for authorization
# Subsequent runs: Uses cached token from .cache file
# Requires: SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env
# Scope: user-library-read
# Output: api/data/saved_tracks.json
```

### Audio Downloads

```bash
# Check which tracks are downloaded vs missing
python download/check_matches.py

# Download missing tracks using spotdl (recommended)
python download/download_missing.py

# Alternative: Download using yt-dlp
python download/download_ytdlp.py

# All scripts use aggressive filename normalization for matching
# Downloads saved to: songs/data/
# Failed downloads logged to: failed_downloads_*.txt
```

### Lyrics

```bash
# Fetch lyrics from Genius API
python lyrics/fetch_lyrics.py

# Features:
# - Smart caching (safe to stop/resume)
# - Progress tracking
# - Multiple output formats
# Requires: GENIUS_ACCESS_TOKEN in .env
# Output: lyrics/data/ (JSON files + individual text files)
```

### Analysis Pipeline

```bash
# Main analysis script - full pipeline
python run_analysis.py

# Common options:
python run_analysis.py --use-cache              # Skip feature extraction (use cached)
python run_analysis.py --mode audio             # Audio features only
python run_analysis.py --mode lyrics            # Lyric features only
python run_analysis.py --mode combined          # Both (default)

# Embedding backend selection:
python run_analysis.py --audio-embedding-backend mert          # MERT: deep semantic understanding
python run_analysis.py --audio-embedding-backend interpretable # Interpretable: 17-dim human-readable features
python run_analysis.py --lyrics-embedding-backend e5           # E5: higher quality lyrics clustering

# Recommended for exploration (interpretable features with adjustable weights):
python run_analysis.py --use-cache --audio-embedding-backend interpretable

# Extract GPT-based interpretable lyrics (requires OPENAI_API_KEY):
python run_analysis.py --extract-interpretable-lyrics
# This extracts theme, language, valence, arousal, moods, etc. via GPT-4o-mini
# Uses existing cache, only makes GPT API calls for lyrics
# Cost: ~$0.01-0.05 depending on library size

# Full interpretable pipeline (best for understanding your clusters):
python run_analysis.py --use-cache  # First: ensure audio+lyric features cached
python run_analysis.py --extract-interpretable-lyrics  # Then: extract GPT lyric features
streamlit run analysis/interactive_interpretability.py  # Finally: explore interactively

# Advanced options (for experimenting with clustering):
python run_analysis.py --use-cache --algorithm birch
python run_analysis.py --use-cache --n-clusters 8
python run_analysis.py --use-cache --pca-components 50

# IMPORTANT: Always use --use-cache when experimenting with parameters
# Feature extraction takes ~90 minutes and should only run once

# Embedding Backends:
# Audio:
#   - essentia (default): 1280-dim EffNetDiscogs embeddings
#   - mert: 768-dim MERT-v1-95M, optimized for music understanding
#   - interpretable: 17-dim human-readable features (BPM, mood, voice gender, genre purity)
# Lyrics:
#   - bge-m3 (default): 1024-dim, 8192 token context, multilingual
#   - e5: 1024-dim E5-large, higher quality, 512 token context
#
# Note: Essentia ALWAYS runs for feature extraction.
# The "interpretable" backend uses Essentia features but constructs a 17-dim
# vector with explicit meaning (see "Interpretable Feature Vector" section).
# Great for understanding WHY songs cluster together.
```

### Interactive Tuning

```bash
# Launch Streamlit app for interactive parameter tuning
streamlit run analysis/interactive_tuner.py

# Features:
# - Live parameter adjustment
# - Real-time visualization updates
# - Compare clustering algorithms
# - Export optimal parameters
# - Interpretable mode with feature weights
```

### Interactive Interpretability (Advanced)

```bash
# Launch Streamlit app for interpretable audio+lyrics clustering
streamlit run analysis/interactive_interpretability.py

# Features:
# - Combined 29-dim interpretable feature vector (audio + lyrics)
# - Adjustable weights for each feature group:
#   - Core audio (BPM, danceability, etc.)
#   - Audio moods (happy, sad, aggressive, etc.)
#   - Lyric emotions (valence, arousal)
#   - Lyric content (explicit, narrative, vocabulary)
#   - Theme and language
# - Automatic filtering of vocal songs without lyrics
# - SHAP-based feature importance analysis
# - Real-time cluster interpretation
# - No separate lyric embeddings used - only interpretable features

# Prerequisites:
# 1. Run audio analysis: python run_analysis.py --use-cache
# 2. Extract interpretable lyrics: python run_analysis.py --extract-interpretable-lyrics
# 3. Launch app: streamlit run analysis/interactive_interpretability.py
# 4. Select "Interpretability" tab → "combined" mode
```

### Model Management

```bash
# Download Essentia TensorFlow models
python analysis/models/download_models.py

# List available models and check installation status
python analysis/models/list_models.py
```

### Playlist Export

```bash
# Preview playlists without creating (no auth needed)
python export/create_playlists.py --dry-run

# Create playlists in Spotify
python export/create_playlists.py

# Options:
python export/create_playlists.py --private             # Create private playlists
python export/create_playlists.py --prefix "My Music"   # Custom playlist prefix
python export/create_playlists.py --audio-only          # Only audio cluster playlists
python export/create_playlists.py --lyrics-only         # Only lyric cluster playlists

# Requires: playlist-modify-public and playlist-modify-private scopes
# Creates up to 10 playlists (5 audio + 5 lyrics)
# Playlist naming:
# - Audio: "Audio Cluster 0: genre_152 - Aggressive"
# - Lyrics: "Lyrics Cluster 0: Style 0 (AR)"
```

### Utilities

```bash
# Build unified track index (links Spotify data with local files)
python tools/build_master_index.py

# Clean lyric data (remove duplicates, normalize text)
python tools/clean_lyrics.py

# Remove duplicate MP3 files
python tools/deduplicate_songs.py

# Verify feature cache integrity
python tools/verify_cache.py

# Validate genre ladder feature (shows pure vs fusion songs)
python tools/validate_genre_ladder.py
# Output example:
#   BANG THAT       | Hip Hop---Cloud Rap | 0.000 (pure)
#   Collage         | Soundtrack          | 1.000 (fusion)
```

## Filename Normalization Strategy

All download scripts share a common normalization approach to ensure Spotify track names match downloaded filenames:

1. **Unicode normalization**: Japanese wave dashes → regular dashes, smart quotes → straight quotes
2. **Special characters**: Remove/replace brackets, punctuation, symbols
3. **Filesystem compatibility**: Colons → dashes, slashes → dashes
4. **Word conversion**: Ampersands → "and", "feat." → "ft"
5. **Whitespace**: Aggressive normalization of spaces, dashes, underscores
6. **Case-insensitive matching**

Example transformations:

- `"Artist: Song (Remix)"` → `artist-song-remix.mp3`
- `"Track & Artist feat. Other"` → `track-and-artist-ft-other.mp3`
- `"Japanese 〜 Title"` → `japanese-title.mp3`

This ensures reliable matching across different systems and download tools.

## Spotify API Endpoints Used

- **GET /v1/me/tracks** - Fetch user's saved/liked tracks with pagination
- **GET /v1/me** - Get current user ID for playlist creation
- **POST /v1/users/{user_id}/playlists** - Create new playlists
- **POST /v1/playlists/{playlist_id}/tracks** - Add tracks to playlists (batch: 100 tracks)

## Requirements

- Python 3.10+
- ~4GB RAM
- ~90 minutes processing time (first run)
- ~500MB disk space for ML models (one-time download)
- Internet connection (for API access and model downloads)
- OpenAI API key (optional, for interpretable lyrics - ~$0.01-0.05 per library)

## Data Management

### Data Files and Storage

- **Spotify metadata**: `api/data/saved_tracks.json` - Raw Spotify API data
- **Master index**: `spotify/master_index.json` - Unified track mapping with MP3/lyrics file paths
- **Downloaded MP3s**: `songs/data/` - Audio files (gitignored for privacy)
- **Lyrics**: `lyrics/data/` - Fetched lyrics in multiple formats (gitignored for privacy)
  - `tracks_with_lyrics.json` - Complete data with lyrics
  - `lyrics_only.json` - Simplified lyrics-only format
  - `lyrics_cache.json` - Cache to avoid re-fetching
  - `individual/` - Individual text files (Artist - Song.txt)
- **Analysis outputs**: `analysis/outputs/`
  - `music_taste_map.html` - Interactive visualization
  - `music_taste_report.md` - Detailed cluster analysis
  - `analysis_data.pkl` - Serialized results
  - `outliers.txt` - Unclustered songs (if any)
- **Feature cache**: `cache/`
  - `audio_features.pkl` - Essentia audio features (~500MB for 1,500 songs)
  - `mert_embeddings_24khz_30s_cls.pkl` - MERT audio embeddings (~9MB for 1,500 songs)
  - `lyric_features.pkl` - BGE-M3 lyric embeddings + interpretable features (~50MB for 1,500 songs)
  - `lyric_features_e5.pkl` - E5 lyric embeddings (~50MB for 1,500 songs)
  - `lyric_interpretable_features.pkl` - GPT-extracted interpretable lyric features (~1MB for 1,500 songs)
- **OAuth tokens**: `.cache` - Spotify OAuth token (gitignored)
- **Logs**: `logging/analysis_YYYYMMDD_HHMMSS.log` - Timestamped execution logs

### Cache Management

The cache system dramatically speeds up re-runs:

- **First run**: ~90 minutes (feature extraction)
- **With cache**: <5 minutes (skip extraction, only clustering/visualization)

**When to clear cache:**

- New songs added to library
- Changed feature extraction logic
- Corrupted cache files

```bash
# Verify cache integrity
python tools/verify_cache.py

# Clear cache manually
rm cache/audio_features.pkl
rm cache/lyric_features.pkl

# Next run will rebuild cache
python run_analysis.py
```

## Troubleshooting

### Common Issues

**"Essentia models not found"**

- Models download automatically on first run
- Ensure internet connection
- Check `~/.essentia/models/` directory
- Manually download: `python analysis/models/download_models.py`

**"Memory error" during feature extraction**

- Reduce batch size in `analysis/pipeline/audio_analysis.py` (line ~134) from 100 to 50
- Close other applications to free RAM
- Consider processing in smaller batches

**"Too many outliers"**

- HAC algorithm doesn't produce outliers by design
- If outliers appear, check data quality (corrupted MP3s, empty lyrics)
- Verify master index is correctly built

**"Missing features" or "No tracks found"**

- Ensure data preparation scripts have been run:

  ```bash
  python api/fetch_audio_data.py
  python download/download_missing.py
  python lyrics/fetch_lyrics.py
  python tools/build_master_index.py
  ```

- Check that files exist in expected locations
- Verify master_index.json has valid file paths

**"Spotify API authentication failed"**

- Check `.env` file has correct credentials
- Verify redirect URI is `http://127.0.0.1:3000/callback`
- Delete `.cache` file and re-authenticate
- Check app settings in Spotify Dashboard

**"Genius API rate limit exceeded"**

- Lyrics fetching respects rate limits automatically
- Script will pause and resume when limit resets
- Use `--delay` flag to slow down requests if needed

**"Download failures" with spotdl or yt-dlp**

- Check internet connection
- Some tracks may not be available on YouTube
- Failed downloads logged to `failed_downloads_*.txt`
- Try alternative download method
- Consider purchasing tracks from official sources

**"Clustering produces only 1 cluster"**

- Increase `--n-clusters` parameter
- Try different clustering algorithm (`--algorithm birch`)
- May indicate homogeneous music taste
- Use interactive tuner to experiment with parameters

**"Visualization not loading in browser"**

- File may be too large (>50MB)
- Try reducing dataset size or number of clusters
- Open HTML file directly instead of through file explorer
- Check browser console for JavaScript errors

### Performance Optimization

**Speed up feature extraction:**

- Use SSD for storage
- Increase batch size if RAM allows
- Close unnecessary applications
- Consider cloud computing for large libraries (>5,000 songs)

**Reduce memory usage:**

- Process in audio-only or lyrics-only mode
- Reduce PCA components (`--pca-components 30`)
- Use smaller batch sizes

**Speed up clustering experiments:**

- **Always use `--use-cache`** when experimenting with parameters
- Use interactive tuner for real-time parameter adjustment
- Start with small dataset to find optimal parameters

## Model Information

### Audio Analysis Models

**Essentia Models (Always runs - for interpretation)**

- **Embedding**: discogs-effnet-bs64-1 (1280-dim)
- **Genre**: genre_discogs400 (400 genre probabilities)
- **Moods**: 5 separate models (happy, sad, aggressive, relaxed, party)
- **Musical Attributes**: BPM, key, danceability, instrumentalness, valence, arousal
- **Voice/Production**: Gender, timbre (bright/dark), acoustic/electronic
- **Download**: Auto-downloads to `~/.essentia/models/` (~22MB)

**MERT (Optional - for clustering)**

- **Model**: m-a-p/MERT-v1-95M (Music-understanding Evaluation and Representation Transformer)
- **Embedding Size**: 768-dim
- **Purpose**: Semantic audio representations optimized for music understanding
- **Preprocessing**: 24kHz mono, 30s center excerpts, CLS token pooling, L2-normalized
- **HuggingFace**: [m-a-p/MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M)
- **Download**: Auto-downloads from HuggingFace (~360MB)
- **Performance**: ~45 min on GPU, ~6 hours on CPU (1,500 songs)

### Lyric Analysis Models

**BGE-M3 (Default)**

- **Model**: BAAI/bge-m3 (multilingual embedding model)
- **Embedding Size**: 1024-dim
- **Context Length**: 8192 tokens (very long lyrics supported)
- **Languages**: 100+ languages
- **HuggingFace**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)

**E5-Large (Optional - higher quality)**

- **Model**: intfloat/multilingual-e5-large
- **Embedding Size**: 1024-dim
- **Context Length**: 512 tokens
- **Languages**: 100+ languages
- **Instruction Format**: Uses "passage: " prefix for encoding
- **HuggingFace**: [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)
- **Download**: Auto-downloads from HuggingFace (~2.2GB)
- **Performance**: ~20 min on GPU, ~2 hours on CPU (1,500 songs)

### Model Storage Locations

- **Essentia**: `~/.essentia/models/`
- **MERT/E5**: `~/.cache/huggingface/` (managed by transformers)
- **Total First-Time Download**: ~2.5GB (if using all models)

### Cache Files

- `cache/audio_features.pkl` - Essentia features (~500MB for 1,500 songs)
- `cache/mert_embeddings_24khz_30s_cls.pkl` - MERT embeddings (~9MB for 1,500 songs)
- `cache/lyric_features.pkl` - BGE-M3 embeddings + interpretable features (~50MB for 1,500 songs)
- `cache/lyric_features_e5.pkl` - E5 embeddings (~50MB for 1,500 songs)
- `cache/lyric_interpretable_features.pkl` - GPT-extracted interpretable lyrics (~1MB for 1,500 songs)

**Note**: MERT/E5/GPT caches are separate and don't modify existing cache files. GPT interpretable features are also merged into `lyric_features.pkl` for convenience.

## Performance

| Stage | Expected Time |
|-------|---------------|
| Audio feature extraction | 60-90 minutes (1,500 songs) |
| Lyric feature extraction | 5-10 minutes |
| GPT interpretable lyrics | 5-10 minutes (API calls) |
| Clustering | 2-5 minutes |
| Visualization | 30-60 seconds |
| **Total (first run)** | **~90 minutes** |
| **Total (with cache)** | **<5 minutes** |
| **Interpretable pipeline** | **~100 minutes** (includes GPT) |

## Development

### Code Quality

The project uses standard Python development tools:

```bash
# Linting and formatting (if ruff is installed)
ruff check .          # Check for issues
ruff format .         # Format code

# Alternative: Use other tools
black .               # Format with black
flake8 .              # Lint with flake8
```

### Testing

Once tests are added, run with pytest:

```bash
pytest                          # Run all tests
pytest -v                       # Verbose output
pytest tests/test_specific.py   # Single test file
pytest -k test_function_name    # Single test function
```

### Jupyter Notebooks

The project supports Jupyter notebooks for exploration:

```bash
jupyter notebook      # Launch Jupyter Notebook
jupyter lab          # Launch JupyterLab
```

### Package Managers Supported

The .gitignore supports various Python package managers:

- **pip**: Standard requirements.txt
- **poetry**: Poetry lock files
- **pipenv**: Pipfile and Pipfile.lock
- **uv**: Modern fast package installer
- **pdm**: PDM project files
- **pixi**: Conda-based package manager

### Git Workflow

Standard Git workflow with comprehensive .gitignore:

- Private data (songs/, lyrics/) excluded
- Cache and output files excluded
- Environment files (.env, .cache) excluded
- IDE configs excluded

## Advanced Usage

### Custom Feature Extraction

Modify feature extraction parameters in `analysis/pipeline/`:

- `audio_analysis.py`: Change Essentia models, embedding dimensions
- `lyric_analysis.py`: Change sentence-transformer model, language processing

### Custom Clustering Algorithms

The clustering pipeline supports multiple algorithms:

- **HAC** (Hierarchical Agglomerative): Default, balanced clusters
- **Birch**: Memory-efficient, good for large datasets
- **Spectral**: Graph-based, finds complex patterns
- **DBSCAN**: Density-based, finds outliers (experimental)

Add custom algorithms in `analysis/pipeline/clustering.py`.

### Playlist Naming Customization

Modify playlist naming logic in `export/create_playlists.py`:

- Customize genre/mood mapping
- Change language codes
- Add custom metadata to names
- Adjust playlist ordering

### Integration with Other Services

The master index (`spotify/master_index.json`) provides a unified data structure for:

- Exporting to other platforms (YouTube Music, Apple Music)
- Integration with music players (Plex, Jellyfin)
- Custom recommendation engines
- Music library management tools

## Use Cases

### Personal Music Discovery

- Discover hidden patterns in your music taste
- Find clusters you didn't know existed
- Understand your mood-based listening habits
- Identify genre blends in your library

### Playlist Curation

- Auto-generate mood-based playlists
- Create workout playlists (high energy clusters)
- Find study music (calm, instrumental clusters)
- Discover new connections between artists

### Music Analysis

- Analyze evolution of taste over time (with historical data)
- Compare libraries with friends
- Study genre distributions
- Identify language diversity in listening

### DJ and Music Production

- Organize sample library by audio characteristics
- Find complementary tracks for mixing
- Identify BPM and key clusters
- Discover sonic patterns in collection

## Performance Benchmarks

Tested on MacBook Pro (M1, 16GB RAM) with 1,500 songs:

| Stage | Time (First Run) | Time (Cached) |
|-------|-----------------|---------------|
| API fetch | 2-3 min | 30 sec |
| Audio download | 30-60 min | N/A |
| Lyrics fetch | 10-15 min | 1 min |
| **Essentia audio features** | **60-90 min** | **5 sec** |
| MERT audio features (optional) | 45 min (GPU) / 6 hrs (CPU) | 5 sec |
| **BGE-M3 lyric features** | **5-10 min** | **2 sec** |
| E5 lyric features (optional) | 20 min (GPU) / 2 hrs (CPU) | 2 sec |
| GPT interpretable lyrics (optional) | 5-10 min | 2 sec |
| Clustering | 2-5 min | 2-5 min |
| Visualization | 30-60 sec | 30-60 sec |
| **Total (default)** | **~2-3 hours** | **<5 min** |
| **Total (with MERT+E5, GPU)** | **~3-4 hours** | **<5 min** |
| **Total (interpretable pipeline)** | **~2.5 hours** | **<5 min** |

**Extraction Time Comparison (1,500 songs):**

- **Default (Essentia + BGE-M3)**: ~90 min first run
- **With MERT (GPU)**: +45 min = ~135 min first run
- **With MERT + E5 (GPU)**: +45 min + 20 min = ~155 min first run
- **With MERT + E5 (CPU)**: +6 hrs + 2 hrs = ~8 hours first run
- **Interpretable (Essentia + BGE-M3 + GPT)**: ~100 min first run

Scales approximately linearly with library size:

- 500 songs: ~45 min (first run, default)
- 1,500 songs: ~90 min (first run, default)
- 5,000 songs: ~4-5 hours (first run, default)

## Privacy and Data Security

### Local Processing

- All audio analysis happens locally on your machine
- Features never leave your computer
- No telemetry or tracking

### Data Storage

- Personal listening data (songs/, lyrics/) is gitignored
- API tokens stored securely in .env (gitignored)
- OAuth tokens cached locally in .cache (gitignored)

### API Access

- Spotify API: Read-only access to your library
- Genius API: Read-only access to public lyrics
- No data shared with third parties
- You can revoke API access anytime via respective dashboards

### Best Practices

- Keep .env file secure
- Don't commit .cache or private data
- Regularly rotate API tokens
- Review API permissions in dashboards

## License

This project uses open-source libraries with various licenses:

- [Essentia](https://essentia.upf.edu/) - AGPL-3.0 (audio analysis)
- [Sentence-Transformers](https://www.sbert.net/) - Apache-2.0 (text embeddings)
- [UMAP](https://umap-learn.readthedocs.io/) - BSD-3-Clause (dimensionality reduction)
- [scikit-learn](https://scikit-learn.org/) - BSD-3-Clause (machine learning)
- [Plotly](https://plotly.com/python/) - MIT (visualization)
- [Spotipy](https://spotipy.readthedocs.io/) - MIT (Spotify API)
- [lyricsgenius](https://lyricsgenius.readthedocs.io/) - MIT (Genius API)

The project code itself is provided as-is for personal use.

## Contributing

This is a personal project for analyzing music taste, but contributions are welcome!

### How to Contribute

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Areas for Improvement

- Additional clustering algorithms
- Alternative audio feature extractors
- Support for more lyrics sources
- Enhanced visualization options
- Export to more platforms
- Performance optimizations
- Better error handling

### Reporting Issues

Please report issues on GitHub with:

- Python version
- Operating system
- Library size (approximate number of songs)
- Full error message and stack trace
- Steps to reproduce

## Acknowledgments

Special thanks to the developers and researchers behind:

- **Essentia Team** ([MTG-UPF](https://www.upf.edu/web/mtg)) - Comprehensive audio analysis models and framework
- **Sentence-Transformers Team** - Multilingual semantic embeddings that make lyric analysis possible
- **scikit-learn Developers** - Robust machine learning algorithms and pipelines
- **UMAP Developers** - Beautiful dimensionality reduction for visualization
- **Plotly Team** - Interactive visualization capabilities
- **Spotipy Maintainers** - Excellent Spotify API wrapper
- **lyricsgenius Maintainers** - Simple Genius API integration

Additional credits:

- Spotify for providing comprehensive Web API
- Genius for lyrics database and API access
- The broader Python data science community

## Related Projects

If you're interested in music analysis, check out:

- [Essentia](https://essentia.upf.edu/) - Audio analysis library
- [librosa](https://librosa.org/) - Audio and music analysis
- [Music Genre Classification](https://github.com/topics/music-genre-classification) - Various approaches
- [Every Noise at Once](https://everynoise.com/) - Genre exploration
- [Spotify API Examples](https://github.com/spotify/web-api-examples) - Official examples

## FAQs

**Q: Do I need to have Spotify Premium?**
A: No, the API works with free Spotify accounts. You just need saved/liked tracks.

**Q: Can I analyze playlists instead of my saved library?**
A: Currently, the project focuses on saved tracks. Playlist support could be added by modifying the API fetch script.

**Q: What if I don't have lyrics for all songs?**
A: The pipeline handles missing lyrics gracefully. Use `--mode audio` to skip lyric analysis entirely.

**Q: Can I use this with local MP3s not from Spotify?**
A: Yes, but you'll need to modify the data preparation pipeline to skip Spotify API fetching and create the master index manually.

**Q: How accurate is the clustering?**
A: Clustering is subjective and depends on your library diversity. Use the interactive tuner to find parameters that match your perception.

**Q: Can I export to other platforms besides Spotify?**
A: Currently, only Spotify export is supported. The master index provides data that could be adapted for other platforms.

**Q: Is there a GUI?**
A: The Streamlit interactive tuner (`analysis/interactive_tuner.py`) provides a web-based GUI for parameter tuning.

**Q: How often should I re-run the analysis?**
A: Re-run when you've added significant new tracks (50+) to your library. Use `--use-cache` to save time.

**Q: Should I use MERT or Essentia for audio clustering?**
A:

- **Essentia (default)**: Fast, interpretable, great for most users
- **MERT**: Higher quality clustering, better semantic understanding
- **Recommendation**: Start with Essentia. Try MERT if you want better clustering quality
- **Note**: Essentia always runs for interpretation (genre/mood/BPM), even when using MERT

**Q: Do I need a GPU to use MERT?**
A: No, but highly recommended:

- **GPU**: ~45 min for 1,500 songs
- **CPU**: ~6 hours for 1,500 songs
- MERT extraction is one-time; subsequent runs use cache

**Q: Will MERT overwrite my existing cache?**
A: No. MERT creates a separate cache (`cache/mert_embeddings_*.pkl`). Your existing `cache/audio_features.pkl` is never modified.

**Q: What's the difference between BGE-M3 and E5 for lyrics?**
A:

- **BGE-M3 (default)**: 8192 token context (very long lyrics), good quality
- **E5**: Higher quality embeddings, 512 token context (shorter)
- **Recommendation**: BGE-M3 is sufficient for most users

**Q: Can I mix and match embedding backends?**
A: Yes! Examples:

- Essentia audio + E5 lyrics
- MERT audio + BGE-M3 lyrics
- MERT audio + E5 lyrics (best quality, slowest extraction)
- Interpretable audio + BGE-M3 lyrics (best for understanding clusters)

**Q: What is the "interpretable" audio backend?**
A: A 17-dimensional feature vector where each dimension has explicit meaning:

- BPM, danceability, instrumentalness, valence, arousal
- Engagement, approachability, 5 mood dimensions
- Voice gender (female↔male), genre ladder (pure↔fusion)
- Key features (3D)

Use `--audio-embedding-backend interpretable` for clustering you can understand and tune.

**Q: What is the genre ladder?**
A: Measures how "categorizable" a song is using Shannon entropy:

- **0.0 = Pure**: AI is 95%+ confident about the genre (e.g., pure Trap)
- **1.0 = Fusion**: AI is confused, song crosses many genres (e.g., experimental)

It captures whether an artist works within genre traditions or breaks boundaries.

**Q: Why not use acoustic↔electronic for the genre ladder?**
A: We analyzed feature correlations and found:

- Acoustic/electronic is **0.71 correlated** with danceability (already in vector)
- Genre entropy has **0.54 uniqueness** - adds genuinely new information
- Essentia already extracts `mood_acoustic` and `mood_electronic` directly

**Q: What does voice gender capture?**
A: The vocal character of a song (female=0, male=1):

- **0.67 uniqueness** - highest of any candidate feature
- Won't be captured by lyrics (text doesn't reveal voice timbre)
- Already computed by Essentia, no extra processing needed

**Q: What is the Interactive Interpretability app?**
A: A Streamlit app (`analysis/interactive_interpretability.py`) that provides:

- 29-dimensional interpretable feature vectors (combined audio + lyrics)
- Adjustable weight sliders for each feature group
- Automatic filtering of vocal songs without lyrics
- SHAP-based feature importance analysis
- Real-time cluster visualization

Launch with: `streamlit run analysis/interactive_interpretability.py`

**Q: How do I extract interpretable lyric features?**
A: Run:

```bash
export OPENAI_API_KEY='sk-your-key'
python run_analysis.py --extract-interpretable-lyrics
```

This uses GPT-4o-mini to extract theme, language, valence, arousal, and other semantic features from your lyrics. Cost is ~$0.01-0.05 depending on library size.

**Q: What happens to vocal songs without lyrics in interpretable mode?**
A: The Interactive Interpretability app automatically filters them out and displays a warning with the list of excluded tracks. This ensures clean clustering when lyric features are important.

**Q: How is the theme/language encoded in the feature vector?**
A: Both are encoded as single 0-1 values on semantic/ordinal scales:

- **Theme**: party=1.0, flex=0.9, love=0.8, ... none=0.0 (ordered by energy/positivity)
- **Language**: english=1.0, spanish=0.86, ... none=0.0 (evenly spaced)

This keeps the vector compact (29 dims) while preserving meaningful relationships.

**Q: How do I run the full interpretable pipeline?**
A: Three steps:

```bash
# 1. Extract audio + lyric features (if not already cached)
python run_analysis.py --use-cache

# 2. Extract GPT-based interpretable lyrics
python run_analysis.py --extract-interpretable-lyrics

# 3. Launch interactive app
streamlit run analysis/interactive_interpretability.py
```

Then select "Interpretability" tab → "combined" mode.

---

**Built with ❤️ for music lovers who want to understand their taste better.**
