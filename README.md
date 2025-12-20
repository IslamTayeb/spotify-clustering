# Music Taste Analysis

Analyzes your Spotify library using deep learning audio features and lyric embeddings to discover clusters in your musical taste.

## Features

- **Audio Analysis**: Extracts 1280-dimensional embeddings using Essentia's deep learning models
- **Lyric Analysis**: Generates multilingual embeddings using sentence-transformers
- **Clustering**: UMAP dimensionality reduction + HDBSCAN clustering
- **Visualization**: Interactive Plotly-based HTML visualization
- **Reporting**: Detailed markdown reports with cluster statistics

## Quick Start

### 1. Setup Environment

Using pyenv:
```bash
pyenv activate spotify-clustering
pip install -r requirements.txt
```

Or using venv:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Essentia Models

Download the required TensorFlow models (~22MB total):
```bash
python download_models.py
```

This downloads models to `~/.essentia/models/` (one-time setup).

### 3. Prepare Data

Ensure you have:
- MP3 files in `songs/data/`
- Lyrics in `lyrics/data/`
- Master index at `spotify/master_index.json`

To fetch your Spotify data:
```bash
python spotify/fetch_audio_data.py
python lyrics/fetch_lyrics.py
python tools/build_master_index.py
```

### 4. Run Analysis

First run (extracts all features, ~90 minutes for 1,500 songs):
```bash
python run_analysis.py
```

Subsequent runs (uses cache):
```bash
python run_analysis.py --use-cache
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
├── analysis/
│   ├── audio_analysis.py       # Essentia-based audio feature extraction
│   ├── lyric_analysis.py       # Sentence-transformer lyric embeddings
│   ├── clustering.py           # UMAP + HDBSCAN clustering
│   └── visualization.py        # Plotly visualizations and reports
├── spotify/
│   ├── fetch_audio_data.py     # Fetch tracks from Spotify API
│   └── master_index.json       # Unified track mapping
├── songs/data/                 # MP3 files
├── lyrics/data/                # Lyric text files
├── cache/                      # Feature cache (speeds up re-runs)
├── outputs/                    # Analysis results
├── logging/                    # Log files with timestamps
├── download_models.py          # Download Essentia TensorFlow models
├── run_analysis.py             # Main pipeline script
└── requirements.txt
```

## How It Works

1. **Audio Feature Extraction**: Uses Essentia's pre-trained models to extract:
   - 1280-dimensional embeddings
   - 400 genre probabilities
   - Mood scores (happy, sad, aggressive, relaxed, party)
   - Musical attributes (BPM, key, danceability, instrumentalness)

2. **Lyric Feature Extraction**: Uses sentence-transformers to create:
   - 384-dimensional multilingual embeddings
   - Language detection
   - Word counts

3. **Dimensionality Reduction**: UMAP reduces high-dimensional features to 2D for visualization

4. **Clustering**: HDBSCAN identifies natural clusters and outliers

5. **Visualization & Reporting**: Generates interactive HTML and detailed markdown reports

## Requirements

- Python 3.10+
- ~4GB RAM
- ~90 minutes processing time (first run)
- Internet connection (for downloading ML models on first run)

## Troubleshooting

**"Essentia models not found"**: Models download automatically on first run. Ensure internet connection.

**"Memory error"**: Reduce batch size in `src/audio_analysis.py` (line ~134) from 100 to 50.

**"Too many outliers"**: Normal if your taste is diverse. Try decreasing `min_cluster_size` in `src/clustering.py`.

**Missing features**: Make sure you've run the data preparation scripts first:
```bash
python spotify/fetch_audio_data.py
python lyrics/fetch_lyrics.py
python tools/build_master_index.py
```

## Model Information

### Essentia Models
- Embeddings: discogs-effnet-bs64-1 (1280-dim)
- Genre: genre_discogs400 (400 classes)
- Mood models: happy, sad, aggressive, relaxed, party
- Musical attributes: danceability, BPM, key detection

### Sentence-Transformers
- Model: paraphrase-multilingual-MiniLM-L12-v2 (384-dim)
- Supports 50+ languages

All models download automatically to:
- Essentia: `~/.essentia/models/`
- Sentence-transformers: `~/.cache/torch/`

Total download size: ~500MB (one-time)

## Performance

| Stage | Expected Time |
|-------|---------------|
| Audio feature extraction | 60-90 minutes (1,500 songs) |
| Lyric feature extraction | 5-10 minutes |
| Clustering | 2-5 minutes |
| Visualization | 30-60 seconds |
| **Total (first run)** | **~90 minutes** |
| **Total (with cache)** | **<5 minutes** |

## License

This project uses:
- [Essentia](https://essentia.upf.edu/) (AGPL-3.0)
- [Sentence-Transformers](https://www.sbert.net/) (Apache-2.0)
- [UMAP](https://umap-learn.readthedocs.io/) (BSD-3-Clause)
- [HDBSCAN](https://hdbscan.readthedocs.io/) (BSD-3-Clause)
- [Plotly](https://plotly.com/python/) (MIT)

## Contributing

This is a personal project for analyzing music taste. Feel free to fork and adapt for your own use.

## Acknowledgments

- Essentia team for audio analysis models
- Sentence-Transformers team for multilingual embeddings
- UMAP and HDBSCAN developers for clustering algorithms
