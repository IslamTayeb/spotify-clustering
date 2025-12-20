# Music Taste Analysis: Technical Specification

**Target**: Full implementation of audio + lyric analysis pipeline with static HTML visualization
**Platform**: MacBook M3, Python 3.10+
**Dataset**: 1,500 MP3 files with separate lyric files
**Deliverable**: Self-contained HTML file + markdown report

---

## 1. Project Structure

```
music_analysis/
├── data/
│   ├── songs/           # MP3 files
│   └── lyrics/          # Text files (same filename as MP3)
├── outputs/
│   ├── music_taste_map.html
│   ├── music_taste_report.md
│   ├── outliers.txt
│   └── analysis_data.pkl  # Serialized results for future use
├── cache/
│   ├── audio_features.pkl
│   └── lyric_features.pkl
├── src/
│   ├── audio_analysis.py
│   ├── lyric_analysis.py
│   ├── clustering.py
│   └── visualization.py
├── requirements.txt
├── run_analysis.py
└── README.md
```

---

## 2. Dependencies

### requirements.txt

```
essentia==2.1b6.dev1110
essentia-tensorflow==2.1b6.dev1110
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
umap-learn>=0.5.3
hdbscan>=0.8.33
plotly>=5.14.0
sentence-transformers>=2.2.2
langdetect>=1.0.9
tqdm>=4.65.0
```

**Installation notes:**

- Essentia requires: `pip install essentia-tensorflow` (includes both essentia and TensorFlow models)
- Sentence-transformers will download `paraphrase-multilingual-MiniLM-L12-v2` on first run (~120MB)
- UMAP may require: `pip install pynndescent` for performance

---

## 3. Audio Feature Extraction (`audio_analysis.py`)

### Input

- Directory of MP3 files
- Caching mechanism to avoid reprocessing

### Models to Use

```python
ESSENTIA_MODELS = {
    'embeddings': 'discogs-effnet-bs64-1',  # Output: 1280-dim vector
    'genre': 'genre_discogs400-discogs-effnet-1',  # Output: 400 genre probs
    'mood_happy': 'mood_happy-discogs-effnet-1',
    'mood_sad': 'mood_sad-discogs-effnet-1',
    'mood_aggressive': 'mood_aggressive-discogs-effnet-1',
    'mood_relaxed': 'mood_relaxed-discogs-effnet-1',
    'mood_party': 'mood_party-discogs-effnet-1',
    'arousal': 'deam-msd-musicnn-2',  # Output: arousal score
    'valence': 'deam-msd-musicnn-2',  # Output: valence score
    'danceability': 'danceability-discogs-effnet-1',
    'voice_instrumental': 'voice_instrumental-discogs-effnet-1',
    'key': 'key_edma',  # Key detection
    'bpm': 'tempocnn'   # Tempo detection
}
```

### Output Schema

```python
audio_features = {
    'filename': str,
    'embedding': np.array(1280,),  # For clustering
    'genre_probs': np.array(400,),
    'top_3_genres': List[Tuple[str, float]],  # Top 3 genres with probabilities
    'mood_happy': float,
    'mood_sad': float,
    'mood_aggressive': float,
    'mood_relaxed': float,
    'mood_party': float,
    'arousal': float,
    'valence': float,
    'danceability': float,
    'instrumentalness': float,  # 0=vocal, 1=instrumental
    'key': str,  # e.g., "C major", "A minor"
    'bpm': float,
    'duration_seconds': float
}
```

### Implementation Requirements

1. **Progress bar** for batch processing (use `tqdm`)
2. **Error handling**:
   - Skip corrupted MP3s, log to `errors.log`
   - Continue processing if one model fails (log and use NaN)
3. **Caching**: Save `audio_features.pkl` after each batch (every 100 songs)
4. **Memory management**: Process in batches of 50 songs if memory issues occur
5. **Normalization**: Keep raw values; normalization happens in clustering stage

### Essentia Usage Pattern

```python
import essentia.standard as es

# Load audio
loader = es.MonoLoader(filename=filepath, sampleRate=16000)
audio = loader()

# Extract embeddings
embedding_model = es.TensorflowPredictEffnetDiscogs(
    graphFilename='discogs-effnet-bs64-1.pb',
    output="PartitionedCall:1"
)
embeddings = embedding_model(audio)

# Extract genre
genre_model = es.TensorflowPredict2D(
    graphFilename='genre_discogs400-discogs-effnet-1.pb'
)
genre_probs = genre_model(embeddings)
```

**Note**: Essentia models are downloaded automatically on first use to `~/.essentia/models/`

---

## 4. Lyric Feature Extraction (`lyric_analysis.py`)

### Input

- Directory of text files (UTF-8 encoded)
- One text file per song, matching MP3 filename

### Model

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

### Chunking Strategy

**Problem**: Model has 512-token limit, lyrics can exceed this
**Solution**: Chunk and average

```python
def embed_lyrics(text, model, max_length=512):
    """
    Split lyrics into chunks, embed each, return mean embedding.
    Handles empty/instrumental tracks.
    """
    if not text or len(text.strip()) < 10:
        return np.zeros(384)  # Return zero vector for instrumentals

    # Tokenize and chunk
    tokens = model.tokenizer.tokenize(text)
    chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]

    # Embed each chunk
    chunk_embeddings = []
    for chunk in chunks:
        chunk_text = model.tokenizer.convert_tokens_to_string(chunk)
        emb = model.encode(chunk_text, convert_to_numpy=True)
        chunk_embeddings.append(emb)

    # Return mean
    return np.mean(chunk_embeddings, axis=0)
```

### Language Detection

```python
from langdetect import detect, LangDetectException

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'
```

### Output Schema

```python
lyric_features = {
    'filename': str,
    'embedding': np.array(384,),  # For clustering
    'language': str,  # ISO 639-1 code (e.g., 'en', 'es', 'ar')
    'word_count': int,
    'is_instrumental': bool  # True if lyrics are empty/very short
}
```

### Implementation Requirements

1. **Handle missing files**: If no lyric file exists, mark as instrumental
2. **Encoding detection**: Try UTF-8, fall back to latin-1 if needed
3. **Progress bar**: Same as audio processing
4. **Caching**: Save `lyric_features.pkl`

---

## 5. Clustering Pipeline (`clustering.py`)

### Three Clustering Modes

1. **Audio-only clustering** (based on audio embeddings)
2. **Lyric-only clustering** (based on lyric embeddings)
3. **Combined clustering** (concatenated normalized embeddings)

### Preprocessing

```python
from sklearn.preprocessing import StandardScaler

def prepare_features(audio_emb, lyric_emb, mode='combined'):
    """
    Normalize and combine embeddings.

    Args:
        audio_emb: (n_songs, 1280)
        lyric_emb: (n_songs, 384)
        mode: 'audio', 'lyrics', 'combined'

    Returns:
        features: (n_songs, feature_dim)
    """
    if mode == 'audio':
        return StandardScaler().fit_transform(audio_emb)
    elif mode == 'lyrics':
        # Filter out instrumental songs (zero vectors)
        non_instrumental = (lyric_emb != 0).any(axis=1)
        return StandardScaler().fit_transform(lyric_emb[non_instrumental])
    else:  # combined
        audio_norm = StandardScaler().fit_transform(audio_emb)
        lyric_norm = StandardScaler().fit_transform(lyric_emb)
        return np.hstack([audio_norm, lyric_norm])
```

### Dimensionality Reduction

```python
import umap

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)

umap_2d = reducer.fit_transform(features)
```

**Parameters explained:**

- `n_neighbors=15`: Local neighborhood size (15 is good for 1,500 songs)
- `min_dist=0.1`: How tightly packed points can be (0.1 = readable but compact)
- `metric='cosine'`: Standard for embeddings

### Clustering

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,      # Minimum songs per cluster
    min_samples=5,            # Conservative (more robust to noise)
    cluster_selection_epsilon=0.5,  # Merge nearby clusters
    metric='euclidean'        # Use on UMAP output
)

cluster_labels = clusterer.fit_predict(umap_2d)
```

**Expected behavior:**

- Will produce 8-15 clusters typically
- 10-20% of songs will be outliers (label=-1)
- If >30% are outliers, decrease `min_cluster_size` to 15

### Cluster Analysis

For each cluster, compute:

```python
def analyze_cluster(cluster_id, df):
    """
    Aggregate statistics for a cluster.

    Returns dict with:
    - n_songs: int
    - top_3_genres: List[Tuple[str, float]]
    - median_bpm: float
    - mood_distribution: dict
    - language_distribution: dict
    - key_distribution: dict (major vs minor %)
    - avg_danceability: float
    - representative_songs: List[str] (5 closest to centroid)
    """
    cluster_df = df[df.cluster == cluster_id]

    # Genre aggregation (across all 400 genres, sum and take top 3)
    genre_matrix = np.vstack(cluster_df['genre_probs'].values)
    avg_genre_probs = genre_matrix.mean(axis=0)
    top_genres = get_top_k_genres(avg_genre_probs, k=3)

    # Mood distribution
    mood_cols = ['mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed', 'mood_party']
    mood_dist = {col: cluster_df[col].mean() for col in mood_cols}

    # Key distribution
    major_count = cluster_df['key'].str.contains('major').sum()
    minor_count = cluster_df['key'].str.contains('minor').sum()
    key_dist = {
        'major': major_count / len(cluster_df),
        'minor': minor_count / len(cluster_df)
    }

    # Representative songs (closest to centroid)
    centroid = cluster_df[['umap_x', 'umap_y']].mean().values
    distances = np.linalg.norm(
        cluster_df[['umap_x', 'umap_y']].values - centroid,
        axis=1
    )
    representative_indices = np.argsort(distances)[:5]
    representative_songs = cluster_df.iloc[representative_indices]['filename'].tolist()

    return {...}
```

### Output Schema

```python
clustering_results = {
    'umap_coords': np.array((n_songs, 2)),
    'cluster_labels': np.array((n_songs,)),  # -1 for outliers
    'cluster_stats': {
        0: {...},  # analyze_cluster output
        1: {...},
        ...
    },
    'outlier_songs': List[str],
    'silhouette_score': float  # Overall cluster quality metric
}
```

---

## 6. Visualization (`visualization.py`)

### Interactive HTML with Plotly

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_interactive_map(df, clustering_results):
    """
    Generate self-contained HTML with:
    - Main scatter plot (UMAP projection)
    - Hover info (song name, cluster, top genre, mood)
    - Dropdown filters (cluster, mood, language)
    - Color by cluster
    - Outliers in gray
    """

    # Main scatter plot
    fig = go.Figure()

    # Add each cluster as separate trace (for legend)
    for cluster_id in sorted(df['cluster'].unique()):
        if cluster_id == -1:
            name = 'Outliers'
            color = 'lightgray'
        else:
            name = f"Cluster {cluster_id}"
            color = None  # Use default color cycle

        cluster_df = df[df['cluster'] == cluster_id]

        fig.add_trace(go.Scatter(
            x=cluster_df['umap_x'],
            y=cluster_df['umap_y'],
            mode='markers',
            name=name,
            marker=dict(
                size=8,
                color=color if cluster_id == -1 else cluster_id,
                colorscale='Viridis' if cluster_id != -1 else None,
                showscale=False,
                line=dict(width=0.5, color='white')
            ),
            text=cluster_df.apply(lambda row: (
                f"<b>{row['filename']}</b><br>"
                f"Cluster: {row['cluster']}<br>"
                f"Genre: {row['top_genre']}<br>"
                f"Mood: {row['dominant_mood']}<br>"
                f"BPM: {row['bpm']:.0f}<br>"
                f"Key: {row['key']}<br>"
                f"Language: {row['language']}"
            ), axis=1),
            hovertemplate='%{text}<extra></extra>'
        ))

    # Layout
    fig.update_layout(
        title='My Music Taste Map',
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        hovermode='closest',
        template='plotly_white',
        width=1400,
        height=900,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Add dropdown filters
    fig.update_layout(
        updatemenus=[
            # Filter by cluster
            dict(
                buttons=[
                    dict(label='All Clusters',
                         method='update',
                         args=[{'visible': [True] * len(fig.data)}]),
                    *[dict(label=f'Cluster {i}',
                           method='update',
                           args=[{'visible': [j == i for j in range(len(fig.data))]}])
                      for i in range(len(fig.data)-1)]  # Exclude outliers
                ],
                direction="down",
                showactive=True,
                x=0.15,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ]
    )

    return fig
```

### Additional Visualizations

**Embedded in the same HTML file:**

1. **Cluster size distribution** (bar chart)
2. **Mood distribution** (radar chart)
3. **Genre word cloud or treemap**
4. **BPM histogram**
5. **Language distribution** (pie chart)
6. **Key distribution** (major vs minor bar chart)

Use `plotly.subplots.make_subplots()` to arrange multiple charts.

### HTML Export

```python
fig.write_html(
    'outputs/music_taste_map.html',
    config={'displayModeBar': True, 'displaylogo': False},
    include_plotlyjs='cdn'  # Use CDN for smaller file size
)
```

**Requirements:**

- File must be self-contained (no external dependencies except Plotly CDN)
- Must work offline after initial load
- Include a "last updated" timestamp in the HTML

---

## 7. Markdown Report (`music_taste_report.md`)

### Template

```markdown
# My Music Taste Analysis Report

**Generated**: {timestamp}
**Total Songs**: {n_songs}
**Clusters Found**: {n_clusters}
**Outliers**: {n_outliers} ({outlier_percentage}%)

---

## Overview Statistics

### Genre Distribution
Top 10 genres across entire library:
1. {genre_1}: {percentage}%
2. {genre_2}: {percentage}%
...

### Mood Profile
- Happy: {avg_happy}%
- Sad: {avg_sad}%
- Aggressive: {avg_aggressive}%
- Relaxed: {avg_relaxed}%
- Party: {avg_party}%

### Musical Characteristics
- **Median BPM**: {median_bpm}
- **Key Distribution**: {major_pct}% major, {minor_pct}% minor
- **Average Danceability**: {avg_danceability}
- **Instrumental Songs**: {instrumental_pct}%

### Language Distribution
- {lang_1}: {count_1} songs ({pct_1}%)
- {lang_2}: {count_2} songs ({pct_2}%)
...

---

## Cluster Breakdown

{for each cluster}

### Cluster {id}: "{suggested_name}"

**Size**: {n_songs} songs ({percentage}% of library)

**Top Genres**:
1. {genre_1} ({prob_1}%)
2. {genre_2} ({prob_2}%)
3. {genre_3} ({prob_3}%)

**Musical Profile**:
- Median BPM: {bpm}
- Key: {major_pct}% major, {minor_pct}% minor
- Avg Danceability: {danceability}

**Mood**:
- Happy: {happy}%
- Sad: {sad}%
- Aggressive: {aggressive}%
- Relaxed: {relaxed}%

**Languages**: {lang_distribution}

**Representative Songs** (closest to cluster center):
1. {song_1}
2. {song_2}
3. {song_3}
4. {song_4}
5. {song_5}

---

{end for each cluster}

## Outliers

{n_outlier} songs didn't fit into any cluster. These might be:
- Unique experiments in your taste
- Guilty pleasures
- Songs that bridge multiple clusters

See [outliers.txt](outliers.txt) for the full list.

---

## Interesting Findings

### Audio-Lyric Contradictions
Songs where sound and words diverge:

**Happy Sound + Sad Lyrics**:
- {song_1}
- {song_2}
...

**Aggressive Sound + Relaxed Lyrics**:
- {song_1}
- {song_2}
...

### Extremes
- **Highest BPM**: {song} ({bpm} BPM)
- **Lowest BPM**: {song} ({bpm} BPM)
- **Most Danceable**: {song}
- **Least Danceable**: {song}
- **Happiest**: {song}
- **Saddest**: {song}

---

## Next Steps

1. **Listen to clusters**: Start with Cluster 0 and see if it makes intuitive sense
2. **Explore outliers**: Check `outliers.txt` for surprising inclusions
3. **Use the interactive map**: Open `music_taste_map.html` to explore visually

**Note**: Cluster names are suggestions based on features. Rename them to match your intuition!
```

---

## 8. Main Execution Script (`run_analysis.py`)

```python
#!/usr/bin/env python3
"""
Music Taste Analysis Pipeline
Run: python run_analysis.py --songs data/songs/ --lyrics data/lyrics/
"""

import argparse
from pathlib import Path
import pickle
from datetime import datetime

from src.audio_analysis import extract_audio_features
from src.lyric_analysis import extract_lyric_features
from src.clustering import run_clustering_pipeline
from src.visualization import create_interactive_map, generate_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--songs', required=True, help='Directory with MP3 files')
    parser.add_argument('--lyrics', required=True, help='Directory with lyric text files')
    parser.add_argument('--use-cache', action='store_true', help='Use cached features if available')
    args = parser.parse_args()

    # Create directories
    Path('outputs').mkdir(parents=True, exist_ok=True)
    Path('cache').mkdir(exist_ok=True)

    print("=" * 60)
    print("MUSIC TASTE ANALYSIS PIPELINE")
    print("=" * 60)

    # Step 1: Audio Features
    print("\n[1/5] Extracting audio features...")
    if args.use_cache and Path('cache/audio_features.pkl').exists():
        print("  Loading from cache...")
        audio_features = pickle.load(open('cache/audio_features.pkl', 'rb'))
    else:
        audio_features = extract_audio_features(args.songs)
        pickle.dump(audio_features, open('cache/audio_features.pkl', 'wb'))
    print(f"  ✓ Processed {len(audio_features)} songs")

    # Step 2: Lyric Features
    print("\n[2/5] Extracting lyric features...")
    if args.use_cache and Path('cache/lyric_features.pkl').exists():
        print("  Loading from cache...")
        lyric_features = pickle.load(open('cache/lyric_features.pkl', 'rb'))
    else:
        lyric_features = extract_lyric_features(args.lyrics)
        pickle.dump(lyric_features, open('cache/lyric_features.pkl', 'wb'))
    print(f"  ✓ Processed {len(lyric_features)} songs")

    # Step 3: Clustering
    print("\n[3/5] Running clustering pipeline...")
    results = run_clustering_pipeline(audio_features, lyric_features)
    print(f"  ✓ Found {results['n_clusters']} clusters")
    print(f"  ✓ Outliers: {len(results['outliers'])} songs")

    # Step 4: Visualization
    print("\n[4/5] Generating interactive visualization...")
    fig = create_interactive_map(results['dataframe'], results)
    fig.write_html('outputs/music_taste_map.html')
    print("  ✓ Saved to outputs/music_taste_map.html")

    # Step 5: Report
    print("\n[5/5] Generating report...")
    generate_report(results, output_dir='outputs')
    print("  ✓ Report saved to outputs/music_taste_report.md")

    # Save complete results
    pickle.dump(results, open('outputs/analysis_data.pkl', 'wb'))

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nOutputs:")
    print("  - outputs/music_taste_map.html (interactive visualization)")
    print("  - outputs/music_taste_report.md (detailed report)")
    print("  - outputs/outliers.txt (unclustered songs)")
    print(f"\nTotal time: {datetime.now()}")

if __name__ == '__main__':
    main()
```

---

## 9. Error Handling & Edge Cases

### Must Handle

1. **Corrupted MP3 files**: Skip and log to `errors.log`
2. **Missing lyrics**: Mark as instrumental, use zero vector
3. **Non-UTF-8 text encoding**: Try latin-1 fallback
4. **Very short songs (<10 seconds)**: Flag but process
5. **Essentia model download failures**: Retry with exponential backoff
6. **Memory errors**: Process in smaller batches (reduce from 50 to 25)
7. **Too few clusters**: If only 2-3 clusters, decrease `min_cluster_size`
8. **Too many outliers (>30%)**: Decrease `min_cluster_size` to 15 or lower

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
```

---

## 10. Success Criteria

### The pipeline succeeds if

1. ✅ All audio features extracted with <5% error rate
2. ✅ All lyric features extracted with <10% error rate (missing lyrics expected)
3. ✅ Clustering produces 5-20 clusters
4. ✅ Outliers are 10-30% of dataset
5. ✅ Silhouette score > 0.2 (indicates meaningful clusters)
6. ✅ HTML file loads and is interactive
7. ✅ Report contains no "NaN" or missing statistics

### Manual validation

- Random sample 5 songs from 3 different clusters
- Verify they sound/feel similar within each cluster
- Check that outliers genuinely don't fit anywhere

---

## 11. Performance Targets

| Stage | Expected Time |
|-------|---------------|
| Audio feature extraction | 60-90 minutes (1,500 songs) |
| Lyric feature extraction | 5-10 minutes |
| Clustering | 2-5 minutes |
| Visualization | 30-60 seconds |
| Report generation | 10-20 seconds |
| **Total** | **~90 minutes** |

**Optimization notes:**

- Essentia extraction is CPU-bound; no practical way to parallelize on M3
- Caching after first run reduces re-runs to <5 minutes
- If memory is an issue, reduce batch size (will increase time proportionally)

---

## 12. Future Extensions (Out of Scope for V1)

These are explicitly NOT required but documented for future reference:

1. **Temporal analysis** (requires timestamp metadata)
2. **BERTopic on lyrics** (adds 20-30 minutes processing)
3. **Recommendation engine** (requires additional similarity search infrastructure)
4. **Streamlit web app** (convert after static HTML validates)
5. **Comparative analysis** (requires second dataset)

---

## 13. Testing Checklist

Before considering the project complete:

- [ ] Run on a small test set (10 songs) to verify pipeline
- [ ] Verify HTML file works offline
- [ ] Confirm report markdown renders correctly
- [ ] Verify all clusters have sensible representative songs
- [ ] Check that genre names are human-readable (not just IDs)
- [ ] Ensure no file path leaks in HTML/report
- [ ] Test with songs that have no lyrics
- [ ] Test with non-English lyrics

---

## 14. README.md Template

```markdown
# Music Taste Analysis

Analyzes 1,500 songs using audio features and lyrics to discover clusters in your musical taste.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_analysis.py --songs data/songs/ --lyrics data/lyrics/
```

On subsequent runs (to use cached features):

```bash
python run_analysis.py --songs data/songs/ --lyrics data/lyrics/ --use-cache
```

## Outputs

- `outputs/music_taste_map.html` - Interactive visualization (double-click to open)
- `outputs/music_taste_report.md` - Detailed cluster analysis
- `outputs/outliers.txt` - Songs that don't fit any cluster

## Requirements

- Python 3.10+
- 1,500 MP3 files
- Corresponding lyric text files (optional)
- ~4GB RAM
- ~90 minutes processing time (first run)

## Troubleshooting

**"Essentia models not found"**: Models download automatically on first run. Ensure internet connection.

**"Memory error"**: Edit `audio_analysis.py` and reduce batch size from 50 to 25.

**"Too many outliers"**: Normal if your taste is very diverse. Try decreasing `min_cluster_size` in `clustering.py`.

```

---

## 15. Implementation Notes for Claude Code

### Order of Implementation

1. Start with `audio_analysis.py` (most complex, test early)
2. Then `lyric_analysis.py` (simple, quick to validate)
3. Then `clustering.py` (depends on 1+2)
4. Then `visualization.py` (depends on 3)
5. Finally `run_analysis.py` (orchestrates everything)

### Testing Strategy

- Create a `tests/` directory with 5-10 test MP3s
- Verify each stage independently before integrating
- Use `pytest` for unit tests (optional but recommended)

### Dependencies on External Data

- Essentia models: Downloaded automatically to `~/.essentia/models/`
- Sentence-transformers model: Downloaded automatically to `~/.cache/torch/`
- Total download size: ~500MB (one-time)

### Code Style

- Type hints on all function signatures
- Docstrings for all public functions
- Use `pathlib.Path` for file operations (not `os.path`)
- Follow PEP 8

---

## 16. Final Deliverable Checklist

When complete, the `outputs/` directory should contain:

```

outputs/
├── music_taste_map.html          # Main interactive visualization
├── music_taste_report.md         # Detailed written analysis
├── analysis_data.pkl             # Serialized results for future use
└── outliers.txt                  # List of unclustered songs

```

---

**END OF SPECIFICATION**

This document is complete and ready for autonomous implementation. All design decisions are made, edge cases are documented, and success criteria are defined.
