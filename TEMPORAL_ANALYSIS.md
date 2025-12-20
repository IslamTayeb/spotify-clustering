# Temporal Analysis - Music Taste Evolution

Your music clustering pipeline now includes comprehensive temporal analysis to understand how your music taste has evolved over time.

## What's New

### 1. Automatic Temporal Data in Clustering

When you run `python run_analysis.py --use-cache`, the clustering pipeline now **automatically**:
- Loads temporal metadata from `spotify/saved_tracks.json`
- Adds these fields to every song in your analysis:
  - `added_at` - When you saved/liked the song
  - `release_date` - When the song was originally released
  - `popularity` - Spotify popularity score
  - `album_name`, `album_type`, `explicit` - Album metadata
  - `age_at_add_years` - How old the song was when you added it
  - `added_year`, `added_month` - Temporal groupings

### 2. Enhanced Visualization Hover Tooltips

When you hover over songs in the 3D interactive map, you now see:
- **Added**: 2025-12-17 (when you saved it)
- **Released**: 2024 (original release year)
- **Age when added**: 0.3 years (how old it was when you added it)

### 3. Dedicated Temporal Analysis Script

Run comprehensive temporal analysis after clustering:

```bash
python analysis/scripts/temporal_analysis.py
```

This generates **7 interactive visualizations** in `analysis/outputs/eda/`:

#### 1. Library Growth (`library_growth.html`)
- Cumulative songs over time
- Shows how your library has grown
- Interactive time series

#### 2. Songs Per Month (`songs_per_month.html`)
- Bar chart of monthly activity
- Identifies your most active music discovery periods
- Seasonal patterns

#### 3. Cluster Timeline (`cluster_timeline.html`)
- Stacked area chart showing when you discovered each cluster
- See how your taste in different clusters evolved
- Understand which clusters are "early loves" vs "recent discoveries"

#### 4. Song Age Distribution (`song_age_at_add.html`)
- Histogram of how old songs were when you added them
- Do you add new releases or discover older music?
- Shows your discovery patterns

#### 5. Cluster-Time Heatmap (`cluster_time_heatmap.html`)
- Heatmap showing cluster preferences across time periods
- Library split into 4 quartiles (Early/Mid-Early/Mid-Late/Recent)
- See if certain clusters dominated specific periods

#### 6. Genre Evolution (`genre_evolution.html`)
- Stacked area chart of top 10 genres over time
- Track shifts in your genre preferences
- Identify genre "phases"

#### 7. Mood Trends (`mood_trends.html`)
- Rolling average of mood scores over time
- Did you go through a "sad phase"?
- Track emotional patterns in your music choices

### 4. Temporal Analysis Report

The script also generates `temporal_analysis_report.md` with:

- **Library Timeline**: Total songs, date range, average rate
- **Most Active Months**: Your top 5 months for music discovery
- **Taste Evolution by Period**: Statistics for each quartile
  - Song counts, date ranges
  - Top genres per period
  - Mood profiles per period
  - Median song age when added
- **Song Discovery Patterns**: Recent vs older music preferences
- **Cluster Evolution**: How each cluster grew over time

## Usage Examples

### Run After Clustering
```bash
# First, run the main analysis
python run_analysis.py --use-cache

# Then run temporal analysis on the results
python analysis/scripts/temporal_analysis.py
```

### Standalone Temporal Analysis
```python
from analysis.scripts.temporal_analysis import run_temporal_analysis

# Run on existing clustering results
df_with_temporal = run_temporal_analysis(
    results_pkl='analysis/outputs/analysis_data.pkl',
    saved_tracks_path='spotify/saved_tracks.json',
    output_dir='analysis/outputs/eda'
)
```

## What You Can Discover

### Music Taste Evolution
- **Early adopter or nostalgic?** Check the song age distribution
- **Genre shifts**: See how your taste changed from early to recent adds
- **Cluster discovery timeline**: Which music styles did you discover when?

### Activity Patterns
- **Most active periods**: When did you add the most music?
- **Seasonal patterns**: Do you discover more music in certain months?
- **Growth trajectory**: Steady growth or bursts of discovery?

### Emotional Journey
- **Mood trends**: Did your mood preferences change over time?
- **Happy vs sad phases**: Track emotional patterns
- **Cluster moods by period**: How cluster preferences correlate with moods

### Musical Characteristics Over Time
- **BPM trends**: Do you like faster or slower music now?
- **Key preferences**: Major vs minor over time
- **Danceability evolution**: Party music phases?

## Data Fields Available

After running the pipeline, your clustering dataframe includes:

**Temporal Fields** (added automatically):
- `added_at` (datetime): When you saved the song
- `release_date` (datetime): When the song was released
- `popularity` (int): Spotify popularity score (0-100)
- `album_name` (str): Album name
- `album_type` (str): album/single/compilation
- `explicit` (bool): Explicit content flag
- `age_at_add_years` (float): How old the song was when you added it
- `added_year` (int): Year you added it
- `added_month` (str): Month you added it (YYYY-MM format)

**Clustering Fields** (already available):
- All audio features (mood, genre, BPM, key, danceability, etc.)
- All lyric features (language, word count, embeddings, etc.)
- Cluster assignments and UMAP coordinates

## Notes

- Temporal data is **NOT** used for clustering (only for analysis/visualization)
- All temporal features are metadata - clustering is still based on audio + lyric embeddings
- If `spotify/saved_tracks.json` is missing, temporal features will be skipped gracefully
- Temporal analysis can be run multiple times on the same clustering results

## Future Enhancements

Possible additions:
- Predict future taste based on trends
- Identify "gateway songs" that led to new clusters
- Analyze listening patterns (if play count data available)
- Correlation between temporal patterns and other metadata
