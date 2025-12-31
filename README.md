# Spotify Music Taste Clustering

A Python pipeline for analyzing and clustering your Spotify library using interpretable audio features and GPT-powered lyric classification. The hallmark of this project is a **33-dimensional interpretable feature vector** that enables meaningful music clustering with explainable results.

## Why This Project Exists

**The Problem:**
- Spotify's discovery algorithms are narrow—they keep recommending similar content without understanding your full taste
- There's no way to automatically organize music by "vibe" or mood
- I wanted deeper analysis of patterns in my own listening habits

**The Solution:**
Local audio analysis using Essentia's pre-trained classifiers combined with GPT-powered lyric classification, producing a 33-dimensional interpretable vector where every dimension has human-readable meaning.

**The Key Insight:**
Interpretable features beat raw embeddings for meaningful music clustering. You can explain *why* songs cluster together (genre, mood, energy) rather than having opaque similarity scores.

---

## The Journey: Design Decisions

This project evolved through several failed approaches before arriving at the current architecture:

### 1. Spotify API Deprecated Audio Features
Spotify deprecated their audio features API (danceability, valence, etc.), which forced local analysis of MP3 files.

### 2. Tried Raw Essentia Embeddings
First attempt: use Essentia's EffNet-Discogs embeddings (1280-dim) directly for clustering. Result: poor clustering quality, no interpretability.

### 3. Tried MERT (Late 2024 SOTA)
Second attempt: MERT (Music-Understanding Evaluation and Representation Transformer), the state-of-the-art music representation model. Result: still poor clustering for my use case.

### 4. Breakthrough: Essentia Classifiers
The breakthrough came from running Essentia's pre-trained **classifiers** on top of the embeddings. These classifiers (trained on labeled datasets) output interpretable features: genre probabilities, mood scores, danceability, etc. Much better clustering results.

### 5. Added GPT Lyric Classification
Sentence transformers (BGE-M3, E5) only separated songs by language, not content. GPT-5-mini with a carefully designed prompt extracts parallel emotional dimensions (valence, arousal, moods) plus lyric-unique features (explicit content, narrative style, theme).

### 6. Result: 33-Dim Interpretable Vector
The final architecture produces a 33-dimensional vector where:
- Every dimension has a human-readable name
- Clustering happens directly on these features (no PCA)
- Results can be explained: "These songs cluster together because they're all high-energy, aggressive, electronic tracks with themes of struggle"

### 7. Added Acoustic/Electronic + Timbre (30 → 32 dims)
**Problem discovered:** Jazz and osu/electronic game music were clustering together—both are instrumental, but the model couldn't distinguish acoustic instruments from synthesizers.

**Solution:** Added two dimensions:
- `electronic_acoustic`: Separates acoustic (jazz, classical) from electronic (synths, game music)
- `timbre_brightness`: Distinguishes dark/mellow sounds from bright/crisp production

### 8. Added Release Year Metadata (32 → 33 dims)
**Problem discovered:** Clustering couldn't distinguish temporal patterns. A 1970s jazz track and a 2024 neo-jazz track would cluster together despite massive production style differences.

**Solution:** Added `release_year` dimension (index 32) using decade bucket encoding:
- **Decade buckets**: Maps 1950s=0.0, 1960s=0.14, ..., 2020s=1.0
- **Why decade buckets?**: Production styles change by decade, not year-by-year. More interpretable and stable than linear year normalization.
- **Default for missing/invalid dates**: 0.5 (centered "unknown", consistent with theme/language defaults)

**Impact:** Expected to enable:
- Era-aware clustering (vintage vs. modern production styles)
- Temporal pattern discovery ("my 2020s music is experimental, my 80s music is pop")
- Era-based playlist generation
- Better separation of genre revivals (80s synthpop vs. 2020s synthwave)

---

## Architecture

### Data Pipeline

```
Spotify API → saved_tracks.json
     ↓
Download MP3s (spotdl/yt-dlp) → songs/data/
     ↓
Fetch lyrics (Genius API) → lyrics/data/
     ↓
Feature Extraction:
  ├─ Audio: Essentia classifiers (16 dims)
  ├─ Key: Circular encoding (3 dims)
  ├─ Lyrics: GPT-5-mini (10 dims)
  └─ Meta: theme, language, popularity, release_year (4 dims)
     ↓
33-dim interpretable vector
     ↓
StandardScaler normalization
     ↓
HAC clustering (5-7 clusters)
     ↓
UMAP 3D visualization
```

### The 33-Dimensional Interpretable Vector

Each dimension has explicit meaning and rationale:

#### Audio Features (Dimensions 0-18)

| Dim | Name | Source | Description |
|-----|------|--------|-------------|
| 0 | `bpm` | Essentia RhythmExtractor | Tempo normalized to [0,1] range |
| 1 | `danceability` | Essentia classifier | How suitable for dancing |
| 2 | `instrumentalness` | voice_instrumental classifier | 0=vocal, 1=instrumental |
| 3 | `valence` | DEAM model (MusiCNN) | Emotional positivity [1-9] → normalized |
| 4 | `arousal` | DEAM model (MusiCNN) | Energy/activation [1-9] → normalized |
| 5 | `engagement` | engagement_regression | Active vs background listening |
| 6 | `approachability` | approachability_regression | Mainstream vs niche |
| 7 | `mood_happy` | mood_happy classifier | Joy/celebration presence |
| 8 | `mood_sad` | mood_sad classifier | Melancholy/grief presence |
| 9 | `mood_aggressive` | mood_aggressive classifier | Anger/intensity presence |
| 10 | `mood_relaxed` | mood_relaxed classifier | Calm/peaceful presence |
| 11 | `mood_party` | mood_party classifier | Upbeat/celebratory presence |
| 12 | `voice_gender` | gender classifier | 0=female, 1=male, 0=instrumental* |
| 13 | `genre_fusion` | Entropy of genre_probs | 0=pure genre, 1=genre fusion |
| 14 | `electronic_acoustic` | mood_acoustic - mood_electronic | 0=electronic, 1=acoustic |
| 15 | `timbre_brightness` | timbre classifier | 0=dark/mellow, 1=bright/crisp |
| 16 | `key_sin` | Essentia RhythmExtractor | sin(2π × pitch/12) × 0.33 |
| 17 | `key_cos` | Essentia RhythmExtractor | cos(2π × pitch/12) × 0.33 |
| 18 | `key_scale` | Essentia RhythmExtractor | 0=minor, 0.33=major |

*Voice gender is set to 0 for instrumental tracks (instrumentalness >= 0.5) since there's no voice to classify.

**Genre Fusion Deep-Dive:**
The genre_fusion measures how "categorizable" a song is, computed from the entropy of the 400-dimensional Discogs genre probability vector:
- Low entropy (→0) = Song clearly belongs to one genre (artist working WITHIN a tradition)
- High entropy (→1) = Song crosses many genres (artist CROSSING boundaries)

**Acoustic/Electronic Deep-Dive:**
Added to solve jazz vs electronic game music clustering. Both are instrumental, but one uses acoustic instruments (saxophone, piano) and one uses synthesizers. Computed as `(mood_acoustic - mood_electronic + 1) / 2` to get [0,1] scale.

**Timbre Brightness Deep-Dive:**
Captures the tonal quality of the production. Dark timbre (mellow jazz, ambient) vs bright timbre (crisp electronic, pop). Helps distinguish mellow rap from energetic party music.

**Key Encoding Deep-Dive:**
Musical keys are cyclical (C is "close" to B, not far). Sin/cos encoding captures octave equivalence. The 0.33 weight ensures 3 key dimensions contribute roughly 1 equivalent dimension of influence.

#### Lyric Features (Dimensions 19-28) — Semantic Weighting Strategies

**Critical Design Decision:** Lyric features use three different weighting strategies based on their semantic type:

**1. Bipolar Scales (valence, arousal):** `0.5 + (raw - 0.5) * (1 - instrumentalness)`
- Pulls toward 0.5 (neutral), not 0
- Rationale: These are negative↔positive scales. An instrumental track isn't "lyrically negative"—it's lyrically absent, which should be neutral.

**2. Presence/Absence (moods, explicit, narrative, vocabulary, repetition):** `raw * (1 - instrumentalness)`
- Pulls toward 0 (absent)
- Rationale: These classifiers output "happy vs non_happy", etc. An instrumental track is definitively non_happy, non_sad, non_explicit, etc.

**3. Categorical (theme, language):** Hard threshold at `instrumentalness > 0.5`
- If instrumental: use 0.5 (centered "none")
- If vocal: use ordinal scale value
- Rationale: Themes are categorical, not continuous. A track at instrumentalness=0.9 doesn't have "10% Japanese"—it either has meaningful lyrics or it doesn't.

| Dim | Name | Source | Weighting | Description |
|-----|------|--------|-----------|-------------|
| 19 | `lyric_valence` | GPT | Bipolar → 0.5 | Emotional tone of lyrics |
| 20 | `lyric_arousal` | GPT | Bipolar → 0.5 | Energy level of lyric content |
| 21 | `lyric_mood_happy` | GPT | Presence → 0 | Joy/celebration in lyrics |
| 22 | `lyric_mood_sad` | GPT | Presence → 0 | Grief/melancholy in lyrics |
| 23 | `lyric_mood_aggressive` | GPT | Presence → 0 | Anger/confrontation in lyrics |
| 24 | `lyric_mood_relaxed` | GPT | Presence → 0 | Peace/calm in lyrics |
| 25 | `lyric_explicit` | GPT | Presence → 0 | Holistic explicit content score |
| 26 | `lyric_narrative` | GPT | Presence → 0 | 0=pure vibes → 1=specific story |
| 27 | `lyric_vocabulary` | Local (TTR) | Presence → 0 | Type-token ratio (vocabulary richness) |
| 28 | `lyric_repetition` | Local | Presence → 0 | 1 - (unique lines / total lines) |

#### Metadata Features (Dimensions 29-32)

| Dim | Name | Weighting | Description |
|-----|------|-----------|-------------|
| 29 | `theme` | Categorical → 0.5 | Hard threshold at instrumentalness > 0.5 |
| 30 | `language` | Categorical → 0.5 | Hard threshold at instrumentalness > 0.5 |
| 31 | `popularity` | None | Spotify popularity [0-100] → normalized |
| 32 | `release_year` | None | Decade bucket encoding [0.0-1.0], 0=1950s, 1=2020s |

**Theme Scale** (ordered by energy/positivity):
```
party=1.0, flex=0.9, love=0.8, social=0.7, spirituality=0.6,
introspection=0.5, street=0.4, heartbreak=0.3, struggle=0.2,
other=0.1, none=0.5 (centered)
```

**Language Scale** (grouped by musical tradition similarity):
```
English=1.0 (Western pop baseline)
Romance (Spanish/Portuguese/French)=0.85
Germanic (German/Swedish/Norwegian)=0.70
Slavic (Russian/Ukrainian/Serbian/Czech)=0.55
Middle Eastern (Arabic/Hebrew/Turkish)=0.40
South Asian (Punjabi)=0.30
East Asian (Korean/Japanese/Chinese/Vietnamese)=0.20
African (Luganda)=0.10
multilingual/unknown/none=0.5 (centered)
```

**Note:** Theme and language use a hard threshold: if `instrumentalness > 0.5`, use 0.5 (centered "none"). This ensures "none" is equidistant from all categories rather than being at an arbitrary edge of the ordinal scale.

---

### GPT Lyric Classification

**Model:** gpt-5-mini-2025-08-07

**Why GPT over sentence transformers?** Sentence transformers (BGE-M3, E5) only separated songs by language, not by emotional or thematic content. The embeddings didn't produce meaningful clusters.

**Prompt Design Principles:**
1. **Detailed scoring guidelines** - Not just "rate 0-1" but specific anchor points (e.g., "0.0-0.2: Deeply negative - despair, hopelessness, nihilism")
2. **Non-mutually exclusive moods** - A song can be both happy (0.7) AND sad (0.4) if bittersweet
3. **Single theme choice** - Forces picking ONE primary theme to avoid multi-label ambiguity
4. **Holistic explicit score** - Combines profanity, sexual content, violence, drugs into one dimension
5. **Narrative scale** - 0=pure vibes/hooks → 1=specific story with beginning/middle/end

**Three-Tier Architecture:**
- **Tier 1 (Parallel to audio):** valence, arousal, happy, sad, aggressive, relaxed — same dimensions as audio for alignment
- **Tier 3 (Lyric-unique):** explicit, narrative, theme, language, vocabulary richness, repetition

**Local Features** (computed without GPT):
- `vocabulary_richness`: Type-token ratio (unique words / total words)
- `repetition_score`: 1 - (unique lines / total lines)

---

### Essentia Model Architecture

**Two Embedding Backbones:**
1. **EffNet-Discogs** (1280-dim) → Most classifiers
2. **MusiCNN** → Valence/arousal (DEAM dataset)

**Classifiers Used:**
- `genre_discogs400` - 400 genre probabilities
- `mood_happy/sad/aggressive/relaxed/party` - Binary mood classifiers
- `danceability` - Dance suitability
- `voice_instrumental` - Vocal vs instrumental
- `approachability_regression` - Mainstream vs niche (continuous)
- `engagement_regression` - Active vs background listening (continuous)
- `gender` - Voice male/female
- `timbre` - Bright vs dark (used in dim 15: timbre_brightness)
- `mood_acoustic` - Acoustic vs electronic (used in dim 14: electronic_acoustic)
- `mtg_jamendo_moodtheme` - 56 mood/theme classes (extracted for reference)
- `mtg_jamendo_instrument` - 40 instrument classes (extracted for reference)
- `moods_mirex` - 5 MIREX mood clusters (extracted for reference)

---

### Filename Normalization Strategy

Matching Spotify metadata to downloaded filenames requires aggressive normalization:

**Unicode Handling:**
- Japanese wave dashes (`～`, `〜`) → hyphen
- Various quote types → removed
- En/em dashes → hyphen

**Character Replacements:**
- Filesystem-illegal characters (`< > " / \ | ? *`) → removed
- Brackets `( ) [ ] { }` → removed
- Colons `:` → dash
- Ampersand `&` → "and"
- Special characters (`# $ % @ + =`) → removed

**Normalization:**
- Lowercase
- Multiple spaces → single space
- Strip whitespace
- Remove periods

**Why so aggressive?** Tracks like `"Artist: Song (Remix) ～"` need to match `artist-song-remix.mp3` across different character sets (Japanese, Arabic, etc.).

---

## Clustering Approach

### No PCA for Interpretable Mode

Unlike MERT/E5 embeddings (which need PCA to ~50 components), the 33-dim interpretable vector clusters directly.

**Only Preprocessing:** `StandardScaler` (zero mean, unit variance normalization)

**Rationale:** Each dimension already has meaning. PCA would create opaque linear combinations, losing interpretability.

### Optional Feature Weighting

`apply_feature_weights()` allows manual scaling of feature groups for experimentation:

| Group | Dimensions | Features |
|-------|------------|----------|
| `core_audio` | 0-6 | BPM, danceability, instrumentalness, valence, arousal, engagement, approachability |
| `mood` | 7-11 | 5 mood dimensions |
| `genre` | 12-15 | Voice gender, genre ladder, acoustic/electronic, timbre brightness |
| `key` | 16-18 | Circular key encoding |
| `lyric_emotion` | 19-24 | Lyric valence/arousal + 4 moods |
| `lyric_content` | 25-28 | Explicit, narrative, vocabulary, repetition |
| `theme` | 29 | Theme ordinal |
| `language` | 30 | Language ordinal |
| `popularity` | 31 | Spotify popularity |

Used in the Streamlit dashboard to experiment (e.g., weight moods 2x, reduce key influence to 0.5x).

### HAC with 5-7 Clusters

**Default:** 5 clusters

**Why 5-7?** At this granularity, clusters show clear genre/vibe isolation. More clusters fragment meaningful groups; fewer merge distinct vibes.

**Tuning:** Use the Streamlit interactive dashboard (`interactive_interpretability.py`) to experiment with cluster counts and feature weights.

### UMAP for Visualization Only

- **Clustering:** Done on standardized 33-dim vector
- **UMAP:** Only for 2D/3D visualization (NOT used for clustering decisions)
- **Parameters:** n_neighbors=20, min_dist=0.2, cosine metric, 3 components

---

## Reproduction Guide

### Prerequisites

- Python 3.9+
- FFmpeg (for audio processing)
- ~2GB disk space for Essentia models

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/spotify-clustering.git
cd spotify-clustering

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### 2. API Setup

Create a `.env` file:

```bash
cp .env.example .env
```

**Spotify API** (required):
1. Create app at [Spotify Dashboard](https://developer.spotify.com/dashboard)
2. Set Redirect URI to `http://127.0.0.1:3000/callback`
3. Add to `.env`:
   ```
   SPOTIFY_CLIENT_ID=your_client_id
   SPOTIFY_CLIENT_SECRET=your_client_secret
   ```

**Genius API** (for lyrics):
1. Create client at [Genius API Clients](https://genius.com/api-clients)
2. Generate "Client Access Token"
3. Add to `.env`:
   ```
   GENIUS_ACCESS_TOKEN=your_token
   ```

**OpenAI API** (for GPT lyric classification):
1. Get API key from [OpenAI](https://platform.openai.com/api-keys)
2. Add to `.env`:
   ```
   OPENAI_API_KEY=your_key
   ```

### 3. Fetch Spotify Library

```bash
python api/fetch_audio_data.py
```

First run opens browser for OAuth. Token cached in `.cache`.

### 4. Download MP3s

```bash
# Check what's missing
python download/check_matches.py

# Download (choose one)
python download/download_missing.py   # spotdl (faster)
python download/download_ytdlp.py     # yt-dlp (fallback)
```

### 5. Fetch Lyrics

```bash
python lyrics/fetch_lyrics.py
```

Safe to stop/resume—uses caching.

### 6. Run Analysis

```bash
python analysis/run_analysis.py --songs songs/data/ --lyrics lyrics/data/individual/
```

First run extracts all features (~90 min for 1,500 songs). Subsequent runs use cache automatically.

Use `--fresh` to force re-extraction.

### 7. (Optional) Export to Spotify Playlists

```bash
# Preview
python export/create_playlists.py --dry-run

# Create playlists
python export/create_playlists.py
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `analysis/pipeline/interpretable_features.py` | 33-dim vector construction, instrumentalness weighting |
| `analysis/pipeline/audio_analysis.py` | Essentia classifier extraction, model loading |
| `analysis/pipeline/lyric_features.py` | GPT lyric classification, prompt design |
| `analysis/pipeline/clustering.py` | HAC clustering, UMAP visualization, feature preparation |
| `analysis/pipeline/config.py` | Theme/language scales, default parameters |
| `analysis/pipeline/genre_fusion.py` | Entropy-based genre fusion calculation |
| `analysis/interactive_interpretability.py` | Streamlit dashboard for parameter tuning |
| `analysis/run_analysis.py` | Main orchestration script |

---

## Technical Nuances

### Caching Strategy
- **Incremental saves:** Every 25 tracks (crash recovery)
- **Resume support:** Track ID lookup to skip already-processed tracks
- **Separate caches:** Audio features vs lyric features in different pickle files

### Silhouette Score
Computed on the standardized 33-dim features (clustering space), NOT on UMAP coordinates (visualization space).

### Edge Cases
- **Instrumental tracks** (instrumentalness > 0.5): Voice gender → 0, lyric valence/arousal → 0.5 (neutral), moods/explicit/narrative/vocab/repetition → 0, theme/language → 0.5 (centered "none")
- **Missing lyrics:** Lyric features default to neutral (0.5 for valence/arousal, 0 for moods)
- **Very short lyrics:** Still analyzed, but narrative score will be low

### Model Dependencies
- Essentia models auto-download to `~/.essentia/models/`
- MusiCNN required for valence/arousal (DEAM model)
- EffNet-Discogs required for all other classifiers

---

## Project Structure

```
spotify-clustering/
├── api/                    # Spotify API integration
│   └── fetch_audio_data.py
├── download/               # MP3 download scripts
│   ├── check_matches.py
│   ├── download_missing.py (spotdl)
│   └── download_ytdlp.py
├── lyrics/                 # Genius lyric fetching
│   └── fetch_lyrics.py
├── analysis/
│   ├── pipeline/           # Core analysis modules
│   │   ├── audio_analysis.py
│   │   ├── lyric_features.py
│   │   ├── interpretable_features.py
│   │   ├── clustering.py
│   │   ├── config.py
│   │   └── genre_fusion.py
│   ├── cache/              # Feature caches (gitignored)
│   ├── outputs/            # Results and visualizations
│   ├── run_analysis.py     # Main entry point
│   └── interactive_interpretability.py  # Streamlit dashboard
├── export/                 # Spotify playlist export
│   └── create_playlists.py
├── songs/data/             # Downloaded MP3s (gitignored)
├── lyrics/data/            # Fetched lyrics (gitignored)
└── spotify/                # Spotify metadata
    └── saved_tracks.json
```

---

## License

MIT
