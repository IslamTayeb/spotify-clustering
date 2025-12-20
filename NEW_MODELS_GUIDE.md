# Guide to New Essentia Models

## Current Situation

You have **16 models** and are using **15 of them** (1 is missing: `deam-msd-musicnn-2.pb`).

## New Models Available (9 total)

### 🔴 TIER 0: MISSING FIX (1 model)

#### `deam-msd-musicnn-2.pb` ⚠️
**What it does:** Arousal & Valence regression (emotional dimensions)
**Why you need it:** This is ALREADY referenced in your code but the file is missing!
**For your library:**
- Currently defaulting to 0.5 for all tracks
- Will provide actual emotional positioning for all genres
**Priority:** Download immediately to fix existing feature

---

### 🟢 TIER 1: CRITICAL (2 models)

#### `gender-discogs-effnet-1.pb` 🎤
**What it does:** Classifies singing voice gender (male/female)
**Output:** 2 probabilities: [female, male]

**Why critical for YOUR library:**
- **Rap:** Separate male rappers (Drake, Kendrick) from female (Nicki Minaj, Cardi B)
- **Jazz:** Instrumental vs vocal, male (Frank Sinatra) vs female (Ella Fitzgerald)
- **Pop/Portuguese/Japanese:** Huge clustering dimension (gender preferences)
- **Folk:** Traditional gender-based vocal styles

**Expected clusters:**
- "Female Japanese pop singers"
- "Male rappers"
- "Female Portuguese fado"
- "Instrumental jazz" (neither male nor female)

**New fields added:**
- `voice_gender_female` (0-1 probability)
- `voice_gender_male` (0-1 probability)

---

#### `timbre-discogs-effnet-1.pb` 🎨
**What it does:** Classifies timbre color (bright vs dark)
**Output:** 2 probabilities: [bright, dark]

**Why critical for YOUR library:**
- **Arabian/Folk:** Acoustic traditional instruments (dark/warm)
- **Japanese:** Traditional shamisen/koto (dark) vs J-pop synths (bright)
- **Jazz:** Acoustic jazz (warm/dark) vs fusion (bright)
- **Rap:** Lo-fi boom-bap (dark) vs modern trap (bright)
- **Pop:** Production style indicator

**Expected clusters:**
- "Dark acoustic folk" (Portuguese fado, Arabian traditional)
- "Bright modern pop" (synth-heavy J-pop, modern rap)
- "Warm jazz" (acoustic recordings)

**New fields added:**
- `timbre_bright` (0-1 probability)
- `timbre_dark` (0-1 probability)

---

### 🟡 TIER 2: HIGH VALUE (2 models)

#### `mood_acoustic-discogs-effnet-1.pb` 🎸
**What it does:** Binary classification: acoustic vs electronic
**Output:** 2 probabilities: [acoustic, electronic]

**Why valuable:**
- **All genres:** Separates production methods
- **Folk/Arabian:** Purely acoustic traditional music
- **Jazz:** Acoustic vs electric/fusion
- **Pop/Rap:** Electronic production vs live instruments
- **Japanese:** Traditional vs modern production

**Expected clusters:**
- "Acoustic folk & traditional" (Portuguese, Arabian)
- "Electronic pop & rap" (modern production)
- "Hybrid jazz" (electronic + acoustic)

**New fields added:**
- `mood_acoustic` (0-1 probability)
- `mood_electronic` (0-1 probability)

---

#### `mtg_jamendo_instrument-discogs-effnet-1.pb` 🎺
**What it does:** Multi-label instrument detection (40 classes)
**Output:** 40 probabilities for different instruments

**Classes include:**
- Strings: guitar, bass, violin, cello, etc.
- Winds: flute, saxophone, trumpet, clarinet
- Keys: piano, synthesizer, organ
- Percussion: drums, percussion
- Voice: male voice, female voice
- And more...

**Why valuable for YOUR library:**
- **Jazz:** Identify instrumentation (piano trio vs big band vs solo sax)
- **Folk:** Traditional instruments (guitar, violin, etc.)
- **Arabian:** Oud, qanun, ney detection (if trained on them)
- **Rap:** Identify sample-based vs synthesizer-based
- **Portuguese:** Identify fado guitar

**Expected clusters:**
- "Piano-based jazz"
- "Guitar-driven folk"
- "Horn-heavy jazz/funk"
- "Synthesizer pop"

**New fields added:**
- `instruments` (40-dimensional vector)
- Top 3 detected instruments with probabilities

---

### 🟠 TIER 3: VALUABLE (3 models)

#### `emomusic-msd-musicnn-2.pb` 💓
**What it does:** Alternative arousal/valence regression
**Output:** 2 values: arousal (1-9), valence (1-9)

**Why valuable:**
- **Ensemble with existing:** Two different emotion models can provide more robust estimates
- **Different training data:** May capture different emotional aspects
- Uses different architecture (MusiCNN vs your current one)

**Use case:**
- Average both arousal models → more stable
- Identify when models disagree → complex emotional content

**New fields added:**
- `arousal_emomusic` (1-9 scale)
- `valence_emomusic` (1-9 scale)

---

#### `mtg_jamendo_genre-discogs-effnet-1.pb` 🎵
**What it does:** Multi-label genre classification (87 classes)
**Output:** 87 probabilities for different genres

**Why valuable:**
- **More granular than 400-class:** Different taxonomy, might align better with your taste
- **Multi-label:** Tracks can have multiple genres (e.g., "jazz-funk", "electro-pop")
- **Modern genres:** Better coverage of contemporary styles

**Expected benefit:**
- Better sub-clustering within your main genres
- Cross-genre identification (e.g., "jazz-influenced hip-hop")

**New fields added:**
- `genre_jamendo_probs` (87-dimensional vector)
- Top 3 MTG-Jamendo genres

---

#### `moods_mirex-msd-musicnn-1.pb` 😊
**What it does:** Mood clustering (5 mood clusters)
**Output:** 5 probabilities for MIREX mood categories

**Mood clusters:**
1. Cluster 1: Passionate, rousing, confident, boisterous, rowdy
2. Cluster 2: Rollicking, cheerful, fun, sweet, amiable/good natured
3. Cluster 3: Literate, poignant, wistful, bittersweet, autumnal, brooding
4. Cluster 4: Humorous, silly, campy, quirky, whimsical, witty, wry
5. Cluster 5: Aggressive, fiery, tense/anxious, intense, volatile, visceral

**Why valuable:**
- **Different mood vocabulary** than your existing 5 binary moods
- **Clusters, not binary:** More nuanced emotional grouping
- **Alternative taxonomy:** May align better with how you experience music

**New fields added:**
- `mood_mirex_cluster1` through `mood_mirex_cluster5` (probabilities)

---

### 🔵 TIER 4: OPTIONAL (1 model)

#### `deepsquare-k16-3.pb` 🥁
**What it does:** Tempo classification (256 BPM classes, 30-286 BPM)
**Output:** 256 probabilities for different BPMs

**Why optional:**
- You already have `RhythmExtractor2013` that gives BPM
- This model provides BPM classification instead of exact value
- Might be useful if rhythm extractor is inaccurate for some genres

**Potential benefit:**
- More accurate for complex tempos (jazz, Arabian maqam)
- Handles tempo variations better

**New fields added:**
- `tempo_cnn_bpm` (predicted BPM class)
- `tempo_cnn_confidence` (confidence score)

---

## Recommendation Summary

### Download These NOW:
1. ✅ `deam-msd-musicnn-2.pb` (FIX MISSING)
2. ✅ `gender-discogs-effnet-1.pb` (CRITICAL)
3. ✅ `timbre-discogs-effnet-1.pb` (CRITICAL)

### Download These for Maximum Value:
4. ✅ `mood_acoustic-discogs-effnet-1.pb` (HIGH VALUE)
5. ✅ `mtg_jamendo_instrument-discogs-effnet-1.pb` (HIGH VALUE)

### Download if You Want Comprehensive Coverage:
6. ⚠️ `emomusic-msd-musicnn-2.pb` (ensemble model)
7. ⚠️ `mtg_jamendo_genre-discogs-effnet-1.pb` (alternative genre taxonomy)
8. ⚠️ `moods_mirex-msd-musicnn-1.pb` (alternative mood taxonomy)

### Skip for Now:
9. ❌ `deepsquare-k16-3.pb` (redundant with existing BPM extraction)

---

## Total New Features

If you download all recommended models (1-8), you'll add:

**New dimensions:**
- 2 (voice gender: male/female probabilities)
- 2 (timbre: bright/dark)
- 2 (acoustic vs electronic)
- 40 (instruments multi-label)
- 2 (alternative arousal/valence)
- 87 (alternative genre taxonomy)
- 5 (MIREX mood clusters)

**Total: ~140 new features**

Combined with your existing features, you'll have **~170 total features** feeding into PCA!

---

## Download & Integration Steps

### Step 1: Download Models
```bash
python download_additional_models.py
```

### Step 2: Verify Download
```bash
python list_available_models.py
```

### Step 3: Update Code
I'll need to update `audio_analysis.py` to:
- Load the new models
- Extract features from them
- Add new fields to the output dictionary
- Update the `update_cached_features()` function to check for these fields

### Step 4: Re-extract Features
```bash
python run_analysis.py --re-classify-audio
```

### Step 5: Test
```bash
streamlit run interactive_tuner.py
```

---

## Expected Impact on Clustering

**Before (current):**
- Clusters like "Happy rap" vs "Sad jazz" vs "Aggressive metal"
- Genre-dominated clustering with mood variation

**After (with new models):**
- "Female Japanese pop (bright, electronic, happy)"
- "Male rappers (bright, electronic, aggressive)"
- "Instrumental jazz (dark, acoustic, relaxed)" - piano vs saxophone sub-clusters
- "Male Portuguese fado (dark, acoustic, sad, guitar-heavy)"
- "Female folk (warm, acoustic, guitar + violin)"
- "Arabian traditional (dark, acoustic, oud + qanun)"

**Key improvement:** Multi-dimensional clustering that respects:
1. Gender/voice type
2. Production style (acoustic vs electronic, bright vs dark)
3. Instrumentation
4. Emotional content
5. Genre

This is **exactly what you need** for your diverse library!
