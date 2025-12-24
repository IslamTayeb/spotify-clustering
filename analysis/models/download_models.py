#!/usr/bin/env python3
"""
Download Essentia TensorFlow models for audio analysis.
Combines core models with additional high-value models for enhanced clustering.
"""
import os
import urllib.request
from pathlib import Path
from tqdm import tqdm


# Models directory
MODELS_DIR = Path.home() / '.essentia' / 'models'
BASE_URL = 'https://essentia.upf.edu/models'


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


# All models organized by priority
MODELS = {
    # TIER 0: CRITICAL FIXES (missing required models)
    'deam-msd-musicnn-2.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/deam/deam-msd-musicnn-2.pb',
        'description': '⚠️  DEAM Arousal/Valence - Required baseline model',
        'priority': 0,
    },

    # TIER 1: CORE MODELS (required for basic pipeline)
    'discogs-effnet-bs64-1.pb': {
        'url': 'https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb',
        'description': '🎵 Feature Extractor (base embeddings)',
        'priority': 1,
    },
    'genre_discogs400-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb',
        'description': '🎸 Genre Classification (400 classes)',
        'priority': 1,
    },
    'mood_happy-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-discogs-effnet-1.pb',
        'description': '😊 Mood: Happy',
        'priority': 1,
    },
    'mood_sad-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-discogs-effnet-1.pb',
        'description': '😢 Mood: Sad',
        'priority': 1,
    },
    'mood_aggressive-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.pb',
        'description': '😠 Mood: Aggressive',
        'priority': 1,
    },
    'mood_relaxed-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.pb',
        'description': '😌 Mood: Relaxed',
        'priority': 1,
    },
    'mood_party-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/mood_party/mood_party-discogs-effnet-1.pb',
        'description': '🎉 Mood: Party',
        'priority': 1,
    },
    'danceability-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/danceability/danceability-discogs-effnet-1.pb',
        'description': '💃 Danceability',
        'priority': 1,
    },
    'voice_instrumental-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb',
        'description': '🎤 Voice vs Instrumental',
        'priority': 1,
    },

    # TIER 2: ENGAGEMENT & APPROACHABILITY (2023 models)
    'approachability_2c-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/approachability/approachability_2c-discogs-effnet-1.pb',
        'description': '🎯 Approachability (2-class: accessible/niche)',
        'priority': 2,
    },
    'approachability_3c-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/approachability/approachability_3c-discogs-effnet-1.pb',
        'description': '🎯 Approachability (3-class)',
        'priority': 2,
    },
    'approachability_regression-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/approachability/approachability_regression-discogs-effnet-1.pb',
        'description': '🎯 Approachability (regression score)',
        'priority': 2,
    },
    'engagement_2c-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/engagement/engagement_2c-discogs-effnet-1.pb',
        'description': '⚡ Engagement (2-class: low/high)',
        'priority': 2,
    },
    'engagement_3c-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/engagement/engagement_3c-discogs-effnet-1.pb',
        'description': '⚡ Engagement (3-class)',
        'priority': 2,
    },
    'engagement_regression-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/engagement/engagement_regression-discogs-effnet-1.pb',
        'description': '⚡ Engagement (regression score)',
        'priority': 2,
    },
    'mtg_jamendo_moodtheme-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb',
        'description': '🎭 MTG-Jamendo Mood/Theme (56 labels)',
        'priority': 2,
    },

    # TIER 3: ENHANCED FEATURES (2024 models - voice, timbre, production)
    'gender-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/gender/gender-discogs-effnet-1.pb',
        'description': '🎤 Voice Gender (male/female) - Critical for vocal music',
        'priority': 3,
    },
    'timbre-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/timbre/timbre-discogs-effnet-1.pb',
        'description': '🎨 Timbre (bright/dark) - Acoustic character',
        'priority': 3,
    },
    'mood_acoustic-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-discogs-effnet-1.pb',
        'description': '🎸 Acoustic vs Electronic - Production style',
        'priority': 3,
    },
    'mtg_jamendo_instrument-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb',
        'description': '🎺 Instruments (40 classes) - Great for jazz/folk/Arabian',
        'priority': 3,
    },

    # TIER 4: ALTERNATIVE MODELS (additional perspectives)
    'emomusic-msd-musicnn-2.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-msd-musicnn-2.pb',
        'description': '💓 Alternative Arousal/Valence - Ensemble with existing',
        'priority': 4,
    },
    'mtg_jamendo_genre-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-discogs-effnet-1.pb',
        'description': '🎵 Genres (87 classes) - More granular than 400-class',
        'priority': 4,
    },
    'moods_mirex-msd-musicnn-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/moods_mirex/moods_mirex-msd-musicnn-1.pb',
        'description': '😊 MIREX Moods (5 clusters) - Alternative mood taxonomy',
        'priority': 4,
    },
}


def download_model(filename: str, model_info: dict) -> bool:
    """Download a single model file"""
    url = model_info['url']
    output_path = MODELS_DIR / filename
    description = model_info['description']

    if output_path.exists():
        print(f"  ✅ Already exists: {filename}")
        return True

    print(f"\n  📥 Downloading: {filename}")
    print(f"     {description}")

    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=f"     Progress") as t:
            urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
        print(f"  ✅ Downloaded successfully!")
        return True
    except Exception as e:
        print(f"  ❌ Failed to download: {e}")
        print(f"     URL: {url}")
        return False


def main():
    print("=" * 80)
    print("ESSENTIA MODEL DOWNLOADER")
    print("=" * 80)
    print(f"\n📁 Download location: {MODELS_DIR}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Sort models by priority (0 = highest)
    sorted_models = sorted(MODELS.items(), key=lambda x: x[1]['priority'])

    # Count by priority
    priority_counts = {}
    for filename, info in sorted_models:
        priority = info['priority']
        output_path = MODELS_DIR / filename
        if not output_path.exists():
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

    if not any(priority_counts.values()):
        print("✅ All models are already downloaded!\n")
        print(f"📁 Models location: {MODELS_DIR}")
        print(f"📊 Total models: {len(MODELS)}")
        return

    print("📊 Models to download:\n")
    tier_names = {
        0: "TIER 0: MISSING FIXES",
        1: "TIER 1: CORE MODELS",
        2: "TIER 2: ENGAGEMENT & APPROACHABILITY",
        3: "TIER 3: ENHANCED FEATURES",
        4: "TIER 4: ALTERNATIVE MODELS"
    }

    for priority in sorted(priority_counts.keys()):
        count = priority_counts[priority]
        print(f"  {tier_names.get(priority, f'TIER {priority}')}: {count} models")

    print("\n" + "=" * 80)

    # Ask for confirmation
    response = input("\nDownload all models? (y/n): ").lower().strip()
    if response != 'y':
        print("\n❌ Download cancelled.")
        return

    print("\n" + "=" * 80)
    print("DOWNLOADING MODELS...")
    print("=" * 80)

    success_count = 0
    fail_count = 0

    for filename, model_info in sorted_models:
        result = download_model(filename, model_info)
        if result:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"\n  ✅ Successfully downloaded/verified: {success_count}")
    print(f"  ❌ Failed: {fail_count}")

    if success_count > 0:
        print(f"\n  📁 Models saved to: {MODELS_DIR}")
        print("\n  🎯 Next steps:")
        print("     1. Run: python analysis/models/list_models.py (to verify)")
        print("     2. Run: python analysis/run_analysis.py --re-classify-audio (to extract features)")
        print("     3. Or run full pipeline: python analysis/run_analysis.py")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
