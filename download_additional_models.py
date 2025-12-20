#!/usr/bin/env python3
"""
Download additional Essentia models for enhanced clustering.
Focuses on models highly relevant for rap/jazz/Japanese/Arabian/folk/Portuguese/pop.
"""
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Models directory
MODELS_DIR = Path.home() / '.essentia' / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# High-value models to download (prioritized for your library)
MODELS_TO_DOWNLOAD = {
    # TIER 1: CRITICAL (Voice & Timbre)
    'gender-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/gender/gender-discogs-effnet-1.pb',
        'description': '🎤 Voice Gender (male/female) - CRITICAL for vocal music',
        'priority': 1,
    },
    'timbre-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/timbre/timbre-discogs-effnet-1.pb',
        'description': '🎨 Timbre (bright/dark) - Acoustic character',
        'priority': 1,
    },

    # TIER 2: HIGH VALUE (Instrumentation & Production)
    'mood_acoustic-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-discogs-effnet-1.pb',
        'description': '🎸 Acoustic vs Electronic - Production style',
        'priority': 2,
    },
    'mtg_jamendo_instrument-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb',
        'description': '🎺 Instruments (40 classes) - Great for jazz/folk/Arabian',
        'priority': 2,
    },

    # TIER 3: VALUABLE (Alternative Models & Granularity)
    'emomusic-msd-musicnn-2.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-msd-musicnn-2.pb',
        'description': '💓 Alternative Arousal/Valence - Ensemble with existing',
        'priority': 3,
    },
    'mtg_jamendo_genre-discogs-effnet-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-discogs-effnet-1.pb',
        'description': '🎵 Genres (87 classes) - More granular than 400-class',
        'priority': 3,
    },
    'moods_mirex-msd-musicnn-1.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/moods_mirex/moods_mirex-msd-musicnn-1.pb',
        'description': '😊 MIREX Moods (5 clusters) - Alternative mood taxonomy',
        'priority': 3,
    },

    # TIER 4: POTENTIALLY USEFUL
    'deepsquare-k16-3.pb': {
        'url': 'https://essentia.upf.edu/models/tempo/tempocnn/deepsquare-k16-3.pb',
        'description': '🥁 TempoCNN (256 BPM classes) - Detailed tempo',
        'priority': 4,
    },

    # MISSING MODEL (should have been downloaded)
    'deam-msd-musicnn-2.pb': {
        'url': 'https://essentia.upf.edu/models/classification-heads/deam/deam-msd-musicnn-2.pb',
        'description': '⚠️  DEAM Arousal/Valence - FIX MISSING MODEL',
        'priority': 0,  # Highest priority - fix missing
    },
}


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_model(filename, model_info):
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
        return False


def main():
    print("=" * 80)
    print("ESSENTIA MODEL DOWNLOADER")
    print("=" * 80)
    print(f"\n📁 Download location: {MODELS_DIR}\n")

    # Sort models by priority (0 = highest)
    sorted_models = sorted(MODELS_TO_DOWNLOAD.items(), key=lambda x: x[1]['priority'])

    # Count by priority
    priority_counts = {}
    for filename, info in sorted_models:
        priority = info['priority']
        output_path = MODELS_DIR / filename
        if not output_path.exists():
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

    if not any(priority_counts.values()):
        print("✅ All models are already downloaded!\n")
        return

    print("📊 Models to download:\n")
    tier_names = {0: "MISSING FIX", 1: "TIER 1: CRITICAL", 2: "TIER 2: HIGH VALUE",
                  3: "TIER 3: VALUABLE", 4: "TIER 4: OPTIONAL"}

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
    print(f"\n  ✅ Successfully downloaded: {success_count}")
    print(f"  ❌ Failed: {fail_count}")

    if success_count > 0:
        print(f"\n  📁 Models saved to: {MODELS_DIR}")
        print("\n  🎯 Next steps:")
        print("     1. Run: python list_available_models.py (to verify)")
        print("     2. Update audio_analysis.py to use new models")
        print("     3. Run: python run_analysis.py --re-classify-audio")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
