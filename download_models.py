#!/usr/bin/env python3
"""Download Essentia TensorFlow models"""

import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path.home() / '.essentia' / 'models'
BASE_URL = 'https://essentia.upf.edu/models'

MODELS = [
    # Feature extractor
    'feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb',

    # Genre classification
    'classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb',

    # Mood classification (existing)
    'classification-heads/mood_happy/mood_happy-discogs-effnet-1.pb',
    'classification-heads/mood_sad/mood_sad-discogs-effnet-1.pb',
    'classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.pb',
    'classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.pb',
    'classification-heads/mood_party/mood_party-discogs-effnet-1.pb',

    # Other existing models
    'classification-heads/danceability/danceability-discogs-effnet-1.pb',
    'classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb',

    # NOTE: Arousal/Valence models (DEAM, emoMusic, MuSe) do NOT have discogs-effnet variants
    # They only exist with msd-musicnn and audioset-vggish backbones
    # To use these, we would need to extract MusiCNN/VGGish embeddings from MP3s (~1-2 hours)
    # Skipping for now - can add later if needed

    # NEW: Approachability (3 variants)
    'classification-heads/approachability/approachability_2c-discogs-effnet-1.pb',
    'classification-heads/approachability/approachability_3c-discogs-effnet-1.pb',
    'classification-heads/approachability/approachability_regression-discogs-effnet-1.pb',

    # NEW: Engagement (3 variants)
    'classification-heads/engagement/engagement_2c-discogs-effnet-1.pb',
    'classification-heads/engagement/engagement_3c-discogs-effnet-1.pb',
    'classification-heads/engagement/engagement_regression-discogs-effnet-1.pb',

    # NEW: MTG-Jamendo mood/theme (56 labels)
    'classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb',
]

def download_model(model_path: str):
    """Download a single model file"""
    url = f'{BASE_URL}/{model_path}'
    filename = Path(model_path).name
    output_path = MODELS_DIR / filename

    if output_path.exists():
        print(f'✓ {filename} already exists')
        return

    print(f'Downloading {filename}...')
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f'✓ Downloaded {filename}')
    except Exception as e:
        print(f'✗ Failed to download {filename}: {e}')
        print(f'  URL: {url}')


def main():
    print('Essentia Model Downloader')
    print('=' * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Models directory: {MODELS_DIR}\n')

    for model_path in MODELS:
        download_model(model_path)

    print('\n' + '=' * 60)
    print('Download complete!')
    print(f'Models saved to: {MODELS_DIR}')


if __name__ == '__main__':
    main()
