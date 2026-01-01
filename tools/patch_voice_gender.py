#!/usr/bin/env python3
"""
Patch Voice Gender for Instrumental Songs

Fixes voice gender in existing cached audio features by setting both
voice_gender_male and voice_gender_female to 0.0 for instrumental songs
(instrumentalness >= 0.5).
"""

import pickle
import sys
from pathlib import Path

def patch_voice_gender(cache_path='cache/audio_features.pkl'):
    """Fix voice gender for instrumental songs in existing cache."""
    cache_file = Path(cache_path)

    if not cache_file.exists():
        print(f"Cache file not found: {cache_path}")
        return

    print(f"Loading cache from: {cache_path}")
    with open(cache_file, 'rb') as f:
        features = pickle.load(f)

    patched = 0
    for track in features:
        instrumentalness = track.get('instrumentalness', 0)
        if instrumentalness >= 0.5:
            # Check if it needs patching
            male = track.get('voice_gender_male', 0)
            female = track.get('voice_gender_female', 0)
            if male != 0.0 or female != 0.0:
                track['voice_gender_male'] = 0.0
                track['voice_gender_female'] = 0.0
                patched += 1

    if patched > 0:
        # Backup original
        backup_path = cache_file.with_suffix('.pkl.backup')
        print(f"Creating backup: {backup_path}")
        with open(backup_path, 'wb') as f:
            pickle.dump(features, f)

        # Save patched version
        print(f"Saving patched cache to: {cache_path}")
        with open(cache_file, 'wb') as f:
            pickle.dump(features, f)

        print(f"✓ Patched {patched} instrumental tracks")
    else:
        print("No tracks needed patching (all instrumental tracks already have voice_gender = 0.0)")


if __name__ == '__main__':
    cache_path = sys.argv[1] if len(sys.argv) > 1 else 'cache/audio_features.pkl'
    patch_voice_gender(cache_path)







