#!/usr/bin/env python3
"""
Verify that cached audio features have all required fields.
Run this after update_cached_features to ensure nothing is missing.
"""
import pickle
from pathlib import Path
from collections import defaultdict

# Complete list of all fields that should be present in audio features
REQUIRED_FIELDS = {
    # Metadata
    'filename',
    'filepath',
    'track_id',
    'track_name',
    'artist',

    # Embeddings
    'embedding',

    # Basic rhythm/key
    'bpm',
    'key',

    # Genre
    'genre_probs',
    'top_3_genres',

    # Basic audio features
    'danceability',
    'instrumentalness',

    # Mood features
    'mood_happy',
    'mood_sad',
    'mood_aggressive',
    'mood_relaxed',
    'mood_party',

    # Emotional features
    'valence',
    'arousal',

    # Advanced features (2023)
    'approachability_score',
    'approachability_2c_accessible',
    'approachability_2c_niche',
    'approachability_3c_probs',

    'engagement_score',
    'engagement_2c_low',
    'engagement_2c_high',
    'engagement_3c_probs',

    'mtg_jamendo_probs',

    # NEW (2024): Voice & Production
    'voice_gender_female',
    'voice_gender_male',
    'timbre_bright',
    'timbre_dark',
    'mood_acoustic',
    'mood_electronic',

    # NEW (2024): Advanced Classifiers
    'mtg_jamendo_instrument_probs',
    'arousal_emomusic',
    'valence_emomusic',
    'mtg_jamendo_genre_probs',
    'moods_mirex_probs',
}


def verify_cache(cache_path='analysis/cache/audio_features.pkl'):
    """Verify all tracks have all required fields"""
    if not Path(cache_path).exists():
        print(f"❌ Cache file not found: {cache_path}")
        return False

    with open(cache_path, 'rb') as f:
        features = pickle.load(f)

    print(f"📊 Checking {len(features)} tracks...\n")

    # Track missing fields across all tracks
    missing_by_field = defaultdict(int)
    tracks_with_missing = []

    for i, track in enumerate(features):
        missing_fields = REQUIRED_FIELDS - set(track.keys())

        if missing_fields:
            tracks_with_missing.append({
                'index': i,
                'track_name': track.get('track_name', 'Unknown'),
                'missing': missing_fields
            })

            for field in missing_fields:
                missing_by_field[field] += 1

    # Categorize fields by presence
    all_fields_present = {}
    all_fields_missing = {}

    for field in REQUIRED_FIELDS:
        present_count = sum(1 for track in features if field in track)
        missing_count = len(features) - present_count

        if missing_count == 0:
            all_fields_present[field] = present_count
        else:
            all_fields_missing[field] = (present_count, missing_count)

    # Print comprehensive status
    print("\n" + "=" * 80)
    print("COMPREHENSIVE FEATURE STATUS")
    print("=" * 80)

    # Group fields by category
    categories = {
        "Metadata": ['filename', 'filepath', 'track_id', 'track_name', 'artist'],
        "Embeddings": ['embedding'],
        "Basic Audio": ['bpm', 'key', 'danceability', 'instrumentalness'],
        "Genre": ['genre_probs', 'top_3_genres'],
        "Moods (Binary)": ['mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed', 'mood_party'],
        "Emotions (Continuous)": ['valence', 'arousal'],
        "Approachability (2023)": ['approachability_score', 'approachability_2c_accessible', 'approachability_2c_niche', 'approachability_3c_probs'],
        "Engagement (2023)": ['engagement_score', 'engagement_2c_low', 'engagement_2c_high', 'engagement_3c_probs'],
        "MTG-Jamendo Mood/Theme (2023)": ['mtg_jamendo_probs'],
        "Voice Gender (2024)": ['voice_gender_female', 'voice_gender_male'],
        "Timbre (2024)": ['timbre_bright', 'timbre_dark'],
        "Production Style (2024)": ['mood_acoustic', 'mood_electronic'],
        "Instruments (2024)": ['mtg_jamendo_instrument_probs'],
        "Alternative Emotions (2024)": ['arousal_emomusic', 'valence_emomusic'],
        "Alternative Genres (2024)": ['mtg_jamendo_genre_probs'],
        "MIREX Moods (2024)": ['moods_mirex_probs'],
    }

    for category, fields in categories.items():
        category_fields = [f for f in fields if f in REQUIRED_FIELDS]
        if not category_fields:
            continue

        present_in_category = [f for f in category_fields if f in all_fields_present]
        missing_in_category = [f for f in category_fields if f in all_fields_missing]

        if missing_in_category:
            status = f"⚠️  INCOMPLETE ({len(present_in_category)}/{len(category_fields)} fields)"
        else:
            status = f"✅ COMPLETE ({len(category_fields)}/{len(category_fields)} fields)"

        print(f"\n{category}: {status}")
        print("-" * 80)

        for field in category_fields:
            if field in all_fields_present:
                print(f"  ✅ {field:40s} Present in all {len(features)} tracks")
            elif field in all_fields_missing:
                present, missing = all_fields_missing[field]
                pct_missing = (missing / len(features)) * 100
                print(f"  ❌ {field:40s} Missing in {missing}/{len(features)} tracks ({pct_missing:.1f}%)")

    print("\n" + "=" * 80)

    # Overall summary
    total_present = len(all_fields_present)
    total_missing = len(all_fields_missing)
    total_fields = len(REQUIRED_FIELDS)

    if not tracks_with_missing:
        print("✅ SUCCESS! All tracks have all required fields.\n")
        print(f"📊 Coverage: {total_present}/{total_fields} fields ({total_present/total_fields*100:.1f}%)")

        # Sample a random track to show values
        import random
        sample = random.choice(features)
        print(f"\n📝 Sample track: {sample.get('track_name', 'Unknown')} by {sample.get('artist', 'Unknown')}")
        print(f"\n   Basic Features:")
        print(f"   - Danceability: {sample.get('danceability', 'MISSING'):.3f}")
        print(f"   - BPM: {sample.get('bpm', 'MISSING'):.1f}")
        print(f"   - Instrumentalness: {sample.get('instrumentalness', 'MISSING'):.3f}")
        print(f"\n   Emotional Features:")
        print(f"   - Valence: {sample.get('valence', 'MISSING'):.3f}")
        print(f"   - Arousal: {sample.get('arousal', 'MISSING'):.3f}")
        print(f"   - Happy: {sample.get('mood_happy', 'MISSING'):.3f}")
        print(f"   - Sad: {sample.get('mood_sad', 'MISSING'):.3f}")
        print(f"\n   NEW (2024) Features:")
        female = sample.get('voice_gender_female', 'MISSING')
        male = sample.get('voice_gender_male', 'MISSING')
        print(f"   - Voice (Female/Male): {female if female == 'MISSING' else f'{female:.3f}'}/{male if male == 'MISSING' else f'{male:.3f}'}")
        bright = sample.get('timbre_bright', 'MISSING')
        dark = sample.get('timbre_dark', 'MISSING')
        print(f"   - Timbre (Bright/Dark): {bright if bright == 'MISSING' else f'{bright:.3f}'}/{dark if dark == 'MISSING' else f'{dark:.3f}'}")
        acoustic = sample.get('mood_acoustic', 'MISSING')
        electronic = sample.get('mood_electronic', 'MISSING')
        print(f"   - Sound (Acoustic/Electronic): {acoustic if acoustic == 'MISSING' else f'{acoustic:.3f}'}/{electronic if electronic == 'MISSING' else f'{electronic:.3f}'}")
        arousal_emo = sample.get('arousal_emomusic', 'MISSING')
        valence_emo = sample.get('valence_emomusic', 'MISSING')
        print(f"   - Alt Arousal/Valence: {arousal_emo if arousal_emo == 'MISSING' else f'{arousal_emo:.3f}'}/{valence_emo if valence_emo == 'MISSING' else f'{valence_emo:.3f}'}")
        instrument_count = len(sample.get('mtg_jamendo_instrument_probs', []))
        genre_count = len(sample.get('mtg_jamendo_genre_probs', []))
        mood_count = len(sample.get('moods_mirex_probs', []))
        print(f"   - Instrument classes: {instrument_count}/40")
        print(f"   - Genre classes: {genre_count}/87")
        print(f"   - MIREX mood clusters: {mood_count}/5")

        return True
    else:
        print(f"❌ PROBLEMS FOUND! {len(tracks_with_missing)} tracks have missing fields.\n")
        print(f"📊 Coverage: {total_present}/{total_fields} fields complete, {total_missing} fields incomplete")

        print("\n" + "=" * 80)
        print("MISSING FIELDS BREAKDOWN")
        print("=" * 80)

        for field, count in sorted(missing_by_field.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(features)) * 100
            print(f"  ❌ {field:40s} Missing in {count}/{len(features)} tracks ({pct:.1f}%)")

        print("\n" + "=" * 80)
        print(f"AFFECTED TRACKS (showing first 5 of {len(tracks_with_missing)})")
        print("=" * 80)

        for track_info in tracks_with_missing[:5]:
            print(f"\n  Track #{track_info['index']}: {track_info['track_name']}")
            missing_list = sorted(track_info['missing'])
            # Group by category for cleaner display
            print(f"    Missing {len(missing_list)} fields:")
            for i in range(0, len(missing_list), 3):
                batch = missing_list[i:i+3]
                print(f"      {', '.join(batch)}")

        print("\n" + "=" * 80)
        print("💡 SOLUTION")
        print("=" * 80)
        print("\n  Run this command to extract all missing features:")
        print("  python analysis/run_analysis.py --re-classify-audio")
        print("\n  This will:")
        print("  - Load your existing cache")
        print("  - Detect all tracks missing features")
        print("  - Extract ONLY the missing features")
        print("  - Save the updated cache")

        return False


if __name__ == '__main__':
    print("=" * 60)
    print("AUDIO CACHE VERIFICATION")
    print("=" * 60)
    print()

    success = verify_cache()

    print("\n" + "=" * 60)
    if success:
        print("✅ Cache is ready to use!")
    else:
        print("❌ Cache needs updating")
    print("=" * 60)
