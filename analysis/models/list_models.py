#!/usr/bin/env python3
"""
List all available Essentia models and identify which ones are currently used vs unused.
"""
from pathlib import Path
import re

# Models currently used in the codebase
CURRENTLY_USED = {
    'discogs-effnet-bs64-1.pb': 'Embeddings (base model)',
    'genre_discogs400-discogs-effnet-1.pb': 'Genre classification (400 classes)',
    'mood_happy-discogs-effnet-1.pb': 'Mood: Happy',
    'mood_sad-discogs-effnet-1.pb': 'Mood: Sad',
    'mood_aggressive-discogs-effnet-1.pb': 'Mood: Aggressive',
    'mood_relaxed-discogs-effnet-1.pb': 'Mood: Relaxed',
    'mood_party-discogs-effnet-1.pb': 'Mood: Party',
    'deam-msd-musicnn-2.pb': 'Valence/Arousal (MusiCNN)',
    'danceability-discogs-effnet-1.pb': 'Danceability',
    'voice_instrumental-discogs-effnet-1.pb': 'Voice/Instrumental',
    'approachability_2c-discogs-effnet-1.pb': 'Approachability (2-class)',
    'approachability_3c-discogs-effnet-1.pb': 'Approachability (3-class)',
    'approachability_regression-discogs-effnet-1.pb': 'Approachability (regression)',
    'engagement_2c-discogs-effnet-1.pb': 'Engagement (2-class)',
    'engagement_3c-discogs-effnet-1.pb': 'Engagement (3-class)',
    'engagement_regression-discogs-effnet-1.pb': 'Engagement (regression)',
    'mtg_jamendo_moodtheme-discogs-effnet-1.pb': 'MTG-Jamendo mood/theme (56 classes)',
}

def categorize_model(filename):
    """Categorize model by name pattern"""
    name = filename.lower()

    # Categorization patterns
    categories = {
        'Mood': ['mood_', 'moodtheme'],
        'Genre': ['genre', 'style'],
        'Voice/Gender': ['voice', 'gender', 'vocal'],
        'Emotional': ['arousal', 'valence', 'emotion'],
        'Rhythm/Tempo': ['rhythm', 'tempo', 'bpm', 'beat', 'danceability'],
        'Timbre': ['timbre', 'brightness', 'tonal'],
        'Engagement': ['engagement', 'approachability'],
        'Instrumentation': ['instrument', 'music_loop'],
        'Embeddings': ['effnet', 'embedding', 'vgg'],
        'Other': [],
    }

    for category, patterns in categories.items():
        if any(pattern in name for pattern in patterns):
            return category

    return 'Other'


def extract_description(filename):
    """Try to extract human-readable description from filename"""
    # Remove .pb extension
    name = filename.replace('.pb', '')

    # Common patterns
    name = name.replace('-discogs-effnet-1', '')
    name = name.replace('-msd-musicnn-2', '')
    name = name.replace('_', ' ')
    name = name.replace('-', ' ')

    return name.title()


def main():
    models_dir = Path.home() / '.essentia' / 'models'

    if not models_dir.exists():
        print(f"❌ Essentia models directory not found: {models_dir}")
        print("\n💡 Try running: python analysis/models/download_models.py")
        print("   Or check if models are in a different location")
        return

    # Get all .pb files
    all_models = sorted([f.name for f in models_dir.glob('*.pb')])

    if not all_models:
        print(f"❌ No .pb model files found in: {models_dir}")
        return

    print("=" * 80)
    print("ESSENTIA MODELS INVENTORY")
    print("=" * 80)
    print(f"\n📁 Location: {models_dir}")
    print(f"📊 Total models found: {len(all_models)}")
    print(f"✅ Currently used: {len(CURRENTLY_USED)}")
    print(f"🆕 Unused available: {len(all_models) - len(CURRENTLY_USED)}")

    # Categorize all models
    categorized = {}
    for model in all_models:
        category = categorize_model(model)
        if category not in categorized:
            categorized[category] = []
        categorized[category].append(model)

    print("\n" + "=" * 80)
    print("CURRENTLY USED MODELS")
    print("=" * 80)

    for model, description in sorted(CURRENTLY_USED.items()):
        if model in all_models:
            print(f"  ✅ {model}")
            print(f"     → {description}")
        else:
            print(f"  ⚠️  {model} (MISSING!)")
            print(f"     → {description}")

    print("\n" + "=" * 80)
    print("UNUSED AVAILABLE MODELS (OPPORTUNITIES!)")
    print("=" * 80)

    unused = [m for m in all_models if m not in CURRENTLY_USED]

    if not unused:
        print("\n  ✅ All available models are already being used!")
    else:
        # Group unused by category
        unused_categorized = {}
        for model in unused:
            category = categorize_model(model)
            if category not in unused_categorized:
                unused_categorized[category] = []
            unused_categorized[category].append(model)

        for category in sorted(unused_categorized.keys()):
            models = unused_categorized[category]
            print(f"\n  📂 {category} ({len(models)} models)")
            print("  " + "-" * 76)
            for model in sorted(models):
                description = extract_description(model)
                print(f"     🆕 {model}")
                print(f"        → {description}")

    print("\n" + "=" * 80)
    print("SUMMARY BY CATEGORY")
    print("=" * 80)

    for category in sorted(categorized.keys()):
        models = categorized[category]
        used_count = sum(1 for m in models if m in CURRENTLY_USED)
        unused_count = len(models) - used_count

        print(f"\n  {category}:")
        print(f"    Total: {len(models)} | Used: {used_count} | Unused: {unused_count}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR YOUR LIBRARY")
    print("=" * 80)

    # Check for high-priority unused models
    priority_patterns = {
        'gender': '🎤 Voice Gender - Separates male/female/instrumental vocals',
        'brightness': '🎨 Timbre/Brightness - Sonic texture dimension',
        'timbre': '🎨 Timbre - Acoustic character of sound',
        'arousal-emomusic': '💓 Alternative Arousal - Ensemble with existing model',
        'valence-emomusic': '😊 Alternative Valence - Ensemble with existing model',
        'music_loop': '🔁 Loop Detection - Repetitive vs developmental structure',
        'rhythm': '🥁 Rhythm Features - Pattern complexity',
        'beat': '🥁 Beat Features - Rhythmic characteristics',
    }

    recommended = []
    for model in unused:
        model_lower = model.lower()
        for pattern, description in priority_patterns.items():
            if pattern in model_lower:
                recommended.append((model, description))
                break

    if recommended:
        print("\n  🌟 HIGH PRIORITY (Based on your library):\n")
        for model, description in recommended:
            print(f"     {description}")
            print(f"        File: {model}\n")
    else:
        print("\n  ℹ️  No obvious high-priority models found in unused set")

    print("\n" + "=" * 80)
    print(f"📋 Full list saved to: {models_dir / 'INVENTORY.txt'}")
    print("=" * 80)

    # Save detailed inventory to file
    with open(models_dir / 'INVENTORY.txt', 'w') as f:
        f.write("ESSENTIA MODELS INVENTORY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Location: {models_dir}\n")
        f.write(f"Generated: {Path(__file__).name}\n\n")

        f.write("ALL MODELS:\n")
        f.write("-" * 80 + "\n")
        for model in all_models:
            status = "USED" if model in CURRENTLY_USED else "UNUSED"
            f.write(f"[{status}] {model}\n")
            if model in CURRENTLY_USED:
                f.write(f"        {CURRENTLY_USED[model]}\n")
            f.write("\n")


if __name__ == '__main__':
    main()
