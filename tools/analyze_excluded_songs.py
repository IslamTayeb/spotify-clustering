#!/usr/bin/env python3
"""Analyze which songs are excluded from the analysis pipeline and why.

This script identifies songs that are "left out" based on the inclusion logic:
- If has_lyrics(song): include
- elif instrumentalness >= 0.5: include (instrumental, lyrics not expected)
- else: exclude (no lyrics, but should have had them)
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict

def main():
    # Load all data sources
    print("Loading data sources...")

    # 1. All saved tracks from Spotify
    with open("spotify/saved_tracks.json", "r") as f:
        saved_tracks = json.load(f)
    saved_tracks_by_id = {t["track_id"]: t for t in saved_tracks}
    print(f"  Spotify library: {len(saved_tracks)} songs")

    # 2. Audio features (songs with MP3s that were analyzed)
    with open("analysis/cache/audio_features.pkl", "rb") as f:
        audio_features = pickle.load(f)
    audio_by_id = {t["track_id"]: t for t in audio_features}
    print(f"  Audio features: {len(audio_features)} songs")

    # 3. Lyric features (songs with lyrics analyzed by GPT)
    with open("analysis/cache/lyric_interpretable_features.pkl", "rb") as f:
        lyric_features = pickle.load(f)
    lyric_by_id = {t["track_id"]: t for t in lyric_features}
    print(f"  Lyric features: {len(lyric_features)} songs")

    # 4. Check lyrics directory for raw lyrics files
    lyrics_dir = Path("lyrics/data")
    lyrics_files = set()
    if lyrics_dir.exists():
        for f in lyrics_dir.glob("*.txt"):
            lyrics_files.add(f.stem)
    print(f"  Raw lyrics files: {len(lyrics_files)} files")

    # 5. Check songs directory for MP3 files
    songs_dir = Path("songs/data")
    mp3_files = set()
    if songs_dir.exists():
        for f in songs_dir.glob("*.mp3"):
            mp3_files.add(f.stem)
    print(f"  MP3 files: {len(mp3_files)} files")

    print("\n" + "="*80)
    print("ANALYSIS: What's excluded at each stage?")
    print("="*80)

    # Stage 1: Spotify -> MP3 downloads
    no_mp3 = []
    for track_id, track in saved_tracks_by_id.items():
        if track_id not in audio_by_id:
            no_mp3.append(track)

    print(f"\n[Stage 1] No MP3/Audio features: {len(no_mp3)} songs")
    if no_mp3:
        print("  (These songs couldn't be downloaded or analyzed)")
        for t in no_mp3[:10]:
            print(f"    - {t['artists'][0]} - {t['track_name']}")
        if len(no_mp3) > 10:
            print(f"    ... and {len(no_mp3) - 10} more")

    # Stage 2: Apply the exclusion logic from the user's description
    # Songs that:
    # - Have audio features (MP3 was downloaded and analyzed)
    # - Don't have lyrics
    # - AND instrumentalness < 0.5 (should have had lyrics)

    excluded_songs = []
    included_with_lyrics = []
    included_instrumental = []

    for track_id, audio in audio_by_id.items():
        has_lyrics = track_id in lyric_by_id
        instrumentalness = audio.get("instrumentalness", 0.5)

        track_info = saved_tracks_by_id.get(track_id, {})
        song_info = {
            "track_id": track_id,
            "track_name": audio.get("track_name", track_info.get("track_name", "Unknown")),
            "artist": audio.get("artist", ", ".join(track_info.get("artists", ["Unknown"]))),
            "instrumentalness": instrumentalness,
            "has_lyrics": has_lyrics,
        }

        if has_lyrics:
            included_with_lyrics.append(song_info)
        elif instrumentalness >= 0.5:
            included_instrumental.append(song_info)
        else:
            # This is the excluded category
            excluded_songs.append(song_info)

    print(f"\n[Stage 2] Applying inclusion logic to {len(audio_by_id)} songs with audio:")
    print(f"  ✓ Included (has lyrics): {len(included_with_lyrics)} songs")
    print(f"  ✓ Included (instrumental, instrumentalness >= 0.5): {len(included_instrumental)} songs")
    print(f"  ✗ EXCLUDED (no lyrics + instrumentalness < 0.5): {len(excluded_songs)} songs")

    total_included = len(included_with_lyrics) + len(included_instrumental)
    total_with_audio = len(audio_by_id)
    print(f"\n  Final dataset: {total_included} / {total_with_audio} ({100*total_included/total_with_audio:.1f}%)")

    print("\n" + "="*80)
    print("EXCLUDED SONGS (no lyrics, but should have had them)")
    print("="*80)

    # Sort by instrumentalness to see which ones are "most vocal"
    excluded_songs.sort(key=lambda x: x["instrumentalness"])

    print(f"\nTotal excluded: {len(excluded_songs)} songs")
    print("\nSorted by instrumentalness (most vocal first):\n")

    for i, song in enumerate(excluded_songs, 1):
        inst = song["instrumentalness"]
        print(f"{i:3}. [{inst:.2f}] {song['artist']} - {song['track_name']}")

    # Save to JSON for further analysis
    output_path = Path("analysis/outputs/excluded_songs.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total_spotify_library": len(saved_tracks),
                "total_with_audio": len(audio_by_id),
                "included_with_lyrics": len(included_with_lyrics),
                "included_instrumental": len(included_instrumental),
                "excluded": len(excluded_songs),
            },
            "excluded_songs": excluded_songs,
        }, f, indent=2)
    print(f"\n\nSaved detailed report to: {output_path}")

    # Analyze WHY lyrics might be missing
    print("\n" + "="*80)
    print("ANALYSIS: Why might lyrics be missing?")
    print("="*80)

    # Group by instrumentalness ranges
    ranges = {
        "0.0-0.1 (very vocal)": [],
        "0.1-0.2": [],
        "0.2-0.3": [],
        "0.3-0.4": [],
        "0.4-0.5": [],
    }
    for song in excluded_songs:
        inst = song["instrumentalness"]
        if inst < 0.1:
            ranges["0.0-0.1 (very vocal)"].append(song)
        elif inst < 0.2:
            ranges["0.1-0.2"].append(song)
        elif inst < 0.3:
            ranges["0.2-0.3"].append(song)
        elif inst < 0.4:
            ranges["0.3-0.4"].append(song)
        else:
            ranges["0.4-0.5"].append(song)

    print("\nDistribution by instrumentalness:")
    for range_name, songs in ranges.items():
        print(f"  {range_name}: {len(songs)} songs")

    print("\n" + "="*80)
    print("MOST LIKELY REASONS FOR MISSING LYRICS:")
    print("="*80)
    print("""
Common reasons why Genius/Musixmatch might not have lyrics:
1. Very niche/underground artists
2. Non-English songs with no translations
3. Very new releases
4. Leaks, unreleased tracks, or unofficial versions
5. Remixes or extended versions
6. SoundCloud/YouTube-only tracks
7. Video game or anime soundtracks
8. Covers or live versions
""")

if __name__ == "__main__":
    main()
