#!/usr/bin/env python3
"""Fetch lyrics for Spotify tracks using Genius API"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import lyricsgenius

# Configuration
load_dotenv()
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")
SPOTIFY_JSON_PATH = "api/saved_tracks.json"
OUTPUT_DIR = Path("lyrics")
CACHE_FILE = OUTPUT_DIR / "lyrics_cache.json"

# Initialize
print("=" * 70)
print("SPOTIFY LYRICS FETCHER")
print("=" * 70)

OUTPUT_DIR.mkdir(exist_ok=True)

# Load Spotify data
print(f"\nLoading {SPOTIFY_JSON_PATH}...")
with open(SPOTIFY_JSON_PATH, "r", encoding="utf-8") as f:
    spotify_tracks = json.load(f)

total_tracks = len(spotify_tracks)
print(f"✓ Loaded {total_tracks} tracks")
print("\nFirst 5 tracks:")
for i, track in enumerate(spotify_tracks[:5]):
    artist = track["artists"][0] if track["artists"] else "Unknown"
    print(f"  {i + 1}. {artist} - {track['track_name']}")

# Initialize Genius
if not GENIUS_ACCESS_TOKEN or GENIUS_ACCESS_TOKEN == "your_genius_access_token_here":
    print("\n" + "!" * 70)
    print("ERROR: Set GENIUS_ACCESS_TOKEN in .env file")
    print("Get token from: https://genius.com/api-clients")
    print("!" * 70)
    exit(1)

genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)
genius.verbose = False
genius.remove_section_headers = True
genius.skip_non_songs = True
genius.excluded_terms = ["(Remix)", "(Live)"]
print("✓ Genius API initialized")

# Cache helpers
def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

cache = load_cache()
print(f"✓ Cache: {len(cache)} entries\n")

# Fetch lyrics
def get_lyrics(track_name, artist_name):
    """Fetch lyrics from Genius with caching"""
    cache_key = f"{artist_name}::{track_name}".lower().strip()

    if cache_key in cache:
        return cache[cache_key]

    try:
        song = genius.search_song(track_name, artist_name)
        if song:
            result = {
                "lyrics": song.lyrics,
                "genius_title": song.title,
                "genius_artist": song.artist,
                "genius_url": song.url,
                "genius_id": getattr(song, 'id', None),  # Handle missing id attribute
                "status": "success",
                "fetched_at": datetime.now().isoformat(),
            }
        else:
            result = {
                "status": "not_found",
                "fetched_at": datetime.now().isoformat()
            }
    except Exception as e:
        result = {
            "status": "error",
            "error": str(e),
            "fetched_at": datetime.now().isoformat()
        }

    cache[cache_key] = result
    return result


# Process all tracks
print("=" * 70)
print("FETCHING LYRICS")
print("=" * 70)
print("Progress saved every 10 songs. Safe to Ctrl+C and resume.\n")

results = []
stats = {"success": 0, "not_found": 0, "error": 0, "cached": 0}

for idx, track in enumerate(spotify_tracks):
    track_name = track["track_name"]
    artist_name = track["artists"][0] if track["artists"] else "Unknown"
    cache_key = f"{artist_name}::{track_name}".lower().strip()
    was_cached = cache_key in cache

    # Progress
    progress = f"[{idx + 1}/{total_tracks}]"
    print(f"{progress} {artist_name} - {track_name}"[:70], end="", flush=True)

    # Fetch
    result = get_lyrics(track_name, artist_name)

    # Build result
    enhanced_track = track.copy()
    enhanced_track["lyrics_status"] = result["status"]

    if result["status"] == "success":
        for key in ["lyrics", "genius_url", "genius_id", "genius_title", "genius_artist"]:
            enhanced_track[key] = result.get(key)
        stats["success"] += 1
        if was_cached:
            stats["cached"] += 1
            print(" ✓ (cached)")
        else:
            print(" ✓")
    elif result["status"] == "not_found":
        stats["not_found"] += 1
        print(" ✗ Not found")
    else:
        stats["error"] += 1
        print(f" ✗ {result.get('error', 'Unknown')[:30]}")

    results.append(enhanced_track)

    # Auto-save every 10 songs
    if (idx + 1) % 10 == 0:
        save_cache(cache)
        print(f"     Saved: {stats['success']} found, {stats['cached']} cached")

    # Rate limit (skip if cached)
    if not was_cached and idx < total_tracks - 1:
        time.sleep(0.5)

save_cache(cache)

# Save results
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Save complete data
with open(OUTPUT_DIR / "tracks_with_lyrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"✓ Complete: tracks_with_lyrics.json")

# Save lyrics-only (cleaner format)
lyrics_only = [
    {
        "spotify_id": t["track_id"],
        "track_name": t["track_name"],
        "artist": t["artists"][0] if t["artists"] else "Unknown",
        "album": t["album_name"],
        "lyrics": t.get("lyrics"),
        "genius_url": t.get("genius_url"),
        "genius_id": t.get("genius_id"),
    }
    for t in results if t["lyrics_status"] == "success"
]

with open(OUTPUT_DIR / "lyrics_only.json", "w", encoding="utf-8") as f:
    json.dump(lyrics_only, f, indent=2, ensure_ascii=False)
print(f"✓ Lyrics only: lyrics_only.json ({len(lyrics_only)} tracks)")

# Save individual text files
lyrics_dir = OUTPUT_DIR / "individual"
lyrics_dir.mkdir(exist_ok=True)

for track in results:
    if track["lyrics_status"] == "success":
        artist = track["artists"][0] if track["artists"] else "Unknown"
        safe_name = f"{artist} - {track['track_name']}"
        # Remove invalid filename chars
        for char in '/\\:?*"<>|':
            safe_name = safe_name.replace(char, "_" if char in "/\\" else "-")
        safe_name = safe_name[:150]

        with open(lyrics_dir / f"{safe_name}.txt", "w", encoding="utf-8") as f:
            f.write(f"Title: {track['track_name']}\n")
            f.write(f"Artist: {artist}\n")
            f.write(f"Album: {track['album_name']}\n")
            f.write(f"Spotify: {track['external_url']}\n")
            f.write(f"Genius: {track.get('genius_url', 'N/A')}\n")
            f.write("\n" + "=" * 60 + "\n\n")
            f.write(track.get("lyrics", ""))

print(f"✓ Individual: {len(lyrics_only)} files in individual/")

# Save failed tracks
failed = [
    {
        "track_name": t["track_name"],
        "artist": t["artists"][0] if t["artists"] else "Unknown",
        "status": t["lyrics_status"],
        "spotify_url": t["external_url"],
    }
    for t in results if t["lyrics_status"] != "success"
]

with open(OUTPUT_DIR / "failed_tracks.json", "w", encoding="utf-8") as f:
    json.dump(failed, f, indent=2, ensure_ascii=False)
print(f"✓ Failed: failed_tracks.json ({len(failed)} tracks)")

# Summary statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(f"Total tracks:         {total_tracks}")
print(f"Successfully matched: {stats['success']:4d} ({stats['success']/total_tracks*100:.1f}%)")
print(f"  - From cache:       {stats['cached']:4d}")
print(f"  - Newly fetched:    {stats['success']-stats['cached']:4d}")
print(f"Not found on Genius:  {stats['not_found']:4d} ({stats['not_found']/total_tracks*100:.1f}%)")
print(f"Errors:               {stats['error']:4d} ({stats['error']/total_tracks*100:.1f}%)")
print(f"\nCache size:           {len(cache)} entries")
print(f"Output directory:     {OUTPUT_DIR.absolute()}")
print("=" * 70)

# Show sample matched tracks
if lyrics_only:
    print("\n✓ Sample matched tracks:")
    for i, track in enumerate(lyrics_only[:3]):
        print(f"  {i+1}. {track['artist']} - {track['track_name']}")
        preview = track['lyrics'][:100].replace("\n", " ") if track.get('lyrics') else ""
        print(f"     {preview}...")

print("\n✓ Done! Lyrics saved to lyrics/ directory")
