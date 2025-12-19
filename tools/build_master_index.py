#!/usr/bin/env python3
"""
Build a master index JSON that connects Spotify metadata, MP3 files, and lyrics.
Uses cascading matching: exact artist+song, exact song only, then fuzzy matching.
"""

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from rapidfuzz import fuzz, process


def normalize_for_matching(text: str) -> str:
    """
    Normalize text for matching.
    """
    # Unicode normalization
    text = unicodedata.normalize('NFC', text)

    # Replace various special characters
    text = text.replace('〜', '-').replace('～', '-')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace('&', 'and')
    text = text.replace(':', '-')

    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^\w\s]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()

    return text


def build_file_indexes(directory: Path, extension: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build two dictionaries for file matching:
    1. Full filename index (normalized full stem -> relative path)
    2. Track name only index (normalized track name -> relative path)

    Returns (full_index, track_only_index)
    """
    full_index = {}
    track_only_index = {}
    data_dir = directory / "data"

    if data_dir.exists():
        for file_path in data_dir.glob(f"*{extension}"):
            rel_path = str(file_path.relative_to(directory))
            stem = file_path.stem

            # Full filename normalized
            full_normalized = normalize_for_matching(stem)
            full_index[full_normalized] = rel_path

            # Track name only (everything after first " - ")
            parts = stem.split(' - ', 1)
            if len(parts) > 1:
                track_normalized = normalize_for_matching(parts[1])
                # If multiple files have same track name, keep first one
                if track_normalized not in track_only_index:
                    track_only_index[track_normalized] = rel_path

    return full_index, track_only_index


def match_file(artist: str, track_name: str, full_index: Dict[str, str],
               track_only_index: Dict[str, str], match_stats: Dict[str, int]) -> Optional[str]:
    """
    Try to match a file using cascading strategy:
    1. Exact match on "artist - track"
    2. Exact match on track name only
    3. Fuzzy match on track name (threshold 85)

    Returns relative path if found, None otherwise.
    Also updates match_stats with which strategy worked.
    """
    # Strategy 1: Exact match on full "artist - track"
    full_query = normalize_for_matching(f"{artist} - {track_name}")
    if full_query in full_index:
        match_stats["exact_full"] += 1
        return full_index[full_query]

    # Strategy 2: Exact match on track name only
    track_query = normalize_for_matching(track_name)
    if track_query in track_only_index:
        match_stats["exact_track"] += 1
        return track_only_index[track_query]

    # Strategy 3: Fuzzy match on track name
    if track_only_index:
        result = process.extractOne(
            track_query,
            track_only_index.keys(),
            scorer=fuzz.ratio,
            score_cutoff=85
        )
        if result:
            matched_key, score, _ = result
            match_stats["fuzzy"] += 1
            return track_only_index[matched_key]

    # No match found
    match_stats["no_match"] += 1
    return None


def build_master_index():
    """
    Build master index connecting Spotify metadata, MP3s, and lyrics.
    """
    # Paths
    base_dir = Path(__file__).parent
    spotify_json = base_dir / "spotify" / "saved_tracks.json"
    songs_dir = base_dir / "songs"
    lyrics_dir = base_dir / "lyrics"
    output_file = base_dir / "master_index.json"

    # Load Spotify tracks
    print("Loading Spotify tracks...")
    with open(spotify_json, 'r', encoding='utf-8') as f:
        spotify_tracks = json.load(f)
    print(f"Found {len(spotify_tracks)} Spotify tracks")

    # Build file indexes
    print("\nIndexing MP3 files...")
    mp3_full_index, mp3_track_index = build_file_indexes(songs_dir, ".mp3")
    print(f"Found {len(mp3_full_index)} MP3 files")

    print("\nIndexing lyrics files...")
    lyrics_full_index, lyrics_track_index = build_file_indexes(lyrics_dir, ".txt")
    print(f"Found {len(lyrics_full_index)} lyrics files")

    # Track which files get matched and how
    matched_mp3s = set()
    matched_lyrics = set()
    mp3_match_stats = {"exact_full": 0, "exact_track": 0, "fuzzy": 0, "no_match": 0}
    lyrics_match_stats = {"exact_full": 0, "exact_track": 0, "fuzzy": 0, "no_match": 0}

    # Build master index
    print("\nMatching tracks (exact artist+song -> exact song -> fuzzy)...")
    master_index = []
    stats = {
        "total_tracks": len(spotify_tracks),
        "has_mp3": 0,
        "has_lyrics": 0,
        "has_both": 0,
        "has_neither": 0
    }

    for i, track in enumerate(spotify_tracks):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(spotify_tracks)} tracks...")

        track_id = track["track_id"]
        track_name = track["track_name"]
        # Use first artist for matching (most relevant)
        artist = track["artists"][0] if track["artists"] else ""

        # Match MP3 and lyrics using cascading strategy
        mp3_path = match_file(artist, track_name, mp3_full_index, mp3_track_index, mp3_match_stats)
        lyrics_path = match_file(artist, track_name, lyrics_full_index, lyrics_track_index, lyrics_match_stats)

        # Track which files were matched
        if mp3_path:
            matched_mp3s.add(mp3_path)
        if lyrics_path:
            matched_lyrics.add(lyrics_path)

        # Update statistics
        has_mp3 = mp3_path is not None
        has_lyrics = lyrics_path is not None

        if has_mp3:
            stats["has_mp3"] += 1
        if has_lyrics:
            stats["has_lyrics"] += 1
        if has_mp3 and has_lyrics:
            stats["has_both"] += 1
        if not has_mp3 and not has_lyrics:
            stats["has_neither"] += 1

        # Build entry with all artists for display
        all_artists = ", ".join(track["artists"])
        entry = {
            "track_id": track_id,
            "track_name": track_name,
            "artist": all_artists,
            "spotify_metadata": f"spotify/saved_tracks.json#{track_id}",
            "mp3_file": f"songs/{mp3_path}" if mp3_path else None,
            "lyrics_file": f"lyrics/{lyrics_path}" if lyrics_path else None
        }

        master_index.append(entry)

    # Calculate match percentages for files
    mp3_match_pct = (len(matched_mp3s) / len(mp3_full_index) * 100) if mp3_full_index else 0
    lyrics_match_pct = (len(matched_lyrics) / len(lyrics_full_index) * 100) if lyrics_full_index else 0

    # Save master index
    print(f"\nSaving master index to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "statistics": {
                **stats,
                "total_mp3_files": len(mp3_full_index),
                "matched_mp3_files": len(matched_mp3s),
                "mp3_match_percentage": round(mp3_match_pct, 2),
                "total_lyrics_files": len(lyrics_full_index),
                "matched_lyrics_files": len(matched_lyrics),
                "lyrics_match_percentage": round(lyrics_match_pct, 2)
            },
            "tracks": master_index
        }, f, indent=2, ensure_ascii=False)

    # Print statistics
    print("\n" + "="*60)
    print("MASTER INDEX STATISTICS")
    print("="*60)
    print(f"\nSpotify Tracks: {stats['total_tracks']}")
    print(f"  - With MP3: {stats['has_mp3']} ({stats['has_mp3']/stats['total_tracks']*100:.1f}%)")
    print(f"  - With lyrics: {stats['has_lyrics']} ({stats['has_lyrics']/stats['total_tracks']*100:.1f}%)")
    print(f"  - With both: {stats['has_both']} ({stats['has_both']/stats['total_tracks']*100:.1f}%)")
    print(f"  - With neither: {stats['has_neither']} ({stats['has_neither']/stats['total_tracks']*100:.1f}%)")

    print(f"\nMP3 Files: {len(mp3_full_index)}")
    print(f"  - Matched to Spotify tracks: {len(matched_mp3s)} ({mp3_match_pct:.1f}%)")
    print(f"  - Unmatched: {len(mp3_full_index) - len(matched_mp3s)} ({100-mp3_match_pct:.1f}%)")
    print(f"  - Match strategy breakdown:")
    print(f"    • Exact (artist + song): {mp3_match_stats['exact_full']}")
    print(f"    • Exact (song only): {mp3_match_stats['exact_track']}")
    print(f"    • Fuzzy match: {mp3_match_stats['fuzzy']}")

    print(f"\nLyrics Files: {len(lyrics_full_index)}")
    print(f"  - Matched to Spotify tracks: {len(matched_lyrics)} ({lyrics_match_pct:.1f}%)")
    print(f"  - Unmatched: {len(lyrics_full_index) - len(matched_lyrics)} ({100-lyrics_match_pct:.1f}%)")
    print(f"  - Match strategy breakdown:")
    print(f"    • Exact (artist + song): {lyrics_match_stats['exact_full']}")
    print(f"    • Exact (song only): {lyrics_match_stats['exact_track']}")
    print(f"    • Fuzzy match: {lyrics_match_stats['fuzzy']}")

    print(f"\nMaster index saved to: {output_file}")
    print("="*60)


if __name__ == "__main__":
    build_master_index()
