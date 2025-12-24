#!/usr/bin/env python3
"""
Script to check which tracks from Spotify library match downloaded MP3 files.
Shows detailed statistics and helps debug matching issues.
"""

import json
import re
from pathlib import Path
from typing import Set, Dict, List, Tuple
from collections import defaultdict


def normalize_filename(text: str) -> str:
    """
    Normalize text for filename comparison.
    Aggressively normalizes to match how files get saved.
    """
    # First, normalize Unicode characters
    text = text.replace('～', '-')  # Japanese wave dash
    text = text.replace('〜', '-')  # Another Japanese wave dash
    text = text.replace('~', '-')  # Regular tilde

    # Replace colon with dash (spotdl/yt-dlp do this)
    text = text.replace(':', '-')

    # Remove or replace other characters that can't be in filenames
    text = re.sub(r'[<>"/\\|?*]', '', text)  # Remove filesystem-illegal chars (note: colon already handled)
    text = re.sub(r'[\'\"‚„""'']', '', text)  # Remove all quotes
    text = re.sub(r'[\(\)\[\]\{\}]', '', text)  # Remove all brackets
    text = re.sub(r'[–—−]', '-', text)  # Normalize all dash types to hyphen
    text = re.sub(r'[,;]', '', text)  # Remove punctuation
    text = re.sub(r'[!¡]', '', text)  # Remove exclamation marks
    text = re.sub(r'[.]', '', text)  # Remove periods (except in extension)
    text = re.sub(r'[&]', 'and', text)  # Replace ampersand
    text = re.sub(r'[＃#]', '', text)  # Remove hash/number signs
    text = re.sub(r'[$￥]', '', text)  # Remove currency symbols
    text = re.sub(r'[%]', '', text)  # Remove percent
    text = re.sub(r'[@]', '', text)  # Remove at sign
    text = re.sub(r'[+]', '', text)  # Remove plus
    text = re.sub(r'[=]', '', text)  # Remove equals

    # Normalize spaces and dashes
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\s*-\s*', '-', text)  # Remove spaces around dashes
    text = re.sub(r'-+', '-', text)  # Multiple dashes to single (after removing spaces)

    text = text.strip().strip('-').lower()  # Strip whitespace and leading/trailing dashes
    return text


def get_downloaded_tracks(songs_dir: Path) -> Tuple[Set[str], Dict[str, str]]:
    """
    Get set of already downloaded track identifiers from songs/ directory.
    Returns: (normalized_ids, normalized_to_original_map)
    """
    normalized_ids = set()
    normalized_to_original = {}

    if not songs_dir.exists():
        print(f"Songs directory not found: {songs_dir}")
        return normalized_ids, normalized_to_original

    for file in songs_dir.glob("*.mp3"):
        # Remove .mp3 extension
        original_filename = file.stem
        normalized = normalize_filename(original_filename)

        normalized_ids.add(normalized)
        normalized_to_original[normalized] = original_filename

    return normalized_ids, normalized_to_original


def get_spotify_tracks(json_path: Path) -> List[Dict]:
    """
    Read all tracks from saved_tracks.json.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        tracks = json.load(f)
    return tracks


def analyze_matches(spotify_tracks: List[Dict], downloaded: Set[str]) -> Dict:
    """
    Analyze which tracks match and which don't.
    Returns detailed statistics and lists.
    """
    matched_tracks = []
    missing_tracks = []
    file_match_count = defaultdict(int)  # Count how many tracks match each file

    for track in spotify_tracks:
        # Create identifier: "artist - track"
        artists = ", ".join(track.get("artists", []))
        track_name = track.get("track_name", "")

        # Create normalized identifier
        identifier = f"{artists} - {track_name}"
        normalized_id = normalize_filename(identifier)

        track_info = {
            "artists": artists,
            "track_name": track_name,
            "album": track.get("album_name", ""),
            "identifier": identifier,
            "normalized": normalized_id,
            "url": track.get("external_url", ""),
        }

        if normalized_id in downloaded:
            matched_tracks.append(track_info)
            file_match_count[normalized_id] += 1
        else:
            missing_tracks.append(track_info)

    # Find files that match multiple tracks
    duplicates = {k: v for k, v in file_match_count.items() if v > 1}

    return {
        "matched": matched_tracks,
        "missing": missing_tracks,
        "duplicates": duplicates,
        "file_match_count": file_match_count,
    }


def print_statistics(analysis: Dict, total_files: int, total_tracks: int):
    """
    Print detailed statistics about the matching.
    """
    matched_count = len(analysis["matched"])
    missing_count = len(analysis["missing"])
    duplicate_count = len(analysis["duplicates"])

    print("\n" + "=" * 70)
    print("MATCHING STATISTICS")
    print("=" * 70)
    print(f"📊 Total tracks in Spotify library:     {total_tracks}")
    print(f"📁 Total MP3 files in songs/:           {total_files}")
    print(f"✅ Successfully matched tracks:         {matched_count}")
    print(f"❌ Missing/unmatched tracks:            {missing_count}")

    if total_tracks > 0:
        match_rate = (matched_count / total_tracks) * 100
        print(f"📈 Match rate:                          {match_rate:.1f}%")

    print(f"🔄 Files matching multiple tracks:      {duplicate_count}")
    print("=" * 70)


def print_duplicates(analysis: Dict, normalized_to_original: Dict, limit: int = 10):
    """
    Print files that match multiple tracks (likely duplicates in library).
    """
    duplicates = analysis["duplicates"]

    if not duplicates:
        print("\n✓ No duplicate matches found!")
        return

    print(f"\n{'=' * 70}")
    print(f"FILES MATCHING MULTIPLE TRACKS (showing top {limit})")
    print(f"{'=' * 70}")

    # Sort by match count (highest first)
    sorted_dups = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)

    for normalized_id, count in sorted_dups[:limit]:
        original_filename = normalized_to_original.get(normalized_id, normalized_id)
        print(f"\n📄 {original_filename}")
        print(f"   Matches {count} tracks in your library")


def print_sample_missing(missing_tracks: List[Dict], limit: int = 20, show_normalized: bool = False):
    """
    Print a sample of missing tracks.
    """
    if not missing_tracks:
        print("\n✓ All tracks are downloaded!")
        return

    print(f"\n{'=' * 70}")
    print(
        f"SAMPLE MISSING TRACKS (showing {min(limit, len(missing_tracks))} of {len(missing_tracks)})"
    )
    print(f"{'=' * 70}")

    for i, track in enumerate(missing_tracks[:limit], 1):
        print(f"{i:3}. {track['artists']} - {track['track_name']}")
        if track["album"]:
            print(f"     Album: {track['album']}")
        if show_normalized:
            print(f"     Normalized: {track['normalized']}")


def print_sample_matched(matched_tracks: List[Dict], limit: int = 10):
    """
    Print a sample of matched tracks.
    """
    if not matched_tracks:
        return

    print(f"\n{'=' * 70}")
    print(
        f"SAMPLE MATCHED TRACKS (showing {min(limit, len(matched_tracks))} of {len(matched_tracks)})"
    )
    print(f"{'=' * 70}")

    for i, track in enumerate(matched_tracks[:limit], 1):
        print(f"{i:3}. {track['artists']} - {track['track_name']}")


def main():
    # Setup paths
    base_dir = Path(__file__).parent
    json_path = base_dir / "api" / "data" / "saved_tracks.json"
    songs_dir = base_dir / "songs"
    missing_output = base_dir / "missing_tracks.txt"

    print("=" * 70)
    print("SPOTIFY LIBRARY MATCHER")
    print("=" * 70)

    # Load data
    print("\n1. Loading Spotify library...")
    spotify_tracks = get_spotify_tracks(json_path)
    print(f"   Loaded {len(spotify_tracks)} tracks from Spotify")

    print("\n2. Scanning downloaded MP3 files...")
    downloaded, normalized_to_original = get_downloaded_tracks(songs_dir)
    print(f"   Found {len(downloaded)} MP3 files in songs/")

    print("\n3. Analyzing matches...")
    analysis = analyze_matches(spotify_tracks, downloaded)

    # Print statistics
    print_statistics(analysis, len(downloaded), len(spotify_tracks))

    # Print duplicates
    print_duplicates(analysis, normalized_to_original, limit=10)

    # Print samples
    print_sample_matched(analysis["matched"], limit=10)
    print_sample_missing(analysis["missing"], limit=20, show_normalized=True)

    print(f"\n{'=' * 70}")
    print("Analysis complete!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
