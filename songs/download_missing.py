#!/usr/bin/env python3
"""
Script to download missing Spotify tracks using spotdl.
Compares saved_tracks.json with already downloaded files in songs/ directory.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import List, Set, Dict
import re


def normalize_filename(text: str) -> str:
    """
    Normalize text for filename comparison.
    Removes special characters and normalizes spaces.
    """
    # Remove common special characters that might differ between API and filename
    text = re.sub(r'[\'\"‚„""'']', '', text)  # Remove quotes
    text = re.sub(r'[\(\)\[\]\{\}]', '', text)  # Remove brackets
    text = re.sub(r'[:\-–—]', ' ', text)  # Replace separators with space
    text = re.sub(r'[/\\|]', ' ', text)  # Replace path separators
    text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces
    text = text.strip().lower()
    return text


def get_downloaded_tracks(songs_dir: Path) -> Set[str]:
    """
    Get set of already downloaded track identifiers from songs/ directory.
    Returns normalized "artist - track" strings.
    """
    downloaded = set()

    if not songs_dir.exists():
        print(f"Songs directory not found: {songs_dir}")
        return downloaded

    for file in songs_dir.glob("*.mp3"):
        # Remove .mp3 extension and normalize
        filename = file.stem
        normalized = normalize_filename(filename)
        downloaded.add(normalized)

    print(f"Found {len(downloaded)} already downloaded tracks")
    return downloaded


def get_spotify_tracks(json_path: Path) -> List[Dict]:
    """
    Read all tracks from saved_tracks.json.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        tracks = json.load(f)

    print(f"Found {len(tracks)} tracks in Spotify library")
    return tracks


def find_missing_tracks(spotify_tracks: List[Dict], downloaded: Set[str]) -> tuple[List[Dict], int]:
    """
    Find tracks that are in Spotify library but not downloaded.
    Returns: (missing_tracks, matched_count)
    """
    missing = []
    matched = 0

    for track in spotify_tracks:
        # Create identifier: "artist - track"
        artists = ", ".join(track.get("artists", []))
        track_name = track.get("track_name", "")

        # Create normalized identifier
        identifier = f"{artists} - {track_name}"
        normalized_id = normalize_filename(identifier)

        if normalized_id not in downloaded:
            missing.append(track)
        else:
            matched += 1

    return missing, matched


def download_tracks(tracks: List[Dict], songs_dir: Path, batch_size: int = 10):
    """
    Download tracks using spotdl.
    Downloads in batches to avoid overwhelming the system.
    """
    if not tracks:
        print("No tracks to download!")
        return

    print(f"\nStarting download of {len(tracks)} tracks...")
    print(f"Output directory: {songs_dir}")

    # Ensure output directory exists
    songs_dir.mkdir(parents=True, exist_ok=True)

    # Process in batches
    for i in range(0, len(tracks), batch_size):
        batch = tracks[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tracks) + batch_size - 1) // batch_size

        print(f"\n{'='*60}")
        print(f"Batch {batch_num}/{total_batches} ({len(batch)} tracks)")
        print(f"{'='*60}")

        # Create list of Spotify URLs for this batch
        urls = [track.get("external_url") for track in batch if track.get("external_url")]

        if not urls:
            print("No valid URLs in this batch, skipping...")
            continue

        # Show what we're downloading
        for track in batch:
            artists = ", ".join(track.get("artists", []))
            print(f"  - {artists} - {track.get('track_name')}")

        # Run spotdl command
        # Use download subcommand and proper output template
        # Template format: https://spotdl.readthedocs.io/en/latest/usage/
        output_template = str(songs_dir / "{artists} - {title}.{output-ext}")
        cmd = [
            "spotdl",
            "download",
            *urls,
            "--output", output_template
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per batch
            )

            if result.returncode != 0:
                print(f"\n⚠️  Warning: Some tracks in batch {batch_num} may have failed")
                if result.stderr:
                    print(f"Error output: {result.stderr[:500]}")
            else:
                print(f"✓ Batch {batch_num} completed successfully")

        except subprocess.TimeoutExpired:
            print(f"⚠️  Batch {batch_num} timed out, moving to next batch")
        except Exception as e:
            print(f"⚠️  Error downloading batch {batch_num}: {e}")

    print(f"\n{'='*60}")
    print("Download process completed!")
    print(f"{'='*60}")


def main():
    # Setup paths
    base_dir = Path(__file__).parent
    json_path = base_dir / "api" / "data" / "saved_tracks.json"
    songs_dir = base_dir / "songs"

    print("="*60)
    print("Spotify Missing Tracks Downloader")
    print("="*60)

    # Check if spotdl is installed
    try:
        subprocess.run(["spotdl", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: spotdl is not installed or not in PATH")
        print("Install with: pip install spotdl")
        return

    # Load data
    print("\n1. Loading track data...")
    spotify_tracks = get_spotify_tracks(json_path)

    print("\n2. Scanning downloaded tracks...")
    downloaded = get_downloaded_tracks(songs_dir)

    print("\n3. Finding missing tracks...")
    missing, matched = find_missing_tracks(spotify_tracks, downloaded)

    # Print detailed statistics
    print("\n" + "="*60)
    print("DETECTION STATISTICS")
    print("="*60)
    print(f"📊 Total tracks in Spotify library:  {len(spotify_tracks)}")
    print(f"📁 Total MP3 files in songs/:        {len(downloaded)}")
    print(f"✅ Successfully matched tracks:      {matched}")
    print(f"❌ Missing/unmatched tracks:         {len(missing)}")
    if len(spotify_tracks) > 0:
        match_rate = (matched / len(spotify_tracks)) * 100
        print(f"📈 Match rate:                       {match_rate:.1f}%")
    print("="*60)

    if not missing:
        print("\n✓ All tracks are already downloaded!")
        return

    # Ask for confirmation
    print(f"\n📥 Ready to download {len(missing)} missing tracks")
    response = input("Continue? (y/n): ").strip().lower()

    if response != 'y':
        print("Download cancelled.")
        return

    print("\n4. Downloading missing tracks...")
    download_tracks(missing, songs_dir, batch_size=10)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
