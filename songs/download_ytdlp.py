#!/usr/bin/env python3
"""
Download missing songs using yt-dlp directly.
Searches YouTube for each track and downloads as MP3.
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict
import re
import time


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


def get_downloaded_tracks(songs_dir: Path) -> set:
    """Get set of already downloaded track identifiers."""
    downloaded = set()
    if not songs_dir.exists():
        return downloaded

    for file in songs_dir.glob("*.mp3"):
        normalized = normalize_filename(file.stem)
        downloaded.add(normalized)

    return downloaded


def get_spotify_tracks(json_path: Path) -> List[Dict]:
    """Read all tracks from saved_tracks.json."""
    with open(json_path, 'r', encoding='utf-8') as f:
        tracks = json.load(f)
    return tracks


def find_missing_tracks(spotify_tracks: List[Dict], downloaded: set) -> List[Dict]:
    """Find tracks that are in Spotify library but not downloaded."""
    missing = []

    for track in spotify_tracks:
        artists = ", ".join(track.get("artists", []))
        track_name = track.get("track_name", "")
        identifier = f"{artists} - {track_name}"
        normalized_id = normalize_filename(identifier)

        if normalized_id not in downloaded:
            missing.append(track)

    return missing


def sanitize_for_filename(text: str) -> str:
    """Sanitize text to be safe for filenames."""
    # Remove characters that are problematic in filenames
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def download_with_ytdlp(artists: str, track_name: str, songs_dir: Path) -> tuple[bool, str, str]:
    """
    Download a track using yt-dlp.
    Returns: (success, message, filename)
    """
    # Get files before download
    files_before = set(songs_dir.glob("*.mp3"))

    # Create search query
    search_query = f"ytsearch1:{artists} - {track_name} audio"

    # Sanitize for filename
    safe_artists = sanitize_for_filename(artists)
    safe_track = sanitize_for_filename(track_name)
    output_template = str(songs_dir / f"{safe_artists} - {safe_track}.%(ext)s")

    # yt-dlp command
    cmd = [
        "yt-dlp",
        search_query,
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",  # Best quality
        "--output", output_template,
        "--no-playlist",
        "--quiet",  # Suppress most output
        "--no-warnings",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )

        # Find new files
        files_after = set(songs_dir.glob("*.mp3"))
        new_files = files_after - files_before

        if new_files:
            # Successfully created a new file
            new_file = list(new_files)[0]
            return True, "Downloaded", new_file.name
        elif result.returncode == 0:
            # Command succeeded but no new file (might already exist)
            return True, "Already exists or no file created", "[No new file]"
        else:
            # Failed
            error = result.stderr if result.stderr else result.stdout
            return False, error[:200], ""

    except subprocess.TimeoutExpired:
        return False, "Download timed out (120s)", ""
    except FileNotFoundError:
        return False, "yt-dlp not found - install with: pip install yt-dlp", ""
    except Exception as e:
        return False, str(e), ""


def main():
    base_dir = Path(__file__).parent
    json_path = base_dir / "api" / "data" / "saved_tracks.json"
    songs_dir = base_dir / "songs"
    failed_file = base_dir / "failed_ytdlp.txt"

    print("="*70)
    print("YT-DLP DOWNLOADER")
    print("="*70)

    # Check if yt-dlp is installed
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("\n❌ yt-dlp is not installed!")
        print("Install with: pip install yt-dlp")
        return
    except Exception as e:
        print(f"\n❌ Error checking yt-dlp: {e}")
        return

    # Load data
    print("\n1. Loading track data...")
    spotify_tracks = get_spotify_tracks(json_path)

    print("2. Scanning downloaded tracks...")
    downloaded = get_downloaded_tracks(songs_dir)

    print("3. Finding missing tracks...")
    missing = find_missing_tracks(spotify_tracks, downloaded)

    if not missing:
        print("\n✓ All tracks are already downloaded!")
        return

    print(f"\nFound {len(missing)} missing tracks")
    print("\nThis will search YouTube and download each track as MP3.")

    response = input(f"\nContinue downloading {len(missing)} tracks? (y/n): ").strip().lower()
    if response != 'y':
        print("Download cancelled.")
        return

    # Track results
    successful = []
    failed = []

    print(f"\n{'='*70}")
    print("DOWNLOADING")
    print(f"{'='*70}\n")

    for i, track in enumerate(missing, 1):
        artists = ", ".join(track.get("artists", []))
        track_name = track.get("track_name", "")

        print(f"[{i}/{len(missing)}] {artists} - {track_name}")

        success, message, filename = download_with_ytdlp(artists, track_name, songs_dir)

        if success and filename and filename != "[No new file]":
            print(f"    ✓ Downloaded: {filename}")
            successful.append(track)
        elif filename == "[No new file]":
            print(f"    ⚠️  {message}")
        else:
            print(f"    ✗ Failed: {message}")
            failed.append((track, message))

        # Brief pause between downloads to avoid rate limiting
        if i < len(missing):
            time.sleep(1)

    # Summary
    print(f"\n{'='*70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Successfully downloaded: {len(successful)}")
    print(f"✗ Failed: {len(failed)}")
    print(f"{'='*70}")

    # Save failed tracks
    if failed:
        print(f"\nSaving failed tracks to {failed_file}...")
        with open(failed_file, 'w', encoding='utf-8') as f:
            f.write("FAILED DOWNLOADS (yt-dlp)\n")
            f.write("="*70 + "\n\n")

            for track, message in failed:
                artists = ", ".join(track.get("artists", []))
                track_name = track.get("track_name", "")
                url = track.get("external_url", "")

                f.write(f"{artists} - {track_name}\n")
                f.write(f"Spotify URL: {url}\n")
                f.write(f"Error: {message}\n")
                f.write("-"*70 + "\n\n")

        print(f"💾 Failed tracks saved to: {failed_file}")

        if len(failed) <= 10:
            print(f"\nFailed tracks ({len(failed)}):")
            for track, _ in failed:
                artists = ", ".join(track.get("artists", []))
                track_name = track.get("track_name", "")
                print(f"  - {artists} - {track_name}")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
