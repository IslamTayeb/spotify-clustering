#!/usr/bin/env python3
"""
Identify and remove duplicate songs from the master index.
Duplicates are detected based on:
1. Same track name + artist (case-insensitive)
2. Normalized track names (removing special chars, parentheses, etc.)
3. Same Spotify track ID (exact duplicates)
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_name(name: str) -> str:
    """Normalize track name for comparison."""
    name = name.lower()
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"\[[^\]]*\]", "", name)
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def find_duplicates(
    master_index_path: str = "../spotify/master_index.json",
) -> Dict[str, List[Dict]]:
    """Find duplicate tracks in the master index."""

    with open(master_index_path, "r") as f:
        master_index = json.load(f)

    tracks = master_index["tracks"]

    track_id_groups = defaultdict(list)
    name_artist_groups = defaultdict(list)
    normalized_groups = defaultdict(list)

    for track in tracks:
        track_id = track["track_id"]
        track_name = track["track_name"]
        artist = track["artist"]

        track_id_groups[track_id].append(track)

        key = f"{track_name.lower()}|{artist.lower()}"
        name_artist_groups[key].append(track)

        normalized_key = f"{normalize_name(track_name)}|{normalize_name(artist)}"
        normalized_groups[normalized_key].append(track)

    duplicates = {
        "exact_id": {k: v for k, v in track_id_groups.items() if len(v) > 1},
        "name_artist": {k: v for k, v in name_artist_groups.items() if len(v) > 1},
        "normalized": {k: v for k, v in normalized_groups.items() if len(v) > 1},
    }

    return duplicates


def select_best_track(duplicates: List[Dict]) -> Tuple[Dict, List[Dict]]:
    """
    Select the best track from duplicates.
    Priority:
    1. Has both MP3 and lyrics
    2. Has MP3 only
    3. Has lyrics only
    4. Neither
    """

    scored = []
    for track in duplicates:
        score = 0
        if track.get("mp3_file"):
            score += 2
        if track.get("lyrics_file"):
            score += 1
        scored.append((score, track))

    scored.sort(key=lambda x: x[0], reverse=True)

    best = scored[0][1]
    to_remove = [item[1] for item in scored[1:]]

    return best, to_remove


def deduplicate_master_index(
    master_index_path: str = "../spotify/master_index.json", dry_run: bool = True
):
    """Remove duplicates from master index. Modifies the file in-place with backup."""

    print("=" * 60)
    print("SPOTIFY TRACK DEDUPLICATION")
    print("=" * 60)

    with open(master_index_path, "r") as f:
        master_index = json.load(f)

    original_count = len(master_index["tracks"])
    print(f"\nOriginal track count: {original_count}")

    duplicates = find_duplicates(master_index_path)

    print("\n--- Duplicate Analysis ---")
    print(f"Exact ID duplicates: {len(duplicates['exact_id'])} groups")
    print(f"Name+Artist duplicates: {len(duplicates['name_artist'])} groups")
    print(f"Normalized duplicates: {len(duplicates['normalized'])} groups")

    if duplicates["exact_id"]:
        print("\n⚠️  EXACT ID DUPLICATES (same Spotify track ID):")
        for track_id, dups in list(duplicates["exact_id"].items())[:5]:
            print(f"\n  Track ID: {track_id}")
            for dup in dups:
                print(f"    - {dup['artist']} - {dup['track_name']}")
                print(
                    f"      MP3: {bool(dup.get('mp3_file'))}, Lyrics: {bool(dup.get('lyrics_file'))}"
                )

    if duplicates["normalized"]:
        print("\n--- NORMALIZED NAME DUPLICATES ---")
        for key, dups in list(duplicates["normalized"].items())[:10]:
            if key not in duplicates["name_artist"]:
                print(f"\n  {key}")
                for dup in dups:
                    print(f"    - {dup['track_name']} by {dup['artist']}")

    tracks_to_keep = {}
    tracks_to_remove = []

    for track_id, dups in duplicates["exact_id"].items():
        best, to_remove = select_best_track(dups)
        tracks_to_keep[track_id] = best
        tracks_to_remove.extend([t["track_id"] for t in to_remove])

    for key, dups in duplicates["normalized"].items():
        if len(dups) > 1:
            track_ids = [d["track_id"] for d in dups]
            if not any(tid in tracks_to_keep for tid in track_ids):
                best, to_remove = select_best_track(dups)
                tracks_to_keep[key] = best
                tracks_to_remove.extend([t["track_id"] for t in to_remove])

    remove_ids = set(tracks_to_remove)
    deduplicated_tracks = [
        track for track in master_index["tracks"] if track["track_id"] not in remove_ids
    ]

    new_count = len(deduplicated_tracks)
    removed_count = original_count - new_count

    print(f"\n--- Deduplication Summary ---")
    print(f"Original tracks: {original_count}")
    print(f"Duplicates to remove: {removed_count}")
    print(f"Final track count: {new_count}")

    if dry_run:
        print("\n🔍 DRY RUN - No changes made")
        print(f"Run with --execute to modify {master_index_path}")
    else:
        from datetime import datetime
        import shutil

        # Create backup
        backup_path = master_index_path.replace(
            ".json", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        shutil.copy2(master_index_path, backup_path)
        print(f"\n📁 Backup created: {backup_path}")

        # Update master index
        new_master_index = master_index.copy()
        new_master_index["tracks"] = deduplicated_tracks

        new_master_index["statistics"] = {
            "total_tracks": new_count,
            "has_mp3": sum(1 for t in deduplicated_tracks if t.get("mp3_file")),
            "has_lyrics": sum(1 for t in deduplicated_tracks if t.get("lyrics_file")),
            "has_both": sum(
                1
                for t in deduplicated_tracks
                if t.get("mp3_file") and t.get("lyrics_file")
            ),
            "has_neither": sum(
                1
                for t in deduplicated_tracks
                if not t.get("mp3_file") and not t.get("lyrics_file")
            ),
        }

        # Write to original file
        with open(master_index_path, "w") as f:
            json.dump(new_master_index, f, indent=2, ensure_ascii=False)

        print(f"\n✓ {master_index_path} updated successfully!")
        print(f"✓ Removed {removed_count} duplicate tracks")
        print(f"\nIf you need to restore, use the backup:")
        print(f"  cp {backup_path} {master_index_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Deduplicate Spotify tracks in master_index.json"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute deduplication (default is dry-run)",
    )
    parser.add_argument(
        "--input",
        default="../spotify/master_index.json",
        help="Master index file to deduplicate",
    )

    args = parser.parse_args()

    deduplicate_master_index(master_index_path=args.input, dry_run=not args.execute)
