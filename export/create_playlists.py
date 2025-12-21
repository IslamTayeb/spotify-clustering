#!/usr/bin/env python3
"""
Export clustering results to Spotify playlists.
Creates 5 playlists for audio clusters and 5 for lyric clusters.

Usage:
    python export/create_playlists.py [--private] [--prefix "My Music"]
    python export/create_playlists.py --dry-run  # Preview without creating
"""

import argparse
import os
import pickle
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OAuth configuration with playlist modification scope
SCOPE = "user-library-read playlist-modify-public playlist-modify-private"
REDIRECT_URI = "http://127.0.0.1:3000/callback"


def create_spotify_client():
    """Create and return authenticated Spotify client with playlist modification scope."""
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth

    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=os.getenv("SPOTIFY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
            redirect_uri=REDIRECT_URI,
            scope=SCOPE,
        )
    )


def load_clustering_results(data_path):
    """Load clustering results from pickle file."""
    with open(data_path, "rb") as f:
        return pickle.load(f)


def group_tracks_by_cluster(dataframe, cluster_labels):
    """Group track IDs by their cluster label."""
    clusters = defaultdict(list)

    for idx, row in dataframe.iterrows():
        cluster_id = cluster_labels[idx]
        track_id = row["track_id"]
        track_name = row["track_name"]
        artist = row["artist"]

        clusters[cluster_id].append(
            {
                "id": track_id,
                "name": track_name,
                "artist": artist,
                "uri": f"spotify:track:{track_id}",
            }
        )

    return clusters


def get_cluster_description(cluster_stats, cluster_id, mode="audio"):
    """Generate a description for the cluster based on its statistics."""
    if cluster_id not in cluster_stats:
        return f"Cluster {cluster_id}"

    stats = cluster_stats[cluster_id]

    if mode == "audio":
        # Extract top genre from top_3_genres list
        top_3_genres = stats.get("top_3_genres", [])
        if top_3_genres:
            top_genre = top_3_genres[0][0]  # Get genre name from first tuple
        else:
            top_genre = "Mixed"

        # Find dominant mood from mood_distribution
        mood_dist = stats.get("mood_distribution", {})
        if mood_dist:
            dominant_mood = max(mood_dist.items(), key=lambda x: x[1])[0]
            # Clean up mood name (remove "mood_" prefix)
            dominant_mood = dominant_mood.replace("mood_", "").capitalize()
        else:
            dominant_mood = "Mixed"

        return f"{top_genre} - {dominant_mood}"
    else:
        # For lyric clusters, use language distribution if available
        lang_dist = stats.get("language_distribution", {})
        if lang_dist:
            top_lang = max(lang_dist.items(), key=lambda x: x[1])[0]
            if top_lang == "unknown":
                return f"Style {cluster_id}"
            else:
                return f"Style {cluster_id} ({top_lang.upper()})"
        return f"Style {cluster_id}"


def create_playlist(sp, user_id, name, description, public=True):
    """Create a new Spotify playlist."""
    playlist = sp.user_playlist_create(
        user=user_id, name=name, public=public, description=description
    )
    return playlist


def add_tracks_to_playlist(sp, playlist_id, track_uris, batch_size=100):
    """Add tracks to a playlist in batches (Spotify API limit: 100 tracks per request)."""
    for i in range(0, len(track_uris), batch_size):
        batch = track_uris[i : i + batch_size]
        sp.playlist_add_items(playlist_id, batch)


def preview_cluster_playlists(clusters, cluster_stats, mode="audio", prefix=""):
    """Preview playlists that would be created for each cluster."""
    print(f"\n{'=' * 70}")
    print(f"{mode.upper()} CLUSTER PLAYLISTS".center(70))
    print(f"{'=' * 70}\n")

    preview_data = []

    for cluster_id in sorted(clusters.keys()):
        tracks = clusters[cluster_id]
        track_count = len(tracks)

        # Generate playlist name
        cluster_desc = get_cluster_description(cluster_stats, cluster_id, mode)
        playlist_name = f"{prefix}{mode.capitalize()} Cluster {cluster_id}: {cluster_desc}"

        print(f"📁 {playlist_name}")
        print(f"   Tracks: {track_count}")

        # Show sample tracks (first 5)
        print(f"   Sample tracks:")
        for i, track in enumerate(tracks[:5]):
            print(f"     {i+1}. {track['artist']} - {track['name']}")

        if track_count > 5:
            print(f"     ... and {track_count - 5} more")

        print()

        preview_data.append({
            "name": playlist_name,
            "track_count": track_count,
        })

    return preview_data


def export_clusters_to_playlists(
    sp, clusters, cluster_stats, mode="audio", prefix="", public=True
):
    """Create playlists for each cluster and add tracks."""
    user_id = sp.current_user()["id"]
    print(f"\n{'=' * 60}")
    print(f"Creating {mode.upper()} cluster playlists for user: {user_id}")
    print(f"{'=' * 60}\n")

    created_playlists = []

    for cluster_id in sorted(clusters.keys()):
        tracks = clusters[cluster_id]
        track_count = len(tracks)

        # Generate playlist name and description
        cluster_desc = get_cluster_description(cluster_stats, cluster_id, mode)
        playlist_name = f"{prefix}{mode.capitalize()} Cluster {cluster_id}: {cluster_desc}"
        playlist_description = (
            f"Auto-generated from {mode} clustering analysis. "
            f"Contains {track_count} tracks with similar {mode} characteristics."
        )

        print(f"Creating playlist: {playlist_name}")
        print(f"  Tracks: {track_count}")
        print(f"  Description: {cluster_desc}")

        # Create the playlist
        playlist = create_playlist(
            sp, user_id, playlist_name, playlist_description, public
        )
        playlist_id = playlist["id"]
        playlist_url = playlist["external_urls"]["spotify"]

        # Add tracks to the playlist
        track_uris = [track["uri"] for track in tracks]
        add_tracks_to_playlist(sp, playlist_id, track_uris)

        created_playlists.append(
            {
                "name": playlist_name,
                "id": playlist_id,
                "url": playlist_url,
                "track_count": track_count,
            }
        )

        print(f"  ✓ Created: {playlist_url}\n")

    return created_playlists


def main():
    parser = argparse.ArgumentParser(
        description="Export clustering results to Spotify playlists"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview playlists without creating them",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private playlists instead of public",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for playlist names (e.g., 'My Music - ')",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="analysis/outputs/analysis_data.pkl",
        help="Path to analysis data pickle file",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Only create audio cluster playlists",
    )
    parser.add_argument(
        "--lyrics-only",
        action="store_true",
        help="Only create lyric cluster playlists",
    )

    args = parser.parse_args()

    # Load clustering results
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please run the analysis pipeline first: python run_analysis.py")
        return

    print("Loading clustering results...")
    data = load_clustering_results(data_path)

    prefix = args.prefix + " - " if args.prefix else ""
    all_playlists = []

    # DRY RUN MODE - Preview only
    if args.dry_run:
        # Preview audio clusters
        if not args.lyrics_only:
            audio_data = data["audio"]
            audio_clusters = group_tracks_by_cluster(
                audio_data["dataframe"], audio_data["cluster_labels"]
            )
            audio_playlists = preview_cluster_playlists(
                audio_clusters, audio_data["cluster_stats"], mode="audio", prefix=prefix
            )
            all_playlists.extend(audio_playlists)

        # Preview lyric clusters
        if not args.audio_only:
            lyrics_data = data["lyrics"]
            lyric_clusters = group_tracks_by_cluster(
                lyrics_data["dataframe"], lyrics_data["cluster_labels"]
            )
            lyric_playlists = preview_cluster_playlists(
                lyric_clusters, lyrics_data["cluster_stats"], mode="lyrics", prefix=prefix
            )
            all_playlists.extend(lyric_playlists)

        # Summary
        total_audio = sum(pl["track_count"] for pl in all_playlists if "Audio" in pl["name"])
        total_lyrics = sum(pl["track_count"] for pl in all_playlists if "Lyrics" in pl["name"])

        print(f"{'=' * 70}")
        print("SUMMARY".center(70))
        print(f"{'=' * 70}\n")

        if not args.lyrics_only:
            audio_count = sum(1 for pl in all_playlists if "Audio" in pl["name"])
            print(f"  Audio clusters:  {audio_count} playlists, {total_audio} total tracks")
        if not args.audio_only:
            lyrics_count = sum(1 for pl in all_playlists if "Lyrics" in pl["name"])
            print(f"  Lyric clusters:  {lyrics_count} playlists, {total_lyrics} total tracks")

        print(f"  Total playlists: {len(all_playlists)}")
        print(f"\n  To create these playlists in Spotify, run:")
        print(f"    python export/create_playlists.py")
        if args.prefix:
            print(f"    (with --prefix \"{args.prefix}\")")
        print()
        return

    # ACTUAL CREATION MODE
    # Create Spotify client
    print("Authenticating with Spotify...")
    sp = create_spotify_client()

    public = not args.private

    # Export audio clusters
    if not args.lyrics_only:
        audio_data = data["audio"]
        audio_clusters = group_tracks_by_cluster(
            audio_data["dataframe"], audio_data["cluster_labels"]
        )
        audio_playlists = export_clusters_to_playlists(
            sp,
            audio_clusters,
            audio_data["cluster_stats"],
            mode="audio",
            prefix=prefix,
            public=public,
        )
        all_playlists.extend(audio_playlists)

    # Export lyric clusters
    if not args.audio_only:
        lyrics_data = data["lyrics"]
        lyric_clusters = group_tracks_by_cluster(
            lyrics_data["dataframe"], lyrics_data["cluster_labels"]
        )
        lyric_playlists = export_clusters_to_playlists(
            sp,
            lyric_clusters,
            lyrics_data["cluster_stats"],
            mode="lyrics",
            prefix=prefix,
            public=public,
        )
        all_playlists.extend(lyric_playlists)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total playlists created: {len(all_playlists)}\n")

    for pl in all_playlists:
        print(f"  • {pl['name']}")
        print(f"    Tracks: {pl['track_count']}")
        print(f"    URL: {pl['url']}\n")

    print("✓ All playlists created successfully!")


if __name__ == "__main__":
    main()
