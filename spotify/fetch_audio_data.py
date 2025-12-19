"""
Fetch metadata for all saved/liked songs from Spotify.
"""

import os
import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# OAuth configuration
SCOPE = "user-library-read"
REDIRECT_URI = "http://127.0.0.1:3000/callback"


def create_spotify_client():
    """Create and return authenticated Spotify client."""
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=REDIRECT_URI,
        scope=SCOPE
    ))


def get_all_saved_tracks(sp):
    """Fetch all user's saved tracks with pagination."""
    tracks = []
    offset = 0
    limit = 50

    print("Fetching saved tracks...")

    while True:
        results = sp.current_user_saved_tracks(limit=limit, offset=offset)

        if not results['items']:
            break

        tracks.extend(results['items'])
        offset += limit

        print(f"  Fetched {len(tracks)} tracks...", end='\r')

        if len(results['items']) < limit:
            break

    print(f"\nTotal tracks fetched: {len(tracks)}")
    return tracks


def extract_track_metadata(saved_tracks):
    """Extract relevant metadata from saved tracks."""
    metadata = []

    for item in saved_tracks:
        track = item['track']

        # Handle multiple artists
        artists = [artist['name'] for artist in track['artists']]
        artist_ids = [artist['id'] for artist in track['artists']]

        metadata.append({
            'track_id': track['id'],
            'track_name': track['name'],
            'artists': artists,
            'artist_ids': artist_ids,
            'album_name': track['album']['name'],
            'album_id': track['album']['id'],
            'album_type': track['album']['album_type'],
            'release_date': track['album']['release_date'],
            'duration_ms': track['duration_ms'],
            'duration_min': round(track['duration_ms'] / 60000, 2),
            'popularity': track['popularity'],
            'explicit': track['explicit'],
            'track_number': track['track_number'],
            'disc_number': track['disc_number'],
            'added_at': item['added_at'],
            'preview_url': track.get('preview_url'),
            'external_url': track['external_urls'].get('spotify'),
            'isrc': track.get('external_ids', {}).get('isrc'),
        })

    return metadata


def save_to_json(data, filename='saved_tracks.json'):
    """Save metadata to JSON file."""
    filepath = os.path.join('data', filename)
    os.makedirs('data', exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nData saved to: {filepath}")


def display_summary(metadata):
    """Display summary statistics."""
    print("\n" + "="*60)
    print("SAVED TRACKS SUMMARY")
    print("="*60 + "\n")

    total_tracks = len(metadata)
    total_duration_ms = sum(track['duration_ms'] for track in metadata)
    total_hours = total_duration_ms / (1000 * 60 * 60)

    # Collect unique artists and albums
    all_artists = set()
    for track in metadata:
        all_artists.update(track['artists'])

    albums = set(track['album_name'] for track in metadata)

    # Average popularity
    avg_popularity = sum(track['popularity'] for track in metadata) / total_tracks if total_tracks > 0 else 0

    # Explicit content count
    explicit_count = sum(1 for track in metadata if track['explicit'])

    print(f"Total Tracks: {total_tracks}")
    print(f"Unique Artists: {len(all_artists)}")
    print(f"Unique Albums: {len(albums)}")
    print(f"Total Duration: {total_hours:.2f} hours")
    print(f"Average Popularity: {avg_popularity:.1f}/100")
    print(f"Explicit Tracks: {explicit_count} ({explicit_count/total_tracks*100:.1f}%)")

    # Show some examples
    print("\n" + "-"*60)
    print("Sample Tracks (first 5):")
    print("-"*60 + "\n")

    for track in metadata[:5]:
        artists_str = ", ".join(track['artists'])
        print(f"• {track['track_name']} - {artists_str}")
        print(f"  Album: {track['album_name']} ({track['release_date']})")
        print(f"  Duration: {track['duration_min']} min | Popularity: {track['popularity']}/100")
        print()


def main():
    """Main execution function."""
    # Create authenticated client
    sp = create_spotify_client()

    # Fetch all saved tracks
    saved_tracks = get_all_saved_tracks(sp)

    if not saved_tracks:
        print("No saved tracks found!")
        return

    # Extract metadata
    metadata = extract_track_metadata(saved_tracks)

    # Display summary
    display_summary(metadata)

    # Save to file
    save_to_json(metadata)

    print("\n" + "="*60)
    print("✓ Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
