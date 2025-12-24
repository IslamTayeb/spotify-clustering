"""Spotify playlist export functionality.

This module handles exporting clustering results to Spotify playlists.
"""

import os
import streamlit as st
import pandas as pd
from collections import Counter
from typing import Optional


def render_export_controls(df: pd.DataFrame, mode: str) -> None:
    """Render Spotify export controls in sidebar.

    Args:
        df: DataFrame with clustering results
        mode: Clustering mode (audio/lyrics/combined)
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎧 Export to Spotify")

    # Cluster selection
    cluster_ids = sorted(df["cluster"].unique())
    export_option = st.sidebar.radio(
        "Export scope",
        ["All Clusters", "Single Cluster"],
        help="Export all clusters or select a specific one",
    )

    selected_export_cluster = None
    if export_option == "Single Cluster":
        selected_export_cluster = st.sidebar.selectbox(
            "Select cluster to export",
            options=cluster_ids,
            format_func=lambda x: f"Cluster {x} ({len(df[df['cluster'] == x])} songs)",
            key="export_cluster_select",
        )

    # Playlist options
    playlist_prefix = st.sidebar.text_input(
        "Playlist name prefix",
        value="",
        placeholder="e.g., My Music",
        help="Optional prefix for playlist names",
    )

    make_private = st.sidebar.checkbox(
        "Make playlists private",
        value=False,
        help="Create private playlists instead of public",
    )

    # Export button
    if st.sidebar.button("🚀 Export to Spotify", type="primary", use_container_width=True):
        export_to_spotify(
            df=df,
            mode=mode,
            cluster_id=selected_export_cluster,
            prefix=playlist_prefix,
            private=make_private,
        )


def export_to_spotify(
    df: pd.DataFrame,
    mode: str,
    cluster_id: Optional[int] = None,
    prefix: str = "",
    private: bool = False,
) -> None:
    """Export clusters to Spotify playlists.

    Args:
        df: DataFrame with track_id and cluster columns
        mode: Clustering mode
        cluster_id: Single cluster ID to export (None = all clusters)
        prefix: Prefix for playlist names
        private: Whether to make playlists private
    """
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
        from dotenv import load_dotenv
    except ImportError:
        st.error(
            "❌ Missing dependencies. Install with: `pip install spotipy python-dotenv`"
        )
        return

    # Load environment variables
    load_dotenv()

    # Check for credentials
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        st.error(
            "❌ Spotify credentials not found!\\n\\n"
            "Please set `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` in your `.env` file."
        )
        return

    # OAuth configuration
    SCOPE = "user-library-read playlist-modify-public playlist-modify-private"
    REDIRECT_URI = "http://127.0.0.1:3000/callback"

    with st.spinner("🔐 Authenticating with Spotify..."):
        try:
            sp = spotipy.Spotify(
                auth_manager=SpotifyOAuth(
                    client_id=client_id,
                    client_secret=client_secret,
                    redirect_uri=REDIRECT_URI,
                    scope=SCOPE,
                )
            )
            user_id = sp.current_user()["id"]
            st.success(f"✓ Authenticated as: {user_id}")
        except Exception as e:
            st.error(f"❌ Authentication failed: {e}")
            st.info("💡 Try deleting the `.cache` file and re-running.")
            return

    # Group tracks by cluster
    clusters_to_export = {}
    if cluster_id is not None:
        # Single cluster export
        cluster_df = df[df["cluster"] == cluster_id]
        clusters_to_export[cluster_id] = cluster_df
    else:
        # All clusters export
        for cid in sorted(df["cluster"].unique()):
            if cid != -1:  # Skip outliers
                clusters_to_export[cid] = df[df["cluster"] == cid]

    # Build prefix string
    prefix_str = f"{prefix} - " if prefix else ""
    public = not private

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    created_playlists = []

    total_clusters = len(clusters_to_export)
    for i, (cid, cluster_df) in enumerate(clusters_to_export.items()):
        progress = (i + 1) / total_clusters
        progress_bar.progress(progress)
        status_text.text(
            f"Creating playlist for Cluster {cid}... ({i + 1}/{total_clusters})"
        )

        # Generate playlist name and description
        top_genre = "Mixed"
        if "top_genre" in cluster_df.columns:
            genre_counts = cluster_df["top_genre"].value_counts()
            if len(genre_counts) > 0:
                top_genre = genre_counts.index[0]

        dominant_mood = "Mixed"
        mood_cols = [
            "mood_happy",
            "mood_sad",
            "mood_aggressive",
            "mood_relaxed",
            "mood_party",
        ]
        available_moods = [col for col in mood_cols if col in cluster_df.columns]
        if available_moods:
            mood_means = {
                col.replace("mood_", "").capitalize(): cluster_df[col].mean()
                for col in available_moods
            }
            dominant_mood = max(mood_means, key=mood_means.get)

        playlist_name = f"{prefix_str}{mode.capitalize()} Cluster {cid}: {top_genre} - {dominant_mood}"
        playlist_description = (
            f"Auto-generated from {mode} clustering analysis. "
            f"Contains {len(cluster_df)} tracks with similar {mode} characteristics."
        )

        try:
            # Create playlist
            playlist = sp.user_playlist_create(
                user=user_id,
                name=playlist_name,
                public=public,
                description=playlist_description,
            )
            playlist_id = playlist["id"]
            playlist_url = playlist["external_urls"]["spotify"]

            # Add tracks in batches (Spotify limit: 100 per request)
            track_uris = [f"spotify:track:{tid}" for tid in cluster_df["track_id"]]
            batch_size = 100
            for j in range(0, len(track_uris), batch_size):
                batch = track_uris[j : j + batch_size]
                sp.playlist_add_items(playlist_id, batch)

            created_playlists.append(
                {
                    "name": playlist_name,
                    "url": playlist_url,
                    "track_count": len(cluster_df),
                    "cluster_id": cid,
                }
            )

        except Exception as e:
            st.warning(f"⚠️ Failed to create playlist for Cluster {cid}: {e}")

    progress_bar.progress(1.0)
    status_text.empty()

    # Show results
    if created_playlists:
        st.success(f"🎉 Created {len(created_playlists)} playlist(s)!")

        # Show created playlists
        for pl in created_playlists:
            st.markdown(
                f"**{pl['name']}** ({pl['track_count']} tracks)  \\n"
                f"[Open in Spotify]({pl['url']})"
            )
    else:
        st.error("❌ No playlists were created.")
