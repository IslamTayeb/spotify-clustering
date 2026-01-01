#!/usr/bin/env python3
"""Simple web export for Bear Blog - using only FREE FOREVER services."""

import sys
from pathlib import Path
import pickle
import json
import os

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from analysis.components.visualization.umap_3d import (
    compute_umap_embedding,
    get_cluster_color,
    OUTLIER_COLOR,
)
from analysis.components.visualization.color_palette import get_subcluster_color
from analysis.pipeline.config import (
    CLUSTER_NAMES,
    SUBCLUSTER_NAMES,
    CLUSTER_PLAYLIST_LINKS,
    SUBCLUSTER_PLAYLIST_LINKS,
)


# Simplified hover text function that only shows song name and artist
def build_hover_text(row):
    """Build simplified hover text with just song name and artist."""
    return f"<b>{row['track_name']}</b><br>Artist: {row['artist']}"


import plotly.graph_objects as go
import pandas as pd

# ============================================================================
# COLOR PALETTES FOR OVERLAY
# ============================================================================

# Blues palette for audio clusters
AUDIO_COLORS = [
    "#08519c",  # Dark blue
    "#3182bd",  # Medium blue
    "#6baed6",  # Light blue
    "#9ecae1",  # Lighter blue
    "#c6dbef",  # Very light blue
]

# Greens palette for lyrics clusters
LYRICS_COLORS = [
    "#006d2c",  # Dark green
    "#31a354",  # Medium green
    "#74c476",  # Light green
    "#a1d99b",  # Lighter green
    "#c7e9c0",  # Very light green
]

# ============================================================================
# LOADING BACKDROP FOR IFRAME EMBEDDING
# ============================================================================


def add_loading_backdrop(html: str) -> str:
    """Add a loading backdrop with hard refresh message for iframe embedding.

    Injects CSS and HTML to show a helpful message if the visualization
    fails to load (common with cached/stale iframes).
    """
    backdrop_css = """
<style>
    #loading-backdrop {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(248, 249, 250, 0.95);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    }
    #loading-backdrop .spinner {
        width: 40px;
        height: 40px;
        border: 3px solid #e0e0e0;
        border-top: 3px solid #1DB954;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 16px;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    #loading-backdrop .message {
        color: #555;
        font-size: 14px;
        text-align: center;
        max-width: 280px;
        line-height: 1.5;
    }
    #loading-backdrop .refresh-hint {
        margin-top: 12px;
        padding: 8px 16px;
        background: #f0f0f0;
        border-radius: 4px;
        font-size: 12px;
        color: #777;
    }
</style>
"""
    backdrop_html = """
<div id="loading-backdrop">
    <div class="spinner"></div>
    <div class="message">Loading visualization...</div>
    <div class="refresh-hint">If this doesn't load, try a hard refresh<br>(Ctrl+Shift+R or Cmd+Shift+R)</div>
</div>
<div id="error-message" style="display:none; text-align:center; padding:40px; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
    <div style="font-size:48px; margin-bottom:16px;">⚠️</div>
    <div style="color:#555; font-size:14px; margin-bottom:12px;">Visualization failed to load</div>
    <div style="color:#888; font-size:12px; margin-bottom:16px;">This can happen if Plotly CDN is blocked by your browser or ad blocker.</div>
    <button onclick="location.reload()" style="padding:8px 16px; background:#1DB954; color:white; border:none; border-radius:4px; cursor:pointer; font-size:14px;">
        Try Again
    </button>
</div>
<script>
    // Hide backdrop once Plotly renders
    function hideBackdrop() {
        var backdrop = document.getElementById('loading-backdrop');
        if (backdrop) {
            backdrop.style.display = 'none';
        }
    }
    // Show error message
    function showError() {
        hideBackdrop();
        var errorMsg = document.getElementById('error-message');
        var plotDiv = document.getElementById('plotly-div');
        if (errorMsg && plotDiv && !plotDiv.querySelector('.plotly')) {
            errorMsg.style.display = 'block';
        }
    }
    // Check if Plotly div exists and has content
    function checkPlotlyLoaded() {
        var plotDiv = document.getElementById('plotly-div');
        if (plotDiv && plotDiv.querySelector('.plotly')) {
            hideBackdrop();
            return true;
        }
        return false;
    }
    // Check if Plotly library loaded
    function checkPlotlyLibrary() {
        if (typeof Plotly === 'undefined') {
            console.error('Plotly library failed to load from CDN');
            showError();
            return false;
        }
        return true;
    }
    // Start checking after DOM loads
    var checkCount = 0;
    var maxChecks = 50; // 5 seconds max (100ms * 50)
    function pollForPlotly() {
        checkCount++;
        if (checkPlotlyLoaded()) {
            return; // Success!
        }
        if (checkCount >= maxChecks) {
            // Timeout - check why
            if (!checkPlotlyLibrary()) {
                return; // CDN failed
            }
            showError(); // Unknown error
            return;
        }
        setTimeout(pollForPlotly, 100);
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(pollForPlotly, 300);
        });
    } else {
        setTimeout(pollForPlotly, 300);
    }
</script>
"""
    # Insert backdrop CSS in head and HTML after body tag
    if "<head>" in html:
        html = html.replace("<head>", "<head>" + backdrop_css)
    if "<body>" in html:
        html = html.replace("<body>", "<body>" + backdrop_html)
    return html


# ============================================================================
# CUSTOM VISUALIZATION WITH NAMED CLUSTERS
# ============================================================================


def create_umap_3d_plot_with_names(
    df, cluster_names=None, is_subcluster=False, parent_cluster=None
):
    """Create 3D UMAP plot with custom cluster names."""
    fig = go.Figure()

    # Get unique labels
    unique_labels = sorted(df["label"].unique())

    # Add trace for each cluster
    for label in unique_labels:
        cluster_points = df[df["label"] == label]
        if cluster_points.empty:
            continue

        if label == -1:
            name = f"Outliers ({len(cluster_points)})"
            color_val = OUTLIER_COLOR
            size = 3
            opacity = 0.3
        else:
            # Use custom names if available
            if is_subcluster and parent_cluster is not None:
                # For subclusters, use the subcluster naming with index
                subcluster_name = SUBCLUSTER_NAMES.get(
                    (parent_cluster, label), f"Subcluster {label}"
                )
                name = f"{subcluster_name} ({parent_cluster}.{label}) • {len(cluster_points)} songs"
                # Use subcluster-specific colors (variants of parent color)
                color_val = get_subcluster_color(parent_cluster, label)
            elif cluster_names and label in cluster_names:
                # For main clusters, include the index
                name = f"{cluster_names[label]} ({label}) • {len(cluster_points)} songs"
                color_val = get_cluster_color(label)
            else:
                name = f"Cluster {label} • {len(cluster_points)} songs"
                color_val = get_cluster_color(label)

            size = 4
            opacity = 0.8

        # Build hover text
        hover_texts = cluster_points.apply(build_hover_text, axis=1)

        fig.add_trace(
            go.Scatter3d(
                x=cluster_points["x"],
                y=cluster_points["y"],
                z=cluster_points["z"],
                mode="markers",
                name=name,
                marker=dict(
                    size=size,
                    color=color_val,
                    opacity=opacity,
                ),
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    return fig


def create_overlay_plot(audio_df, lyrics_df):
    """Create overlay plot of audio and lyrics clusters."""
    fig = go.Figure()

    # Plot audio clusters in blues
    unique_audio_labels = sorted(audio_df["label"].unique())
    for label in unique_audio_labels:
        if label == -1:
            continue  # Skip outliers

        cluster_points = audio_df[audio_df["label"] == label]
        if cluster_points.empty:
            continue

        # Use blue palette
        color_val = AUDIO_COLORS[label % len(AUDIO_COLORS)]
        name = f"Audio: {CLUSTER_NAMES.get(label, f'Cluster {label}')} ({len(cluster_points)})"

        # Build hover text
        hover_texts = cluster_points.apply(build_hover_text, axis=1)

        fig.add_trace(
            go.Scatter3d(
                x=cluster_points["x"],
                y=cluster_points["y"],
                z=cluster_points["z"],
                mode="markers",
                name=name,
                marker=dict(
                    size=5,
                    color=color_val,
                    opacity=0.7,
                    symbol="circle",  # Circles for audio
                    line=dict(width=0),
                ),
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                legendgroup="audio",
                legendgrouptitle_text="Audio Clusters",
            )
        )

    # Plot lyrics clusters in greens
    unique_lyrics_labels = sorted(lyrics_df["label"].unique())
    for label in unique_lyrics_labels:
        if label == -1:
            continue  # Skip outliers

        cluster_points = lyrics_df[lyrics_df["label"] == label]
        if cluster_points.empty:
            continue

        # Use green palette
        color_val = LYRICS_COLORS[label % len(LYRICS_COLORS)]
        name = f"Lyrics: {CLUSTER_NAMES.get(label, f'Cluster {label}')} ({len(cluster_points)})"

        # Build hover text
        hover_texts = cluster_points.apply(build_hover_text, axis=1)

        fig.add_trace(
            go.Scatter3d(
                x=cluster_points["x"],
                y=cluster_points["y"],
                z=cluster_points["z"],
                mode="markers",
                name=name,
                marker=dict(
                    size=5,
                    color=color_val,
                    opacity=0.7,
                    symbol="diamond",  # Diamonds for lyrics
                    line=dict(width=0),
                ),
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                legendgroup="lyrics",
                legendgrouptitle_text="Lyrics Clusters",
            )
        )

    return fig


# ============================================================================
# TRULY FREE FOREVER OPTIONS (No trials, no limits, no BS)
# ============================================================================


def export_for_bearblog(
    mode="combined", input_file="analysis/outputs/analysis_data.pkl", output_dir=None
):
    """The simplest solution: Netlify Drop - 100% free forever for static sites."""

    print("\n🎯 SIMPLE SOLUTION FOR BEAR BLOG")
    print("=" * 50)

    # Load and prepare data
    print(f"Loading data...")
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    df = data[mode]["dataframe"].copy()
    if "label" not in df.columns and "cluster" in df.columns:
        df["label"] = df["cluster"]

    # Compute UMAP if needed
    if "x" not in df.columns:
        print("Computing 3D coordinates...")
        if "pca_features" in data[mode]:
            features = data[mode]["pca_features"]
        else:
            exclude_cols = [
                "track_name",
                "artist",
                "cluster",
                "label",
                "x",
                "y",
                "z",
                "preview_url",
                "track_id",
                "album_id",
                "artist_id",
                "genre",
                "key",
                "scale",
            ]
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            features = df[feature_cols].values

        coords = compute_umap_embedding(features)
        df["x"] = coords[:, 0]
        df["y"] = coords[:, 1]
        df["z"] = coords[:, 2]

    # Create visualization with custom cluster names
    fig = create_umap_3d_plot_with_names(df, cluster_names=CLUSTER_NAMES)

    # Update layout for web
    fig.update_layout(
        title="",  # No title - cleaner for embedding
        # No fixed height - let iframe control it
        autosize=True,
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
        margin=dict(l=0, r=0, t=30, b=0),  # More top margin for legend
        scene=dict(
            xaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
            ),
            yaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
            ),
            zaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
            ),
            bgcolor="rgba(0,0,0,0)",  # Transparent background
            camera=dict(
                eye=dict(x=1.25, y=1.25, z=1.25),  # Slightly closer view
                center=dict(x=0, y=0, z=0),  # Center on origin
            ),
            aspectmode="data",  # Maintain data proportions
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
            font=dict(size=12),
            orientation="v",
            tracegroupgap=5,  # More spacing between legend items
        ),
    )

    # Use Plotly's built-in HTML generation (handles JSON serialization properly)
    html = fig.to_html(
        include_plotlyjs="cdn",
        config={
            "displayModeBar": "hover",
            "displaylogo": False,
            "responsive": True,
            "fillFrame": True,  # Fill the entire frame
        },
        div_id="plotly-div",
    )

    # Add loading backdrop with hard refresh message
    html = add_loading_backdrop(html)

    # Determine output directory
    if output_dir is None:
        output_dir = f"export/visualizations/{mode}"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "index.html")
    with open(output_file, "w") as f:
        f.write(html)

    file_size = len(html) / 1024
    print(f"\n✅ Created {output_file} ({file_size:.1f} KB)")

    print("\n" + "=" * 50)
    print("FREE FOREVER HOSTING OPTIONS:")
    print("=" * 50)

    print(f"""
1️⃣  NETLIFY DROP (Recommended - Easiest)
   ✅ FREE FOREVER for static sites
   ✅ No account needed
   ✅ No credit card
   ✅ No limits for static HTML

   Steps:
   1. Go to: https://app.netlify.com/drop
   2. Drag the '{output_dir}' FOLDER into the browser
   3. Get instant URL like: https://amazing-viz-123.netlify.app
   4. Add to Bear Blog: <iframe src="YOUR_URL" width="100%" height="600"></iframe>

2️⃣  GITHUB PAGES (If you have GitHub)
   ✅ FREE FOREVER with GitHub account
   ✅ No limits
   ✅ Custom domains supported

   Steps:
   1. Create new GitHub repo
   2. Upload the {output_dir} folder contents
   3. Settings → Pages → Deploy from main branch
   4. URL: https://yourusername.github.io/repo-name/

3️⃣  VERCEL (Alternative to Netlify)
   ✅ FREE FOREVER for personal use
   ✅ Generous limits
   ✅ Fast global CDN

   Steps:
   1. Go to: https://vercel.com
   2. Sign up with GitHub (free)
   3. Drag and drop the {output_dir} folder
   4. Get URL instantly

⚠️  AVOID THESE (Not truly free):
   ❌ Plotly Chart Studio - Deprecated
   ❌ Plotly Cloud - Requires paid plans for privacy
   ❌ Observable - Limited free tier
   ❌ CodePen - Limits on free tier
""")

    print(f"\n📋 BEAR BLOG EMBED CODE:")
    print("-" * 50)
    print('<iframe src="YOUR_NETLIFY_URL_HERE"')
    print('        width="100%"')
    print('        height="600"')
    print('        frameborder="0">')
    print("</iframe>")

    print(f"\n💡 That's it! Just drag the '{output_dir}' folder to Netlify Drop!")

    return output_file


def export_audio_lyrics_overlay(input_file="analysis/outputs/analysis_data.pkl"):
    """Export overlay visualization of audio and lyrics clusters."""

    print("\n🎨 EXPORTING AUDIO VS LYRICS OVERLAY")
    print("=" * 50)

    # Load data
    print("Loading data...")
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    if "audio" not in data or "lyrics" not in data:
        print("  ⚠️  Need both audio and lyrics data for overlay")
        return None

    # Prepare audio dataframe
    audio_df = data["audio"]["dataframe"].copy()
    if "label" not in audio_df.columns and "cluster" in audio_df.columns:
        audio_df["label"] = audio_df["cluster"]

    # Compute UMAP for audio if needed
    if "x" not in audio_df.columns:
        print("Computing 3D coordinates for audio...")
        if "pca_features" in data["audio"]:
            features = data["audio"]["pca_features"]
        else:
            exclude_cols = [
                "track_name",
                "artist",
                "cluster",
                "label",
                "x",
                "y",
                "z",
                "preview_url",
                "track_id",
                "album_id",
                "artist_id",
                "genre",
                "key",
                "scale",
            ]
            feature_cols = [col for col in audio_df.columns if col not in exclude_cols]
            features = audio_df[feature_cols].values

        coords = compute_umap_embedding(features)
        audio_df["x"] = coords[:, 0]
        audio_df["y"] = coords[:, 1]
        audio_df["z"] = coords[:, 2]

    # Prepare lyrics dataframe
    lyrics_df = data["lyrics"]["dataframe"].copy()
    if "label" not in lyrics_df.columns and "cluster" in lyrics_df.columns:
        lyrics_df["label"] = lyrics_df["cluster"]

    # Compute UMAP for lyrics if needed
    if "x" not in lyrics_df.columns:
        print("Computing 3D coordinates for lyrics...")
        if "pca_features" in data["lyrics"]:
            features = data["lyrics"]["pca_features"]
        else:
            exclude_cols = [
                "track_name",
                "artist",
                "cluster",
                "label",
                "x",
                "y",
                "z",
                "preview_url",
                "track_id",
                "album_id",
                "artist_id",
                "genre",
                "key",
                "scale",
            ]
            feature_cols = [col for col in lyrics_df.columns if col not in exclude_cols]
            features = lyrics_df[feature_cols].values

        coords = compute_umap_embedding(features)
        lyrics_df["x"] = coords[:, 0]
        lyrics_df["y"] = coords[:, 1]
        lyrics_df["z"] = coords[:, 2]

    # Create overlay visualization
    print("Creating overlay visualization...")
    fig = create_overlay_plot(audio_df, lyrics_df)

    # Update layout
    fig.update_layout(
        title="",  # No title
        # No fixed height - let iframe control it
        autosize=True,
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
            ),
            yaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
            ),
            zaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
            ),
            bgcolor="rgba(0,0,0,0)",  # Transparent background
            camera=dict(
                eye=dict(x=1.25, y=1.25, z=1.25),  # Slightly closer view
                center=dict(x=0, y=0, z=0),  # Center on origin
            ),
            aspectmode="data",  # Maintain data proportions
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
            font=dict(size=11),
            orientation="v",
            tracegroupgap=10,  # More spacing between groups
            groupclick="toggleitem",  # Allow toggling individual items
        ),
    )

    # Generate HTML
    html = fig.to_html(
        include_plotlyjs="cdn",
        config={
            "displayModeBar": "hover",
            "displaylogo": False,
            "responsive": True,
            "fillFrame": True,
        },
        div_id="plotly-div",
    )

    # Add loading backdrop with hard refresh message
    html = add_loading_backdrop(html)

    # Save to appropriate directory
    output_dir = "export/visualizations/audio-vs-lyrics"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "index.html")
    with open(output_file, "w") as f:
        f.write(html)

    file_size = len(html) / 1024
    print(f"\n✅ Created {output_file} ({file_size:.1f} KB)")

    print("\n📊 Visualization Legend:")
    print("  • Blue circles = Audio clustering")
    print("  • Green diamonds = Lyrics clustering")
    print("  • Toggle clusters on/off by clicking legend items")

    return output_file


def export_combined_and_subclusters(input_file="analysis/outputs/analysis_data.pkl"):
    """Export combined mode and any saved subclusters - the most common use case."""

    print("\n🎯 EXPORTING COMBINED + SUBCLUSTERS")
    print("=" * 50)

    exported_paths = []

    # Export combined mode
    print("\n📊 Exporting combined clustering...")
    output_path = export_for_bearblog("combined", input_file)
    exported_paths.append(output_path)

    # Export saved subclusters
    print("\n🔍 Looking for saved subclusters...")
    subcluster_paths = export_saved_subclusters()
    exported_paths.extend(subcluster_paths)

    print("\n" + "=" * 50)
    print(f"✅ EXPORT COMPLETE")
    print("=" * 50)
    print(f"\n📁 Created {len(exported_paths)} visualizations:")
    print(f"   • Combined clustering: export/visualizations/combined/")
    if len(subcluster_paths) > 0:
        print(
            f"   • {len(subcluster_paths)} subclusters in: export/visualizations/subclusters/"
        )
    print("\n💡 Each folder can be uploaded separately to Netlify for different URLs!")

    return exported_paths


def export_saved_subclusters():
    """Export visualizations for saved subclusters from analysis/outputs/subclusters/."""
    import glob

    subcluster_dir = "analysis/outputs/subclusters"
    subcluster_files = glob.glob(f"{subcluster_dir}/*.pkl")

    exported_paths = []

    if not subcluster_files:
        print("  ⚠️  No saved subclusters found")
        return exported_paths

    for pkl_file in subcluster_files:
        # Extract cluster name from filename
        filename = os.path.basename(pkl_file)
        cluster_name = filename.replace(".pkl", "")

        # Load subcluster data
        with open(pkl_file, "rb") as f:
            subcluster_data = pickle.load(f)

        # Get parent cluster info
        parent_cluster = subcluster_data.get("parent_cluster", None)

        # Create a more descriptive folder name
        if parent_cluster is not None:
            parent_name = CLUSTER_NAMES.get(parent_cluster, f"Cluster{parent_cluster}")
            # Create a cleaner folder name
            folder_name = f"{parent_name}_subclusters"
            print(
                f"\n📊 Exporting subclusters of {parent_name} (Cluster {parent_cluster})"
            )
        else:
            folder_name = cluster_name
            print(f"\n📊 Exporting subcluster: {cluster_name}")

        # Prepare dataframe - subclusters use 'subcluster_df' key
        df = subcluster_data["subcluster_df"].copy()

        # Add cluster labels from subcluster_labels
        if "subcluster_labels" in subcluster_data:
            df["label"] = subcluster_data["subcluster_labels"]
        elif "label" not in df.columns and "cluster" in df.columns:
            df["label"] = df["cluster"]

        # Use existing UMAP coords if available, otherwise compute
        if "x" not in df.columns:
            if "umap_coords" in subcluster_data:
                coords = subcluster_data["umap_coords"]
                df["x"] = coords[:, 0]
                df["y"] = coords[:, 1]
                df["z"] = coords[:, 2]
            else:
                print("  Computing 3D coordinates...")
                if "pca_features_subset" in subcluster_data:
                    features = subcluster_data["pca_features_subset"]
                elif "pca_features" in subcluster_data:
                    features = subcluster_data["pca_features"]
                else:
                    exclude_cols = [
                        "track_name",
                        "artist",
                        "cluster",
                        "label",
                        "x",
                        "y",
                        "z",
                        "preview_url",
                        "track_id",
                        "album_id",
                        "artist_id",
                        "genre",
                        "key",
                        "scale",
                    ]
                    feature_cols = [
                        col for col in df.columns if col not in exclude_cols
                    ]
                    features = df[feature_cols].values

                coords = compute_umap_embedding(features)
                df["x"] = coords[:, 0]
                df["y"] = coords[:, 1]
                df["z"] = coords[:, 2]

        # Create visualization with subcluster names
        fig = create_umap_3d_plot_with_names(
            df, is_subcluster=True, parent_cluster=parent_cluster
        )

        # Update layout
        fig.update_layout(
            title="",  # No title
            # No fixed height - let iframe control it
            autosize=True,
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
            margin=dict(l=0, r=0, t=10, b=0),
            scene=dict(
                xaxis=dict(
                    visible=False,
                    showbackground=False,
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                ),
                yaxis=dict(
                    visible=False,
                    showbackground=False,
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                ),
                zaxis=dict(
                    visible=False,
                    showbackground=False,
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                ),
                bgcolor="rgba(0,0,0,0)",  # Transparent background
                camera=dict(
                    eye=dict(x=1.25, y=1.25, z=1.25),  # Slightly closer view
                    center=dict(x=0, y=0, z=0),  # Center on origin
                ),
                aspectmode="data",  # Maintain data proportions
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1,
                # Add padding inside the legend
                font=dict(size=12),
                # Add margin above legend
                orientation="v",
                # More spacing between items
                tracegroupgap=5,
            ),
        )

        # Generate HTML
        html = fig.to_html(
            include_plotlyjs="cdn",
            config={
                "displayModeBar": "hover",
                "displaylogo": False,
                "responsive": True,
                "fillFrame": True,  # Fill the entire frame
            },
            div_id="plotly-div",
        )

        # Add loading backdrop with hard refresh message
        html = add_loading_backdrop(html)

        # Save to appropriate directory
        output_dir = f"export/visualizations/subclusters/{folder_name}"
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, "index.html")
        with open(output_file, "w") as f:
            f.write(html)

        file_size = len(html) / 1024
        print(f"  ✅ Created {output_file} ({file_size:.1f} KB)")
        exported_paths.append(output_file)

    return exported_paths


# ============================================================================
# KEY ENCODING VISUALIZATION
# ============================================================================


def export_key_encoding_visualization():
    """Export circular key encoding explainer visualization."""
    import numpy as np
    from plotly.subplots import make_subplots

    print("\n🎹 EXPORTING KEY ENCODING VISUALIZATION")
    print("=" * 50)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "<b>1. The Problem</b>",
            "<b>2. The Solution</b>",
            "<b>3. Distance Comparison</b>",
            "<b>4. The Formula</b>",
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Panel 1: Linear number line showing C far from B
    keys = ["C", "", "D", "", "E", "F", "", "G", "", "A", "", "B"]
    for i, key in enumerate(keys):
        if key:
            color = "#E74C3C" if key in ["C", "B"] else "#95A5A6"
            fig.add_trace(
                go.Scatter(
                    x=[i],
                    y=[0],
                    mode="markers+text",
                    marker=dict(size=20, color=color),
                    text=[key],
                    textposition="top center",
                    textfont=dict(size=10),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=[-0.5, 11.5],
            y=[0, 0],
            mode="lines",
            line=dict(color="#ccc", width=2),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.add_annotation(
        x=5.5,
        y=-0.4,
        text="C=0, B=11: Distance=11 (wrong!)",
        showarrow=False,
        font=dict(size=9, color="#E74C3C"),
        xref="x",
        yref="y",
    )

    # Panel 2: Circle
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            line=dict(color="#ccc", width=1, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    for i, key in enumerate(["C", "D", "E", "F", "G", "A", "B"]):
        pitch = [0, 2, 4, 5, 7, 9, 11][i]
        angle = 2 * np.pi * pitch / 12
        x, y = np.cos(angle), np.sin(angle)
        color = "#27AE60" if key in ["C", "B"] else "#95A5A6"
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(size=18, color=color),
                text=[key],
                textposition="top center" if y >= 0 else "bottom center",
                textfont=dict(size=10),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )

    fig.add_annotation(
        x=0,
        y=-1.3,
        text="C and B are neighbors!",
        showarrow=False,
        font=dict(size=9, color="#27AE60"),
        xref="x2",
        yref="y2",
    )

    # Panel 3: Bar chart comparison
    pairs = ["C→D", "C→B", "B→C"]
    linear_d = [2, 11, 11]
    circular_d = [2, 1, 1]

    fig.add_trace(
        go.Bar(
            x=pairs,
            y=linear_d,
            name="Linear",
            marker_color="#E74C3C",
            text=linear_d,
            textposition="auto",
            showlegend=True,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=pairs,
            y=circular_d,
            name="Circular",
            marker_color="#27AE60",
            text=circular_d,
            textposition="auto",
            showlegend=True,
        ),
        row=2,
        col=1,
    )

    # Panel 4: Formula
    fig.add_annotation(
        x=0.5,
        y=0.7,
        text="<b>key_sin</b> = sin(2π × pitch/12)",
        showarrow=False,
        font=dict(size=12, color="#9B59B6"),
        xref="x4",
        yref="y4",
    )
    fig.add_annotation(
        x=0.5,
        y=0.4,
        text="<b>key_cos</b> = cos(2π × pitch/12)",
        showarrow=False,
        font=dict(size=12, color="#27AE60"),
        xref="x4",
        yref="y4",
    )
    fig.add_annotation(
        x=0.5,
        y=0.1,
        text="<b>key_scale</b> = 0.33 (major) or 0 (minor)",
        showarrow=False,
        font=dict(size=12, color="#3498DB"),
        xref="x4",
        yref="y4",
    )

    # Update axes
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-1.2, 1.2],
        row=1,
        col=1,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=2)
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-1.5, 1.5],
        scaleanchor="x2",
        row=1,
        col=2,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, row=2, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, range=[0, 14], row=2, col=1)
    fig.update_xaxes(
        showgrid=False, zeroline=False, showticklabels=False, range=[0, 1], row=2, col=2
    )
    fig.update_yaxes(
        showgrid=False, zeroline=False, showticklabels=False, range=[0, 1], row=2, col=2
    )

    fig.update_layout(
        height=500,
        width=900,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=60, b=40),
        barmode="group",
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.25),
        title=dict(
            text="<b>Circular Key Encoding Explained</b>", x=0.5, font=dict(size=18)
        ),
    )

    # Generate and save HTML
    html = fig.to_html(
        include_plotlyjs="cdn",
        config={"displayModeBar": "hover", "displaylogo": False, "responsive": True},
        div_id="plotly-div",
    )
    html = add_loading_backdrop(html)

    output_dir = "export/visualizations/key_encoding"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "index.html")

    with open(output_file, "w") as f:
        f.write(html)

    print(f"  ✅ Created {output_file}")
    return output_file


def export_all_modes(
    input_file="analysis/outputs/analysis_data.pkl", include_subclusters=False
):
    """Export visualizations for all available modes and optionally subclusters."""

    print("\n🚀 EXPORTING ALL MODES")
    print("=" * 50)

    # Load data to see what's available
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    exported_paths = []

    # Export main modes
    for mode in ["audio", "lyrics", "combined"]:
        if mode in data:
            print(f"\n📊 Exporting {mode} mode...")
            output_path = export_for_bearblog(mode, input_file)
            exported_paths.append(output_path)

    # Export audio vs lyrics overlay
    if "audio" in data and "lyrics" in data:
        print("\n🎨 Exporting audio vs lyrics overlay...")
        overlay_path = export_audio_lyrics_overlay(input_file)
        if overlay_path:
            exported_paths.append(overlay_path)

    # Export saved subclusters if requested
    subcluster_paths = []
    if include_subclusters:
        print("\n🔍 Looking for saved subclusters...")
        subcluster_paths = export_saved_subclusters()
        exported_paths.extend(subcluster_paths)

    # Export key encoding visualization
    print("\n🎹 Exporting key encoding explainer...")
    key_path = export_key_encoding_visualization()
    exported_paths.append(key_path)

    print("\n" + "=" * 50)
    print(f"✅ EXPORT COMPLETE - Created {len(exported_paths)} visualizations")
    print("=" * 50)

    print("\n📁 Created folders:")
    print("   • export/visualizations/audio/")
    print("   • export/visualizations/lyrics/")
    print("   • export/visualizations/combined/")
    print("   • export/visualizations/audio-vs-lyrics/")
    print("   • export/visualizations/key_encoding/")
    if len(subcluster_paths) > 0:
        print(
            f"   • export/visualizations/subclusters/ ({len(subcluster_paths)} subclusters)"
        )
        for path in subcluster_paths:
            folder_name = os.path.dirname(path).split("/")[-1]
            print(f"       - {folder_name}/")

    print("\n💡 Each folder can be uploaded separately to Netlify for different URLs!")
    print("   Or upload the entire 'visualizations' folder for one site with subpaths.")

    return exported_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple export for Bear Blog")
    parser.add_argument(
        "--mode",
        choices=["combined", "audio", "lyrics"],
        help="Which clustering mode to export (default: combined + subclusters)",
    )
    parser.add_argument(
        "--input",
        default="analysis/outputs/analysis_data.pkl",
        help="Path to analysis data pickle file",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export ALL modes (audio, lyrics, combined, overlay) + subclusters",
    )
    parser.add_argument(
        "--subclusters",
        action="store_true",
        help="Export combined + saved subclusters (default behavior)",
    )
    parser.add_argument(
        "--only",
        action="store_true",
        help="Only export the specified mode, no subclusters",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Export audio vs lyrics overlay comparison",
    )

    args = parser.parse_args()

    if args.overlay:
        # Export audio vs lyrics overlay
        export_audio_lyrics_overlay(args.input)
    elif args.all:
        # Export all three clustering modes + subclusters
        export_all_modes(args.input, include_subclusters=True)
    elif args.mode and args.only:
        # Export only the specified mode
        export_for_bearblog(args.mode, args.input)
    elif args.mode and not args.only:
        # Export specified mode (but warn if not combined)
        if args.mode != "combined":
            print(
                f"\n⚠️  Note: Exporting only {args.mode} mode. Use --subclusters or no flags for combined + subclusters"
            )
        export_for_bearblog(args.mode, args.input)
    else:
        # Default: Export combined + subclusters
        export_combined_and_subclusters(args.input)

    # Don't auto-open browser
    print("\n📌 To upload: Go to https://app.netlify.com/drop")
