#!/usr/bin/env python3
"""Create a rotating video of the UMAP 3D clustering visualization.

Generates a smooth 360° rotation animation of the 3D cluster scatter plot
and exports it as an MP4 video.

Usage:
    python tools/create_umap_video.py
    python tools/create_umap_video.py --duration 15 --fps 30 --output my_video.mp4
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.components.visualization.color_palette import (
    get_cluster_color,
    OUTLIER_COLOR,
)


def load_data(input_file: str, mode: str = "combined"):
    """Load the analysis data from pickle file.

    Args:
        input_file: Path to the pickle file
        mode: Which mode to load ('combined', 'audio', or 'lyrics')

    Returns:
        DataFrame with UMAP coordinates and cluster labels
    """
    print(f"Loading data from {input_file}...")
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    # Handle nested structure: data[mode]['dataframe']
    if isinstance(data, dict):
        if mode in data and isinstance(data[mode], dict):
            df = data[mode].get("dataframe")
            if df is not None:
                print(f"Loaded {len(df)} tracks from '{mode}' mode")
                return df

        # Try direct access
        df = data.get("df") or data.get("dataframe")
        if df is not None:
            print(f"Loaded {len(df)} tracks")
            return df

        # Try first mode that has a dataframe
        for m in ["combined", "audio", "lyrics"]:
            if m in data and isinstance(data[m], dict):
                df = data[m].get("dataframe")
                if df is not None:
                    print(f"Loaded {len(df)} tracks from '{m}' mode")
                    return df

        raise ValueError(
            f"Could not find DataFrame in pickle file. Keys: {list(data.keys())}"
        )
    else:
        # Assume it's already a DataFrame
        print(f"Loaded {len(data)} tracks")
        return data


def create_umap_video(
    input_file: str = "analysis/outputs/analysis_data.pkl",
    output_file: str = "umap_3d_rotation.mp4",
    duration: float = 10.0,
    fps: int = 30,
    resolution: tuple = (1920, 1080),
    bg_color: str = "#0a0a0a",
    elevation: float = 20.0,
    show_legend: bool = True,
    show_title: bool = True,
    marker_size: float = 15.0,
    bitrate: int = 3000,
    zoom: float = 1.0,
):
    """Create rotating video of UMAP 3D clusters.

    Args:
        input_file: Path to analysis_data.pkl
        output_file: Output video filename
        duration: Video duration in seconds
        fps: Frames per second
        resolution: Video resolution (width, height)
        bg_color: Background color (hex)
        elevation: Camera elevation angle
        show_legend: Whether to show cluster legend
        marker_size: Size of scatter points
    """
    # Load data
    df = load_data(input_file)

    # Check for UMAP columns
    umap_cols = ["umap_x", "umap_y", "umap_z"]
    if not all(col in df.columns for col in umap_cols):
        raise ValueError(f"Missing UMAP columns. Found: {df.columns.tolist()}")

    # Determine cluster column
    cluster_col = "cluster" if "cluster" in df.columns else "label"
    if cluster_col not in df.columns:
        raise ValueError("No cluster/label column found")

    # Determine if light or dark theme
    is_light_bg = bg_color.lower() in ("#ffffff", "#fff", "white", "#f5f5f5", "#fafafa")
    text_color = "#1a1a1a" if is_light_bg else "white"
    legend_bg = "#ffffff" if is_light_bg else "#1a1a1a"
    legend_edge = "#cccccc" if is_light_bg else "#333333"

    # Set up the figure
    dpi = 100
    fig = plt.figure(figsize=(resolution[0] / dpi, resolution[1] / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Background
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(bg_color)
    ax.yaxis.pane.set_edgecolor(bg_color)
    ax.zaxis.pane.set_edgecolor(bg_color)

    # Hide axes
    ax.set_axis_off()

    # Get unique clusters and sort
    clusters = sorted(df[cluster_col].unique())

    # Plot each cluster
    scatter_handles = []
    for cluster_id in clusters:
        cluster_df = df[df[cluster_col] == cluster_id]

        if cluster_id == -1:
            color = OUTLIER_COLOR
            alpha = 0.3
            size = marker_size * 0.6
            label = f"Outliers ({len(cluster_df)})"
        else:
            color = get_cluster_color(cluster_id)
            alpha = 0.85
            size = marker_size
            label = f"Cluster {cluster_id} ({len(cluster_df)})"

        scatter = ax.scatter(
            cluster_df["umap_x"],
            cluster_df["umap_y"],
            cluster_df["umap_z"],
            c=color,
            s=size,
            alpha=alpha,
            label=label,
            edgecolors="none",
        )
        scatter_handles.append(scatter)

    # Add legend
    if show_legend:
        ax.legend(
            loc="lower center",
            fontsize=8,
            framealpha=0.85,
            facecolor=legend_bg,
            edgecolor=legend_edge,
            labelcolor=text_color,
            markerscale=1.5,
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,  # Spread horizontally
        )

    # Title (optional)
    if show_title:
        ax.set_title(
            "Spotify Library Clusters (UMAP 3D)",
            fontsize=18,
            fontweight="bold",
            color=text_color,
            pad=20,
        )

    # Remove all margins to fill the frame
    plt.subplots_adjust(left=-0.1, right=1.1, top=1.1, bottom=-0.1)

    # Zoom: adjust camera distance (default dist=10, lower=closer)
    ax.dist = 10 / zoom

    # Calculate frames
    total_frames = int(duration * fps)

    # Animation function
    def animate(frame):
        # Rotate 360 degrees over the duration
        angle = (frame / total_frames) * 360
        ax.view_init(elev=elevation, azim=angle)
        ax.dist = 10 / zoom  # Maintain zoom during animation
        return scatter_handles

    print(f"Creating animation: {duration}s at {fps}fps = {total_frames} frames")
    print(f"Resolution: {resolution[0]}x{resolution[1]}")

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=total_frames,
        interval=1000 / fps,
        blit=False,
    )

    # Save video
    print(f"Saving video to {output_file}...")
    print("(This may take a few minutes)")

    # Try different writers
    try:
        # Optimized settings for X/Twitter upload
        # H.264 codec, AAC audio, optimized bitrate
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=bitrate,
            extra_args=[
                "-pix_fmt",
                "yuv420p",  # Required for compatibility
                "-preset",
                "slow",  # Better compression
                "-crf",
                "23",  # Quality (18-28, lower=better)
                "-movflags",
                "+faststart",  # Web optimization
            ],
        )
        anim.save(output_file, writer=writer)
    except Exception as e:
        print(f"FFmpeg not available ({e}), trying Pillow for GIF...")
        gif_file = output_file.replace(".mp4", ".gif")
        writer = animation.PillowWriter(fps=fps)
        anim.save(gif_file, writer=writer)
        print(f"Saved as GIF instead: {gif_file}")
        return gif_file

    plt.close(fig)
    print(f"✓ Video saved: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Create rotating video of UMAP 3D clustering"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="analysis/outputs/analysis_data.pkl",
        help="Input pickle file with analysis data",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="umap_3d_rotation.mp4",
        help="Output video file (default: umap_3d_rotation.mp4)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=10.0,
        help="Video duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second (default: 30)"
    )
    parser.add_argument(
        "--width", type=int, default=1920, help="Video width (default: 1920)"
    )
    parser.add_argument(
        "--height", type=int, default=1080, help="Video height (default: 1080)"
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=20.0,
        help="Camera elevation angle (default: 20)",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=15.0,
        help="Marker size for scatter points (default: 15)",
    )
    parser.add_argument(
        "--no-legend", action="store_true", help="Hide the cluster legend"
    )
    parser.add_argument("--no-title", action="store_true", help="Hide the title")
    parser.add_argument(
        "--bitrate",
        type=int,
        default=3000,
        help="Video bitrate in kbps (default: 3000)",
    )
    parser.add_argument(
        "--twitter",
        "--x",
        action="store_true",
        help="Optimize for X/Twitter: 720p, 10s, compact size",
    )
    parser.add_argument(
        "--square",
        action="store_true",
        help="Use 1:1 square aspect ratio (better for mobile)",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Use white background instead of dark",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Zoom level for the cluster (default: 1.0, try 1.5 for closer)",
    )

    args = parser.parse_args()

    # Apply X/Twitter preset
    width, height = args.width, args.height
    duration = args.duration
    bitrate = args.bitrate
    fps = args.fps
    bg_color = "#ffffff" if args.light else "#0a0a0a"

    if args.twitter:
        # Optimized for X/Twitter
        width, height = 1280, 720  # 720p
        duration = min(duration, 15.0)  # Keep short for engagement
        bitrate = 2500  # Good quality, small file
        fps = 30
        print("🐦 Using X/Twitter optimized settings: 720p, compressed")

    if args.light:
        print("☀️ Using light (white) background")

    if args.square:
        # Square format for mobile
        size = min(width, height)
        width, height = size, size
        print("⬛ Using square (1:1) aspect ratio")

    create_umap_video(
        input_file=args.input,
        output_file=args.output,
        duration=duration,
        fps=fps,
        resolution=(width, height),
        bg_color=bg_color,
        elevation=args.elevation,
        marker_size=args.marker_size,
        show_legend=not args.no_legend,
        show_title=not args.no_title,
        bitrate=bitrate,
        zoom=args.zoom,
    )


if __name__ == "__main__":
    main()
