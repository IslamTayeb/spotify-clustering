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

from analysis.components.visualization.umap_3d import compute_umap_embedding, build_hover_text, get_cluster_color, OUTLIER_COLOR
import plotly.graph_objects as go
import pandas as pd

# ============================================================================
# CLUSTER NAME MAPPINGS
# ============================================================================

CLUSTER_NAMES = {
    0: "Hard-Rap",
    1: "Narrative-Rap",
    2: "Jazz-Fusion",
    3: "Rhythm-Game-EDM",
    4: "Mellow"
}

SUBCLUSTER_NAMES = {
    # Cluster 0 (Hard-Rap) subclusters
    (0, 0): "Hard-Rap-Aggro",
    (0, 1): "Hard-Rap-Acoustic",
    # Cluster 4 (Mellow) subclusters
    (4, 0): "Mellow-Hopecore",
    (4, 1): "Mellow-Sadcore"
}

# ============================================================================
# CUSTOM VISUALIZATION WITH NAMED CLUSTERS
# ============================================================================

def create_umap_3d_plot_with_names(df, cluster_names=None, is_subcluster=False, parent_cluster=None):
    """Create 3D UMAP plot with custom cluster names."""
    fig = go.Figure()

    # Get unique labels
    unique_labels = sorted(df['label'].unique())

    # Add trace for each cluster
    for label in unique_labels:
        cluster_points = df[df['label'] == label]
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
                subcluster_name = SUBCLUSTER_NAMES.get((parent_cluster, label), f"Subcluster {label}")
                name = f"{subcluster_name} ({parent_cluster}.{label}) • {len(cluster_points)} songs"
            elif cluster_names and label in cluster_names:
                # For main clusters, include the index
                name = f"{cluster_names[label]} ({label}) • {len(cluster_points)} songs"
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

# ============================================================================
# TRULY FREE FOREVER OPTIONS (No trials, no limits, no BS)
# ============================================================================

def export_for_bearblog(mode="combined", input_file="analysis/outputs/analysis_data.pkl", output_dir=None):
    """The simplest solution: Netlify Drop - 100% free forever for static sites."""

    print("\n🎯 SIMPLE SOLUTION FOR BEAR BLOG")
    print("="*50)

    # Load and prepare data
    print(f"Loading data...")
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    df = data[mode]['dataframe'].copy()
    if 'label' not in df.columns and 'cluster' in df.columns:
        df['label'] = df['cluster']

    # Compute UMAP if needed
    if 'x' not in df.columns:
        print("Computing 3D coordinates...")
        if 'pca_features' in data[mode]:
            features = data[mode]['pca_features']
        else:
            exclude_cols = ['track_name', 'artist', 'cluster', 'label', 'x', 'y', 'z',
                          'preview_url', 'track_id', 'album_id', 'artist_id',
                          'genre', 'key', 'scale']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            features = df[feature_cols].values

        coords = compute_umap_embedding(features)
        df['x'] = coords[:, 0]
        df['y'] = coords[:, 1]
        df['z'] = coords[:, 2]

    # Create visualization with custom cluster names
    fig = create_umap_3d_plot_with_names(df, cluster_names=CLUSTER_NAMES)

    # Update layout for web
    fig.update_layout(
        title='',  # No title - cleaner for embedding
        height=700,
        autosize=True,
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
        plot_bgcolor="rgba(0,0,0,0)",   # Transparent plot background
        margin=dict(l=0, r=0, t=30, b=0),  # More top margin for legend
        scene=dict(
            xaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False
            ),
            yaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False
            ),
            zaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False
            ),
            bgcolor="rgba(0,0,0,0)",  # Transparent background
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Default camera position
            )
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
            tracegroupgap=5  # More spacing between legend items
        )
    )

    # Use Plotly's built-in HTML generation (handles JSON serialization properly)
    html = fig.to_html(
        include_plotlyjs='cdn',
        config={'displayModeBar': True, 'displaylogo': False, 'responsive': True}
    )

    # Determine output directory
    if output_dir is None:
        output_dir = f"export/visualizations/{mode}"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "index.html")
    with open(output_file, 'w') as f:
        f.write(html)

    file_size = len(html) / 1024
    print(f"\n✅ Created {output_file} ({file_size:.1f} KB)")

    print("\n" + "="*50)
    print("FREE FOREVER HOSTING OPTIONS:")
    print("="*50)

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
    print("-"*50)
    print('<iframe src="YOUR_NETLIFY_URL_HERE"')
    print('        width="100%"')
    print('        height="600"')
    print('        frameborder="0">')
    print('</iframe>')

    print(f"\n💡 That's it! Just drag the '{output_dir}' folder to Netlify Drop!")

    return output_file


def export_combined_and_subclusters(input_file="analysis/outputs/analysis_data.pkl"):
    """Export combined mode and any saved subclusters - the most common use case."""

    print("\n🎯 EXPORTING COMBINED + SUBCLUSTERS")
    print("="*50)

    exported_paths = []

    # Export combined mode
    print("\n📊 Exporting combined clustering...")
    output_path = export_for_bearblog("combined", input_file)
    exported_paths.append(output_path)

    # Export saved subclusters
    print("\n🔍 Looking for saved subclusters...")
    subcluster_paths = export_saved_subclusters()
    exported_paths.extend(subcluster_paths)

    print("\n" + "="*50)
    print(f"✅ EXPORT COMPLETE")
    print("="*50)
    print(f"\n📁 Created {len(exported_paths)} visualizations:")
    print(f"   • Combined clustering: export/visualizations/combined/")
    if len(subcluster_paths) > 0:
        print(f"   • {len(subcluster_paths)} subclusters in: export/visualizations/subclusters/")
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
        cluster_name = filename.replace('.pkl', '')

        # Load subcluster data
        with open(pkl_file, 'rb') as f:
            subcluster_data = pickle.load(f)

        # Get parent cluster info
        parent_cluster = subcluster_data.get('parent_cluster', None)

        # Create a more descriptive folder name
        if parent_cluster is not None:
            parent_name = CLUSTER_NAMES.get(parent_cluster, f"Cluster{parent_cluster}")
            # Create a cleaner folder name
            folder_name = f"{parent_name}_subclusters"
            print(f"\n📊 Exporting subclusters of {parent_name} (Cluster {parent_cluster})")
        else:
            folder_name = cluster_name
            print(f"\n📊 Exporting subcluster: {cluster_name}")

        # Prepare dataframe - subclusters use 'subcluster_df' key
        df = subcluster_data['subcluster_df'].copy()

        # Add cluster labels from subcluster_labels
        if 'subcluster_labels' in subcluster_data:
            df['label'] = subcluster_data['subcluster_labels']
        elif 'label' not in df.columns and 'cluster' in df.columns:
            df['label'] = df['cluster']

        # Use existing UMAP coords if available, otherwise compute
        if 'x' not in df.columns:
            if 'umap_coords' in subcluster_data:
                coords = subcluster_data['umap_coords']
                df['x'] = coords[:, 0]
                df['y'] = coords[:, 1]
                df['z'] = coords[:, 2]
            else:
                print("  Computing 3D coordinates...")
                if 'pca_features_subset' in subcluster_data:
                    features = subcluster_data['pca_features_subset']
                elif 'pca_features' in subcluster_data:
                    features = subcluster_data['pca_features']
                else:
                    exclude_cols = ['track_name', 'artist', 'cluster', 'label', 'x', 'y', 'z',
                                  'preview_url', 'track_id', 'album_id', 'artist_id',
                                  'genre', 'key', 'scale']
                    feature_cols = [col for col in df.columns if col not in exclude_cols]
                    features = df[feature_cols].values

                coords = compute_umap_embedding(features)
                df['x'] = coords[:, 0]
                df['y'] = coords[:, 1]
                df['z'] = coords[:, 2]

        # Create visualization with subcluster names
        fig = create_umap_3d_plot_with_names(df, is_subcluster=True, parent_cluster=parent_cluster)

        # Update layout
        fig.update_layout(
            title='',  # No title
            height=700,
            autosize=True,
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
            plot_bgcolor="rgba(0,0,0,0)",   # Transparent plot background
            margin=dict(l=0, r=0, t=10, b=0),
            scene=dict(
                xaxis=dict(
                    visible=False,
                    showbackground=False,
                    showgrid=False,
                    showline=False,
                    showticklabels=False
                ),
                yaxis=dict(
                    visible=False,
                    showbackground=False,
                    showgrid=False,
                    showline=False,
                    showticklabels=False
                ),
                zaxis=dict(
                    visible=False,
                    showbackground=False,
                    showgrid=False,
                    showline=False,
                    showticklabels=False
                ),
                bgcolor="rgba(0,0,0,0)",  # Transparent background
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)  # Default camera position
                )
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
                tracegroupgap=5
            )
        )

        # Generate HTML
        html = fig.to_html(
            include_plotlyjs='cdn',
            config={'displayModeBar': True, 'displaylogo': False, 'responsive': True}
        )

        # Save to appropriate directory
        output_dir = f"export/visualizations/subclusters/{folder_name}"
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, "index.html")
        with open(output_file, 'w') as f:
            f.write(html)

        file_size = len(html) / 1024
        print(f"  ✅ Created {output_file} ({file_size:.1f} KB)")
        exported_paths.append(output_file)

    return exported_paths


def export_all_modes(input_file="analysis/outputs/analysis_data.pkl", include_subclusters=False):
    """Export visualizations for all available modes and optionally subclusters."""

    print("\n🚀 EXPORTING ALL MODES")
    print("="*50)

    # Load data to see what's available
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    exported_paths = []

    # Export main modes
    for mode in ['audio', 'lyrics', 'combined']:
        if mode in data:
            print(f"\n📊 Exporting {mode} mode...")
            output_path = export_for_bearblog(mode, input_file)
            exported_paths.append(output_path)

    # Export saved subclusters if requested
    subcluster_paths = []
    if include_subclusters:
        print("\n🔍 Looking for saved subclusters...")
        subcluster_paths = export_saved_subclusters()
        exported_paths.extend(subcluster_paths)

    print("\n" + "="*50)
    print(f"✅ EXPORT COMPLETE - Created {len(exported_paths)} visualizations")
    print("="*50)

    print("\n📁 Created folders:")
    print("   • export/visualizations/audio/")
    print("   • export/visualizations/lyrics/")
    print("   • export/visualizations/combined/")
    if len(subcluster_paths) > 0:
        print(f"   • export/visualizations/subclusters/ ({len(subcluster_paths)} subclusters)")
        for path in subcluster_paths:
            folder_name = os.path.dirname(path).split('/')[-1]
            print(f"       - {folder_name}/")

    print("\n💡 Each folder can be uploaded separately to Netlify for different URLs!")
    print("   Or upload the entire 'visualizations' folder for one site with subpaths.")

    return exported_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple export for Bear Blog")
    parser.add_argument("--mode", choices=["combined", "audio", "lyrics"],
                       help="Which clustering mode to export (default: combined + subclusters)")
    parser.add_argument("--input", default="analysis/outputs/analysis_data.pkl",
                       help="Path to analysis data pickle file")
    parser.add_argument("--all", action="store_true",
                       help="Export ALL modes (audio, lyrics, combined) + subclusters")
    parser.add_argument("--subclusters", action="store_true",
                       help="Export combined + saved subclusters (default behavior)")
    parser.add_argument("--only", action="store_true",
                       help="Only export the specified mode, no subclusters")

    args = parser.parse_args()

    if args.all:
        # Export all three clustering modes + subclusters
        export_all_modes(args.input, include_subclusters=True)
    elif args.mode and args.only:
        # Export only the specified mode
        export_for_bearblog(args.mode, args.input)
    elif args.mode and not args.only:
        # Export specified mode (but warn if not combined)
        if args.mode != "combined":
            print(f"\n⚠️  Note: Exporting only {args.mode} mode. Use --subclusters or no flags for combined + subclusters")
        export_for_bearblog(args.mode, args.input)
    else:
        # Default: Export combined + subclusters
        export_combined_and_subclusters(args.input)

    # Don't auto-open browser
    print("\n📌 To upload: Go to https://app.netlify.com/drop")