#!/usr/bin/env python3
"""Centralized configuration for music analysis pipeline.

This module provides a single source of truth for all configuration settings
used across the CLI (analysis/run_analysis.py) and Streamlit dashboard
(interactive_interpretability.py).
"""

from typing import Dict, Any

# =============================================================================
# PCA CONFIGURATION
# =============================================================================

# PCA Components for 75% cumulative variance (per mode)
# Determined empirically via tools/find_optimal_pca.py
PCA_COMPONENTS_MAP: Dict[str, int] = {
    "audio": 118,      # 75.01% variance explained
    "lyrics": 162,     # 75.02% variance explained
    "combined": 142,   # 75.04% variance (audio) + 75.04% variance (lyrics)
}


# =============================================================================
# CLUSTERING CONFIGURATION
# =============================================================================

# Default clustering parameters for Hierarchical Agglomerative Clustering (HAC)
DEFAULT_CLUSTERING_PARAMS: Dict[str, Any] = {
    "clustering_algorithm": "hac",
    "n_clusters_hac": 5,
    "linkage_method": "ward",
}

# Default UMAP parameters for dimensionality reduction to 3D
DEFAULT_UMAP_PARAMS: Dict[str, Any] = {
    "umap_n_neighbors": 20,
    "umap_min_dist": 0.2,
    "umap_n_components": 3,
}


# =============================================================================
# INTERPRETABLE FEATURE SCALES
# =============================================================================

# Theme scale for interpretable lyric features
# Maps lyric themes to ordinal values (energetic/positive → heavy/dark)
# 10 themes evenly spaced across [0, 1] (spacing = 1/9 ≈ 0.111)
THEME_SCALE: Dict[str, float] = {
    "party": 1.0,
    "flex": 0.889,
    "love": 0.778,
    "social": 0.667,
    "spirituality": 0.556,
    "introspection": 0.444,
    "street": 0.333,
    "heartbreak": 0.222,
    "struggle": 0.111,
    "other": 0.0,
    "none": 0.5,  # Centered for instrumental tracks (equidistant from all categories)
}

# Language scale for interpretable lyric features
# Maps detected language to ordinal encoding
# Language family buckets - grouped by musical tradition similarity
# 8 families evenly spaced across [0, 1] (spacing = 1/7 ≈ 0.143)
LANGUAGE_SCALE: Dict[str, float] = {
    # English (standalone - dominant Western pop/hip-hop baseline)
    "english": 1.0,
    # Romance (Latin pop, reggaeton, shared production)
    "spanish": 0.857,
    "portuguese": 0.857,
    "french": 0.857,
    # Germanic (European pop traditions)
    "german": 0.714,
    "swedish": 0.714,
    "norwegian": 0.714,
    # Slavic (Eastern European traditions)
    "russian": 0.571,
    "ukrainian": 0.571,
    "serbian": 0.571,
    "czech": 0.571,
    # Middle Eastern (maqam scales, shared instrumentation)
    "arabic": 0.429,
    "hebrew": 0.429,
    "turkish": 0.429,
    # South Asian
    "punjabi": 0.286,
    # East Asian (K-pop/J-pop/C-pop aesthetics)
    "korean": 0.143,
    "japanese": 0.143,
    "chinese": 0.143,
    "vietnamese": 0.143,
    # African
    "luganda": 0.0,
    # Other/Unknown - centered for instrumental tracks (equidistant from all categories)
    "multilingual": 0.5,
    "unknown": 0.5,
    "none": 0.5,
}


# =============================================================================
# CACHE PATHS
# =============================================================================

# Feature cache file paths
CACHE_PATHS: Dict[str, str] = {
    "audio": "analysis/cache/audio_features.pkl",
    "lyrics_bge": "analysis/cache/lyric_features.pkl",          # Legacy default (bge-m3)
    "lyrics_e5": "analysis/cache/lyric_features_e5.pkl",        # Higher quality (E5)
    "mert": "analysis/cache/mert_embeddings_24khz_30s_cls.pkl",
    "lyric_interpretable": "analysis/cache/lyric_interpretable_features.pkl",
}


# =============================================================================
# CLUSTER NAME MAPPINGS
# =============================================================================

# User-defined cluster names (maps cluster index to human-readable label)
CLUSTER_NAMES: Dict[int, str] = {
    0: "Hard-Rap",
    1: "Narrative-Rap",
    2: "Jazz-Fusion",
    3: "Rhythm-Game-EDM",
    4: "Mellow",
}

# Spotify playlist links for each cluster (add your playlist URLs here)
# Format: cluster_index -> Spotify playlist URL
CLUSTER_PLAYLIST_LINKS: Dict[int, str] = {
    # Example: 0: "https://open.spotify.com/playlist/xxxxx",
    # Fill in your actual playlist URLs after running export_clusters_as_playlists.py
}

# Subcluster names (maps (parent_cluster, subcluster_index) to label)
SUBCLUSTER_NAMES: Dict[tuple, str] = {
    # Cluster 0 (Hard-Rap) subclusters
    (0, 0): "Hard-Rap-Aggro",
    (0, 1): "Hard-Rap-Acoustic",
    # Cluster 4 (Mellow) subclusters
    (4, 0): "Mellow-Hopecore",
    (4, 1): "Mellow-Sadcore",
}

# Spotify playlist links for each subcluster
# Format: (parent_cluster, subcluster_index) -> Spotify playlist URL
SUBCLUSTER_PLAYLIST_LINKS: Dict[tuple, str] = {
    # Example: (0, 0): "https://open.spotify.com/playlist/xxxxx",
    # Fill in your actual playlist URLs after running export_clusters_as_playlists.py
}


def get_cluster_name(cluster_id: int) -> str:
    """Get human-readable cluster name from ID.

    Args:
        cluster_id: Cluster index

    Returns:
        Cluster name if defined, else 'Cluster {id}'
    """
    return CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")


def get_cluster_name_with_link(cluster_id: int, markdown: bool = True) -> str:
    """Get cluster name with Spotify playlist link if available.

    Args:
        cluster_id: Cluster index
        markdown: If True, return markdown link. If False, return plain text with URL.

    Returns:
        Cluster name with link, or just name if no link available
    """
    name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
    link = CLUSTER_PLAYLIST_LINKS.get(cluster_id)

    if link:
        if markdown:
            return f"[{name}]({link})"
        else:
            return f"{name} ({link})"
    return name


def get_subcluster_name(parent_cluster: int, subcluster_id: int) -> str:
    """Get human-readable subcluster name.

    Args:
        parent_cluster: Parent cluster index
        subcluster_id: Subcluster index within parent

    Returns:
        Subcluster name if defined, else 'Subcluster {id}'
    """
    return SUBCLUSTER_NAMES.get((parent_cluster, subcluster_id), f"Subcluster {subcluster_id}")


def get_subcluster_name_with_link(parent_cluster: int, subcluster_id: int, markdown: bool = True) -> str:
    """Get subcluster name with Spotify playlist link if available.

    Args:
        parent_cluster: Parent cluster index
        subcluster_id: Subcluster index within parent
        markdown: If True, return markdown link. If False, return plain text with URL.

    Returns:
        Subcluster name with link, or just name if no link available
    """
    name = SUBCLUSTER_NAMES.get((parent_cluster, subcluster_id), f"Subcluster {subcluster_id}")
    link = SUBCLUSTER_PLAYLIST_LINKS.get((parent_cluster, subcluster_id))

    if link:
        if markdown:
            return f"[{name}]({link})"
        else:
            return f"{name} ({link})"
    return name


def generate_cluster_list_markdown() -> str:
    """Generate a markdown list of all clusters with their playlist links.

    Useful for embedding in blog posts or documentation.

    Returns:
        Markdown formatted list of clusters and subclusters with links
    """
    lines = ["## Clusters\n"]

    for cluster_id, cluster_name in sorted(CLUSTER_NAMES.items()):
        link = CLUSTER_PLAYLIST_LINKS.get(cluster_id, "")
        if link:
            lines.append(f"- **{cluster_name}** (Cluster {cluster_id}) - [Spotify Playlist]({link})")
        else:
            lines.append(f"- **{cluster_name}** (Cluster {cluster_id})")

        # Add subclusters under this cluster
        for (parent, sub_id), sub_name in sorted(SUBCLUSTER_NAMES.items()):
            if parent == cluster_id:
                sub_link = SUBCLUSTER_PLAYLIST_LINKS.get((parent, sub_id), "")
                if sub_link:
                    lines.append(f"  - {sub_name} ({parent}.{sub_id}) - [Spotify Playlist]({sub_link})")
                else:
                    lines.append(f"  - {sub_name} ({parent}.{sub_id})")

    return "\n".join(lines)


def get_lyric_cache_path(backend: str) -> str:
    """Get cache path for lyric embedding backend.

    Args:
        backend: Lyric embedding backend ("bge-m3", "e5", or custom)

    Returns:
        Path to cache file for specified backend
    """
    if backend == "bge-m3":
        return CACHE_PATHS["lyrics_bge"]
    elif backend == "e5":
        return CACHE_PATHS["lyrics_e5"]
    else:
        # Custom backend - generate cache path
        return f"analysis/cache/lyric_features_{backend}.pkl"
