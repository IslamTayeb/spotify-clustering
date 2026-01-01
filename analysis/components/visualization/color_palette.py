"""Centralized Color Palette - Single source of truth for all dashboard colors.

This module provides consistent color definitions for:
- Clusters (12 primary colors + 4 extended = 16 total)
- Genre families (12 colors)
- Moods (warm tones, distinct from clusters)
- States (semantic colors for success/warning/error/info)
- Special colors (outliers, brand)

Color Consistency Guarantee:
- Cluster 0 is ALWAYS #6BA3E8 (bright blue) across ALL visualizations
- All categories (genres, subclusters, etc.) get distinct colors up to 12 items
- No conflicts between cluster and mood colors
"""

from typing import List

# ============================================================================
# CLUSTER COLORS - Seaborn "Muted" Palette
# ============================================================================

# Primary 12 colors - Brighter, more vibrant palette
# Higher luminance for better visibility on dark backgrounds
# Expanded to 12 colors to handle genres, subclusters, and other multi-category visualizations
CLUSTER_COLORS = [
    "#6BA3E8",  # Cluster 0: Bright blue (vibrant, clear)
    "#8CE08C",  # Cluster 1: Bright green (fresh, lively)
    "#F08080",  # Cluster 2: Light coral/red (warm, visible)
    "#D4A5E5",  # Cluster 3: Light purple (soft, distinct)
    "#E8D48A",  # Cluster 4: Light gold (warm, sunny)
    "#8DD6EE",  # Cluster 5: Light cyan (bright, calm)
    "#F5A86B",  # Cluster 6: Bright orange (warm, energetic)
    "#A8E6CF",  # Cluster 7: Mint green (fresh, calm)
    "#E890B8",  # Cluster 8: Bright rose/pink (vibrant)
    "#B8C4E8",  # Cluster 9: Periwinkle/lavender blue (soft)
    "#C8E86B",  # Cluster 10: Lime green (bright, fresh)
    "#E8B8A8",  # Cluster 11: Peach/salmon (warm, soft)
]

# Extended palette for >12 clusters (adds 4 more distinct colors)
EXTENDED_CLUSTER_COLORS = CLUSTER_COLORS + [
    "#7B68EE",  # Cluster 12: Medium slate blue
    "#20B2AA",  # Cluster 13: Light sea green
    "#DDA0DD",  # Cluster 14: Plum
    "#F0E68C",  # Cluster 15: Khaki/pale gold
]

# ============================================================================
# SUBCLUSTER COLORS - Variants of parent cluster colors
# ============================================================================
# Maps (parent_cluster, subcluster_index) to color
# Colors are darker/lighter variants of the parent cluster color

SUBCLUSTER_COLORS = {
    # Hard-Rap (cluster 0, parent: #6BA3E8 bright blue) subclusters
    (0, 0): "#4A7BBF",  # Hard-Rap-Aggro: darker blue
    (0, 1): "#9DC4F5",  # Hard-Rap-Acoustic: lighter blue
    # Mellow (cluster 4, parent: #E8D48A light gold) subclusters
    (4, 0): "#F5E8A8",  # Mellow-Hopecore: lighter gold
    (4, 1): "#C4A85A",  # Mellow-Sadcore: darker gold
}


def get_subcluster_color(parent_cluster: int, subcluster_idx: int) -> str:
    """Get color for a subcluster based on parent cluster.

    Args:
        parent_cluster: Parent cluster index
        subcluster_idx: Subcluster index within parent

    Returns:
        Hex color string. Falls back to regular cluster color if not defined.
    """
    return SUBCLUSTER_COLORS.get(
        (parent_cluster, subcluster_idx),
        get_cluster_color(subcluster_idx)  # Fallback to regular palette
    )


# ============================================================================
# MOOD COLORS - Same luminance as clusters, different hues
# ============================================================================

MOOD_COLORS = {
    "mood_happy": "#E8D86B",     # Bright yellow/gold (cheerful, sunny)
    "mood_sad": "#6B8BE8",       # Bright blue (melancholic, cool)
    "mood_aggressive": "#E86B6B", # Bright red (intense, fiery)
    "mood_relaxed": "#6BE8A8",   # Bright mint/green (calm, soothing)
    "mood_party": "#E86BD8",     # Bright magenta/pink (fun, vibrant)
}

# ============================================================================
# GENRE FAMILY COLORS - Same luminance as clusters, different hues
# ============================================================================

GENRE_FAMILY_COLORS = [
    "#C8A878",  # Bright tan (Hip Hop)
    "#78C8B8",  # Bright seafoam (Electronic)
    "#C8B078",  # Bright olive/khaki (Rock)
    "#E890B8",  # Bright rose (R&B/Soul)
    "#B8A8D8",  # Bright lavender (Pop)
    "#90C890",  # Bright sage (Jazz/Blues)
    "#E8C078",  # Bright amber (Latin)
    "#A8B8C8",  # Bright silver/gray (World/Folk)
    "#D8A8A8",  # Dusty rose (Classical)
    "#A8D8C8",  # Pale teal (Metal)
    "#D8C8A8",  # Cream/tan (Country)
    "#C8A8D8",  # Light orchid (Reggae)
]

# ============================================================================
# STATE COLORS - Semantic colors for UI feedback
# ============================================================================

STATE_COLORS = {
    "success": "#28A745",  # Green (positive outcome)
    "warning": "#FFC107",  # Amber (caution)
    "error": "#DC3545",    # Red (problem)
    "info": "#17A2B8",     # Cyan (informational)
}

# ============================================================================
# SPECIAL COLORS
# ============================================================================

OUTLIER_COLOR = "#CCCCCC"  # Light gray for outlier points (cluster -1)
SPOTIFY_GREEN = "#1DB954"  # Brand color (used in config.toml primary color)

# ============================================================================
# HELPER FUNCTIONS - Consistent color access
# ============================================================================


def get_cluster_color(cluster_idx: int) -> str:
    """Get consistent color for a cluster index.

    Args:
        cluster_idx: Cluster index (0-based). Use -1 for outliers.

    Returns:
        Hex color string (e.g., "#6BA3E8")

    Examples:
        >>> get_cluster_color(0)
        '#6BA3E8'  # Bright blue
        >>> get_cluster_color(-1)
        '#CCCCCC'  # Light gray (outliers)
    """
    if cluster_idx == -1:
        return OUTLIER_COLOR

    if cluster_idx < len(EXTENDED_CLUSTER_COLORS):
        return EXTENDED_CLUSTER_COLORS[cluster_idx]

    # Fallback for very large cluster counts (cycle through base palette)
    return CLUSTER_COLORS[cluster_idx % len(CLUSTER_COLORS)]


def get_all_cluster_colors(n_clusters: int) -> List[str]:
    """Get list of colors for n clusters.

    Args:
        n_clusters: Number of clusters to get colors for

    Returns:
        List of hex color strings

    Examples:
        >>> get_all_cluster_colors(3)
        ['#6BA3E8', '#8CE08C', '#F08080']
    """
    return [get_cluster_color(i) for i in range(n_clusters)]


def get_cluster_colorscale() -> List[List]:
    """Get discrete colorscale for Plotly continuous color mapping.

    Returns:
        List of [position, color] pairs for Plotly colorscale

    Note:
        This is used when Plotly needs a colorscale parameter.
        For discrete colors, use get_cluster_color() directly.
    """
    n = len(CLUSTER_COLORS)
    return [[i / (n - 1), color] for i, color in enumerate(CLUSTER_COLORS)]


def get_mood_color(mood_name: str) -> str:
    """Get color for a specific mood.

    Args:
        mood_name: Mood name (case-insensitive). Examples: "happy", "sad", "party"

    Returns:
        Hex color string. Returns gray (#808080) if mood not found.

    Examples:
        >>> get_mood_color("happy")
        '#FFD700'  # Gold
        >>> get_mood_color("PARTY")
        '#FF6347'  # Tomato (case-insensitive)
    """
    return MOOD_COLORS.get(mood_name.lower(), "#808080")


def get_state_color(state_name: str) -> str:
    """Get color for a UI state.

    Args:
        state_name: State name. One of: "success", "warning", "error", "info"

    Returns:
        Hex color string. Returns gray (#808080) if state not found.

    Examples:
        >>> get_state_color("success")
        '#28A745'  # Green
    """
    return STATE_COLORS.get(state_name.lower(), "#808080")


# ============================================================================
# COLOR PALETTE EXPORTS - For use in visualizations
# ============================================================================

# Export color lists for direct use in plotly color_discrete_sequence
# Example: px.bar(df, color="cluster", color_discrete_sequence=CLUSTER_COLORS)

__all__ = [
    # Color lists
    "CLUSTER_COLORS",
    "EXTENDED_CLUSTER_COLORS",
    "SUBCLUSTER_COLORS",
    "MOOD_COLORS",
    "GENRE_FAMILY_COLORS",
    "STATE_COLORS",
    # Special colors
    "OUTLIER_COLOR",
    "SPOTIFY_GREEN",
    # Helper functions
    "get_cluster_color",
    "get_all_cluster_colors",
    "get_cluster_colorscale",
    "get_subcluster_color",
    "get_mood_color",
    "get_state_color",
]
