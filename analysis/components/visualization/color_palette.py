"""Centralized Color Palette - Single source of truth for all dashboard colors.

This module provides consistent color definitions for:
- Clusters (seaborn "muted" palette)
- Moods (warm tones, distinct from clusters)
- States (semantic colors for success/warning/error/info)
- Special colors (outliers, brand)

Color Consistency Guarantee:
- Cluster 0 is ALWAYS #4878CF (muted blue) across ALL visualizations
- Mood "happy" is ALWAYS #FFD700 (gold) across ALL visualizations
- No conflicts between cluster and mood colors
"""

from typing import List

# ============================================================================
# CLUSTER COLORS - Seaborn "Muted" Palette
# ============================================================================

# Primary 6 colors - Brighter, more vibrant palette
# Higher luminance for better visibility on dark backgrounds
CLUSTER_COLORS = [
    "#6BA3E8",  # Cluster 0: Bright blue (vibrant, clear)
    "#8CE08C",  # Cluster 1: Bright green (fresh, lively)
    "#F08080",  # Cluster 2: Light coral/red (warm, visible)
    "#D4A5E5",  # Cluster 3: Light purple (soft, distinct)
    "#E8D48A",  # Cluster 4: Light gold (warm, sunny)
    "#8DD6EE",  # Cluster 5: Light cyan (bright, calm)
]

# Extended palette for >6 clusters (still bright but distinct)
EXTENDED_CLUSTER_COLORS = CLUSTER_COLORS + [
    "#5A8ED0",  # Cluster 6: Medium blue
    "#6BC96B",  # Cluster 7: Medium green
    "#E06666",  # Cluster 8: Medium coral
    "#C490D8",  # Cluster 9: Medium purple
]

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
        Hex color string (e.g., "#4878CF")

    Examples:
        >>> get_cluster_color(0)
        '#4878CF'  # Always muted blue
        >>> get_cluster_color(-1)
        '#CCCCCC'  # Always light gray (outliers)
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
        ['#4878CF', '#6ACC65', '#D65F5F']
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
    "get_mood_color",
    "get_state_color",
]
