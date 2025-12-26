"""EDA Explorer Tab Component - Re-exports from eda/ module.

This file is kept for backward compatibility.
Implementation has been split into analysis/components/tabs/eda/ module.
"""

# Re-export from new module location
from analysis.components.tabs.eda import render_eda_explorer
from analysis.components.tabs.eda.utils import group_small_slices, get_pie_colors, OTHER_SLICE_COLOR

__all__ = ["render_eda_explorer", "group_small_slices", "get_pie_colors", "OTHER_SLICE_COLOR"]
