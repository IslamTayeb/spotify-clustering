"""EDA utility functions for pie charts and data grouping."""

import pandas as pd

# Gray color for "Other" slice
OTHER_SLICE_COLOR = "rgba(128, 128, 128, 0.6)"


def group_small_slices(
    series: pd.Series, threshold_pct: float = 1.0
) -> tuple[pd.Series, int]:
    """Group slices below threshold percentage into 'Other'.

    Returns:
        tuple: (grouped_series, num_items_in_other)
    """
    total = series.sum()
    if total == 0:
        return series, 0

    pct = series / total * 100

    # Separate large and small slices
    large = series[pct >= threshold_pct].copy()
    small = series[pct < threshold_pct]

    # If there are small slices, combine them
    num_other = len(small)
    if num_other > 0 and small.sum() > 0:
        other_label = f"Other ({num_other})"
        large[other_label] = small.sum()

    return large.sort_values(ascending=False), num_other


def get_pie_colors(names: list, base_colors: list) -> list:
    """Get colors for pie chart, using gray for 'Other' slices."""
    colors = []
    color_idx = 0
    for name in names:
        if str(name).startswith("Other"):
            colors.append(OTHER_SLICE_COLOR)
        else:
            colors.append(base_colors[color_idx % len(base_colors)])
            color_idx += 1
    return colors
