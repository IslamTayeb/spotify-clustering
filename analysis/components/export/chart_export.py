"""Shared chart export functionality for web hosting.

This module provides reusable functions for exporting any Plotly chart
to standalone HTML files suitable for Netlify/web hosting.

Usage in any visualization module:
    from analysis.components.export.chart_export import render_chart_with_export, render_export_section

    # Render chart with export checkbox
    render_chart_with_export(fig, "chart_id", "Chart Title")

    # At the end of your section, render the export button
    render_export_section()
"""

import os
import streamlit as st
import plotly.graph_objects as go
from typing import Optional


def export_chart_to_html(
    fig: go.Figure,
    filename: str,
    output_dir: str = "export/visualizations"
) -> str:
    """Export a plotly figure to HTML file in its own folder.

    Args:
        fig: Plotly figure to export
        filename: Name for the chart folder
        output_dir: Parent directory to save to

    Returns:
        Path to the exported file

    Creates: output_dir/filename/index.html
    """
    # Create folder for this chart
    chart_dir = os.path.join(output_dir, filename)
    os.makedirs(chart_dir, exist_ok=True)

    # Apply web-friendly layout
    fig.update_layout(
        autosize=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=20, t=40, b=60),
    )

    html = fig.to_html(
        include_plotlyjs='cdn',
        config={
            'displayModeBar': 'hover',
            'displaylogo': False,
            'responsive': True,
            'fillFrame': True
        },
        div_id="plotly-div"
    )

    output_path = os.path.join(chart_dir, "index.html")
    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def render_chart_with_export(
    fig: go.Figure,
    chart_id: str,
    chart_title: str,
    category: str = "general"
):
    """Render a chart with an export checkbox beside it.

    Args:
        fig: Plotly figure to display
        chart_id: Unique identifier for this chart
        chart_title: Human-readable title for the chart
        category: Category for organizing exports (e.g., "temporal", "genre", "cluster")
    """
    col1, col2 = st.columns([20, 1])

    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Export checkbox
        export_key = f"export_{category}_{chart_id}"
        if st.checkbox("", key=export_key, help=f"Select to export '{chart_title}'"):
            if "charts_to_export" not in st.session_state:
                st.session_state["charts_to_export"] = {}
            st.session_state["charts_to_export"][f"{category}_{chart_id}"] = {
                "fig": fig,
                "title": chart_title,
                "category": category,
                "id": chart_id
            }
        else:
            # Remove from export list if unchecked
            full_id = f"{category}_{chart_id}"
            if "charts_to_export" in st.session_state and full_id in st.session_state["charts_to_export"]:
                del st.session_state["charts_to_export"][full_id]


def render_export_section(
    default_dir: str = "export/visualizations",
    section_key: str = "main"
):
    """Render the export section for selected charts.

    Args:
        default_dir: Default output directory
        section_key: Unique key for this export section (to avoid widget key conflicts)
    """
    st.markdown("---")
    st.subheader("📤 Export Selected Charts")

    charts_to_export = st.session_state.get("charts_to_export", {})

    if not charts_to_export:
        st.info("Select charts to export by checking the boxes next to each visualization above.")
        return

    # Group by category
    by_category = {}
    for full_id, chart_info in charts_to_export.items():
        category = chart_info.get("category", "general")
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(chart_info)

    st.write(f"**{len(charts_to_export)} chart(s) selected for export:**")
    for category, charts in by_category.items():
        st.write(f"  *{category.title()}:*")
        for chart_info in charts:
            st.write(f"    - {chart_info['title']}")

    output_dir = st.text_input(
        "Output directory",
        value=default_dir,
        key=f"export_dir_{section_key}"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        export_clicked = st.button(
            f"Export {len(charts_to_export)} Chart(s)",
            type="primary",
            key=f"export_btn_{section_key}"
        )
    with col2:
        clear_clicked = st.button(
            "Clear Selection",
            key=f"clear_btn_{section_key}"
        )

    if clear_clicked:
        st.session_state["charts_to_export"] = {}
        st.rerun()

    if export_clicked:
        os.makedirs(output_dir, exist_ok=True)
        exported = []

        with st.spinner("Exporting charts..."):
            for full_id, chart_info in charts_to_export.items():
                try:
                    # Use category as subfolder
                    category = chart_info.get("category", "general")
                    chart_output_dir = os.path.join(output_dir, category)
                    os.makedirs(chart_output_dir, exist_ok=True)

                    output_path = export_chart_to_html(
                        chart_info["fig"],
                        chart_info["id"],
                        chart_output_dir
                    )
                    exported.append(output_path)
                except Exception as e:
                    st.warning(f"Failed to export {chart_info['title']}: {e}")

        if exported:
            st.success(f"Exported {len(exported)} chart(s) to `{output_dir}/`")

            st.markdown("**Exported folders:**")
            for path in exported:
                # Get the chart folder name (parent of index.html)
                chart_folder = os.path.dirname(path)
                rel_path = os.path.relpath(chart_folder, output_dir)
                file_size = os.path.getsize(path) / 1024
                st.text(f"  {rel_path}/ ({file_size:.1f} KB)")

            st.info(f"""
**Next steps:**
1. Go to [Netlify Drop](https://app.netlify.com/drop)
2. Drag the `{output_dir}` folder into the browser
3. Get instant URL like: `https://amazing-viz-123.netlify.app`
4. Embed in Bear Blog:
   ```html
   <iframe src="YOUR_URL/category/chart_name/" width="100%" height="600"></iframe>
   ```
            """)

            # Clear selected charts after export
            st.session_state["charts_to_export"] = {}


def get_selected_chart_count() -> int:
    """Get the number of charts currently selected for export."""
    return len(st.session_state.get("charts_to_export", {}))
