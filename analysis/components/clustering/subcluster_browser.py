"""
Browser UI for saved subclusters.

This module provides a collapsible sidebar interface for:
- Browsing all saved subclusters
- Filtering by parent cluster
- Sorting by date or quality
- Loading saved subclusters back into session
- Deleting saved subclusters
"""

import os
import pickle
import streamlit as st
from datetime import datetime
from itertools import groupby
from typing import Dict, Any

from .subcluster_persistence import (
    list_saved_subclusters,
    load_subcluster,
    delete_subcluster,
)


def render_subcluster_browser() -> None:
    """
    Render collapsible browser for saved subclusters in sidebar.

    Features:
    - List all saved subclusters grouped by parent cluster
    - Filter by parent cluster
    - Sort by date (newest first) or silhouette score (best first)
    - Load and Delete buttons for each saved subcluster
    """
    with st.sidebar.expander("📂 Saved Sub-Clusters", expanded=False):
        # Get all saved subclusters
        saved = list_saved_subclusters()

        if not saved:
            st.info("No saved subclusters yet.")
            return

        # Filter controls
        parent_ids = sorted(set(s['parent_cluster'] for s in saved))
        parent_filter = st.selectbox(
            "Filter by parent",
            ["All"] + [f"Parent {pid}" for pid in parent_ids],
            key="subcluster_browser_filter",
        )

        # Sort controls
        sort_by = st.selectbox(
            "Sort by",
            ["Date (newest)", "Quality (best)"],
            key="subcluster_browser_sort",
        )

        # Apply filters
        filtered_saved = saved.copy()
        if parent_filter != "All":
            parent_id = int(parent_filter.split()[-1])
            filtered_saved = [s for s in filtered_saved if s['parent_cluster'] == parent_id]

        # Apply sorting
        if sort_by == "Quality (best)":
            filtered_saved = sorted(
                filtered_saved,
                key=lambda x: x.get('silhouette_score', 0),
                reverse=True
            )
        else:
            filtered_saved = sorted(
                filtered_saved,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )

        # Display count
        st.caption(f"Showing {len(filtered_saved)} of {len(saved)} saved subclusters")
        st.markdown("---")

        # Group by parent cluster for organized display
        if parent_filter == "All":
            # Group by parent when showing all
            filtered_saved_sorted = sorted(filtered_saved, key=lambda x: x['parent_cluster'])
            for parent_id, group in groupby(filtered_saved_sorted, key=lambda x: x['parent_cluster']):
                group_list = list(group)
                st.markdown(f"**📁 Parent {parent_id}** ({len(group_list)} save{'s' if len(group_list) != 1 else ''})")

                for metadata in group_list:
                    render_subcluster_card(metadata)

                st.markdown("")  # Spacing between groups
        else:
            # Just show the filtered list
            for metadata in filtered_saved:
                render_subcluster_card(metadata)


def render_subcluster_card(metadata: Dict[str, Any]) -> None:
    """
    Render a single subcluster entry card with Load/Delete buttons.

    Args:
        metadata: Dictionary containing subcluster metadata
    """
    name = metadata.get('custom_name', '')
    if not name:
        name = "(untitled)"

    timestamp = metadata.get('timestamp', '')
    score = metadata.get('silhouette_score', 0)
    n_clusters = metadata.get('n_subclusters', 0)
    file_name = metadata['file_name']
    algorithm = metadata.get('algorithm', 'unknown')

    # Format timestamp for display
    try:
        dt = datetime.fromisoformat(timestamp)
        display_time = dt.strftime("%b %d, %H:%M")
    except:
        display_time = timestamp

    # Card layout with 3 columns
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.markdown(f"**{name}**")
        st.caption(f"{display_time} | ⭐{score:.2f} | {n_clusters} clusters | {algorithm.upper()}")

    with col2:
        if st.button("Load", key=f"load_{file_name}", use_container_width=True):
            handle_load_subcluster(metadata)

    with col3:
        if st.button("🗑️", key=f"delete_{file_name}", use_container_width=True):
            # Use session state to track delete confirmation
            confirm_key = f"confirm_delete_{file_name}"
            if confirm_key not in st.session_state:
                st.session_state[confirm_key] = False

            # Toggle confirmation state
            st.session_state[confirm_key] = True
            st.rerun()

    # Show delete confirmation if flagged
    confirm_key = f"confirm_delete_{file_name}"
    if confirm_key in st.session_state and st.session_state[confirm_key]:
        st.warning(f"Delete '{name}'?")
        col_yes, col_no = st.columns(2)

        with col_yes:
            if st.button("✓ Yes", key=f"confirm_yes_{file_name}", use_container_width=True):
                handle_delete_subcluster(metadata)
                st.session_state[confirm_key] = False
                st.rerun()

        with col_no:
            if st.button("✗ No", key=f"confirm_no_{file_name}", use_container_width=True):
                st.session_state[confirm_key] = False
                st.rerun()

    st.markdown("---")  # Separator between cards


def handle_load_subcluster(metadata: Dict[str, Any]) -> None:
    """
    Load saved subcluster into session state.

    This function:
    1. Loads the pickle file
    2. Handles the pca_features dependency (uses saved subset)
    3. Populates st.session_state["subcluster_data"]
    4. Triggers a rerun to display results

    Args:
        metadata: Dictionary containing subcluster metadata
    """
    file_path = os.path.join("analysis/outputs/subclusters", metadata['file_name'])
    name = metadata.get('custom_name', '') or "(untitled)"

    try:
        # Load data
        data = load_subcluster(file_path)

        # Tier 1: Use saved pca_features_subset (self-contained)
        if 'pca_features_subset' in data:
            st.session_state["subcluster_data"] = data
            st.sidebar.success(f"✅ Loaded: {name}")
            st.rerun()

        # Tier 2: Fall back to loading from parent analysis file
        elif 'parent_analysis_file' in data:
            parent_file = data['parent_analysis_file']
            if os.path.exists(parent_file):
                with open(parent_file, 'rb') as f:
                    parent_data = pickle.load(f)

                # Check if pca_features exist in parent
                if 'pca_features' in parent_data:
                    st.session_state["pca_features"] = parent_data["pca_features"]
                    st.session_state["subcluster_data"] = data
                    st.sidebar.success(f"✅ Loaded: {name} (from parent analysis)")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Parent analysis missing pca_features")
            else:
                st.sidebar.error(f"❌ Parent analysis file not found: {parent_file}")

        # Tier 3: Check if pca_features already in session
        elif "pca_features" in st.session_state:
            # Assume it's compatible (user has current analysis loaded)
            st.session_state["subcluster_data"] = data
            st.sidebar.warning(f"⚠️ Loaded {name} using current session pca_features")
            st.rerun()

        else:
            st.sidebar.error("❌ Cannot load: missing pca_features. Please load parent analysis first.")

    except FileNotFoundError:
        st.sidebar.error(f"❌ File not found: {file_path}")
    except pickle.UnpicklingError:
        st.sidebar.error(f"❌ Corrupted file: {metadata['file_name']}")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load: {str(e)}")


def handle_delete_subcluster(metadata: Dict[str, Any]) -> None:
    """
    Delete saved subcluster with confirmation.

    Args:
        metadata: Dictionary containing subcluster metadata
    """
    file_path = os.path.join("analysis/outputs/subclusters", metadata['file_name'])
    name = metadata.get('custom_name', '') or "(untitled)"

    try:
        success = delete_subcluster(file_path)
        if success:
            st.sidebar.success(f"✅ Deleted: {name}")
        else:
            st.sidebar.error("❌ Failed to delete file")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
