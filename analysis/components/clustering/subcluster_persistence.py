"""
Persistence module for saving and loading subclustering results.

This module provides functions to:
- Save subcluster analyses to disk with metadata
- Load saved subclusters back into memory
- List and browse saved subclusters
- Delete saved subclusters
- Manage metadata index for fast browsing
"""

import os
import pickle
import json
import re
from datetime import datetime
from typing import Optional, List, Dict, Any


# Constants
SUBCLUSTERS_DIR = "analysis/outputs/subclusters"
METADATA_FILE = "metadata.json"


def save_subcluster(data: dict, custom_name: Optional[str] = None) -> str:
    """
    Save subcluster results to pickle file with metadata.

    Args:
        data: Dictionary containing subcluster results (from run_subcluster_pipeline)
        custom_name: Optional user-provided label for this subcluster

    Returns:
        str: Path to the saved file

    Raises:
        ValueError: If required keys are missing from data
        IOError: If file write fails
    """
    # Validate required keys
    required_keys = ['parent_cluster', 'n_subclusters', 'silhouette_score']
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise ValueError(f"Missing required keys in data: {missing_keys}")

    # Sanitize custom name
    if custom_name:
        custom_name = sanitize_filename(custom_name)
        if not custom_name:  # If sanitization resulted in empty string
            custom_name = None

    # Generate filename
    parent_id = data['parent_cluster']
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if custom_name:
        filename = f"parent_{parent_id}_{timestamp}_{custom_name}.pkl"
    else:
        # Default: use algorithm name or "subcluster"
        algorithm = data.get('algorithm', 'subcluster')
        filename = f"parent_{parent_id}_{timestamp}_{algorithm}.pkl"

    # Create directory if it doesn't exist
    os.makedirs(SUBCLUSTERS_DIR, exist_ok=True)

    # Full file path
    file_path = os.path.join(SUBCLUSTERS_DIR, filename)

    # Add timestamp to data if not present
    if 'timestamp' not in data:
        data['timestamp'] = datetime.now().isoformat()

    # Add custom_name to data
    data['custom_name'] = custom_name or ""

    # Save pickle file
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise IOError(f"Failed to save subcluster to {file_path}: {str(e)}")

    # Update metadata index
    metadata_entry = {
        'file_name': filename,
        'parent_cluster': data['parent_cluster'],
        'n_subclusters': data['n_subclusters'],
        'silhouette_score': data['silhouette_score'],
        'timestamp': data['timestamp'],
        'custom_name': custom_name or "",
        'algorithm': data.get('algorithm', 'unknown'),
        'parent_cluster_size': data.get('parent_cluster_size', 0),
        'source_mode': data.get('source_mode', 'unknown'),
    }

    update_metadata_index('add', metadata_entry)

    return file_path


def load_subcluster(file_path: str) -> dict:
    """
    Load subcluster from pickle file.

    Args:
        file_path: Path to the pickle file

    Returns:
        dict: Subcluster data

    Raises:
        FileNotFoundError: If file doesn't exist
        pickle.UnpicklingError: If file is corrupted
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Subcluster file not found: {file_path}")

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Corrupted pickle file: {file_path}")
    except Exception as e:
        raise IOError(f"Failed to load subcluster from {file_path}: {str(e)}")


def list_saved_subclusters() -> List[Dict[str, Any]]:
    """
    List all saved subclusters from metadata.json.

    Returns:
        list: List of metadata dictionaries, sorted by timestamp (newest first)
    """
    metadata_path = os.path.join(SUBCLUSTERS_DIR, METADATA_FILE)

    # If directory doesn't exist, return empty list
    if not os.path.exists(SUBCLUSTERS_DIR):
        return []

    # If metadata.json doesn't exist or is corrupted, rebuild it
    if not os.path.exists(metadata_path):
        rebuild_metadata_index()

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        subclusters = metadata.get('subclusters', [])

        # Sort by timestamp (newest first)
        subclusters = sorted(
            subclusters,
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )

        return subclusters

    except (json.JSONDecodeError, IOError):
        # If metadata is corrupted, rebuild it
        rebuild_metadata_index()
        # Try again
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return sorted(
                metadata.get('subclusters', []),
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )
        except:
            # If still fails, return empty list
            return []


def delete_subcluster(file_path: str) -> bool:
    """
    Delete saved subcluster.

    Args:
        file_path: Path to the pickle file to delete

    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        # Delete the file
        if os.path.exists(file_path):
            os.remove(file_path)

        # Update metadata index
        filename = os.path.basename(file_path)
        update_metadata_index('remove', {'file_name': filename})

        return True

    except Exception as e:
        print(f"Error deleting subcluster: {str(e)}")
        return False


def update_metadata_index(action: str, data: dict) -> None:
    """
    Update metadata.json index.

    Args:
        action: Either 'add' or 'remove'
        data: Metadata entry for 'add', or {'file_name': ...} for 'remove'
    """
    metadata_path = os.path.join(SUBCLUSTERS_DIR, METADATA_FILE)

    # Create directory if it doesn't exist
    os.makedirs(SUBCLUSTERS_DIR, exist_ok=True)

    # Load existing metadata or create new
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError):
            metadata = {'subclusters': []}
    else:
        metadata = {'subclusters': []}

    # Update based on action
    if action == 'add':
        # Remove any existing entry with same file_name (for overwrite case)
        metadata['subclusters'] = [
            s for s in metadata['subclusters']
            if s.get('file_name') != data.get('file_name')
        ]
        # Add new entry
        metadata['subclusters'].append(data)

    elif action == 'remove':
        # Remove entry with matching file_name
        filename = data.get('file_name')
        metadata['subclusters'] = [
            s for s in metadata['subclusters']
            if s.get('file_name') != filename
        ]

    # Write updated metadata
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error updating metadata index: {str(e)}")


def rebuild_metadata_index() -> None:
    """
    Rebuild metadata.json from existing .pkl files.

    This is used as a fallback when metadata.json is missing or corrupted.
    Scans the subclusters directory, loads each pickle file, and extracts metadata.
    """
    if not os.path.exists(SUBCLUSTERS_DIR):
        return

    metadata = {'subclusters': []}

    # Scan directory for .pkl files
    for filename in os.listdir(SUBCLUSTERS_DIR):
        if not filename.endswith('.pkl'):
            continue

        file_path = os.path.join(SUBCLUSTERS_DIR, filename)

        try:
            # Load pickle to extract metadata
            data = load_subcluster(file_path)

            # Create metadata entry
            entry = {
                'file_name': filename,
                'parent_cluster': data.get('parent_cluster', 0),
                'n_subclusters': data.get('n_subclusters', 0),
                'silhouette_score': data.get('silhouette_score', 0.0),
                'timestamp': data.get('timestamp', ''),
                'custom_name': data.get('custom_name', ''),
                'algorithm': data.get('algorithm', 'unknown'),
                'parent_cluster_size': data.get('parent_cluster_size', 0),
                'source_mode': data.get('source_mode', 'unknown'),
            }

            metadata['subclusters'].append(entry)

        except Exception as e:
            print(f"Error loading {filename} during rebuild: {str(e)}")
            continue

    # Write metadata.json
    metadata_path = os.path.join(SUBCLUSTERS_DIR, METADATA_FILE)
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error writing metadata index: {str(e)}")


def sanitize_filename(name: str, max_length: int = 30) -> str:
    """
    Sanitize a string to be safe for use in filenames.

    Args:
        name: Input string
        max_length: Maximum length of output (default 30)

    Returns:
        str: Sanitized string (alphanumeric + hyphens only)
    """
    # Replace spaces with hyphens
    name = name.replace(' ', '-')

    # Keep only alphanumeric and hyphens
    name = re.sub(r'[^a-zA-Z0-9-]', '', name)

    # Limit length
    name = name[:max_length]

    # Remove leading/trailing hyphens
    name = name.strip('-')

    return name


def get_subcluster_stats() -> Dict[str, Any]:
    """
    Get statistics about saved subclusters.

    Returns:
        dict: Statistics including total count, counts by parent, etc.
    """
    subclusters = list_saved_subclusters()

    if not subclusters:
        return {
            'total': 0,
            'by_parent': {},
            'avg_silhouette': 0.0,
        }

    # Count by parent cluster
    by_parent = {}
    for s in subclusters:
        parent = s['parent_cluster']
        by_parent[parent] = by_parent.get(parent, 0) + 1

    # Average silhouette score
    avg_silhouette = sum(s['silhouette_score'] for s in subclusters) / len(subclusters)

    return {
        'total': len(subclusters),
        'by_parent': by_parent,
        'avg_silhouette': avg_silhouette,
    }
