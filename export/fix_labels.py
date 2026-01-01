#!/usr/bin/env python3
"""
Script to properly simplify labels in exported visualizations.
Keeps only song name and artist, removing all other details.
"""

import re
import glob
import os

def simplify_labels(html_content):
    """
    Simplify labels to keep only song name and artist.
    """

    # Find the "text" array in the JavaScript
    # Pattern to match the entire text array
    text_pattern = r'("text":\[)(.*?)(\],"x":)'

    def process_text_array(match):
        prefix = match.group(1)  # "text":[
        text_content = match.group(2)  # the array content
        suffix = match.group(3)  # ],"x":

        # Split the content into individual label strings
        # Each label is a quoted string
        labels = []
        current_label = ""
        in_quote = False
        escape_next = False

        for char in text_content:
            if escape_next:
                current_label += char
                escape_next = False
            elif char == '\\':
                current_label += char
                escape_next = True
            elif char == '"' and not escape_next:
                if in_quote:
                    # End of a label
                    labels.append('"' + current_label + '"')
                    current_label = ""
                    in_quote = False
                else:
                    in_quote = True
            elif in_quote:
                current_label += char

        # Process each label to keep only song name and artist
        simplified_labels = []
        for label in labels:
            if label.strip() and label != '""':
                # Remove the quotes temporarily for processing
                label_content = label[1:-1]

                # Pattern to extract just the song name and artist
                # Look for pattern: \u003cb\u003eSongName\u003c/b\u003e\u003cbr\u003eArtist: ArtistName
                # and remove everything after the artist name

                # Find the position after "Artist: [artist name]"
                artist_match = re.search(r'(\\u003cb\\u003e.*?\\u003c\\u002fb\\u003e\\u003cbr\\u003eArtist: [^\\]+?)(?:\\u003cbr\\u003e|$)', label_content)

                if artist_match:
                    # Keep only the song and artist part
                    simplified_content = artist_match.group(1)
                    simplified_labels.append('"' + simplified_content + '"')
                else:
                    # If pattern doesn't match, keep original (shouldn't happen)
                    simplified_labels.append(label)

        # Reconstruct the text array
        return prefix + ','.join(simplified_labels) + suffix

    # Process the HTML content
    modified_content = re.sub(text_pattern, process_text_array, html_content, flags=re.DOTALL)

    return modified_content

def process_file(filepath):
    """Process a single HTML file to simplify labels."""
    print(f"Processing: {filepath}")

    try:
        # Read the file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Count original labels for verification
        original_count = content.count('\\u003cbr\\u003eCluster:')

        # Simplify the labels
        modified_content = simplify_labels(content)

        # Count modified labels to verify removal
        modified_count = modified_content.count('\\u003cbr\\u003eCluster:')

        # Write back the modified content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(modified_content)

        print(f"  ✓ Removed details from {original_count} labels (remaining: {modified_count})")
        return True

    except Exception as e:
        print(f"  ✗ Error processing {filepath}: {str(e)}")
        return False

def main():
    """Process all HTML files in the dimensions-of-taste-viz directory."""
    base_dir = "/Users/islamtayeb/Documents/spotify-clustering/export/dimensions-of-taste-viz"

    # Find all HTML files
    html_files = glob.glob(os.path.join(base_dir, "**/*.html"), recursive=True)

    if not html_files:
        print("No HTML files found in the dimensions-of-taste-viz directory.")
        return

    print(f"Found {len(html_files)} HTML files to process.\n")

    success_count = 0
    for filepath in html_files:
        if process_file(filepath):
            success_count += 1

    print(f"\n{'='*50}")
    print(f"Processing complete: {success_count}/{len(html_files)} files successfully processed.")

if __name__ == "__main__":
    main()