#!/usr/bin/env python3
"""
Remove metadata from lyrics files by detecting separator line and save cleaned versions.
"""

import os
from pathlib import Path

# Define directories
DATA_DIR = Path("lyrics/data")
CLEANED_DIR = Path("lyrics/data-cleaned")

# Create cleaned directory if it doesn't exist
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

# Get all .txt files in data directory
txt_files = list(DATA_DIR.glob("*.txt"))

print(f"Found {len(txt_files)} lyrics files to clean")
print(f"Output directory: {CLEANED_DIR}")
print()

# Process each file
cleaned_count = 0
error_count = 0

for txt_file in txt_files:
    try:
        # Read the file
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find the separator line (5+ consecutive ='s)
        separator_index = -1
        for i, line in enumerate(lines):
            if '=====' in line:
                separator_index = i
                break

        # If separator found, skip everything up to and including separator + 1 line after
        if separator_index >= 0 and len(lines) > separator_index + 2:
            # Start from line after the empty line following the separator
            cleaned_lines = lines[separator_index + 2:]

            # Write to cleaned directory with same filename
            output_file = CLEANED_DIR / txt_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(cleaned_lines)

            cleaned_count += 1
            if cleaned_count % 100 == 0:
                print(f"Processed {cleaned_count} files...")
        else:
            print(f"Warning: {txt_file.name} - separator not found or file too short, skipping")
            error_count += 1

    except Exception as e:
        print(f"Error processing {txt_file.name}: {e}")
        error_count += 1

print()
print(f"✓ Successfully cleaned {cleaned_count} files")
if error_count > 0:
    print(f"✗ Encountered errors with {error_count} files")
print(f"Cleaned files saved to: {CLEANED_DIR}")
