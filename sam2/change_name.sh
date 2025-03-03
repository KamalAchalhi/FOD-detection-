#!/bin/bash

# Directory containing the folders to rename
BASE_DIR="./segmented_masks_syntheses_comparaison"

# Find all DSC_* directories, extract numbers, and sort them
mapfile -t folders < <(ls -d ${BASE_DIR}/DSC_* 2>/dev/null | grep -oP '\d+' | sort -n)

# Initialize the new index (i) for renaming
i=0

# Loop through sorted DSC_j folders and rename them
for j in "${folders[@]}"; do
    OLD_NAME="${BASE_DIR}/DSC_${j}"
    NEW_NAME="${BASE_DIR}/frame_$(printf "%05d" $i)"  # Format as frame_0000i

    # Rename folder
    mv "$OLD_NAME" "$NEW_NAME"
    echo "Renamed: $OLD_NAME -> $NEW_NAME"

    # Increment index
    ((i++))
done

echo "✅ All folders renamed successfully!"

