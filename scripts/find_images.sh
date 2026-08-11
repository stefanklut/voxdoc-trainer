#!/usr/bin/env bash
# find_images.sh — find all images, shuffle them, save filenames to a list
set -euo pipefail

# Config
SEARCH_DIR="${1:-.}"          # directory to search (default: current)
OUTPUT_LIST="${2:-images.txt}" # output list file
EXTENSIONS='jpg jpeg png gif webp bmp tiff svg'  # add/remove as needed

# Build the find expression: -iname '*.jpg' -o -iname '*.png' ...
find_args=()
for ext in $EXTENSIONS; do
  find_args+=( -iname "*.$ext" -o )
done
# Remove the trailing "-o"
unset 'find_args[${#find_args[@]}-1]'

# Find, shuffle, write to file
find "$SEARCH_DIR" -type f \( "${find_args[@]}" \) -print0 \
  | shuf -z \
  | tr '\0' '\n' > "$OUTPUT_LIST"

echo "Wrote $(wc -l < "$OUTPUT_LIST") image paths to $OUTPUT_LIST"
