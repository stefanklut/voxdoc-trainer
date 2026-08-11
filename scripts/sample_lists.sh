#!/usr/bin/env bash
# sample_lists.sh — sample from the top of multiple shuffled lists
# Usage:
#   ./sample_lists.sh --fixed 100 list1.txt list2.txt list3.txt
#   ./sample_lists.sh --relative 0.2 list1.txt list2.txt
#   ./sample_lists.sh --fixed 100 --relative 0.1 list1.txt list2.txt
set -euo pipefail

MODE=""      # "fixed" or "relative"
FIXED_N=0
REL_FRAC=0.0
OUTPUT="sampled.txt"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fixed)
      MODE="fixed"; FIXED_N="$2"; shift 2 ;;
    --relative)
      MODE="relative"; REL_FRAC="$2"; shift 2 ;;
    --output)
      OUTPUT="$2"; shift 2 ;;
    -*)
      echo "Unknown option: $1" >&2; exit 1 ;;
    *)
      LISTS+=("$1"); shift ;;
  esac
done

if [[ ${#LISTS[@]} -eq 0 ]]; then
  echo "Error: no list files provided" >&2
  echo "Usage: $0 --fixed N | --relative FRAC [--output FILE] list1.txt ..." >&2
  exit 1
fi

: > "$OUTPUT"   # truncate output

for list in "${LISTS[@]}"; do
  [[ -f "$list" ]] || { echo "Warning: $list not found, skipping" >&2; continue; }

  total=$(wc -l < "$list")

  # Determine how many to take from this list
  case "$MODE" in
    fixed)
      take=$FIXED_N ;;
    relative)
      take=$(awk -v t="$total" -v f="$REL_FRAC" 'BEGIN{printf "%d", t*f}') ;;
    *)
      echo "Error: must specify --fixed or --relative" >&2; exit 1 ;;
  esac

  # Cap at the number of lines available
  if (( take > total )); then take=$total; fi

  echo "  $list: $total lines, taking $take" >&2
  head -n "$take" "$list" >> "$OUTPUT"
done

echo "Wrote $(wc -l < "$OUTPUT") sampled paths to $OUTPUT"
