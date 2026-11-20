"""Shared, unclipped metrics for training rewards and document evaluation."""

from __future__ import annotations

import re


def bbox_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    """Return continuous-coordinate IoU for two validated xyxy boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip the ends."""
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_text(text: str) -> str:
    """Casefold text and collapse whitespace, matching the training rewards."""
    return collapse_whitespace(text).casefold()


def edit_distance(first: str, second: str) -> int:
    """Return Unicode-codepoint Levenshtein distance using linear memory."""
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for row, first_character in enumerate(first, start=1):
        current = [row]
        for column, second_character in enumerate(second, start=1):
            current.append(
                min(
                    previous[column - 1] + (first_character != second_character),
                    previous[column] + 1,
                    current[column - 1] + 1,
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(prediction: str, reference: str) -> float:
    """Return unclipped CER; empty references score zero if equal, else one."""
    if not reference:
        return 0.0 if not prediction else 1.0
    return edit_distance(prediction, reference) / len(reference)
