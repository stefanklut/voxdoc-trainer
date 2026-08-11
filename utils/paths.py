"""Path helpers for voxdoc-trainer."""

from __future__ import annotations

from pathlib import Path

from src.config import RunConfig


def run_output_dir(config: RunConfig) -> Path:
    """Return the output directory for a run, creating it if needed.

    Args:
        config (RunConfig): The run configuration.

    Returns:
        Path: The output directory path.
    """
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
