"""Logging helpers for voxdoc-trainer.

Provides a configured logger and helpers to record run metadata (git commit,
environment, config hash) for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from src.config import RunConfig


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def git_commit() -> str:
    """Return the current git commit hash, or 'unknown' if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def config_hash(config: RunConfig) -> str:
    """Return a short hash of the config for run identification."""
    payload = json.dumps(config.to_dict(), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def write_run_metadata(run_dir: Path, config: RunConfig) -> Path:
    """Write run metadata (config + git commit) to the run directory.

    Args:
        run_dir (Path): Directory to write metadata into.
        config (RunConfig): The configuration for this run.

    Returns:
        Path: The path to the written metadata file.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "git_commit": git_commit(),
        "config": config.to_dict(),
    }
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return metadata_path
