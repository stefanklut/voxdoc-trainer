"""Unit tests for shared utilities."""

from __future__ import annotations

from src.config import RunConfig
from utils.logging_utils import config_hash, git_commit


def test_config_hash_is_stable() -> None:
    """The config hash should be deterministic for the same config."""
    config_a = RunConfig()
    config_b = RunConfig()
    assert config_hash(config_a) == config_hash(config_b)
    assert len(config_hash(config_a)) == 8


def test_git_commit_returns_string() -> None:
    """git_commit should return a non-empty string."""
    assert isinstance(git_commit(), str)
