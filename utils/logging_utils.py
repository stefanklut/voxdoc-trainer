"""Unified logging for voxdoc-trainer.

Based on detectron2.utils.logger.py. Provides auto-derived ``voxdoc-trainer.*``
logger names, a colored console + optional file logger with distributed-rank
support (:func:`setup_logger`), and reproducibility helpers (git commit, config
hash, run metadata).
"""

import atexit
import functools
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from termcolor import colored

if TYPE_CHECKING:
    from src.config import RunConfig

LOGGER_BASE_NAME: str = "voxdoc-trainer"
DEFAULT_LOG_BUFFER_SIZE: int = 1024 * 1024  # 1MB
LOG_BUFFER_SIZE_KEY: str = "LOG_BUFFER_SIZE"


def get_logger_name():
    frame = sys._getframe(1)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logging_utils.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = LOGGER_BASE_NAME
            else:
                mod_name = f"{LOGGER_BASE_NAME}." + mod_name
            # return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
            return mod_name
        frame = frame.f_back


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``voxdoc-trainer`` namespace.

    If ``name`` is None the logger name is derived from the caller module via
    :func:`get_logger_name`. Handlers are attached by :func:`setup_logger`, so
    call that once (e.g. in ``main``) before expecting formatted output.

    Args:
        name (str | None): Explicit logger name. Defaults to the auto-derived name.

    Returns:
        logging.Logger: The logger (handlers configured by setup_logger).
    """
    return logging.getLogger(name if name is not None else get_logger_name())


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record: logging.LogRecord) -> str:
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        elif record.levelno == logging.DEBUG:
            prefix = colored("DEBUG", "blue", attrs=["blink"])
        elif record.levelno == logging.INFO:
            return log
        else:
            return log
        return prefix + " " + log


class _PlainFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs) -> None:
        super(_PlainFormatter, self).__init__(*args, **kwargs)

    @staticmethod
    def remove_ansi(message: str) -> str:
        pattern = re.compile(r"\x1B\[\d+(;\d+){0,2}m")
        stripped = pattern.sub("", message)
        return stripped

    def formatMessage(self, record: logging.LogRecord) -> str:
        message = super().formatMessage(record)
        message = self.remove_ansi(message)
        return message


def _get_log_stream_buffer_size(filename: str) -> int:
    if "://" not in filename:
        # Local file, no extra caching is necessary
        return -1
    # Remote file requires a larger cache to avoid many small writes.
    if LOG_BUFFER_SIZE_KEY in os.environ:
        return int(os.environ[LOG_BUFFER_SIZE_KEY])
    return DEFAULT_LOG_BUFFER_SIZE


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename: str):
    io = open(filename, "a", buffering=_get_log_stream_buffer_size(filename))
    atexit.register(io.close)
    return io


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(output=None, distributed_rank=0, *, color=True, name=LOGGER_BASE_NAME, abbrev_name=None):
    """
    Initialize the voxdoc-trainer logger and set its verbosity level to "DEBUG".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    plain_formatter = _PlainFormatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        output_path = Path(output)
        if output_path.suffix in [".txt", ".log"]:
            filename = output_path
        else:
            filename = output_path.joinpath("log.txt")
        if distributed_rank > 0:
            filename = Path(f"{filename}.rank{distributed_rank}")
        filename.parent.mkdir(parents=True, exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(str(filename)))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

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


def config_hash(config: "RunConfig") -> str:
    """Return a short hash of the config for run identification."""
    payload = json.dumps(config.to_dict(), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def write_run_metadata(run_dir: Path, config: "RunConfig") -> Path:
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
