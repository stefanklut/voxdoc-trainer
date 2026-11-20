"""Generate repeated document predictions or score saved outputs without inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.runner import load_config, run_experiment, score_prediction_file, score_saved_run
from utils.logging_utils import setup_logger


def main() -> None:
    """Run the document evaluator CLI with separate generation and offline commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    generate = subcommands.add_parser("generate", help="Generate repeated attempts; automatically resume compatible runs.")
    generate.add_argument("--config", required=True, type=Path)
    score = subcommands.add_parser("score", help="Offline scoring; never contacts an endpoint.")
    source = score.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--predictions", type=Path, help="External prediction JSONL; also requires --config and --out.")
    score.add_argument("--config", type=Path)
    score.add_argument("--out", type=Path)
    score.add_argument("--iou-threshold", type=float, help="Override threshold when rescoring a saved run.")
    args = parser.parse_args()
    setup_logger()
    if args.command == "generate":
        report = run_experiment(load_config(args.config))
    elif args.run_dir is not None:
        if args.config is not None or args.out is not None:
            parser.error("--run-dir cannot be combined with --config or --out")
        report = score_saved_run(args.run_dir, args.iou_threshold)
    else:
        if args.config is None or args.out is None or args.iou_threshold is not None:
            parser.error("--predictions requires --config and --out; set iou_threshold in the config")
        report = score_prediction_file(load_config(args.config), args.predictions, args.out)
    table = Table(title=f"Document evaluation: {report['completed_attempts']}/{report['planned_attempts']} attempts")
    for column in ("Model", "Variant", "Task", "Valid", "IoU", "Label Accuracy", "Corpus CER"):
        table.add_column(column)
    for group in report["groups"]:
        metrics = group["all_attempts"]
        formatted = [
            "-" if metrics.get(key) is None else f"{metrics[key]:.4f}" for key in ("mean_iou", "label_accuracy", "corpus_cer")
        ]
        table.add_row(group["model"], group["variant"], group["task"], f"{group['valid_rate']:.1%}", *formatted)
    Console().print(table)


if __name__ == "__main__":
    main()
