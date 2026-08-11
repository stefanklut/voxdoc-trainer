"""Post-training evaluation using lm-eval-harness.

lm-eval-harness is external to TRL, so we wrap it here for post-training
benchmark evals. For in-loop eval during training, use TRL's native
`eval_dataset` + `compute_metrics` instead.
"""

from __future__ import annotations

import argparse


def evaluate(model: str, tasks: list[str], limit: int | None = None) -> None:
    """Run lm-eval-harness on a model for the given tasks.

    Args:
        model (str): The model id or path.
        tasks (list[str]): The lm-eval task names.
        limit (int | None): Optional limit on the number of examples.
    """
    from lm_eval import simple_evaluate

    results = simple_evaluate(model=model, tasks=tasks, limit=limit)
    print(results)


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a model with lm-eval-harness.")
    parser.add_argument("--model", required=True, help="Model id or path.")
    parser.add_argument("--tasks", nargs="+", required=True, help="lm-eval task names.")
    parser.add_argument("--limit", type=int, default=None, help="Optional example limit.")
    args = parser.parse_args()
    evaluate(args.model, args.tasks, args.limit)


if __name__ == "__main__":
    main()
