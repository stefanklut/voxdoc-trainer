"""Pretrain reward functions for GRPO pretraining (e.g. learning Old Dutch).

This module is deliberately separate from :mod:`src.rewards` so the *pretrain*
stage (teaching the model a language / task from scratch) has its own reward
surface, distinct from the *finetune* stage (``format_reward`` +
``accuracy_reward`` in :mod:`src.rewards`).

* :func:`cer_reward` — a continuous ``[0, 1]`` reward based on the **character
  error rate** (CER) between the model's final answer and the ground truth.
  CER is the standard OCR/transcription metric, so it is the natural objective
  for pretraining a model to transcribe historical (Old Dutch) documents.
  It is **case-sensitive** and counts whitespace, matching the primary
  (unnormalized) CER reported by the document evaluation.
* :func:`cer_reward_normalized` — the same reward after casefolding and
  collapsing whitespace on both sides, available as an alternative via
  ``grpo.cer_metric: normalized``.

CER is *added* to the reward toolkit, not a replacement for the finetune
rewards: the two stages select different reward functions (see
``GRPOConfig.stage`` and :mod:`src.train_grpo`).

The reward scores the **final answer only**: the reasoning tokens
(``think ... endthink``) are stripped before scoring, so the chain-of-thought
is never reinforced — only the corrected prediction. This matches the
finetune rewards and the goal of keeping the model's reasoning behaviour while
supervising just the answer.

Reward-function contract (TRL 1.13.0): each function is called as
``func(prompts=..., completions=..., completion_ids=..., **dataset_columns)``
and must return exactly one float per completion. The ``ground_truth`` column
(added by :func:`tools.convert_data.to_grpo`) is therefore received as a kwarg.
"""

from __future__ import annotations

from src.rewards import _completion_text, extract_answer
from utils.metrics import character_error_rate as _cer
from utils.metrics import normalize_text as _normalize_text

__all__ = ["cer_reward", "cer_reward_normalized"]


def _cer_scores(
    completions, ground_truth, normalize: bool, log_metric
) -> list[float | None]:
    """Score each completion as ``max(0.0, 1.0 - CER)`` against its ground truth.

    Args:
        completions: Decoded completions (one per generation) — strings or
            lists of message dicts, depending on the prompt format.
        ground_truth (list[str]): Ground-truth answers (one per completion).
        normalize (bool): When ``True``, casefold and collapse whitespace on
            both sides before scoring; when ``False``, score the raw text.
        log_metric: Optional ``name -> value`` logger for the
            ``reward/truncated`` metric.

    Returns:
        list[float | None]: One reward per completion, each in ``[0, 1]``.
    """
    scores: list[float | None] = []
    for completion, gold in zip(completions, ground_truth):
        text = _completion_text(completion)
        answer = extract_answer(text)
        if normalize:
            prediction, reference = _normalize_text(answer), _normalize_text(gold)
        else:
            prediction, reference = answer, gold
        scores.append(max(0.0, 1.0 - _cer(prediction, reference)))
        if log_metric is not None:
            log_metric("reward/truncated", 0.0 if "endthink" in text else 1.0)
    return scores


def cer_reward(completions, ground_truth, **kwargs) -> list[float | None]:
    """Reward the closeness of the final answer to the ground truth via CER.

    The reward is ``max(0.0, 1.0 - CER)`` on the **raw** final answer, so it is
    case-sensitive and counts whitespace — matching the primary (unnormalized)
    CER reported by the document evaluation. It is a continuous score in
    ``[0, 1]``: ``1.0`` for an exact match, decreasing with the number of
    character edits, and clamped to ``0.0`` when the prediction is far worse
    than the reference. Continuous partial credit keeps GRPO groups from
    collapsing to a single reward value (which would yield zero advantage and
    no gradient).

    Also logs a ``reward/truncated`` metric (1.0 when the completion has no
    ``endthink`` marker, i.e. it likely hit ``max_completion_length`` mid-
    reasoning) so a too-small cap is visible in the training metrics.

    Args:
        completions: Decoded completions (one per generation) — strings or
            lists of message dicts, depending on the prompt format.
        ground_truth (list[str]): Ground-truth answers (one per completion).
        **kwargs: Extra TRL reward kwargs (``trainer_state``, ``log_metric``, ...).

    Returns:
        list[float | None]: One reward per completion, each in ``[0, 1]``.
    """
    return _cer_scores(completions, ground_truth, normalize=False, log_metric=kwargs.get("log_metric"))


def cer_reward_normalized(completions, ground_truth, **kwargs) -> list[float | None]:
    """Case-insensitive variant of :func:`cer_reward`.

    Identical to :func:`cer_reward` except both sides are casefolded and
    whitespace-collapsed before scoring, so case and whitespace differences
    earn no penalty. Select it with ``grpo.cer_metric: normalized``.

    Args:
        completions: Decoded completions (one per generation) — strings or
            lists of message dicts, depending on the prompt format.
        ground_truth (list[str]): Ground-truth answers (one per completion).
        **kwargs: Extra TRL reward kwargs (``trainer_state``, ``log_metric``, ...).

    Returns:
        list[float | None]: One reward per completion, each in ``[0, 1]``.
    """
    return _cer_scores(completions, ground_truth, normalize=True, log_metric=kwargs.get("log_metric"))
