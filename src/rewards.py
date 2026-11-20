"""Reward functions for GRPO post-training of the document-understanding VLM.

Two reward functions are exposed, following the format + accuracy pattern used
across the GRPO reasoning literature:

* :func:`format_reward` — ``1.0`` when the model's final answer is well-formed
  for the task (parseable DocLang, a valid bbox / bbox list, an integer, ...),
  else ``0.0``.
* :func:`accuracy_reward` — a continuous ``[0, 1]`` score for how close the
  final answer is to the ground truth (DocLang similarity, bbox IoU, text
  similarity, ...). Partial credit keeps GRPO groups from collapsing to a single
  reward value, which would yield zero advantage and no gradient.

Both functions score the **final answer only**: the reasoning tokens
(``think ... endthink``) are stripped before scoring, so the chain-of-thought is
never reinforced — only the corrected prediction is. This matches the training
goal of keeping the model's reasoning behaviour while supervising just the
answer the annotators fixed.

The task is inferred from the *shape of the ground truth* rather than an
explicit column, so the same rewards work across every data creator's output
format: DocLang documents, single bboxes, bbox lists, DocLang ``<location>``
runs, plain transcriptions, line counts, reading-order index lists, and ordinal
phrases such as ``line 3 of 7``.

Reward-function contract (TRL 1.13.0): each function is called as
``func(prompts=..., completions=..., completion_ids=..., **dataset_columns)``
and must return exactly one float per completion. The ``ground_truth`` column
(added by :func:`tools.convert_data.to_grpo`) is therefore received as a kwarg.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from src.doclang_structured.parse import doclang_xml_to_model
from src.doclang_structured.serialize import model_to_doclang_xml
from utils.metrics import bbox_iou as _iou
from utils.metrics import collapse_whitespace as _collapse_whitespace

__all__ = ["extract_answer", "format_reward", "accuracy_reward"]

# --------------------------------------------------------------------------- #
# Task classification (by ground-truth shape)
# --------------------------------------------------------------------------- #

_INT = r"-?\d+"
_BBOX_RE = re.compile(rf"\[\s*({_INT})\s*,\s*({_INT})\s*,\s*({_INT})\s*,\s*({_INT})\s*\]")
_LOCATION_RE = re.compile(r'<location\s+value="(-?\d+)"\s*/>')
_ORDINAL_RE = re.compile(r"line\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE)


def _classify(gold: str) -> str:
    """Infer the task kind from the shape of the ground-truth answer.

    Args:
        gold (str): The ground-truth completion text.

    Returns:
        str: One of ``doclang``, ``bbox``, ``bbox_doclang``, ``integer``,
        ``ordinal``, ``index_list``, or ``text``.
    """
    s = (gold or "").strip()
    if s.startswith("<doclang"):
        return "doclang"
    if s.startswith("<text>") and "<location" in s:
        return "bbox_doclang"
    if s.startswith("["):
        return "bbox"
    if re.fullmatch(_INT, s):
        return "integer"
    if _ORDINAL_RE.fullmatch(s):
        return "ordinal"
    if "," in s and re.fullmatch(r"[\d,\s]+", s):
        return "index_list"
    return "text"


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #


def _content_text(content) -> str:
    """Extract text from a message ``content`` (a string or a list of blocks).

    Args:
        content: A message content — either a plain string or a list of
            content blocks such as ``[{"type": "text", "text": "..."}]``.

    Returns:
        str: The concatenated text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return "" if content is None else str(content)


def _completion_text(completion) -> str:
    """Normalize a TRL completion to a plain string.

    TRL passes completions to reward functions either as plain strings
    (single-turn prompts) or as a list of message dicts (conversational
    prompts), e.g. ``[{"role": "assistant", "content": "..."}]``. This returns
    the assistant message's text in either case.

    Args:
        completion: A completion — a string or a list of message dicts.

    Returns:
        str: The completion text.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, (list, tuple)):
        for message in reversed(completion):
            if isinstance(message, dict) and "content" in message:
                return _content_text(message["content"])
        return ""
    return "" if completion is None else str(completion)


def extract_answer(completion) -> str:
    """Return the final answer: the text after the last ``endthink`` marker.

    The Qwen3.5 thinking template renders the assistant turn as
    ``think\\n{reasoning}\\nendthink\\n\\n{answer}``; the ``think`` prefix is part
    of the prompt, so the completion is ``{reasoning}\\nendthink\\n\\n{answer}``.
    When the model emits no ``endthink`` (thinking disabled, or a completion
    truncated mid-reasoning), the whole completion is returned.

    Args:
        completion: The completion — a decoded string or a list of message
            dicts (TRL passes either, depending on the prompt format).

    Returns:
        str: The final answer with surrounding whitespace stripped.
    """
    text = _completion_text(completion)
    if not text:
        return ""
    idx = text.rfind("endthink")
    if idx == -1:
        return text.strip()
    return text[idx + len("endthink") :].strip()


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #


def _parse_boxes(text: str) -> list[tuple[int, int, int, int]] | None:
    """Parse ``[x1, y1, x2, y2]`` or ``[[...], ...]`` into a list of boxes.

    Args:
        text (str): The answer text.

    Returns:
        list[tuple[int, int, int, int]] | None: The parsed boxes, or ``None``
        when no well-formed box is present.
    """
    boxes = [(int(m[0]), int(m[1]), int(m[2]), int(m[3])) for m in _BBOX_RE.findall(text or "")]
    return boxes or None


def _parse_doclang_locations(text: str) -> tuple[int, int, int, int] | None:
    """Extract the first four ``<location value="N"/>`` values as a box.

    Args:
        text (str): The answer text (a DocLang ``<text>`` element).

    Returns:
        tuple[int, int, int, int] | None: ``(x1, y1, x2, y2)``, or ``None``
        when fewer than four locations are present.
    """
    values = [int(v) for v in _LOCATION_RE.findall(text or "")]
    return tuple(values[:4]) if len(values) >= 4 else None  # type: ignore[return-value]


def _parse_ints(text: str) -> list[int]:
    """Return all integers found in ``text`` (in order)."""
    return [int(v) for v in re.findall(r"-?\d+", text or "")]


# --------------------------------------------------------------------------- #
# Scoring primitives
# --------------------------------------------------------------------------- #


def _score_bbox_list(pred: list[tuple[int, int, int, int]], gold: list[tuple[int, int, int, int]]) -> float:
    """Mean IoU over gold boxes, each matched to its best unused prediction.

    Args:
        pred (list[tuple[int, int, int, int]]): Predicted boxes.
        gold (list[tuple[int, int, int, int]]): Ground-truth boxes.

    Returns:
        float: Mean IoU in ``[0, 1]``.
    """
    if not gold:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    used = [False] * len(pred)
    total = 0.0
    for g in gold:
        best_i, best_iou = -1, 0.0
        for i, p in enumerate(pred):
            if used[i]:
                continue
            value = _iou(p, g)
            if value > best_iou:
                best_iou, best_i = value, i
        if best_i >= 0:
            used[best_i] = True
            total += best_iou
    return total / len(gold)


def _similarity(a: str, b: str) -> float:
    """Character-level similarity in ``[0, 1]`` (1.0 when identical)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _text_similarity(a: str, b: str) -> float:
    """Case-sensitive text similarity in ``[0, 1]`` with reduced case penalty.

    Whitespace is collapsed on both sides, then the score blends the strict
    (case-sensitive) similarity with the casefolded similarity:

    ``0.5 * strict + 0.5 * casefolded``

    A prediction that differs from the reference only in letter case still
    costs a little (the strict component is not exact) but stays high because
    the casefolded component is exact, while a prediction with a *different
    letter* scores lower than one with the *right letter in the wrong case* —
    e.g. against ``"A letter."`` the prediction ``"a letter."`` scores
    ``0.952`` but ``"b letter."`` scores ``0.905``.

    Args:
        a (str): The predicted text.
        b (str): The reference text.

    Returns:
        float: Similarity in ``[0, 1]``.
    """
    a, b = _collapse_whitespace(a), _collapse_whitespace(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    strict = SequenceMatcher(None, a, b, autojunk=False).ratio()
    casefolded = SequenceMatcher(None, a.casefold(), b.casefold(), autojunk=False).ratio()
    return 0.5 * strict + 0.5 * casefolded


def _score_int(answer: str, gold: str) -> float:
    """Score a predicted integer against a gold integer with near-miss credit."""
    match = re.search(_INT, answer or "")
    if not match:
        return 0.0
    try:
        gold_value = int(gold.strip())
    except ValueError:
        return 0.0
    pred_value = int(match.group(0))
    if pred_value == gold_value:
        return 1.0
    return max(0.0, 1.0 - abs(pred_value - gold_value) / max(1, abs(gold_value)))


def _score_doclang(answer: str, gold: str) -> float:
    """Similarity between canonical DocLang renderings of pred and gold.

    Both sides are parsed into a :class:`DocLang` model and re-serialized to a
    canonical form, so semantically-equal documents compare as identical. A
    prediction that fails to parse scores ``0.0``.

    Args:
        answer (str): The predicted DocLang XML.
        gold (str): The ground-truth DocLang XML.

    Returns:
        float: Similarity in ``[0, 1]``.
    """
    try:
        pred_canon = model_to_doclang_xml(doclang_xml_to_model(answer))
    except Exception:
        return 0.0
    try:
        gold_canon = model_to_doclang_xml(doclang_xml_to_model(gold))
    except Exception:
        gold_canon = gold  # gold is expected to parse; fall back to raw text
    return _similarity(pred_canon, gold_canon)


def _score(task: str, answer: str, gold: str) -> float:
    """Dispatch to the task-specific accuracy scorer.

    Args:
        task (str): The classified task kind.
        answer (str): The extracted final answer.
        gold (str): The ground-truth answer.

    Returns:
        float: Accuracy score in ``[0, 1]``.
    """
    if task == "doclang":
        return _score_doclang(answer, gold)
    if task == "bbox":
        pred = _parse_boxes(answer)
        gold_boxes = _parse_boxes(gold)
        if pred is None or gold_boxes is None:
            return 0.0
        return _score_bbox_list(pred, gold_boxes)
    if task == "bbox_doclang":
        pred = _parse_doclang_locations(answer)
        gold_box = _parse_doclang_locations(gold)
        if pred is None or gold_box is None:
            return 0.0
        return _iou(pred, gold_box)
    if task == "integer":
        return _score_int(answer, gold)
    if task == "ordinal":
        pred_match = re.search(r"\d+", answer or "")
        gold_match = re.search(r"\d+", gold or "")
        if not pred_match or not gold_match:
            return 0.0
        return _score_int(pred_match.group(0), gold_match.group(0))
    if task == "index_list":
        pred = _parse_ints(answer)
        gold_ints = _parse_ints(gold)
        if not gold_ints:
            return 1.0 if not pred else 0.0
        overlap = min(len(pred), len(gold_ints))
        if overlap == 0:
            return 0.0
        matches = sum(1 for i in range(overlap) if pred[i] == gold_ints[i])
        return matches / len(gold_ints)
    # text
    return _text_similarity(answer or "", gold or "")


def _is_well_formed(task: str, answer: str) -> bool:
    """Return ``True`` when ``answer`` is structurally valid for ``task``.

    Args:
        task (str): The classified task kind.
        answer (str): The extracted final answer.

    Returns:
        bool: Whether the answer is well-formed.
    """
    s = (answer or "").strip()
    if task == "doclang":
        try:
            doclang_xml_to_model(s)
            return True
        except Exception:
            return False
    if task == "bbox":
        return _parse_boxes(s) is not None
    if task == "bbox_doclang":
        return _parse_doclang_locations(s) is not None
    if task == "integer":
        return re.fullmatch(_INT, s) is not None
    if task == "ordinal":
        return _ORDINAL_RE.fullmatch(s) is not None
    if task == "index_list":
        return "," in s and re.fullmatch(r"[\d,\s]+", s) is not None
    # text
    return bool(s)


# --------------------------------------------------------------------------- #
# Public reward functions
# --------------------------------------------------------------------------- #


def format_reward(completions, ground_truth, **kwargs) -> list[float | None]:
    """Reward well-formed final answers (1.0) vs. malformed ones (0.0).

    Args:
        completions: Decoded completions (one per generation) — strings or
            lists of message dicts, depending on the prompt format.
        ground_truth (list[str]): Ground-truth answers (one per completion).
        **kwargs: Extra TRL reward kwargs (``trainer_state``, ``log_metric``, ...).

    Returns:
        list[float | None]: One reward per completion.
    """
    scores: list[float | None] = []
    for completion, gold in zip(completions, ground_truth):
        answer = extract_answer(completion)
        scores.append(1.0 if _is_well_formed(_classify(gold), answer) else 0.0)
    return scores


def accuracy_reward(completions, ground_truth, **kwargs) -> list[float | None]:
    """Reward the closeness of the final answer to the ground truth in ``[0, 1]``.

    Also logs a ``reward/truncated`` metric (1.0 when the completion has no
    ``endthink`` marker, i.e. it likely hit ``max_completion_length`` mid-
    reasoning) so a too-small cap is visible in the training metrics.

    Args:
        completions: Decoded completions (one per generation) — strings or
            lists of message dicts, depending on the prompt format.
        ground_truth (list[str]): Ground-truth answers (one per completion).
        **kwargs: Extra TRL reward kwargs (``trainer_state``, ``log_metric``, ...).

    Returns:
        list[float | None]: One reward per completion.
    """
    log_metric = kwargs.get("log_metric")
    scores: list[float | None] = []
    for completion, gold in zip(completions, ground_truth):
        text = _completion_text(completion)
        answer = extract_answer(text)
        scores.append(_score(_classify(gold), answer, gold))
        if log_metric is not None:
            log_metric("reward/truncated", 0.0 if "endthink" in text else 1.0)
    return scores
