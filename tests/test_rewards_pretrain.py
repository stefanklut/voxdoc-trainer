"""Unit tests for the GRPO pretrain reward functions in :mod:`src.rewards_pretrain`.

Covers the CER reward: exact match, partial credit, wrong answer, case and
whitespace sensitivity (raw) vs. insensitivity (normalized), thinking stripped
before scoring, and clamping when the prediction is far worse than the
reference (CER > 1).
"""

from __future__ import annotations

import pytest

from src.config import GRPOConfig, RunConfig
from src.rewards import accuracy_reward, format_reward
from src.rewards_pretrain import cer_reward, cer_reward_normalized
from src.train_grpo import select_reward_funcs


def _wrap(answer: str, reasoning: str = "let me think") -> str:
    """Build a completion in the Qwen3.5 completion form (reasoning + endthink + answer)."""
    return f"{reasoning}\nendthink\n\n{answer}"


# --------------------------------------------------------------------------- #
# cer_reward
# --------------------------------------------------------------------------- #


def test_cer_exact_match() -> None:
    scores = cer_reward(completions=[_wrap("A letter.")], ground_truth=["A letter."])
    assert scores == [1.0]


def test_cer_partial_credit() -> None:
    # One character differs: strictly between 0 and 1.
    scores = cer_reward(completions=[_wrap("A lettur.")], ground_truth=["A letter."])
    assert 0.0 < scores[0] < 1.0


def test_cer_monotonic_in_edits() -> None:
    # More edits -> lower reward.
    one_edit = cer_reward(completions=[_wrap("A lettur.")], ground_truth=["A letter."])
    two_edits = cer_reward(completions=[_wrap("A lettzz.")], ground_truth=["A letter."])
    assert one_edit[0] > two_edits[0]


def test_cer_wrong_answer_near_zero() -> None:
    # A completely different, short answer scores near 0 (but >= 0).
    scores = cer_reward(completions=[_wrap("zzz")], ground_truth=["A letter from 1800."])
    assert 0.0 <= scores[0] < 0.5


def test_cer_case_sensitive() -> None:
    # Raw CER is case-sensitive: a case difference is a real edit.
    scores = cer_reward(completions=[_wrap("a letter.")], ground_truth=["A Letter."])
    assert 0.0 < scores[0] < 1.0


def test_cer_whitespace_sensitive() -> None:
    # Raw CER counts whitespace: extra spaces are real edits.
    scores = cer_reward(completions=[_wrap("  A   letter.  ")], ground_truth=["A letter."])
    assert 0.0 < scores[0] < 1.0


def test_cer_normalized_case_insensitive() -> None:
    # The normalized variant casefolds both sides, so case earns no penalty.
    scores = cer_reward_normalized(completions=[_wrap("a letter.")], ground_truth=["A Letter."])
    assert scores == [1.0]


def test_cer_normalized_whitespace_insensitive() -> None:
    # The normalized variant collapses whitespace on both sides.
    scores = cer_reward_normalized(completions=[_wrap("  A   letter.  ")], ground_truth=["A letter."])
    assert scores == [1.0]


def test_cer_normalized_partial_credit() -> None:
    # A real character difference still costs credit in the normalized variant.
    scores = cer_reward_normalized(completions=[_wrap("A lettur.")], ground_truth=["A letter."])
    assert 0.0 < scores[0] < 1.0


def test_cer_strips_thinking() -> None:
    # The reasoning is irrelevant; only the final answer is scored.
    completions = [_wrap("A letter.", reasoning="I will guess the answer is wrong but here it is")]
    assert cer_reward(completions=completions, ground_truth=["A letter."]) == [1.0]


def test_cer_clamps_when_prediction_much_longer() -> None:
    # A very long wrong prediction gives CER > 1, which clamps the reward to 0.
    scores = cer_reward(completions=[_wrap("x" * 100)], ground_truth=["A letter."])
    assert scores == [0.0]


def test_cer_empty_prediction_vs_nonempty_reference() -> None:
    # Empty answer against a non-empty reference: CER = 1.0 -> reward 0.0.
    scores = cer_reward(completions=[_wrap("")], ground_truth=["A letter."])
    assert scores == [0.0]


def test_cer_empty_both() -> None:
    # Both empty: CER = 0.0 -> reward 1.0.
    scores = cer_reward(completions=[_wrap("")], ground_truth=[""])
    assert scores == [1.0]


def test_cer_conversational_message_list() -> None:
    # TRL passes conversational completions as a list of message dicts.
    completion = [{"role": "assistant", "content": _wrap("A letter.")}]
    assert cer_reward(completions=[completion], ground_truth=["A letter."]) == [1.0]


def test_cer_logs_truncation_metric() -> None:
    logged: list[tuple[str, float]] = []
    log_metric = lambda name, value: logged.append((name, value))  # noqa: E731
    completions = [_wrap("A letter."), "truncated mid reasoning with no marker"]
    cer_reward(completions=completions, ground_truth=["A letter.", "A letter."], log_metric=log_metric)
    assert logged == [("reward/truncated", 0.0), ("reward/truncated", 1.0)]


# --------------------------------------------------------------------------- #
# select_reward_funcs (stage -> reward selection)
# --------------------------------------------------------------------------- #


def _config(stage: str, reward_weights: list[float], cer_metric: str = "raw") -> RunConfig:
    """Build a RunConfig with the given GRPO stage, weights, and CER metric."""
    return RunConfig(grpo=GRPOConfig(stage=stage, reward_weights=reward_weights, cer_metric=cer_metric))


def test_select_reward_funcs_pretrain() -> None:
    funcs = select_reward_funcs(_config("pretrain", [1.0]))
    assert funcs == [cer_reward]


def test_select_reward_funcs_pretrain_normalized() -> None:
    funcs = select_reward_funcs(_config("pretrain", [1.0], cer_metric="normalized"))
    assert funcs == [cer_reward_normalized]


def test_select_reward_funcs_unknown_cer_metric() -> None:
    with pytest.raises(ValueError, match="Unknown grpo.cer_metric"):
        select_reward_funcs(_config("pretrain", [1.0], cer_metric="bogus"))


def test_select_reward_funcs_finetune() -> None:
    funcs = select_reward_funcs(_config("finetune", [0.2, 0.8]))
    assert funcs == [format_reward, accuracy_reward]


def test_select_reward_funcs_unknown_stage() -> None:
    with pytest.raises(ValueError, match="Unknown grpo.stage"):
        select_reward_funcs(_config("bogus", [1.0]))


def test_select_reward_funcs_weight_count_mismatch() -> None:
    # Pretrain selects one reward function but two weights are provided.
    with pytest.raises(ValueError, match="one weight per reward function"):
        select_reward_funcs(_config("pretrain", [0.5, 0.5]))
