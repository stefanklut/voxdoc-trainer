"""Unit tests for the GRPO reward functions in :mod:`src.rewards`.

Covers answer extraction (thinking stripped), task classification by ground-truth
shape, the format reward (well-formed vs. malformed), and the accuracy reward
(exact / partial / wrong) for every task kind the data creators emit.
"""

from __future__ import annotations

from src.rewards import accuracy_reward, extract_answer, format_reward

# A minimal, XSD-valid DocLang document (round-trips through parse/serialize).
DOCLANG_A = '<doclang xmlns="https://www.doclang.ai/ns/v0" version="0.7"><text>Hello world</text></doclang>'
DOCLANG_B = '<doclang xmlns="https://www.doclang.ai/ns/v0" version="0.7"><text>Goodbye world</text></doclang>'

BBOX = "[10, 20, 100, 200]"
BBOX_LIST = "[[10, 20, 100, 200], [30, 40, 300, 400]]"
BBOX_DOCLANG = '<text><location value="1"/><location value="2"/><location value="3"/><location value="4"/>hi</text>'


def _wrap(answer: str, reasoning: str = "let me think") -> str:
    """Build a completion in the Qwen3.5 completion form (reasoning + endthink + answer)."""
    return f"{reasoning}\nendthink\n\n{answer}"


# --------------------------------------------------------------------------- #
# extract_answer
# --------------------------------------------------------------------------- #


def test_extract_answer_strips_thinking() -> None:
    assert extract_answer(_wrap("the answer")) == "the answer"


def test_extract_answer_no_marker_returns_whole() -> None:
    assert extract_answer("just an answer") == "just an answer"


def test_extract_answer_empty() -> None:
    assert extract_answer("") == ""


def test_extract_answer_uses_last_marker() -> None:
    assert extract_answer("a endthink b endthink c") == "c"


def test_extract_answer_truncated_mid_reasoning() -> None:
    # No endthink because the completion hit max_completion_length mid-thought.
    assert extract_answer("I am still thinking about") == "I am still thinking about"


def test_extract_answer_conversational_message_list() -> None:
    # TRL passes conversational completions as a list of message dicts.
    completion = [{"role": "assistant", "content": _wrap("the answer")}]
    assert extract_answer(completion) == "the answer"


def test_extract_answer_conversational_block_content() -> None:
    # Content may be a list of blocks rather than a plain string.
    completion = [{"role": "assistant", "content": [{"type": "text", "text": _wrap("the answer")}]}]
    assert extract_answer(completion) == "the answer"


# --------------------------------------------------------------------------- #
# format_reward
# --------------------------------------------------------------------------- #


def test_format_reward_doclang() -> None:
    completions = [_wrap(DOCLANG_A), _wrap("this is not doclang")]
    assert format_reward(completions=completions, ground_truth=[DOCLANG_A, DOCLANG_A]) == [1.0, 0.0]


def test_format_reward_bbox() -> None:
    completions = [_wrap(BBOX), _wrap("no box here")]
    assert format_reward(completions=completions, ground_truth=[BBOX, BBOX]) == [1.0, 0.0]


def test_format_reward_bbox_list() -> None:
    completions = [_wrap(BBOX_LIST), _wrap("[10, 20]")]
    assert format_reward(completions=completions, ground_truth=[BBOX_LIST, BBOX_LIST]) == [1.0, 0.0]


def test_format_reward_bbox_doclang() -> None:
    completions = [_wrap(BBOX_DOCLANG), _wrap("<text>no locations</text>")]
    assert format_reward(completions=completions, ground_truth=[BBOX_DOCLANG, BBOX_DOCLANG]) == [1.0, 0.0]


def test_format_reward_integer() -> None:
    completions = [_wrap("7"), _wrap("seven")]
    assert format_reward(completions=completions, ground_truth=["7", "7"]) == [1.0, 0.0]


def test_format_reward_ordinal() -> None:
    completions = [_wrap("line 3 of 7"), _wrap("third")]
    assert format_reward(completions=completions, ground_truth=["line 3 of 7", "line 3 of 7"]) == [1.0, 0.0]


def test_format_reward_index_list() -> None:
    completions = [_wrap("2, 1, 3"), _wrap("two one three")]
    assert format_reward(completions=completions, ground_truth=["2, 1, 3", "2, 1, 3"]) == [1.0, 0.0]


def test_format_reward_text() -> None:
    completions = [_wrap("A letter."), _wrap("")]
    assert format_reward(completions=completions, ground_truth=["A letter.", "A letter."]) == [1.0, 0.0]


# --------------------------------------------------------------------------- #
# accuracy_reward
# --------------------------------------------------------------------------- #


def test_accuracy_doclang_exact() -> None:
    scores = accuracy_reward(completions=[_wrap(DOCLANG_A)], ground_truth=[DOCLANG_A])
    assert scores == [1.0]


def test_accuracy_doclang_partial() -> None:
    scores = accuracy_reward(completions=[_wrap(DOCLANG_B)], ground_truth=[DOCLANG_A])
    assert 0.0 < scores[0] < 1.0


def test_accuracy_doclang_unparseable_pred() -> None:
    scores = accuracy_reward(completions=[_wrap("not xml at all")], ground_truth=[DOCLANG_A])
    assert scores == [0.0]


def test_accuracy_bbox_exact() -> None:
    scores = accuracy_reward(completions=[_wrap(BBOX)], ground_truth=[BBOX])
    assert scores == [1.0]


def test_accuracy_bbox_partial() -> None:
    # Same box shifted by 10 in x: overlap but not identical.
    scores = accuracy_reward(completions=[_wrap("[20, 20, 100, 200]")], ground_truth=[BBOX])
    assert 0.0 < scores[0] < 1.0


def test_accuracy_bbox_missing() -> None:
    scores = accuracy_reward(completions=[_wrap("no box")], ground_truth=[BBOX])
    assert scores == [0.0]


def test_accuracy_bbox_list_partial() -> None:
    # One box exact, one box shifted: mean IoU strictly between 0 and 1.
    pred = "[[10, 20, 100, 200], [40, 40, 300, 400]]"
    scores = accuracy_reward(completions=[_wrap(pred)], ground_truth=[BBOX_LIST])
    assert 0.0 < scores[0] < 1.0


def test_accuracy_bbox_doclang_exact() -> None:
    scores = accuracy_reward(completions=[_wrap(BBOX_DOCLANG)], ground_truth=[BBOX_DOCLANG])
    assert scores == [1.0]


def test_accuracy_integer_exact_and_near() -> None:
    exact = accuracy_reward(completions=[_wrap("7")], ground_truth=["7"])
    near = accuracy_reward(completions=[_wrap("8")], ground_truth=["7"])
    assert exact == [1.0]
    assert 0.0 < near[0] < 1.0


def test_accuracy_integer_missing() -> None:
    scores = accuracy_reward(completions=[_wrap("seven")], ground_truth=["7"])
    assert scores == [0.0]


def test_accuracy_ordinal() -> None:
    exact = accuracy_reward(completions=[_wrap("line 3 of 7")], ground_truth=["line 3 of 7"])
    wrong = accuracy_reward(completions=[_wrap("line 5 of 7")], ground_truth=["line 3 of 7"])
    assert exact == [1.0]
    assert 0.0 < wrong[0] < 1.0


def test_accuracy_index_list_partial() -> None:
    # 2 of 3 positions correct.
    scores = accuracy_reward(completions=[_wrap("2, 1, 1")], ground_truth=["2, 1, 3"])
    assert abs(scores[0] - 2.0 / 3.0) < 1e-9


def test_accuracy_text_exact_and_partial() -> None:
    exact = accuracy_reward(completions=[_wrap("A letter.")], ground_truth=["A letter."])
    partial = accuracy_reward(completions=[_wrap("A letter from 1800.")], ground_truth=["A letter."])
    assert exact == [1.0]
    assert 0.0 < partial[0] < 1.0


def test_accuracy_text_case_sensitive_with_reduced_case_penalty() -> None:
    # A case-only difference still costs a little (case-sensitive) but stays
    # high because the casefolded component is exact.
    case_only = accuracy_reward(completions=[_wrap("a letter.")], ground_truth=["A letter."])
    assert 0.9 < case_only[0] < 1.0
    # A different letter scores lower than the same letter in the wrong case.
    wrong_letter = accuracy_reward(completions=[_wrap("b letter.")], ground_truth=["A letter."])
    assert wrong_letter[0] < case_only[0]
    # Whitespace differences are ignored.
    whitespace = accuracy_reward(completions=[_wrap("  A   letter.  ")], ground_truth=["A letter."])
    assert whitespace == [1.0]


def test_accuracy_ignores_thinking_content() -> None:
    # The reasoning is stale/irrelevant; only the final answer is scored.
    completions = [_wrap(BBOX, reasoning="I will guess the box is wrong but here it is")]
    assert accuracy_reward(completions=completions, ground_truth=[BBOX]) == [1.0]


def test_accuracy_logs_truncation_metric() -> None:
    logged: list[tuple[str, float]] = []
    log_metric = lambda name, value: logged.append((name, value))  # noqa: E731
    completions = [_wrap(BBOX), "truncated mid reasoning with no marker"]
    accuracy_reward(completions=completions, ground_truth=[BBOX, BBOX], log_metric=log_metric)
    assert logged == [("reward/truncated", 0.0), ("reward/truncated", 1.0)]


def test_rewards_return_one_score_per_completion() -> None:
    completions = [_wrap(BBOX), _wrap("x"), _wrap(DOCLANG_A)]
    gold = [BBOX, BBOX, DOCLANG_A]
    assert len(format_reward(completions=completions, ground_truth=gold)) == 3
    assert len(accuracy_reward(completions=completions, ground_truth=gold)) == 3


def test_rewards_accept_conversational_completions() -> None:
    # TRL passes conversational completions as lists of message dicts; the
    # rewards must score them identically to the plain-string form.
    completions = [[{"role": "assistant", "content": _wrap(BBOX)}]]
    gold = [BBOX]
    assert format_reward(completions=completions, ground_truth=gold) == [1.0]
    assert accuracy_reward(completions=completions, ground_truth=gold) == [1.0]