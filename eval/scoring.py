"""Strict document evaluation contracts, response parsing, and scoring.

Unlike training rewards, evaluation never guesses a task or coordinate space
from an answer, repairs malformed boxes, or clips character error rates.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from statistics import mean, stdev
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator, model_validator
from scipy.optimize import linear_sum_assignment

from utils.metrics import bbox_iou, character_error_rate, edit_distance, normalize_text

Units = Literal["native", "qwen"]
AnswerFormat = Literal["auto", "boxes", "single_box", "labeled_boxes", "doclang_box", "text"]


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Box(EvaluationModel):
    bbox: tuple[float, float, float, float]
    label: Optional[StrictStr] = None

    @field_validator("bbox", mode="before")
    @classmethod
    def validate_coordinates(cls, coordinates: Any) -> Any:
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
            raise ValueError("geometry: bbox must contain exactly four numbers")
        try:
            finite_numbers = all(type(value) in (int, float) and math.isfinite(value) for value in coordinates)
        except OverflowError:
            finite_numbers = False
        if not finite_numbers:
            raise ValueError("geometry: coordinates must be finite numbers, not booleans or strings")
        if coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]:
            raise ValueError("geometry: bbox must have positive width and height")
        return coordinates


class Sample(EvaluationModel):
    sample_id: StrictStr = Field(min_length=1)
    document_id: Optional[StrictStr] = None
    task: Literal["bbox", "labeled_bbox", "transcription"]
    prompt: StrictStr = Field(min_length=1)
    images: list[StrictStr] = Field(default_factory=list)
    messages: Optional[list[dict[str, Any]]] = None
    width: Optional[StrictInt] = Field(default=None, gt=0)
    height: Optional[StrictInt] = Field(default=None, gt=0)
    gt_units: Units = "native"
    output_format: AnswerFormat = "auto"
    boxes: Optional[list[Box]] = None
    text: Optional[StrictStr] = None
    label_vocabulary: Optional[list[StrictStr]] = None

    @model_validator(mode="after")
    def validate_task(self) -> Sample:
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height must be supplied together")
        expected: dict[str, AnswerFormat] = {"bbox": "boxes", "labeled_bbox": "labeled_boxes", "transcription": "text"}
        if self.output_format == "auto":
            self.output_format = expected[self.task]
        if self.task == "transcription":
            if self.text is None or self.boxes is not None or self.output_format != "text":
                raise ValueError("transcription requires text GT and the text output format, without boxes")
            return self
        if self.boxes is None or self.text is not None:
            raise ValueError("bbox tasks require boxes GT, without transcription text")
        if self.gt_units == "native" and self.width is None:
            raise ValueError("native bbox GT requires annotation width and height")
        if self.task == "labeled_bbox" and self.output_format != "labeled_boxes":
            raise ValueError("labeled_bbox requires labeled_boxes output")
        if self.task == "bbox" and self.output_format not in ("boxes", "single_box", "doclang_box"):
            raise ValueError("bbox requires boxes, single_box or doclang_box output")
        if self.output_format in ("single_box", "doclang_box") and len(self.boxes) != 1:
            raise ValueError("single-box formats require exactly one GT box")
        for box in self.boxes:
            check_bounds(box, self.gt_units, self.width, self.height)
            if self.task == "labeled_bbox" and not box.label:
                raise ValueError("every GT box needs a nonempty label")
            if self.task == "bbox" and box.label is not None:
                raise ValueError("use labeled_bbox to evaluate labels")
            if self.label_vocabulary is not None and box.label not in self.label_vocabulary:
                raise ValueError("GT label is not in label_vocabulary")
        return self


class Variant(EvaluationModel):
    name: StrictStr = Field(min_length=1)
    units: Units = "native"
    prompt_template: StrictStr = "{question}"
    answer_extraction: Literal["none", "endthink", "think_tag"] = "none"


class ResponseError(ValueError):
    """An invalid response with a stable machine-readable failure category."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


def check_bounds(box: Box, units: Units, width: Optional[int], height: Optional[int]) -> None:
    """Validate declared coordinate bounds without clamping or guessing units."""
    if units == "qwen":
        maximum_x, maximum_y = 1000, 1000
        if any(not value.is_integer() for value in box.bbox):
            raise ValueError("geometry: qwen coordinates must be integers")
    else:
        if width is None or height is None:
            raise ValueError("native coordinates require annotation dimensions")
        maximum_x, maximum_y = width, height
    if not (0 <= box.bbox[0] < box.bbox[2] <= maximum_x and 0 <= box.bbox[1] < box.bbox[3] <= maximum_y):
        raise ValueError("bounds: box is outside its declared coordinate space")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(text: str) -> Any:
    """Decode one complete JSON value, rejecting duplicate keys and nonfinite numbers."""

    def reject_constant(value: str) -> Any:
        raise ValueError(f"nonfinite JSON value: {value}")

    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=reject_constant)


def extract_final_answer(raw: str, mode: str) -> str:
    """Extract only explicitly configured reasoning delimiters; preserve text whitespace."""
    if mode == "none":
        return raw
    marker = "endthink" if mode == "endthink" else "</think>"
    if marker not in raw:
        raise ResponseError("answer_marker", f"expected final-answer marker {marker!r}")
    answer = raw.rsplit(marker, 1)[1]
    for separator in ("\r\n\r\n", "\n\n", "\r\n", "\n"):
        if answer.startswith(separator):
            return answer[len(separator) :]
    return answer


def parse_prediction(answer: str, sample: Sample, units: Units) -> list[Box] | str:
    """Parse and validate the entire response against an explicit task contract."""
    if sample.task == "transcription":
        return answer
    if not answer.strip():
        raise ResponseError("empty_response", "no structured answer")
    try:
        if sample.output_format == "doclang_box":
            if "<!" in answer or "<?" in answer:
                raise ValueError("XML declarations and entities are not accepted")
            root = ET.fromstring(answer)
            if root.tag != "text" or root.attrib or len(root) != 4:
                raise ValueError("expected one text element with exactly four locations")
            if any(child.tag != "location" or set(child.attrib) != {"value"} or len(child) or child.text for child in root):
                raise ValueError("expected empty location elements with only value attributes")
            data = [[int(child.attrib["value"]) for child in root]]
        else:
            data = strict_json(answer)
            if sample.output_format == "single_box":
                data = [data]
        if not isinstance(data, list):
            raise ValueError("expected a JSON list")
        if sample.output_format in ("single_box", "doclang_box") and len(data) != 1:
            raise ValueError("expected exactly one box")
        boxes = []
        for item in data:
            if sample.output_format == "labeled_boxes":
                if not isinstance(item, dict) or set(item) != {"bbox", "label"}:
                    raise ValueError("each labeled box must contain exactly bbox and label")
                box = Box.model_validate(item)
                if not box.label:
                    raise ValueError("label must be nonempty")
                if sample.label_vocabulary is not None and box.label not in sample.label_vocabulary:
                    raise ResponseError("unknown_label", f"unknown label: {box.label!r}")
            else:
                box = Box.model_validate({"bbox": item})
            check_bounds(box, units, sample.width, sample.height)
            boxes.append(box)
        return boxes
    except ResponseError:
        raise
    except (ValueError, TypeError, ET.ParseError) as error:
        message = str(error)
        category = "bounds" if "bounds:" in message else "geometry" if "geometry:" in message else "schema"
        if isinstance(error, (json.JSONDecodeError, ET.ParseError)):
            category = "syntax"
        raise ResponseError(category, message) from error


def validate_variant(sample: Sample, variant: Variant) -> None:
    """Reject coordinate comparisons that cannot be grounded in exact native GT."""
    if sample.task != "transcription" and sample.gt_units != variant.units:
        if sample.gt_units != "native" or sample.width is None:
            raise ValueError("cross-unit comparison requires exact native GT and annotation dimensions")


def _to_reference(box: Box, sample: Sample, units: Units) -> Box:
    if units == sample.gt_units:
        return box
    if sample.width is None or sample.height is None:
        raise ValueError("coordinate conversion requires annotation dimensions")
    scale_x, scale_y = sample.width / 1000, sample.height / 1000
    return Box(
        bbox=(box.bbox[0] * scale_x, box.bbox[1] * scale_y, box.bbox[2] * scale_x, box.bbox[3] * scale_y), label=box.label
    )


def _box_metrics(predicted: list[Box], gold: list[Box], threshold: float, valid: bool, labeled: bool) -> dict[str, Any]:
    assigned: list[tuple[int, int, float]] = []
    if predicted and gold:
        overlaps = [[bbox_iou(reference.bbox, prediction.bbox) for prediction in predicted] for reference in gold]
        rows, columns = linear_sum_assignment(overlaps, maximize=True)
        assigned = [(int(row), int(column), overlaps[row][column]) for row, column in zip(rows, columns)]
    eligible = [(row, column) for row, column, overlap in assigned if overlap >= threshold]
    denominator = max(len(predicted), len(gold))
    score = sum(overlap for _, _, overlap in assigned) / denominator if denominator else 1.0
    return {
        "iou": score if valid else 0.0,
        "gt_boxes": len(gold),
        "predicted_boxes": len(predicted),
        "true_positives": len(eligible),
        "label_total": len(gold) if labeled else 0,
        "label_correct": sum(predicted[column].label == gold[row].label for row, column in eligible) if labeled else 0,
        "label_matched": len(eligible) if labeled else 0,
        "matches": [{"gt": row, "prediction": column, "iou": overlap} for row, column, overlap in assigned],
    }


def _text_metrics(prediction: str, reference: str) -> dict[str, Any]:
    normalized_prediction, normalized_reference = normalize_text(prediction), normalize_text(reference)
    return {
        "cer": character_error_rate(prediction, reference),
        "edits": edit_distance(prediction, reference),
        "reference_characters": len(reference),
        "normalized_cer": character_error_rate(normalized_prediction, normalized_reference),
        "normalized_edits": edit_distance(normalized_prediction, normalized_reference),
        "normalized_reference_characters": len(normalized_reference),
    }


def score_response(
    sample: Sample,
    variant: Variant,
    raw_response: Optional[str],
    *,
    finish_reason: str = "stop",
    transport_error: Optional[str] = None,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Score one attempt, retaining failures and penalizing invalid output.

    Args:
        sample (Sample): Validated GT and task contract.
        variant (Variant): Explicit output units and answer extraction mode.
        raw_response (Optional[str]): Unmodified assistant content.
        finish_reason (str): Endpoint finish reason; only stop is complete.
        transport_error (Optional[str]): Transport error, distinct from bad output.
        iou_threshold (float): Detection/label-match threshold in (0, 1].

    Returns:
        dict: Status, parsed prediction, and unclipped metric counts and scores.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    validate_variant(sample, variant)
    status, error, answer = "ok", None, None
    parsed: list[Box] | str = "" if sample.task == "transcription" else []
    try:
        if transport_error is not None:
            raise ResponseError("transport_error", transport_error)
        if raw_response is None:
            raise ResponseError("empty_response", "missing assistant content")
        answer = extract_final_answer(raw_response, variant.answer_extraction)
        if finish_reason != "stop":
            raise ResponseError("truncated" if finish_reason == "length" else "incomplete", finish_reason)
        if not answer and (sample.task != "transcription" or sample.text):
            raise ResponseError("empty_response", "empty assistant answer")
        parsed = parse_prediction(answer, sample, variant.units)
    except ResponseError as failure:
        status, error = failure.category, str(failure)
    valid = status == "ok"
    if sample.task == "transcription":
        metrics = _text_metrics(parsed if isinstance(parsed, str) else "", sample.text or "")
        if not valid and answer is not None:
            metrics["partial_cer"] = character_error_rate(answer, sample.text or "")
        prediction: Any = parsed
    else:
        boxes = [_to_reference(box, sample, variant.units) for box in parsed] if isinstance(parsed, list) else []
        metrics = _box_metrics(boxes, sample.boxes or [], iou_threshold, valid, sample.task == "labeled_bbox")
        prediction = [box.model_dump(mode="json") for box in boxes]
    return {"status": status, "valid": valid, "error": error, "answer": answer, "prediction": prediction, "metrics": metrics}


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def _reduce_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [record["score"]["metrics"] for record in records]
    result: dict[str, Any] = {"attempts": len(records)}
    if not metrics:
        return result
    if "iou" in metrics[0]:
        gold = sum(item["gt_boxes"] for item in metrics)
        predicted = sum(item["predicted_boxes"] for item in metrics)
        true_positives = sum(item["true_positives"] for item in metrics)
        result.update(
            mean_iou=mean(item["iou"] for item in metrics),
            gt_boxes=gold,
            predicted_boxes=predicted,
            true_positives=true_positives,
            precision=_ratio(true_positives, predicted),
            recall=_ratio(true_positives, gold),
            f1=_ratio(2 * true_positives, gold + predicted),
        )
        labeled = sum(item["label_total"] for item in metrics)
        correct = sum(item["label_correct"] for item in metrics)
        matched = sum(item["label_matched"] for item in metrics)
        result.update(
            label_total=labeled,
            label_correct=correct,
            label_accuracy=_ratio(correct, labeled),
            matched_label_accuracy=_ratio(correct, matched),
        )
    else:
        result.update(
            macro_cer=mean(item["cer"] for item in metrics),
            macro_normalized_cer=mean(item["normalized_cer"] for item in metrics),
            empty_references=sum(item["reference_characters"] == 0 for item in metrics),
        )
        for prefix in ("", "normalized_"):
            edits = sum(item[f"{prefix}edits"] for item in metrics)
            characters = sum(item[f"{prefix}reference_characters"] for item in metrics)
            result.update(
                {
                    f"{prefix}edits": edits,
                    f"{prefix}reference_characters": characters,
                    f"corpus_{prefix}cer": _ratio(edits, characters),
                }
            )
    return result


def summarize(records: list[dict[str, Any]], variant_order: Optional[list[str]] = None) -> dict[str, Any]:
    """Aggregate all attempts and valid-only scores, per-run variability, and paired deltas."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["model"], record["variant"], record["task"])].append(record)
    summaries = []
    for (model, variant, task), attempts in sorted(groups.items()):
        by_run: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for attempt in attempts:
            by_run[attempt["run"]].append(attempt)
        runs = [{"run": run, **_reduce_metrics(items)} for run, items in sorted(by_run.items())]
        variation = {}
        for metric in ("mean_iou", "label_accuracy", "corpus_cer", "corpus_normalized_cer", "macro_cer"):
            values = [run[metric] for run in runs if run.get(metric) is not None]
            if values:
                variation[metric] = {
                    "mean": mean(values),
                    "std": stdev(values) if len(values) > 1 else None,
                    "runs": len(values),
                }
        valid = [item for item in attempts if item["score"]["valid"]]
        summaries.append(
            {
                "model": model,
                "variant": variant,
                "task": task,
                "status_counts": dict(Counter(item["score"]["status"] for item in attempts)),
                "valid_rate": len(valid) / len(attempts),
                "all_attempts": _reduce_metrics(attempts),
                "valid_only": _reduce_metrics(valid),
                "runs": runs,
                "run_variation": variation,
            }
        )
    order = variant_order or sorted({record["variant"] for record in records})
    paired = []
    if order:
        baseline = order[0]
        for group in summaries:
            if group["variant"] == baseline:
                continue
            reference = next(
                (
                    item
                    for item in summaries
                    if item["model"] == group["model"] and item["task"] == group["task"] and item["variant"] == baseline
                ),
                None,
            )
            if reference is None:
                continue
            current_attempts = groups[(group["model"], group["variant"], group["task"])]
            reference_attempts = groups[(group["model"], baseline, group["task"])]
            paired_runs = []
            for run in sorted({item["run"] for item in current_attempts} & {item["run"] for item in reference_attempts}):
                current = {item["sample_id"]: item for item in current_attempts if item["run"] == run}
                previous = {item["sample_id"]: item for item in reference_attempts if item["run"] == run}
                common = sorted(current.keys() & previous.keys())
                if common:
                    paired_runs.append(
                        (
                            run,
                            len(common),
                            _reduce_metrics([current[key] for key in common]),
                            _reduce_metrics([previous[key] for key in common]),
                        )
                    )
            for metric in ("mean_iou", "label_accuracy", "corpus_cer", "corpus_normalized_cer"):
                differences = []
                for run, count, current_run, reference_run in paired_runs:
                    if current_run.get(metric) is not None and reference_run.get(metric) is not None:
                        differences.append(
                            {"run": run, "paired_samples": count, "delta": current_run[metric] - reference_run[metric]}
                        )
                if differences:
                    values = [item["delta"] for item in differences]
                    paired.append(
                        {
                            "model": group["model"],
                            "task": group["task"],
                            "baseline": baseline,
                            "variant": group["variant"],
                            "metric": metric,
                            "runs": differences,
                            "mean_delta": mean(values),
                            "std_delta": stdev(values) if len(values) > 1 else None,
                        }
                    )
    return {"groups": summaries, "paired_differences": paired}
