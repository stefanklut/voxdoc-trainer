"""Tests for strict evaluation, paired experiments, and offline rescoring."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from datasets import Dataset
from openai.types.chat import ChatCompletion
from PIL import Image

from eval.benchmarks import evaluate
from eval.scoring import Box, Sample, Variant, score_response, strict_json, summarize
from eval.runner import (
    DataSource,
    ExperimentConfig,
    build_request_messages,
    load_config,
    load_samples,
    run_experiment,
    score_prediction_file,
    score_saved_run,
)
from utils.image_loading import ImageNotFoundError
from utils.metrics import character_error_rate


def bbox_sample(**changes: object) -> Sample:
    values = {
        "sample_id": "page-1",
        "task": "bbox",
        "prompt": "Find the text.",
        "width": 200,
        "height": 400,
        "boxes": [{"bbox": [20, 40, 100, 200]}],
    }
    return Sample.model_validate({**values, **changes})


def test_native_and_grid_predictions_share_reference_space() -> None:
    sample = bbox_sample()
    native = score_response(sample, Variant(name="pixels"), "[[20,40,100,200]]")
    grid = score_response(sample, Variant(name="grid", units="qwen"), "[[100,100,500,500]]")
    assert native["metrics"]["iou"] == grid["metrics"]["iou"] == 1
    assert grid["prediction"] == native["prediction"]


@pytest.mark.parametrize(
    "answer,category",
    [
        ("A box: [[20,40,100,200]]", "syntax"),
        ("[[20,40,100,200]] trailing", "syntax"),
        ("```json\n[[20,40,100,200]]\n```", "syntax"),
        ("[[20,40,100]]", "geometry"),
        ("[[20,40,20,200]]", "geometry"),
        ("[[true,40,100,200]]", "geometry"),
        ("[[0,0,Infinity,200]]", "schema"),
        ("[[20,40,201,200]]", "bounds"),
        ("", "empty_response"),
    ],
)
def test_malformed_predictions_are_failures(answer: str, category: str) -> None:
    result = score_response(bbox_sample(), Variant(name="pixels"), answer)
    assert result["status"] == category
    assert result["metrics"]["iou"] == 0
    assert not result["valid"]


def test_extra_predictions_penalize_iou() -> None:
    score = score_response(bbox_sample(), Variant(name="pixels"), "[[20,40,100,200],[20,40,100,200]]")
    assert score["valid"]
    assert score["metrics"]["iou"] == 0.5
    assert score["metrics"]["true_positives"] == 1


def test_label_does_not_influence_spatial_matching() -> None:
    sample = bbox_sample(
        task="labeled_bbox", boxes=[{"bbox": [20, 40, 100, 200], "label": "text"}], label_vocabulary=["text", "table"]
    )
    result = score_response(sample, Variant(name="pixels"), '[{"bbox":[20,40,100,200],"label":"table"}]')
    assert result["metrics"]["iou"] == 1
    assert result["metrics"]["label_correct"] == 0
    assert result["metrics"]["label_total"] == 1
    missing = score_response(sample, Variant(name="pixels"), "[]")
    assert missing["valid"]
    assert missing["metrics"]["label_total"] == 1
    unknown = score_response(sample, Variant(name="pixels"), '[{"bbox":[20,40,100,200],"label":"other"}]')
    assert unknown["status"] == "unknown_label"


def test_numeric_transcription_is_text_and_raw_whitespace_is_preserved() -> None:
    sample = Sample(sample_id="text", task="transcription", prompt="Read.", text=" 123\n")
    result = score_response(sample, Variant(name="text"), "123")
    assert result["metrics"]["cer"] == 2 / 5
    assert result["metrics"]["normalized_cer"] == 0
    assert character_error_rate("abcdef", "a") == 5


def test_truncated_text_is_not_silently_counted_as_complete() -> None:
    sample = Sample(sample_id="text", task="transcription", prompt="Read.", text="hello")
    result = score_response(sample, Variant(name="text"), "hello", finish_reason="length")
    assert result["status"] == "truncated"
    assert result["metrics"]["cer"] == 1
    assert result["metrics"]["partial_cer"] == 0
    missing = score_response(sample, Variant(name="text", answer_extraction="endthink"), "hello")
    assert missing["status"] == "answer_marker"


def test_empty_sets_and_transcriptions() -> None:
    assert score_response(bbox_sample(boxes=[]), Variant(name="pixels"), "[]")["metrics"]["iou"] == 1
    sample = Sample(sample_id="text", task="transcription", prompt="Read.", text="")
    assert score_response(sample, Variant(name="text"), "")["metrics"]["cer"] == 0
    assert not score_response(sample, Variant(name="text"), None)["valid"]


def test_ground_truth_is_validated_before_generation() -> None:
    with pytest.raises(ValueError, match="positive width"):
        bbox_sample(boxes=[{"bbox": [20, 40, 10, 200]}])
    with pytest.raises(ValueError, match="requires annotation"):
        bbox_sample(width=None, height=None)
    with pytest.raises(ValueError, match="cross-unit"):
        score_response(bbox_sample(gt_units="qwen", width=None, height=None), Variant(name="pixels"), "[]")
    with pytest.raises(ValueError, match="duplicate"):
        strict_json('{"bbox": [], "bbox": []}')


def test_single_box_and_doclang_are_explicit_formats() -> None:
    single = score_response(bbox_sample(output_format="single_box"), Variant(name="pixels"), "[20,40,100,200]")
    assert single["metrics"]["iou"] == 1
    xml = '<text><location value="20"/><location value="40"/><location value="100"/><location value="200"/>hello</text>'
    assert score_response(bbox_sample(output_format="doclang_box"), Variant(name="pixels"), xml)["valid"]
    assert not score_response(bbox_sample(), Variant(name="pixels"), xml)["valid"]


def test_summary_retains_failures_and_run_variability() -> None:
    sample = bbox_sample()
    records = []
    for variant in (Variant(name="pixels"), Variant(name="grid", units="qwen")):
        for run in range(2):
            score = score_response(
                sample, variant, "[]" if run else ("[[20,40,100,200]]" if variant.units == "native" else "[[100,100,500,500]]")
            )
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "model": "model",
                    "variant": variant.name,
                    "task": sample.task,
                    "run": run,
                    "score": score,
                }
            )
    report = summarize(records, ["pixels", "grid"])
    assert report["groups"][0]["run_variation"]["mean_iou"]["mean"] == 0.5
    assert report["paired_differences"][0]["mean_delta"] == 0


def test_transport_failure_is_distinct() -> None:
    result = score_response(bbox_sample(), Variant(name="pixels"), None, transport_error="connection refused")
    assert result["status"] == "transport_error"
    assert result["metrics"]["iou"] == 0


def test_box_rejects_nonfinite_and_non_numeric_coordinates() -> None:
    for invalid in ([0, 0, float("nan"), 1], [0, 0, "1", 1], [False, 0, 1, 1]):
        with pytest.raises(ValueError):
            Box.model_validate({"bbox": invalid})


class FakeEndpoint:
    def __init__(self) -> None:
        self.chat = self
        self.completions = self
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> ChatCompletion:
        self.requests.append(request)
        if len(self.requests) == 1:
            content = "not JSON"
        elif len(self.requests) == 2:
            raise ConnectionError("offline")
        else:
            text = request["messages"][0]["content"][-1]["text"]
            content = "[[100,100,500,500]]" if "1000 by 1000" in text else "[[20,40,100,200]]"
        return ChatCompletion.model_validate(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "model",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            }
        )


def experiment(tmp_path: Path) -> ExperimentConfig:
    manifest = tmp_path / "samples.jsonl"
    rows = [bbox_sample(sample_id=f"page-{index}").model_dump(mode="json") for index in range(2)]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return ExperimentConfig.model_validate(
        {"data": {"path": str(manifest)}, "model": "model", "output_dir": str(tmp_path / "run"), "comparison": "grid_vs_native"}
    )


def test_paired_runs_resume_and_offline_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = experiment(tmp_path)
    endpoint = FakeEndpoint()
    report = run_experiment(config, endpoint)
    assert len(endpoint.requests) == 20
    assert report["planned_attempts"] == report["completed_attempts"] == 20
    assert {request["seed"] for request in endpoint.requests} == set(range(5))
    assert sum(group["status_counts"].get("transport_error", 0) for group in report["groups"]) == 1
    assert sum(group["status_counts"].get("syntax", 0) for group in report["groups"]) == 1
    assert run_experiment(config, endpoint) == report
    assert len(endpoint.requests) == 20
    monkeypatch.setattr("eval.runner.OpenAI", lambda **kwargs: pytest.fail("offline scoring initialized a client"))
    Path(config.data.path).unlink()
    assert score_saved_run(Path(config.output_dir)) == report
    records = [json.loads(line) for line in (Path(config.output_dir) / "attempts.jsonl").read_text().splitlines()]
    assert records[0]["raw_response"] == "not JSON"
    assert records[1]["transport_error"].startswith("ConnectionError")


def test_resume_rejects_changed_sampling_configuration(tmp_path: Path) -> None:
    config = experiment(tmp_path)
    run_experiment(config, FakeEndpoint())
    config.temperature = 0.8
    with pytest.raises(ValueError, match="resume refused"):
        run_experiment(config, FakeEndpoint())


def test_external_predictions_do_not_require_images_or_credentials(tmp_path: Path) -> None:
    config = experiment(tmp_path)
    rows = [bbox_sample(sample_id="external", images=["missing-image.png"]).model_dump(mode="json")]
    Path(config.data.path).write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"sample_id": "external", "variant": "grid", "run": 0, "raw_response": "[[100,100,500,500]]"}) + "\n",
        encoding="utf-8",
    )
    report = score_prediction_file(config, predictions, tmp_path / "offline")
    assert report["completed_attempts"] == 1
    assert report["groups"][0]["all_attempts"]["mean_iou"] == 1


def test_config_paths_and_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model: model\ndata:\n  path: samples.jsonl\noutput_dir: run\n", encoding="utf-8")
    config = load_config(path)
    assert config.data.path == str(tmp_path / "samples.jsonl")
    with pytest.raises(ValueError, match="Extra inputs"):
        ExperimentConfig.model_validate({"data": {"path": "a"}, "model": "model", "output_dir": "b", "temperatue": 1})


def test_optimal_matching_beats_gt_order_greedy() -> None:
    sample = bbox_sample(boxes=[{"bbox": [0, 0, 10, 10]}, {"bbox": [5, 0, 15, 10]}])
    result = score_response(sample, Variant(name="pixels"), "[[2,0,12,10],[0,0,5,10]]")
    assert result["metrics"]["iou"] == pytest.approx((0.5 + 7 / 13) / 2)
    assert [(match["gt"], match["prediction"]) for match in result["metrics"]["matches"]] == [(0, 1), (1, 0)]


def test_iou_threshold_is_inclusive() -> None:
    result = score_response(bbox_sample(), Variant(name="pixels"), "[[20,40,60,200]]")
    assert result["metrics"]["iou"] == 0.5
    assert result["metrics"]["true_positives"] == 1


def test_quantized_grid_may_differ_from_exact_pixels() -> None:
    sample = bbox_sample(width=3, height=3, boxes=[{"bbox": [1, 0, 2, 1]}])
    result = score_response(sample, Variant(name="grid", units="qwen"), "[[333,0,667,333]]")
    assert 0.99 < result["metrics"]["iou"] < 1
    invalid = score_response(sample, Variant(name="grid", units="qwen"), "[[333.5,0,667,333]]")
    assert invalid["status"] == "geometry"


def test_large_numeric_output_is_a_failure_not_an_exception() -> None:
    result = score_response(bbox_sample(), Variant(name="pixels"), "[[0,0," + "9" * 400 + ",100]]")
    assert result["status"] == "geometry"


def test_thinking_extraction_preserves_answer_whitespace() -> None:
    sample = Sample(sample_id="text", task="transcription", prompt="Read.", text=" hello\n")
    result = score_response(sample, Variant(name="text", answer_extraction="endthink"), "reason\nendthink\n\n hello\n")
    assert result["metrics"]["cer"] == 0
    tagged = score_response(sample, Variant(name="text", answer_extraction="think_tag"), "<think>reason</think>\n\n hello\n")
    assert tagged["metrics"]["cer"] == 0


def test_empty_reference_corpus_is_null_and_insertions_are_retained() -> None:
    sample = Sample(sample_id="empty", task="transcription", prompt="Read.", text="")
    score = score_response(sample, Variant(name="text"), "inserted")
    report = summarize(
        [{"sample_id": sample.sample_id, "model": "model", "variant": "text", "task": sample.task, "run": 0, "score": score}]
    )
    metrics = report["groups"][0]["all_attempts"]
    assert metrics["macro_cer"] == 1
    assert metrics["corpus_cer"] is None
    assert metrics["edits"] == 8


def test_pairing_uses_common_samples_not_unrelated_run_means() -> None:
    records = []
    for variant, sample_id, answer in (
        ("baseline", "shared", "[]"),
        ("other", "shared", "[]"),
        ("other", "unpaired", "[[20,40,100,200]]"),
    ):
        sample = bbox_sample(sample_id=sample_id)
        records.append(
            {
                "sample_id": sample_id,
                "model": "model",
                "variant": variant,
                "task": "bbox",
                "run": 0,
                "score": score_response(sample, Variant(name=variant), answer),
            }
        )
    report = summarize(records, ["baseline", "other"])
    assert report["paired_differences"][0]["mean_delta"] == 0
    assert report["paired_differences"][0]["runs"][0]["paired_samples"] == 1


def hf_dataset(
    tmp_path: Path,
    *,
    reference: str = "123",
    task: str = "transcription",
    images: int = 1,
    placeholders: int = 1,
    **options: Any,
) -> DataSource:
    row = {
        "prompt": [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in range(placeholders)] + [{"type": "text", "text": "Read this document."}],
            }
        ],
        "completion": [{"role": "assistant", "content": [{"type": "text", "text": reference}]}],
        "images": [Image.new("RGB", (8, 8), color=color) for color in ("white", "black")[:images]],
    }
    path = tmp_path / "hf"
    Dataset.from_list([row]).save_to_disk(str(path))
    return DataSource.model_validate({"path": str(path), "format": "hf", "task": task, **options})


def test_hf_adapter_preserves_text_and_image_order(tmp_path: Path) -> None:
    source = hf_dataset(tmp_path, images=2, placeholders=2)
    sample = load_samples(source)[0]
    assert sample.task == "transcription"
    assert sample.text == "123"
    assert len(sample.images) == 2
    messages = build_request_messages(sample, Variant(name="default"), sample.images)
    assert messages[0]["content"][0]["image_url"]["url"] == sample.images[0]
    assert messages[0]["content"][1]["image_url"]["url"] == sample.images[1]
    assert sample.images[0] != sample.images[1]


def test_hf_native_dimensions_are_not_inferred_from_raster(tmp_path: Path) -> None:
    source = hf_dataset(tmp_path, reference="[[20,40,100,200]]", task="bbox", gt_units="native")
    with pytest.raises(ValueError, match="requires annotation"):
        load_samples(source)
    source.width, source.height = 200, 400
    sample = load_samples(source)[0]
    assert sample.width == 200 and sample.height == 400
    assert sample.boxes is not None and sample.boxes[0].bbox == (20, 40, 100, 200)


def test_hf_grid_gt_is_usable_but_cannot_be_inverted_for_native_comparison(tmp_path: Path) -> None:
    source = hf_dataset(tmp_path, reference="[100,100,500,500]", task="bbox", gt_units="qwen", output_format="single_box")
    sample = load_samples(source)[0]
    assert score_response(sample, Variant(name="grid", units="qwen"), "[100,100,500,500]")["metrics"]["iou"] == 1
    config = ExperimentConfig.model_validate(
        {"data": source, "model": "model", "comparison": "grid_vs_native", "output_dir": str(tmp_path / "run")}
    )
    endpoint = FakeEndpoint()
    with pytest.raises(ValueError, match="exact native GT"):
        run_experiment(config, endpoint)
    assert not endpoint.requests


def test_hf_image_placeholder_mismatch_fails_before_api(tmp_path: Path) -> None:
    source = hf_dataset(tmp_path, placeholders=0)
    with pytest.raises(ValueError, match="placeholder count"):
        load_samples(source)


def test_hf_labeled_gt_is_explicit(tmp_path: Path) -> None:
    source = hf_dataset(
        tmp_path,
        reference='[{"bbox":[20,40,100,200],"label":"text"}]',
        task="labeled_bbox",
        gt_units="native",
        width=200,
        height=400,
    )
    sample = load_samples(source)[0]
    assert sample.boxes is not None and sample.boxes[0].label == "text"


def hf_reference_dataset(
    tmp_path: Path,
    *,
    reference: str = "123",
    task: str = "transcription",
    image_root: str | None = None,
    **options: Any,
) -> DataSource:
    """Save a reference-mode HF dataset (image paths, no bytes) and its source.

    Args:
        tmp_path (Path): Base directory.
        reference (str): The ground-truth completion text.
        task (str): The explicit HF task.
        image_root (str | None): Remapping root to record on the data source.
        **options (Any): Extra DataSource fields.

    Returns:
        DataSource: The configured source.
    """
    corpus = tmp_path / "corpus"
    (corpus / "document").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color="white").save(corpus / "document" / "doc.png")
    row = {
        "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Read this document."}]}],
        "completion": [{"role": "assistant", "content": [{"type": "text", "text": reference}]}],
        "images": ["document/doc.png"],
    }
    path = tmp_path / "hf"
    Dataset.from_list([row]).save_to_disk(str(path))
    (path / "image_manifest.json").write_text(
        json.dumps({"version": 1, "image_mode": "reference", "roots": [str(corpus)]}), encoding="utf-8"
    )
    source = {"path": str(path), "format": "hf", "task": task, **options}
    if image_root is not None:
        source["image_root"] = image_root
    return DataSource.model_validate(source)


def test_hf_reference_dataset_reads_image_from_disk(tmp_path: Path) -> None:
    """A dataset storing paths loads the original file instead of rejecting the row."""
    source = hf_reference_dataset(tmp_path)
    sample = load_samples(source)[0]
    assert sample.images == [str((tmp_path / "corpus" / "document" / "doc.png").resolve())]


def test_hf_reference_and_embedded_fingerprints_match(tmp_path: Path) -> None:
    """The same page yields the same fingerprint whether bytes are inlined or referenced."""
    embedded = hf_dataset(tmp_path, images=1)
    referenced = hf_reference_dataset(tmp_path / "ref")
    embedded_run, referenced_run = tmp_path / "embedded_run", tmp_path / "referenced_run"
    for source, output_dir in ((embedded, embedded_run), (referenced, referenced_run)):
        config = ExperimentConfig.model_validate(
            {"data": source, "model": "model", "output_dir": str(output_dir), "variants": [{"name": "default"}]}
        )
        run_experiment(config, FakeEndpoint())
    embedded_meta = json.loads((embedded_run / "metadata.json").read_text())
    referenced_meta = json.loads((referenced_run / "metadata.json").read_text())
    assert embedded_meta["images"]["0"] == referenced_meta["images"]["0"]


def test_hf_reference_image_root_remaps_moved_corpus(tmp_path: Path) -> None:
    """A corpus that moved is found through the configured image root."""
    source = hf_reference_dataset(tmp_path, image_root=str(tmp_path / "moved"))
    corpus = tmp_path / "corpus"
    (tmp_path / "moved" / "document").mkdir(parents=True)
    Image.new("RGB", (8, 8), color="black").save(tmp_path / "moved" / "document" / "doc.png")
    corpus.joinpath("document", "doc.png").unlink()
    sample = load_samples(source)[0]
    assert sample.images == [str((tmp_path / "moved" / "document" / "doc.png").resolve())]


def test_hf_missing_reference_image_reports_searched_dirs(tmp_path: Path) -> None:
    """A stored path with no file behind it names the path and the roots tried."""
    source = hf_reference_dataset(tmp_path)
    (tmp_path / "corpus" / "document" / "doc.png").unlink()
    with pytest.raises(ImageNotFoundError, match="document/doc.png"):
        load_samples(source)


def test_hf_unsupported_image_value_still_rejected(tmp_path: Path) -> None:
    """A non-string, non-image entry is still an error rather than a silent skip."""
    row = {
        "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Read this document."}]}],
        "completion": [{"role": "assistant", "content": [{"type": "text", "text": "123"}]}],
        "images": [1234],
    }
    path = tmp_path / "hf"
    Dataset.from_list([row]).save_to_disk(str(path))
    source = DataSource.model_validate({"path": str(path), "format": "hf", "task": "transcription"})
    with pytest.raises(ValueError, match="decoded PIL images or paths"):
        load_samples(source)


def test_manifest_images_are_hashed_and_raster_size_is_distinct(tmp_path: Path, tiny_image: Path) -> None:
    config = experiment(tmp_path)
    rows = [bbox_sample(images=[tiny_image.name]).model_dump(mode="json")]
    Path(config.data.path).write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    endpoint = FakeEndpoint()
    run_experiment(config, endpoint)
    metadata = json.loads((Path(config.output_dir) / "metadata.json").read_text())
    assert metadata["images"]["page-1"][0]["raster_size"] == [8, 8]
    snapshot = json.loads((Path(config.output_dir) / "samples.jsonl").read_text())
    assert snapshot["width"] == 200
    Image.new("RGB", (8, 8), color="black").save(tiny_image)
    with pytest.raises(ValueError, match="resume refused"):
        run_experiment(config, endpoint)
    tiny_image.unlink()
    assert score_saved_run(Path(config.output_dir))["completed_attempts"] == 10


def test_changed_frozen_gt_is_rejected(tmp_path: Path) -> None:
    config = experiment(tmp_path)
    run_experiment(config, FakeEndpoint())
    snapshot = Path(config.output_dir) / "samples.jsonl"
    rows = [json.loads(line) for line in snapshot.read_text().splitlines()]
    rows[0]["boxes"][0]["bbox"][0] += 1
    snapshot.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot changed"):
        score_saved_run(Path(config.output_dir))


@pytest.mark.parametrize(
    "bad_field",
    [{"sample_id": []}, {"run": True}, {"run": -1}, {"raw_response": 123}, {"finish_reason": None}, {"transport_error": {}}],
)
def test_external_prediction_contract_is_strict(tmp_path: Path, bad_field: dict[str, Any]) -> None:
    config = experiment(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    row = {"sample_id": "page-0", "variant": "grid", "run": 0, "raw_response": "[]", **bad_field}
    predictions.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        score_prediction_file(config, predictions, tmp_path / "offline")


def test_external_duplicate_attempts_fail(tmp_path: Path) -> None:
    config = experiment(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    row = {"sample_id": "page-0", "variant": "grid", "run": 0, "raw_response": "[]"}
    predictions.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        score_prediction_file(config, predictions, tmp_path / "offline")


def test_send_seed_false_is_explicit_in_saved_request(tmp_path: Path) -> None:
    config = experiment(tmp_path)
    config.send_seed = False
    endpoint = FakeEndpoint()
    run_experiment(config, endpoint)
    assert all("seed" not in request for request in endpoint.requests)


def test_thinking_options_are_explicit_and_saved(tmp_path: Path) -> None:
    base = experiment(tmp_path)
    values = base.model_dump(exclude={"variants"})
    config = ExperimentConfig.model_validate({**values, "answer_extraction": "endthink", "enable_thinking": True})
    endpoint = FakeEndpoint()
    report = run_experiment(config, endpoint)
    assert endpoint.requests[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert all(variant.answer_extraction == "endthink" for variant in config.variants)
    assert any(group["status_counts"].get("answer_marker") for group in report["groups"])
    assert score_saved_run(Path(config.output_dir)) == report


def test_cli_can_rescore_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from eval.documents import main

    config = experiment(tmp_path)
    run_experiment(config, FakeEndpoint())
    monkeypatch.setattr("eval.runner.OpenAI", lambda **kwargs: pytest.fail("CLI attempted inference"))
    monkeypatch.setattr("sys.argv", ["eval.documents", "score", "--run-dir", config.output_dir])
    main()
    assert "20/20 attempts" in capsys.readouterr().out


@pytest.mark.parametrize(
    "entrypoint",
    [
        ["-m", "eval.documents"],
        ["eval/documents.py"],
        ["-m", "eval.benchmarks"],
        ["eval/benchmarks.py"],
    ],
)
def test_evaluation_entrypoints_start(entrypoint: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, *entrypoint, "--help"],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_relocated_example_config_preserves_data_and_output_paths() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "eval" / "configs" / "documents.yaml")
    assert config.data.path == str(root / "data" / "evaluation.jsonl")
    assert config.output_dir == str(root / "outputs" / "evaluation")


def test_benchmark_wrapper_forwards_harness_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = []

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"results": {"task": 1}}

    monkeypatch.setitem(sys.modules, "lm_eval", SimpleNamespace(simple_evaluate=fake_evaluate))
    evaluate("hf", ["task"], limit=2)
    assert calls == [{"model": "hf", "tasks": ["task"], "limit": 2}]
    assert "results" in capsys.readouterr().out
