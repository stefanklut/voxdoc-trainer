"""Dataset adapters and auditable repeated evaluation through a chat endpoint.

Generation persists raw attempts without output repair or automatic retries.
Offline rescoring uses the frozen GT snapshot, not the original image files.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import random
import string
import time
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlsplit

import yaml
from datasets import Dataset, DatasetDict, load_from_disk
from openai import OpenAI, OpenAIError
from PIL import Image
from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from eval.scoring import (
    AnswerFormat,
    EvaluationModel,
    Sample,
    Units,
    Variant,
    parse_prediction,
    score_response,
    strict_json,
    summarize,
    validate_variant,
)
from utils.image_loading import read_image_manifest, resolve_image_path
from utils.logging_utils import get_logger

LOGGER = get_logger()
ARTIFACT_VERSION = 1
COMPARISON_TEMPLATE = "{question}\nReturn {output_format}, without commentary. Use {units}."


class DataSource(EvaluationModel):
    path: StrictStr
    format: Literal["manifest", "hf"] = "manifest"
    split: Optional[StrictStr] = None
    task: Optional[Literal["bbox", "labeled_bbox", "transcription"]] = None
    gt_units: Optional[Units] = None
    output_format: AnswerFormat = "auto"
    width: Optional[StrictInt] = Field(default=None, gt=0)
    height: Optional[StrictInt] = Field(default=None, gt=0)
    label_vocabulary: Optional[list[StrictStr]] = None
    limit: Optional[StrictInt] = Field(default=None, gt=0)
    sampling_seed: Optional[StrictInt] = None
    image_root: Optional[StrictStr] = None
    """Directory to remap stored image paths onto, for HF datasets built with --image-mode reference."""

    @model_validator(mode="after")
    def validate_source(self) -> DataSource:
        if self.format == "hf" and self.task is None:
            raise ValueError("HF input requires an explicit task")
        if self.format == "hf" and self.task != "transcription" and self.gt_units is None:
            raise ValueError("HF bbox input requires explicit gt_units")
        if self.format == "manifest" and any(
            (self.split, self.task, self.gt_units, self.width, self.height, self.label_vocabulary, self.output_format != "auto")
        ):
            raise ValueError("manifest task/GT metadata belongs in each row, not data configuration")
        return self


class ExperimentConfig(EvaluationModel):
    data: DataSource
    model: StrictStr = Field(min_length=1)
    model_revision: Optional[StrictStr] = None
    server_revision: Optional[StrictStr] = None
    output_dir: StrictStr
    base_url: StrictStr = "http://localhost:8000/v1"
    api_key_env: StrictStr = "OPENAI_API_KEY"
    comparison: Optional[Literal["grid_vs_native"]] = None
    answer_extraction: Literal["none", "endthink", "think_tag"] = "none"
    enable_thinking: Optional[StrictBool] = None
    variants: list[Variant] = Field(default_factory=list)
    seeds: list[StrictInt] = Field(default_factory=lambda: list(range(5)), min_length=1)
    send_seed: StrictBool = True
    temperature: StrictFloat = Field(default=0.7, ge=0)
    top_p: StrictFloat = Field(default=1.0, gt=0, le=1)
    max_tokens: StrictInt = Field(default=2048, gt=0)
    timeout: StrictFloat = Field(default=120.0, gt=0)
    iou_threshold: StrictFloat = Field(default=0.5, gt=0, le=1)

    @model_validator(mode="after")
    def validate_experiment(self) -> ExperimentConfig:
        endpoint = urlsplit(self.base_url)
        if (
            endpoint.scheme not in ("http", "https")
            or not endpoint.netloc
            or endpoint.username
            or endpoint.password
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError("base_url must be an HTTP(S) endpoint without credentials, query or fragment")
        if len(set(self.seeds)) != len(self.seeds) or any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be distinct nonnegative integers")
        if self.comparison:
            if self.variants:
                raise ValueError("choose comparison or variants, not both")
            self.variants = [
                Variant(
                    name="grid", units="qwen", prompt_template=COMPARISON_TEMPLATE, answer_extraction=self.answer_extraction
                ),
                Variant(
                    name="pixels", units="native", prompt_template=COMPARISON_TEMPLATE, answer_extraction=self.answer_extraction
                ),
            ]
        elif not self.variants:
            self.variants = [
                Variant(name="default", units=self.data.gt_units or "native", answer_extraction=self.answer_extraction)
            ]
        elif self.answer_extraction != "none":
            raise ValueError("custom variants must set answer_extraction per variant, not globally")
        if len({variant.name for variant in self.variants}) != len(self.variants):
            raise ValueError("variant names must be unique")
        return self


class PredictionRecord(EvaluationModel):
    model_config = {"extra": "allow", "allow_inf_nan": False}
    sample_id: StrictStr
    variant: StrictStr
    run: StrictInt = Field(ge=0)
    seed: Optional[StrictInt] = None
    model: Optional[StrictStr] = None
    raw_response: Optional[StrictStr] = None
    finish_reason: StrictStr = "stop"
    transport_error: Optional[StrictStr] = None


def load_config(path: Path) -> ExperimentConfig:
    """Load strict YAML configuration with paths relative to its own directory."""
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = ExperimentConfig.model_validate(values)
    config.data.path = str((path.resolve().parent / config.data.path).resolve())
    config.output_dir = str((path.resolve().parent / config.output_dir).resolve())
    return config


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = strict_json(line)
                if not isinstance(value, dict):
                    raise ValueError("expected a JSON object")
                rows.append(value)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict) or value[0].get("role") != "assistant":
        raise ValueError("GT completion must be text or one assistant message")
    content = value[0].get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or any(
        not isinstance(block, dict) or block.get("type") != "text" or not isinstance(block.get("text"), str)
        for block in content
    ):
        raise ValueError("GT completion must contain only text blocks")
    return "".join(block["text"] for block in content)


def _pil_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _hf_sample(
    row: dict[str, Any],
    index: int,
    source: DataSource,
    image_roots: list[Path] | None = None,
) -> Sample:
    if "ground_truth" not in row and "completion" not in row:
        raise ValueError("HF rows require ground_truth or completion")
    reference = _completion_text(row["ground_truth"] if "ground_truth" in row else row["completion"])
    if "ground_truth" in row and "completion" in row and reference != _completion_text(row["completion"]):
        raise ValueError("conflicting ground_truth and completion columns")
    images = []
    for image in row.get("images") or []:
        if isinstance(image, Image.Image):
            images.append(_pil_data_uri(image))
        elif isinstance(image, str):
            # Reference-mode datasets store the source image path; _image_payload
            # reads and fingerprints the file, so the bytes are used as-is.
            resolved = resolve_image_path(image, image_roots, source.image_root)
            images.append(str(resolved))
        else:
            raise ValueError("HF images must be decoded PIL images or paths to them")
    messages = row.get("prompt")
    if isinstance(messages, str):
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": messages}]
        messages = [{"role": "user", "content": content}]
    if not isinstance(messages, list) or not messages:
        raise ValueError("HF prompt must be text or nonempty chat messages")
    messages = [
        (
            {
                **message,
                "content": [
                    {key: value for key, value in block.items() if value is not None} if isinstance(block, dict) else block
                    for block in message["content"]
                ],
            }
            if isinstance(message, dict) and isinstance(message.get("content"), list)
            else message
        )
        for message in messages
    ]
    _validate_messages(messages, len(images))
    last_content = messages[-1]["content"]
    question = (
        last_content
        if isinstance(last_content, str)
        else "".join(block["text"] for block in last_content if block["type"] == "text")
    )
    task = source.task
    if task is None:
        raise ValueError("HF source requires an explicit task")
    output_format = source.output_format
    if output_format == "auto":
        output_format = {"bbox": "boxes", "labeled_bbox": "labeled_boxes", "transcription": "text"}[task]
    fields = {
        "sample_id": str(row.get("sample_id", index)),
        "document_id": row.get("document_id"),
        "task": task,
        "prompt": question,
        "messages": messages,
        "images": images,
        "width": source.width,
        "height": source.height,
        "gt_units": source.gt_units or "native",
        "output_format": output_format,
        "label_vocabulary": source.label_vocabulary,
    }
    if task == "transcription":
        fields["text"] = reference
    else:
        if source.gt_units == "native" and (source.width is None or source.height is None):
            raise ValueError("native bbox GT requires annotation width and height")
        parser_contract = Sample.model_construct(**fields)
        if source.gt_units is None:
            raise ValueError("HF bbox source requires explicit gt_units")
        boxes = parse_prediction(reference, parser_contract, source.gt_units)
        if not isinstance(boxes, list):
            raise ValueError("expected bbox GT")
        fields["boxes"] = [box.model_dump(mode="json") for box in boxes]
    return Sample.model_validate(fields)


def _validate_messages(messages: list[dict[str, Any]], image_count: int) -> None:
    placeholders = 0
    for message in messages:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in ("system", "user", "assistant")
        ):
            raise ValueError("chat messages require only role and content")
        content = message["content"]
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError("chat content must be text or content blocks")
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("invalid chat content block")
            if block == {"type": "image"}:
                placeholders += 1
            elif set(block) == {"type", "text"} and block["type"] == "text" and isinstance(block["text"], str):
                continue
            else:
                raise ValueError("only text blocks and TRL image placeholders are supported")
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("prompt history must end with a user message")
    if placeholders != image_count:
        raise ValueError("image placeholder count does not match images")


def load_samples(source: DataSource) -> list[Sample]:
    """Load explicit manifest or HF tasks without guessing GT types or dimensions."""
    path = Path(source.path)
    if source.format == "manifest":
        rows = _read_jsonl(path)
        samples = []
        for row in rows:
            sample = Sample.model_validate(row)
            sample.images = [
                image if image.startswith("data:") else str((path.resolve().parent / image).resolve())
                for image in sample.images
            ]
            samples.append(sample)
    else:
        dataset = load_from_disk(str(path))
        if isinstance(dataset, DatasetDict):
            if source.split is None or source.split not in dataset:
                raise ValueError("DatasetDict input requires an existing explicit split")
            dataset = dataset[source.split]
        elif source.split is not None:
            raise ValueError("split is only valid for a DatasetDict")
        if not isinstance(dataset, Dataset):
            raise ValueError("expected a saved HF Dataset")
        indices = list(range(len(dataset)))
        if source.sampling_seed is not None:
            random.Random(source.sampling_seed).shuffle(indices)
        indices = indices[: source.limit]
        manifest = read_image_manifest(path)
        image_roots = [Path(root) for root in manifest.get("roots", [])] if manifest else []
        samples = [_hf_sample(dataset[index], index, source, image_roots) for index in indices]
    if source.format == "manifest":
        if source.sampling_seed is not None:
            random.Random(source.sampling_seed).shuffle(samples)
        samples = samples[: source.limit]
    if not samples:
        raise ValueError("evaluation dataset is empty")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("sample_id values must be unique")
    return samples


def _image_payload(image: str) -> tuple[str, dict[str, Any]]:
    if image.startswith("data:"):
        header, encoded = image.split(",", 1)
        if header not in ("data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"):
            raise ValueError("images must be PNG, JPEG or WebP base64 data URIs")
        payload = base64.b64decode(encoded, validate=True)
        declared_mime = header[5:-7]
    else:
        payload = Path(image).read_bytes()
        declared_mime = None
    with Image.open(io.BytesIO(payload)) as raster:
        mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}.get(raster.format or "")
        dimensions = list(raster.size)
        if mime is None or getattr(raster, "n_frames", 1) != 1:
            raise ValueError("endpoint images must be single-frame PNG, JPEG or WebP")
        raster.verify()
    if declared_mime is not None and declared_mime != mime:
        raise ValueError("image MIME type does not match payload")
    uri = f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")
    return uri, {"sha256": hashlib.sha256(payload).hexdigest(), "raster_size": dimensions, "mime": mime}


def render_prompt(sample: Sample, variant: Variant) -> str:
    """Render safe, explicit coordinate/output instructions, never rewrite legacy prompts."""
    formats = {
        "boxes": "a JSON list of [xmin,ymin,xmax,ymax] boxes",
        "single_box": "one JSON [xmin,ymin,xmax,ymax] array",
        "labeled_boxes": 'a JSON list of objects with exactly "bbox" ([xmin,ymin,xmax,ymax]) and "label" (string)',
        "doclang_box": 'one <text> element containing exactly four <location value="integer"/> elements',
        "text": "only the transcription text",
    }
    units = "integer coordinates on a 1000 by 1000 grid, ranging from 0 to 1000"
    if variant.units == "native":
        units = f"original-pixel coordinates in the {sample.width} by {sample.height} annotation space"
    values = {
        "question": sample.prompt,
        "units": units,
        "width": sample.width,
        "height": sample.height,
        "output_format": formats[sample.output_format],
    }
    if sample.label_vocabulary is not None:
        values["output_format"] += "; allowed labels: " + _json(sample.label_vocabulary)
    for _, field, specification, conversion in string.Formatter().parse(variant.prompt_template):
        if field is not None and (field not in values or specification or conversion):
            raise ValueError("prompt placeholders may only be question, units, width, height or output_format")
    return variant.prompt_template.format_map(values)


def build_request_messages(sample: Sample, variant: Variant, image_uris: list[str]) -> list[dict[str, Any]]:
    """Build endpoint messages while preserving HF history and image placeholder order."""
    if sample.messages is not None:
        if variant.prompt_template != "{question}" or (sample.task != "transcription" and variant.units != sample.gt_units):
            raise ValueError(
                "legacy HF/chat prompts cannot be rewritten for coordinate experiments; use a manifest with native GT"
            )
        _validate_messages(sample.messages, len(image_uris))
        images = iter(image_uris)
        result = []
        for message in sample.messages:
            content = message["content"]
            if isinstance(content, list):
                content = [
                    {"type": "image_url", "image_url": {"url": next(images)}} if block["type"] == "image" else dict(block)
                    for block in content
                ]
            result.append({"role": message["role"], "content": content})
        return result
    content = [{"type": "image_url", "image_url": {"url": image}} for image in image_uris]
    content.append({"type": "text", "text": render_prompt(sample, variant)})
    return [{"role": "user", "content": content}]


def _attempt_id(sample_id: str, variant: str, run: int) -> str:
    return _digest([sample_id, variant, run])


def _identity(config: ExperimentConfig) -> dict[str, Any]:
    return config.model_dump(mode="json", exclude={"output_dir", "comparison"})


def _score_records(config: ExperimentConfig, samples: list[Sample], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed_samples = {sample.sample_id: sample for sample in samples}
    indexed_variants = {variant.name: variant for variant in config.variants}
    seen = set()
    scored = []
    for record in records:
        validated = PredictionRecord.model_validate(record)
        sample_id, variant_name, run = validated.sample_id, validated.variant, validated.run
        if (
            sample_id not in indexed_samples
            or variant_name not in indexed_variants
            or type(run) is not int
            or not 0 <= run < len(config.seeds)
        ):
            raise ValueError("prediction references an unknown sample, variant or run")
        key = _attempt_id(sample_id, variant_name, run)
        if record.get("attempt_id", key) != key:
            raise ValueError("attempt_id does not match sample, variant and run")
        if key in seen:
            raise ValueError("duplicate prediction attempt")
        seen.add(key)
        if record.get("model", config.model) != config.model or record.get("seed", config.seeds[run]) != config.seeds[run]:
            raise ValueError("prediction model/seed does not match the experiment")
        raw = record.get("raw_response")
        if raw is not None and not isinstance(raw, str):
            raise ValueError("raw_response must be text or null")
        sample, variant = indexed_samples[sample_id], indexed_variants[variant_name]
        score = score_response(
            sample,
            variant,
            raw,
            finish_reason=record.get("finish_reason", "stop"),
            transport_error=record.get("transport_error"),
            iou_threshold=config.iou_threshold,
        )
        scored.append(
            {
                **record,
                "attempt_id": key,
                "sample_id": sample_id,
                "variant": variant_name,
                "model": config.model,
                "seed": config.seeds[run],
                "task": sample.task,
                "score": score,
            }
        )
    return scored


def _report(config: ExperimentConfig, samples: list[Sample], scored: list[dict[str, Any]]) -> dict[str, Any]:
    planned = len(samples) * len(config.variants) * len(config.seeds)
    return {
        "artifact_version": ARTIFACT_VERSION,
        "planned_attempts": planned,
        "missing_attempts": planned - len(scored),
        "completed_attempts": len(scored),
        "iou_threshold": config.iou_threshold,
        **summarize(scored, [variant.name for variant in config.variants]),
    }


def _config_from_metadata(values: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig.model_validate({**values, "comparison": None, "answer_extraction": "none"})


def score_saved_run(run_dir: Path, iou_threshold: Optional[float] = None) -> dict[str, Any]:
    """Rescore frozen attempts without loading images, initializing a client, or requiring credentials."""
    metadata = strict_json((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["artifact_version"] != ARTIFACT_VERSION:
        raise ValueError("unsupported artifact version")
    config = _config_from_metadata(metadata["config"])
    if iou_threshold is not None:
        config = ExperimentConfig.model_validate(
            {**config.model_dump(exclude={"comparison", "answer_extraction"}), "iou_threshold": iou_threshold}
        )
    samples = [Sample.model_validate(row) for row in _read_jsonl(run_dir / "samples.jsonl")]
    if _digest([sample.model_dump(mode="json") for sample in samples]) != metadata["samples_sha256"]:
        raise ValueError("frozen GT snapshot changed")
    records = _read_jsonl(run_dir / "attempts.jsonl")
    if any(record.get("fingerprint") != metadata["fingerprint"] for record in records):
        raise ValueError("attempt fingerprint does not match metadata")
    scored = _score_records(config, samples, records)
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for record in scored:
            handle.write(_json(record) + "\n")
    report = _report(config, samples, scored)
    _write_json(run_dir / "summary.json", report)
    return report


def score_prediction_file(config: ExperimentConfig, predictions: Path, output: Path) -> dict[str, Any]:
    """Score externally generated JSONL responses, without endpoint access or image reads."""
    if predictions.resolve() in ((output / "scores.jsonl").resolve(), (output / "summary.json").resolve()):
        raise ValueError("scoring output must not overwrite raw predictions")
    samples = load_samples(config.data)
    scored = _score_records(config, samples, _read_jsonl(predictions))
    report = _report(config, samples, scored)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for record in scored:
            handle.write(_json(record) + "\n")
    _write_json(output / "summary.json", report)
    return report


def run_experiment(config: ExperimentConfig, client: Optional[Any] = None) -> dict[str, Any]:
    """Generate every paired sample/variant/seed attempt once and persist failures for audit."""
    samples = load_samples(config.data)
    requests: dict[tuple[str, str], list[dict[str, Any]]] = {}
    image_metadata = {}
    for sample in samples:
        if config.comparison and (sample.task == "transcription" or sample.gt_units != "native"):
            raise ValueError("grid_vs_native requires bbox tasks with exact native GT")
        payloads = [_image_payload(image) for image in sample.images]
        image_metadata[sample.sample_id] = [description for _, description in payloads]
        for variant in config.variants:
            validate_variant(sample, variant)
            requests[(sample.sample_id, variant.name)] = build_request_messages(sample, variant, [uri for uri, _ in payloads])
    snapshot = [sample.model_dump(mode="json") for sample in samples]
    fingerprint = _digest({"config": _identity(config), "samples": snapshot, "images": image_metadata})
    run_dir = Path(config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path, attempts_path = run_dir / "metadata.json", run_dir / "attempts.jsonl"
    if metadata_path.exists():
        metadata = strict_json(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != fingerprint or metadata.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("resume refused: configuration, dataset or image content changed; use a new output directory")
        existing = _read_jsonl(attempts_path)
        _score_records(config, samples, existing)
        if any(record.get("fingerprint") != fingerprint for record in existing):
            raise ValueError("attempt fingerprint does not match metadata")
    else:
        if any(run_dir.iterdir()):
            raise ValueError("output directory is not empty and has no evaluation metadata")
        existing = []
        metadata = {
            "artifact_version": ARTIFACT_VERSION,
            "fingerprint": fingerprint,
            "samples_sha256": _digest(snapshot),
            "implementation_sha256": {
                str(path.relative_to(Path(__file__).resolve().parent.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (
                    Path(__file__).resolve(),
                    Path(__file__).with_name("scoring.py").resolve(),
                    Path(__file__).resolve().parent.parent / "utils" / "metrics.py",
                )
            },
            "config": config.model_dump(mode="json"),
            "images": image_metadata,
            "python": platform.python_version(),
            "versions": {name: importlib.metadata.version(name) for name in ("openai", "datasets", "scipy", "pydantic")},
            "seed_reproducibility": "best effort; dependent on backend, model and hardware",
        }
        with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
            for sample in snapshot:
                handle.write(_json(sample) + "\n")
        attempts_path.touch()
        _write_json(metadata_path, metadata)
    completed = {record["attempt_id"] for record in existing}
    planned = len(samples) * len(config.variants) * len(config.seeds)
    if len(completed) == planned:
        return score_saved_run(run_dir)
    api_key = os.environ.get(config.api_key_env, "EMPTY")
    owned_client = client is None
    if client is None:
        client = OpenAI(base_url=config.base_url, api_key=api_key, max_retries=0, timeout=config.timeout)
    try:
        with attempts_path.open("a", encoding="utf-8") as handle:
            for run, seed in enumerate(config.seeds):
                for sample in samples:
                    for variant in config.variants:
                        attempt_id = _attempt_id(sample.sample_id, variant.name, run)
                        if attempt_id in completed:
                            continue
                        messages = requests[(sample.sample_id, variant.name)]
                        parameters = {
                            "model": config.model,
                            "temperature": config.temperature,
                            "top_p": config.top_p,
                            "max_tokens": config.max_tokens,
                        }
                        if config.send_seed:
                            parameters["seed"] = seed
                        if config.enable_thinking is not None:
                            parameters["extra_body"] = {"chat_template_kwargs": {"enable_thinking": config.enable_thinking}}
                        record = {
                            "attempt_id": attempt_id,
                            "fingerprint": fingerprint,
                            "model": config.model,
                            "sample_id": sample.sample_id,
                            "document_id": sample.document_id,
                            "variant": variant.name,
                            "run": run,
                            "seed": seed,
                            "request_parameters": parameters,
                            "request_sha256": _digest({"messages": messages, **parameters}),
                            "prompt": render_prompt(sample, variant) if sample.messages is None else sample.messages,
                            "images": image_metadata[sample.sample_id],
                            "raw_response": None,
                            "finish_reason": "stop",
                            "transport_error": None,
                            "usage": None,
                        }
                        started = time.monotonic()
                        fatal = False
                        try:
                            response = client.chat.completions.create(messages=messages, **parameters)
                            record["response_id"] = response.id
                            record["response_model"] = response.model
                            record["system_fingerprint"] = getattr(response, "system_fingerprint", None)
                            if response.choices:
                                choice = response.choices[0]
                                record["raw_response"] = choice.message.content
                                record["finish_reason"] = choice.finish_reason
                                record["message"] = choice.message.model_dump(mode="json")
                            else:
                                record["finish_reason"] = "no_choice"
                            if response.usage is not None:
                                record["usage"] = response.usage.model_dump(mode="json")
                        except (OpenAIError, TimeoutError, ConnectionError) as error:
                            description = f"{type(error).__name__}: {error}"
                            record["transport_error"] = (
                                description.replace(api_key, "[REDACTED]") if api_key != "EMPTY" else description
                            )
                            fatal = getattr(error, "status_code", None) in (400, 401, 403, 404, 422)
                        record["latency_seconds"] = time.monotonic() - started
                        handle.write(_json(record) + "\n")
                        handle.flush()
                        completed.add(attempt_id)
                        LOGGER.info(
                            "Completed %s/%s: %s %s run %s", len(completed), planned, sample.sample_id, variant.name, run
                        )
                        if fatal:
                            score_saved_run(run_dir)
                            raise ValueError(
                                "endpoint rejected the request; attempt saved. Check model, credentials and seed support; no options were silently dropped"
                            )
    finally:
        if owned_client:
            client.close()
    return score_saved_run(run_dir)
