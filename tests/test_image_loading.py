"""Unit tests for lazy image storage and on-demand image resolution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset, load_from_disk
from PIL import Image

from utils.image_loading import (
    IMAGE_MODES,
    MANIFEST_NAME,
    ImageNotFoundError,
    LazyImageCollator,
    add_image_mode_argument,
    dataset_output_dir,
    image_field,
    image_roots,
    is_reference_dataset,
    read_image_manifest,
    relative_image_path,
    resolve_image_path,
    save_image_dataset,
    wrap_trainer_collator,
    wrap_trainer_images,
)


def _make_image(path: Path, size: tuple[int, int] = (8, 8)) -> Path:
    """Write an incompressible test image so embedded bytes are not trivially small.

    Args:
        path (Path): Where to write the image.
        size (tuple[int, int]): (width, height) of the image.

    Returns:
        Path: The image path.
    """
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)
    return path


def _records(image_paths: list[Path], image_mode: str = "reference", roots: list[Path] | None = None) -> list[dict]:
    """Build minimal prompt/completion records referencing the given images.

    Args:
        image_paths (list[Path]): One image per record.
        image_mode (str): One of ``utils.image_loading.IMAGE_MODES``.
        roots (list[Path] | None): Corpus roots for reference mode.

    Returns:
        list[dict]: Records shaped like the data creators' output.
    """
    return [
        {
            "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Read this."}]}],
            "completion": [{"role": "assistant", "content": [{"type": "text", "text": f"Output {index}."}]}],
            "images": image_field(path, image_mode, roots),
        }
        for index, path in enumerate(image_paths)
    ]


def _passthrough_collator(examples: list[dict]) -> list[dict]:
    """Collator stand-in that returns the batch, so tests can inspect what it received.

    Args:
        examples (list[dict]): The batch it was given.

    Returns:
        list[dict]: The same batch.
    """
    return examples


def _dir_size(directory: Path) -> int:
    """Total size in bytes of every file under a directory.

    Args:
        directory (Path): Directory to measure.

    Returns:
        int: Summed file sizes.
    """
    return sum(entry.stat().st_size for entry in directory.rglob("*") if entry.is_file())


class TestImageField:
    """Tests for building a single ``images`` column value."""

    def test_embed_opens_the_image(self, tmp_path: Path) -> None:
        """Embed mode stores a decoded image object."""
        image = _make_image(tmp_path / "doc.png")
        values = image_field(image, "embed")
        assert len(values) == 1
        assert isinstance(values[0], Image.Image)

    def test_reference_stores_relative_string(self, tmp_path: Path) -> None:
        """Reference mode stores a path string relative to the corpus root."""
        image = _make_image(tmp_path / "corpus" / "doc.png")
        values = image_field(image, "reference", [tmp_path / "corpus"])
        assert values == ["doc.png"]

    def test_unknown_mode_rejected(self, tmp_path: Path) -> None:
        """An unsupported mode raises rather than silently embedding."""
        image = _make_image(tmp_path / "doc.png")
        with pytest.raises(ValueError, match="Unknown image mode"):
            image_field(image, "jpeg")


class TestImageRoots:
    """Tests for collecting the roots stored paths are relative to."""

    def test_image_dirs_come_first(self, tmp_path: Path) -> None:
        """Explicit image directories take precedence over corpus roots."""
        (tmp_path / "corpus").mkdir()
        images = _make_image(tmp_path / "images" / "keep.png").parent
        roots = image_roots([tmp_path / "corpus"], [images])
        assert roots == [images.resolve(), (tmp_path / "corpus").resolve()]

    def test_single_path_accepted(self, tmp_path: Path) -> None:
        """A bare path is accepted where a sequence is expected."""
        (tmp_path / "corpus").mkdir()
        assert image_roots(tmp_path / "corpus") == [(tmp_path / "corpus").resolve()]

    def test_missing_and_duplicate_roots_dropped(self, tmp_path: Path) -> None:
        """Nonexistent and repeated roots are removed."""
        (tmp_path / "corpus").mkdir()
        roots = image_roots([tmp_path / "corpus", tmp_path / "corpus", tmp_path / "gone"])
        assert roots == [(tmp_path / "corpus").resolve()]


class TestRelativeAndResolve:
    """Tests for the path round-trip between build time and load time."""

    def test_relative_to_matching_root(self, tmp_path: Path) -> None:
        """A path inside a root is stored relative to it."""
        image = _make_image(tmp_path / "corpus" / "doc" / "page.png")
        assert relative_image_path(image, [tmp_path / "corpus"]) == "doc/page.png"

    def test_outside_all_roots_stays_absolute(self, tmp_path: Path) -> None:
        """A path outside every root is stored absolute rather than guessed."""
        image = _make_image(tmp_path / "elsewhere" / "page.png")
        assert relative_image_path(image, [tmp_path / "corpus"]) == str(image.resolve())

    def test_relative_input_resolved_against_roots(self, tmp_path: Path) -> None:
        """A relative input is located under the roots before being stored."""
        image = _make_image(tmp_path / "corpus" / "doc" / "page.png")
        assert relative_image_path("doc/page.png", [tmp_path / "corpus"]) == "doc/page.png"

    def test_round_trip(self, tmp_path: Path) -> None:
        """Storing and resolving returns the original file."""
        image = _make_image(tmp_path / "corpus" / "doc" / "page.png")
        stored = relative_image_path(image, [tmp_path / "corpus"])
        assert resolve_image_path(stored, [tmp_path / "corpus"]) == image.resolve()

    def test_image_root_remaps_moved_corpus(self, tmp_path: Path) -> None:
        """A moved corpus is found through the remapped root."""
        image = _make_image(tmp_path / "original" / "doc" / "page.png")
        stored = relative_image_path(image, [tmp_path / "original"])
        moved = _make_image(tmp_path / "moved" / "doc" / "page.png")
        assert resolve_image_path(stored, [tmp_path / "original"], tmp_path / "moved") == moved.resolve()

    def test_image_root_wins_over_recorded_root(self, tmp_path: Path) -> None:
        """The remapped root is preferred when both contain the file."""
        image = _make_image(tmp_path / "original" / "page.png")
        _make_image(tmp_path / "moved" / "page.png")
        resolved = resolve_image_path("page.png", [tmp_path / "original"], tmp_path / "moved")
        assert resolved == (tmp_path / "moved" / "page.png").resolve()
        assert resolved != image.resolve()

    def test_absolute_path_used_directly(self, tmp_path: Path) -> None:
        """An absolute stored path resolves without any root."""
        image = _make_image(tmp_path / "page.png")
        assert resolve_image_path(str(image)) == image.resolve()

    def test_missing_image_lists_searched_dirs(self, tmp_path: Path) -> None:
        """A missing image names the stored path and every directory tried."""
        (tmp_path / "a").mkdir()
        with pytest.raises(ImageNotFoundError) as excinfo:
            resolve_image_path("doc/page.png", [tmp_path / "a"])
        message = str(excinfo.value)
        assert "doc/page.png" in message
        assert str(tmp_path / "a") in message

    def test_deleted_root_is_still_named_in_the_error(self, tmp_path: Path) -> None:
        """A corpus that was deleted is still reported, so the error says where to look."""
        gone = tmp_path / "corpus"
        with pytest.raises(ImageNotFoundError) as excinfo:
            resolve_image_path("document/page.png", [gone])
        message = str(excinfo.value)
        assert "document/page.png" in message
        assert str(gone) in message


class TestManifest:
    """Tests for the ``image_manifest.json`` sidecar."""

    def test_reference_mode_writes_manifest(self, tmp_path: Path) -> None:
        """Reference datasets record their mode and corpus roots."""
        image = _make_image(tmp_path / "corpus" / "page.png")
        output = tmp_path / "ds"
        save_image_dataset(_records([image]), output, "reference", [tmp_path / "corpus"])
        manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["image_mode"] == "reference"
        assert manifest["roots"] == [str((tmp_path / "corpus").resolve())]

    def test_embed_mode_writes_no_manifest(self, tmp_path: Path) -> None:
        """Embedded datasets stay exactly as they were, with no sidecar."""
        image = _make_image(tmp_path / "corpus" / "page.png")
        output = tmp_path / "ds"
        save_image_dataset(_records([image]), output, "embed", [tmp_path / "corpus"])
        assert not (output / MANIFEST_NAME).exists()
        assert read_image_manifest(output) is None

    def test_corrupt_manifest_raises(self, tmp_path: Path) -> None:
        """An unreadable sidecar is reported rather than treated as embedded."""
        (tmp_path / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
        with pytest.raises(ImageNotFoundError, match="Could not read image manifest"):
            read_image_manifest(tmp_path)

    def test_non_object_manifest_raises(self, tmp_path: Path) -> None:
        """A sidecar holding a list is rejected."""
        (tmp_path / MANIFEST_NAME).write_text("[]", encoding="utf-8")
        with pytest.raises(ImageNotFoundError, match="not a JSON object"):
            read_image_manifest(tmp_path)


class TestSaveImageDataset:
    """Tests for building and saving a dataset in either mode."""

    def test_reference_dataset_holds_no_image_bytes(self, tmp_path: Path) -> None:
        """The whole point: a reference dataset is orders of magnitude smaller."""
        images = [_make_image(tmp_path / "corpus" / f"page{i}.png", (200, 200)) for i in range(4)]
        embedded = tmp_path / "embedded"
        referenced = tmp_path / "referenced"
        save_image_dataset(_records(images, "embed"), embedded, "embed")
        save_image_dataset(_records(images, "reference", [tmp_path / "corpus"]), referenced, "reference", [tmp_path / "corpus"])
        embedded_size = _dir_size(embedded)
        referenced_size = _dir_size(referenced)
        assert embedded_size > 4 * 1024
        assert referenced_size < embedded_size / 10

    def test_reference_column_is_strings(self, tmp_path: Path) -> None:
        """The images column keeps its name but becomes a list of strings."""
        image = _make_image(tmp_path / "corpus" / "page.png")
        output = tmp_path / "ds"
        records = _records([image], "reference", [tmp_path / "corpus"])
        save_image_dataset(records, output, "reference", [tmp_path / "corpus"])
        dataset = load_from_disk(output)
        assert dataset[0]["images"] == ["page.png"]

    def test_unknown_mode_rejected(self, tmp_path: Path) -> None:
        """An unsupported mode raises before anything is written."""
        with pytest.raises(ValueError, match="Unknown image mode"):
            save_image_dataset([], tmp_path / "ds", "jpeg")

    def test_modes_cover_documented_values(self) -> None:
        """The public mode tuple matches what the CLI advertises."""
        assert IMAGE_MODES == ("embed", "reference")


class TestIsReferenceDataset:
    """Tests for detecting a dataset that stores paths instead of bytes."""

    def test_string_column_detected(self) -> None:
        """A list-of-strings images column is recognised."""
        dataset = Dataset.from_list([{"images": ["a.png"], "prompt": "x"}])
        assert is_reference_dataset(dataset) is True

    def test_embedded_column_not_detected(self, tmp_path: Path) -> None:
        """A dataset holding real images is not mistaken for a reference one."""
        _make_image(tmp_path / "page.png")
        with Image.open(tmp_path / "page.png") as opened:
            dataset = Dataset.from_list([{"images": [opened.copy()], "prompt": "x"}])
        assert is_reference_dataset(dataset) is False

    def test_no_image_column(self) -> None:
        """A text-only dataset is treated as needing no wrapping."""
        dataset = Dataset.from_list([{"prompt": "x", "completion": "y"}])
        assert is_reference_dataset(dataset) is False

    def test_singular_image_column_detected(self) -> None:
        """The TRL singular ``image`` column is recognised too."""
        dataset = Dataset.from_list([{"image": ["a.png"], "prompt": "x"}])
        assert is_reference_dataset(dataset) is True


class TestLazyImageCollator:
    """Tests for the collator wrapper that opens images before collation."""

    def test_paths_replaced_with_images(self, tmp_path: Path) -> None:
        """The wrapped collator receives opened images instead of path strings."""
        image = _make_image(tmp_path / "corpus" / "page.png")
        seen: list = []

        def collator(examples: list[dict]) -> dict:
            seen.extend(image for example in examples for image in example["images"])
            return {"count": len(examples)}

        wrapper = LazyImageCollator(collator, [tmp_path / "corpus"])
        result = wrapper([{"images": ["page.png"], "prompt": "x"}, {"images": ["page.png"], "prompt": "y"}])
        assert result == {"count": 2}
        assert all(isinstance(opened, Image.Image) for opened in seen)
        assert len(seen) == 2

    def test_images_survive_after_call(self, tmp_path: Path) -> None:
        """Decoded pixels stay readable once the file handle is closed."""
        image = _make_image(tmp_path / "corpus" / "page.png")
        wrapper = LazyImageCollator(lambda examples: examples, [tmp_path / "corpus"])
        opened = wrapper([{"images": ["page.png"]}])[0]["images"][0]
        assert opened.size == (8, 8)
        assert opened.getpixel((0, 0)) is not None

    def test_return_tensors_forwarded(self, tmp_path: Path) -> None:
        """A collator that requires return_tensors still gets it."""
        _make_image(tmp_path / "corpus" / "page.png")
        wrapper = LazyImageCollator(lambda examples, return_tensors=None: return_tensors, [tmp_path / "corpus"])
        assert wrapper([{"images": ["page.png"]}], return_tensors="pt") == "pt"

    def test_default_call_omits_return_tensors(self, tmp_path: Path) -> None:
        """A collator without a return_tensors parameter is called with the batch alone."""
        _make_image(tmp_path / "corpus" / "page.png")
        wrapper = LazyImageCollator(lambda examples: "ok", [tmp_path / "corpus"])
        assert wrapper([{"images": ["page.png"]}]) == "ok"

    def test_rows_without_paths_untouched(self) -> None:
        """Already-decoded rows are passed through unchanged."""
        sentinel = object()
        wrapper = LazyImageCollator(lambda examples: examples)
        row = {"images": [sentinel], "prompt": "x"}
        assert wrapper([row])[0] is row

    def test_missing_image_raises(self, tmp_path: Path) -> None:
        """A stored path with no file behind it fails loudly."""
        wrapper = LazyImageCollator(lambda examples: examples, [tmp_path / "corpus"])
        with pytest.raises(ImageNotFoundError):
            wrapper([{"images": ["gone.png"]}])

    def test_attributes_forwarded(self) -> None:
        """The wrapper is transparent for the attributes Trainer reads."""

        class Inner:
            processor = "processor"
            pad_token_id = 0

        wrapper = LazyImageCollator(Inner())
        assert wrapper.processor == "processor"
        assert wrapper.pad_token_id == 0
        assert wrapper.wrapped_collator.processor == "processor"

    def test_add_roots_extends_search(self, tmp_path: Path) -> None:
        """Extra roots let one wrapper serve several datasets."""
        _make_image(tmp_path / "second" / "page.png")
        wrapper = LazyImageCollator(lambda examples: examples, [tmp_path / "first"])
        with pytest.raises(ImageNotFoundError):
            wrapper([{"images": ["page.png"]}])
        wrapper.add_roots([tmp_path / "second"])
        assert wrapper([{"images": ["page.png"]}])[0]["images"][0].size == (8, 8)


class TestWrapTrainer:
    """Tests for wiring the wrapper into a constructed trainer."""

    class Trainer:
        """Minimal stand-in exposing only the attribute the wrapper touches."""

        def __init__(self) -> None:
            self.data_collator = _passthrough_collator

    def test_embedded_dataset_left_alone(self, tmp_path: Path) -> None:
        """An embedded dataset keeps the trainer's collator untouched."""
        _make_image(tmp_path / "page.png")
        with Image.open(tmp_path / "page.png") as opened:
            dataset = Dataset.from_list([{"images": [opened.copy()], "prompt": "x"}])
        trainer = self.Trainer()
        assert wrap_trainer_collator(trainer, dataset) is False
        assert trainer.data_collator is _passthrough_collator

    def test_reference_dataset_wrapped(self, tmp_path: Path) -> None:
        """A reference dataset gets its manifest roots into the wrapper."""
        image = _make_image(tmp_path / "corpus" / "page.png")
        output = tmp_path / "ds"
        save_image_dataset(_records([image]), output, "reference", [tmp_path / "corpus"])
        trainer = self.Trainer()
        assert wrap_trainer_collator(trainer, load_from_disk(output), dataset_dir=output) is True
        assert isinstance(trainer.data_collator, LazyImageCollator)
        assert trainer.data_collator.roots == [(tmp_path / "corpus").resolve()]

    def test_image_root_overrides_manifest(self, tmp_path: Path) -> None:
        """The configured image root reaches the wrapper."""
        image = _make_image(tmp_path / "corpus" / "page.png")
        output = tmp_path / "ds"
        save_image_dataset(_records([image]), output, "reference", [tmp_path / "corpus"])
        trainer = self.Trainer()
        wrap_trainer_collator(trainer, load_from_disk(output), tmp_path / "elsewhere", output)
        assert trainer.data_collator._image_root == tmp_path / "elsewhere"

    def test_both_datasets_share_one_wrapper(self, tmp_path: Path) -> None:
        """Train and eval roots accumulate in a single wrapper, not nested ones."""
        train_image = _make_image(tmp_path / "train_corpus" / "page.png")
        eval_image = _make_image(tmp_path / "eval_corpus" / "page.png")
        train_dir, eval_dir = tmp_path / "train", tmp_path / "eval"
        save_image_dataset(_records([train_image]), train_dir, "reference", [tmp_path / "train_corpus"])
        save_image_dataset(_records([eval_image]), eval_dir, "reference", [tmp_path / "eval_corpus"])
        trainer = self.Trainer()
        wrapped = wrap_trainer_images(trainer, load_from_disk(train_dir), load_from_disk(eval_dir))
        assert wrapped is True
        assert isinstance(trainer.data_collator, LazyImageCollator)
        assert trainer.data_collator.wrapped_collator is _passthrough_collator
        assert trainer.data_collator([{"images": ["page.png"]}])[0]["images"][0].size == (8, 8)

    def test_eval_only_reference_dataset_wrapped(self, tmp_path: Path) -> None:
        """A reference eval dataset with an embedded train dataset still wraps."""
        train = Dataset.from_list([{"prompt": "x", "completion": "y"}])
        eval_image = _make_image(tmp_path / "eval_corpus" / "page.png")
        eval_dir = tmp_path / "eval"
        save_image_dataset(_records([eval_image]), eval_dir, "reference", [tmp_path / "eval_corpus"])
        trainer = self.Trainer()
        assert wrap_trainer_images(trainer, train, load_from_disk(eval_dir)) is True
        assert isinstance(trainer.data_collator, LazyImageCollator)

    def test_no_datasets_reference(self) -> None:
        """Two text-only datasets leave the collator alone."""
        train = Dataset.from_list([{"prompt": "x"}])
        trainer = self.Trainer()
        assert wrap_trainer_images(trainer, train, train) is False
        assert trainer.data_collator is _passthrough_collator


class TestDatasetOutputDir:
    """Tests for recovering the directory a dataset was loaded from."""

    def test_loaded_dataset_reports_its_directory(self, tmp_path: Path) -> None:
        """A dataset loaded from disk knows where it came from."""
        image = _make_image(tmp_path / "corpus" / "page.png")
        output = tmp_path / "ds"
        save_image_dataset(_records([image]), output, "reference", [tmp_path / "corpus"])
        assert dataset_output_dir(load_from_disk(output)) == output.resolve()

    def test_in_memory_dataset_has_no_directory(self) -> None:
        """An in-memory dataset reports no directory instead of guessing."""
        assert dataset_output_dir(Dataset.from_list([{"prompt": "x"}])) is None

    def test_wrapper_finds_manifest_without_hint(self, tmp_path: Path) -> None:
        """The wrapper reads the sidecar from the dataset itself when not told where."""
        image = _make_image(tmp_path / "corpus" / "page.png")
        output = tmp_path / "ds"
        save_image_dataset(_records([image]), output, "reference", [tmp_path / "corpus"])

        class Trainer:
            data_collator = "collator"

        trainer = Trainer()
        assert wrap_trainer_collator(trainer, load_from_disk(output)) is True
        assert trainer.data_collator.roots == [(tmp_path / "corpus").resolve()]


class TestAddImageModeArgument:
    """Tests for the shared ``--image-mode`` flag."""

    def _parser(self, default: str = "embed"):
        """Build a parser carrying only the shared flag.

        Args:
            default (str): Default mode to register.

        Returns:
            argparse.ArgumentParser: The parser.
        """
        import argparse

        parser = argparse.ArgumentParser()
        add_image_mode_argument(parser, default)
        return parser

    def test_default_is_embed(self) -> None:
        """Omitting the flag keeps the historical embedding behaviour."""
        assert self._parser().parse_args([]).image_mode == "embed"

    def test_both_modes_accepted(self) -> None:
        """Both documented modes parse."""
        assert self._parser().parse_args(["--image-mode", "reference"]).image_mode == "reference"
        assert self._parser().parse_args(["--image-mode", "embed"]).image_mode == "embed"

    def test_unknown_mode_rejected(self) -> None:
        """An unsupported mode is rejected by argparse."""
        with pytest.raises(SystemExit):
            self._parser().parse_args(["--image-mode", "jpeg"])

    def test_custom_default(self) -> None:
        """A caller can advertise a different default."""
        assert self._parser("reference").parse_args([]).image_mode == "reference"
