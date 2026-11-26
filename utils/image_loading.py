"""Lazy image storage for saved HF datasets.

By default the data creators embed fully decoded images into the dataset they
save: ``datasets`` re-encodes every ``PIL.Image.Image`` to PNG and inlines the
bytes into the Arrow file, which costs roughly 4.5 GB per 1000 document pages
and duplicates those bytes once per dataset that references the same page.

``--image-mode reference`` stores the *path* of each source image instead, in a
plain string column that keeps the ``images`` name TRL expects. The dataset then
holds no image bytes at all, and the original file is opened on demand:

- during training, :func:`wrap_trainer_collator` wraps the collator TRL built,
  so decoding happens once per batch inside the dataloader worker;
- during evaluation, ``eval.runner`` hands the path straight to its existing
  image payload handling.

Why a collator and not something more idiomatic:

- ``datasets``' ``Image(decode=False)`` defers *decoding* only. ``save_to_disk``
  reads through the feature and inlines the bytes anyway, so the dataset is no
  smaller.
- ``Dataset.with_transform`` is rejected outright by ``SFTTrainer``, and
  ``cast_column`` to an ``Image`` feature gets materialized because TRL's
  dataset preparation calls ``Dataset.map()``, which reads every row through
  the formatter and bakes the bytes back into Arrow.
- The collator is the one hook in the training path that never writes anything.

The paths themselves are recorded relative to the corpus root they were
discovered under (see :func:`relative_image_path`), with the roots written to an
``image_manifest.json`` sidecar next to ``state.json``. That makes a dataset
portable: pass ``image_root`` to remap the recorded root when the corpus has
moved or is mounted elsewhere.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from datasets import Dataset
from PIL import Image as PILImage

from utils.logging_utils import get_logger

logger = get_logger()

__all__ = [
    "DEFAULT_MAX_PIXELS",
    "IMAGE_MODES",
    "MANIFEST_NAME",
    "MANIFEST_VERSION",
    "ImageNotFoundError",
    "LazyImageCollator",
    "add_image_mode_argument",
    "cap_processor_pixels",
    "dataset_output_dir",
    "image_field",
    "image_roots",
    "is_reference_dataset",
    "load_processor",
    "read_image_manifest",
    "relative_image_path",
    "resolve_image_path",
    "save_image_dataset",
    "wrap_trainer_collator",
    "wrap_trainer_images",
]

#: Supported values for ``--image-mode``.
IMAGE_MODES = ("embed", "reference")

#: Ceiling on the pixels a document page contributes to a vision-language model.
#:
#: Scanned pages are routinely 20 megapixels, and a vision encoder turns every
#: patch into a token in the language model's sequence: an uncapped 5000x3900
#: page costs over 16000 tokens, which dominates both memory and step time while
#: adding little that a downsampled page does not still carry. This ceiling keeps
#: a full page near 1000 tokens.
DEFAULT_MAX_PIXELS = 1_003_520

#: Name of the sidecar that records the image mode and the corpus roots.
MANIFEST_NAME = "image_manifest.json"

#: Schema version of the sidecar written by :func:`write_image_manifest`.
MANIFEST_VERSION = 1

#: Dataset columns that may hold images, in the order TRL looks for them.
_IMAGE_COLUMNS = ("images", "image")

_IMAGE_MODE_HELP = (
    "How source images are stored in the saved dataset. 'embed' (default) inlines image bytes, so the "
    "dataset is self-contained but large (~4.5 GB per 1000 pages). 'reference' stores image paths instead "
    "and opens the original files on demand, which keeps the dataset tiny but requires the images to stay "
    "available at train/eval time (use image_root to remap them)."
)


class ImageNotFoundError(FileNotFoundError):
    """Raised when a stored image path cannot be resolved to an existing file."""


def cap_processor_pixels(processor: Any, max_pixels: int = DEFAULT_MAX_PIXELS) -> bool:
    """Limit how many pixels a vision-language processor feeds to its image encoder.

    Document scans are far larger than a vision encoder can use, and every patch
    becomes a token in the language model's sequence, so an uncapped page inflates
    memory and step time without adding signal. The ceiling is written to the
    processor's ``size`` mapping, which is what its resize step actually reads:
    ``max_pixels``/``min_pixels`` attributes exist on some processors but are only
    consulted while constructing ``size``, so setting them on a built processor
    has no effect.

    Args:
        processor (Any): The processor whose image encoder should be capped. Its
            image processor is capped directly; a bare image processor works too.
        max_pixels (int): Largest number of pixels the encoder may receive.

    Returns:
        bool: True when a ceiling was applied, False when the processor has no
            resizable image encoder to cap.
    """
    image_processor = getattr(processor, "image_processor", processor)
    size = getattr(image_processor, "size", None)
    # ``size`` is a transformers SizeDict (a dataclass that merely *behaves* like a
    # mapping), so it has to be probed for the key rather than tested with isinstance.
    if size is None or "longest_edge" not in size:
        return False
    if size["longest_edge"] <= max_pixels:
        return True
    # Copy before mutating: the mapping may be shared with the class-level defaults.
    try:
        capped: Any = type(size)(**dict(size))
    except (TypeError, ValueError):
        capped = dict(size)
    capped["longest_edge"] = max_pixels
    image_processor.size = capped
    return True


def load_processor(model_id: str, trust_remote_code: bool = False, max_pixels: int | None = DEFAULT_MAX_PIXELS) -> Any:
    """Load a model's processor and cap the pixels its image encoder receives.

    Trainers hand this to TRL as ``processing_class``. Passing the processor (not
    just a tokenizer) is what lets a vision-language model tokenize its image
    placeholders, and capping it here is what keeps a 20 megapixel scan from
    becoming a 16000 token prompt.

    Args:
        model_id (str): Model identifier or local path.
        trust_remote_code (bool): Whether to run model code from the hub.
        max_pixels (int | None): Largest number of pixels the image encoder may
            receive, or None to leave the processor's own limits in place.

    Returns:
        Any: The processor, ready to pass as ``processing_class``.
    """
    # Imported lazily: the data creators import this module for its path helpers
    # and should not pay for loading transformers.
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if max_pixels is not None and cap_processor_pixels(processor, max_pixels):
        logger.info("Capping image input at %s pixels", f"{max_pixels:,}")
    return processor


def _draft_size(size: tuple[int, int], max_pixels: int) -> tuple[int, int]:
    """Scale ``size`` down to at most ``max_pixels``, for a JPEG decoder's draft mode.

    Args:
        size (tuple[int, int]): The image's native ``(width, height)``.
        max_pixels (int): Largest number of pixels to allow.

    Returns:
        tuple[int, int]: The ``(width, height)`` to request from ``Image.draft``.
    """
    width, height = size
    if max_pixels <= 0 or width * height <= max_pixels:
        return size
    scale = (max_pixels / (width * height)) ** 0.5
    # Floor, not round: rounding either edge up can put the product back over the ceiling.
    return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))


def _resize_to_max_pixels(image: PILImage.Image, max_pixels: int) -> PILImage.Image:
    """Return ``image`` shrunk so it holds at most ``max_pixels``, unchanged if it already does.

    Args:
        image (PILImage.Image): The decoded image.
        max_pixels (int): Largest number of pixels to allow.

    Returns:
        PILImage.Image: The image at, or below, the ceiling.
    """
    width, height = image.size
    if max_pixels <= 0 or width * height <= max_pixels:
        return image
    target_width, target_height = _draft_size((width, height), max_pixels)
    return image.resize((target_width, target_height), PILImage.LANCZOS)


def _resolve_paths(paths: Sequence[Path | str] | Path | str | None) -> list[Path]:
    """Normalise paths to a de-duplicated list of resolved paths, order preserved.

    Unlike :func:`_dedupe_paths`, paths that no longer exist are kept: they are
    still the directories a caller asked us to search, and naming them is what
    makes a missing-image error actionable after a corpus has been deleted or
    moved.

    Args:
        paths (Sequence[Path | str] | Path | str | None): One path or an iterable of them.

    Returns:
        list[Path]: Resolved paths, without duplicates.
    """
    if paths is None:
        return []
    candidates: Sequence[Path | str] = [paths] if isinstance(paths, (str, Path)) else list(paths)
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = Path(candidate).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _dedupe_paths(paths: Sequence[Path | str] | Path | str | None) -> list[Path]:
    """Normalise paths to a de-duplicated list of resolved directories, order preserved.

    Args:
        paths (Sequence[Path | str] | Path | str | None): One path or an iterable of them.

    Returns:
        list[Path]: Existing directories, resolved, without duplicates.
    """
    return [path for path in _resolve_paths(paths) if path.is_dir()]


def add_image_mode_argument(parser: Any, default: str = "embed") -> None:
    """Add a ``--image-mode`` flag to an argparse parser.

    Shared by every surface that writes a dataset so the flag, its choices and
    its help text stay identical everywhere.

    Args:
        parser (Any): The ``argparse.ArgumentParser`` to extend.
        default (str): The default mode. Embedding stays the default so that
            existing commands and saved datasets keep behaving exactly as before.
    """
    parser.add_argument("--image-mode", choices=IMAGE_MODES, default=default, help=_IMAGE_MODE_HELP)


def write_image_manifest(output_dir: Path | str, roots: Sequence[Path | str], image_mode: str) -> None:
    """Record the image mode and corpus roots alongside a saved dataset.

    Only reference-mode datasets get a sidecar; the absence of one means the
    dataset embeds its images, which is what datasets saved before this feature
    existed do.

    Args:
        output_dir (Path | str): Directory the dataset was saved to.
        roots (Sequence[Path | str]): Corpus roots the stored paths are relative to.
        image_mode (str): One of :data:`IMAGE_MODES`.
    """
    if image_mode != "reference":
        return
    manifest = {
        "version": MANIFEST_VERSION,
        "image_mode": image_mode,
        "roots": [str(root) for root in _dedupe_paths(roots)],
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def read_image_manifest(dataset_dir: Path | str) -> dict[str, Any] | None:
    """Read the image sidecar of a saved dataset.

    Args:
        dataset_dir (Path | str): Directory containing ``state.json``.

    Returns:
        dict[str, Any] | None: The manifest, or None when the dataset embeds its images.

    Raises:
        ImageNotFoundError: If the sidecar exists but is not readable JSON.
    """
    manifest_path = Path(dataset_dir) / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ImageNotFoundError(f"Could not read image manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ImageNotFoundError(f"Image manifest {manifest_path} is not a JSON object")
    return manifest


def dataset_output_dir(dataset: Dataset) -> Path | None:
    """Locate the directory a dataset was loaded from, for sidecar lookup.

    Args:
        dataset (Dataset): A dataset produced by ``load_from_disk``.

    Returns:
        Path | None: The dataset directory, or None when it cannot be determined
            (for example a dataset built in memory with ``from_list``).
    """
    cache_files = getattr(dataset, "cache_files", None) or []
    for entry in cache_files:
        filename = entry.get("filename") if isinstance(entry, dict) else None
        if filename:
            return Path(filename).parent
    return None


def relative_image_path(image_path: Path | str, roots: Sequence[Path | str] | Path | str | None) -> str:
    """Express an image path relative to the corpus root it was discovered under.

    Relative paths keep a saved dataset meaningful when the corpus is moved or
    remounted; :func:`resolve_image_path` puts them back together with a root.
    A relative input is first located under the roots, and images outside every
    root are stored absolute rather than guessed at.

    Args:
        image_path (Path | str): The source image, absolute or relative to a root.
        roots (Sequence[Path | str] | Path | str | None): Corpus roots to relativise against.

    Returns:
        str: The path relative to the first matching root, else the absolute path.
    """
    given = Path(image_path)
    resolved = (
        given
        if given.is_absolute()
        else next(
            (base / given for base in _dedupe_paths(roots) if (base / given).is_file()),
            given.resolve(),
        )
    )
    for root in _dedupe_paths(roots):
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            continue
    return str(resolved)


def resolve_image_path(
    value: Path | str,
    roots: Sequence[Path | str] | Path | str | None = None,
    image_root: Path | str | None = None,
) -> Path:
    """Turn a stored image path back into an existing file.

    Candidates are tried nearest-first: the remapped ``image_root`` (so a corpus
    that moved or is mounted elsewhere still works), then each root recorded in
    the manifest, then the working directory for a bare relative path, and
    finally the value as given.

    Args:
        value (Path | str): The stored path, relative or absolute.
        roots (Sequence[Path | str] | Path | str | None): Roots recorded in the manifest.
        image_root (Path | str | None): Replacement root for a moved corpus.

    Returns:
        Path: The first candidate that exists.

    Raises:
        ImageNotFoundError: If no candidate is an existing file. Names the stored
            path and lists every directory that was tried.
    """
    stored = Path(value)
    if stored.is_absolute() and stored.is_file():
        return stored

    tried: list[Path] = []
    for base in [Path(image_root).resolve()] if image_root else []:
        tried.append(base)
    tried.extend(_resolve_paths(roots))

    for base in tried:
        candidate = base / stored
        if candidate.is_file():
            return candidate
    if stored.is_file():
        return stored

    searched = ", ".join(str(base) for base in tried) or "no roots recorded"
    raise ImageNotFoundError(f"Source image '{value}' not found. Searched for it under: {searched}")


def image_roots(
    input_dirs: Sequence[Path | str] | Path | str,
    image_dirs: Sequence[Path | str] | Path | str | None = None,
) -> list[Path]:
    """Collect the roots that stored image paths should be relative to.

    Explicit image directories come first because ``utils.discovery.resolve_image``
    prefers them; the corpus roots follow, which cover images found from the
    corpus layout (the PAGE XML ``imageFilename`` attribute or the directory above
    a ``page`` directory).

    Args:
        input_dirs (Sequence[Path | str] | Path | str): Corpus root(s) given to the builder.
        image_dirs (Sequence[Path | str] | Path | str | None): Explicit image director(ies).

    Returns:
        list[Path]: Existing directories, resolved, image directories first.
    """
    explicit_dirs = None if image_dirs is None else ([image_dirs] if isinstance(image_dirs, (str, Path)) else list(image_dirs))
    corpus_dirs = [input_dirs] if isinstance(input_dirs, (str, Path)) else list(input_dirs)
    return _dedupe_paths([*(explicit_dirs or []), *corpus_dirs])


def image_field(
    image_path: Path | str,
    image_mode: str,
    roots: Sequence[Path | str] | Path | str | None = None,
) -> list[Any]:
    """Build the value of a dataset ``images`` column for one source image.

    Embedding opens the image so ``datasets`` inlines its bytes; referencing
    stores the path relative to the corpus root instead.

    Args:
        image_path (Path | str): The resolved source image.
        image_mode (str): One of :data:`IMAGE_MODES`.
        roots (Sequence[Path | str] | Path | str | None): Corpus roots to relativise
            against in reference mode.

    Returns:
        list[Any]: A one-element list holding a ``PIL.Image.Image`` or a path string,
            matching the shape TRL expects for a single-image sample.

    Raises:
        ValueError: If ``image_mode`` is not one of :data:`IMAGE_MODES`.
    """
    if image_mode == "reference":
        return [relative_image_path(image_path, roots)]
    if image_mode == "embed":
        with PILImage.open(image_path) as opened:
            opened.load()
            return [opened]
    raise ValueError(f"Unknown image mode '{image_mode}'. Expected one of: {', '.join(IMAGE_MODES)}")


def _string_column(dataset: Dataset, column: str) -> bool:
    """Report whether a dataset column holds plain strings rather than decoded images.

    Introspects the feature only, so it never triggers an image decode.

    Args:
        dataset (Dataset): The dataset to inspect.
        column (str): Column name.

    Returns:
        bool: True when the column is string-typed.
    """
    features = getattr(dataset, "features", None)
    if features is None or column not in features:
        return False
    feature = features[column]
    while hasattr(feature, "feature"):
        feature = feature.feature
    return getattr(feature, "dtype", None) == "string"


def is_reference_dataset(dataset: Dataset) -> bool:
    """Detect a dataset that stores image paths instead of image bytes.

    Args:
        dataset (Dataset): The dataset to inspect.

    Returns:
        bool: True when an image column exists and is string-typed.
    """
    return any(_string_column(dataset, column) for column in _IMAGE_COLUMNS)


def save_image_dataset(
    records: Sequence[dict[str, Any]],
    output_dir: Path | str,
    image_mode: str,
    roots: Sequence[Path | str] | Path | str | None = None,
) -> Dataset:
    """Build and save a dataset, recording how its images are stored.

    Args:
        records (Sequence[dict[str, Any]]): The rows to save.
        output_dir (Path | str): Directory to save the dataset to.
        image_mode (str): One of :data:`IMAGE_MODES`.
        roots (Sequence[Path | str] | Path | str | None): Corpus roots the stored
            image paths are relative to.

    Returns:
        Dataset: The dataset that was saved.

    Raises:
        ValueError: If ``image_mode`` is not one of :data:`IMAGE_MODES`.
    """
    if image_mode not in IMAGE_MODES:
        raise ValueError(f"Unknown image mode '{image_mode}'. Expected one of: {', '.join(IMAGE_MODES)}")
    dataset = Dataset.from_list(list(records))
    dataset.save_to_disk(str(output_dir))
    write_image_manifest(output_dir, roots, image_mode)
    return dataset


class LazyImageCollator:
    """Collator wrapper that opens reference-mode images just before collation.

    Wrapping the collator TRL built (rather than replacing it) keeps TRL
    responsible for choosing the right collator and its processor, padding and
    truncation settings. Anything not defined here is forwarded to the wrapped
    collator, so the wrapper is transparent to the ``Trainer``.
    """

    def __init__(
        self,
        collator: Any,
        roots: Sequence[Path | str] | Path | str | None = None,
        image_root: Path | str | None = None,
        max_pixels: int | None = DEFAULT_MAX_PIXELS,
    ) -> None:
        """Store the wrapped collator and the roots its image paths resolve against.

        Args:
            collator (Any): The collator to delegate to once images are decoded.
            roots (Sequence[Path | str] | Path | str | None): Roots recorded in the manifest.
            image_root (Path | str | None): Replacement root for a moved corpus.
            max_pixels (int | None): Largest number of pixels to decode an image at,
                or None to decode at the native resolution of the source file.
        """
        self._collator = collator
        self._roots = _resolve_paths(roots)
        self._image_root = image_root
        self._max_pixels = max_pixels

    @property
    def wrapped_collator(self) -> Any:
        """The collator this wrapper delegates to."""
        return self._collator

    @property
    def roots(self) -> list[Path]:
        """The directories stored image paths are resolved against."""
        return list(self._roots)

    def add_roots(self, roots: Sequence[Path | str] | Path | str | None) -> None:
        """Extend the roots used to resolve image paths.

        Lets one wrapper serve several datasets, each with its own manifest roots.

        Args:
            roots (Sequence[Path | str] | Path | str | None): Additional roots to try.
        """
        self._roots = _resolve_paths([*self._roots, *(roots or [])])

    def __call__(self, examples: list[dict[str, Any]], return_tensors: str | None = None) -> Any:
        """Decode every image path in the batch, then delegate to the wrapped collator.

        Args:
            examples (list[dict[str, Any]]): Raw rows, whose image column holds path strings.
            return_tensors (str | None): Forwarded to the wrapped collator; when omitted
                the collator's own default applies.

        Returns:
            Any: The collated batch produced by the wrapped collator.

        Raises:
            ImageNotFoundError: If a stored path no longer resolves to a file.
        """
        decoded = [self._decode_images(example) for example in examples]
        if return_tensors is None:
            return self._collator(decoded)
        return self._collator(decoded, return_tensors=return_tensors)

    def _decode_images(self, example: dict[str, Any]) -> dict[str, Any]:
        """Replace string image entries with opened images, leaving other rows untouched.

        Args:
            example (dict[str, Any]): One raw dataset row.

        Returns:
            dict[str, Any]: A copy of the row with decoded images, or the row itself
                when it carries no image paths.
        """
        for column in _IMAGE_COLUMNS:
            values = example.get(column)
            if not isinstance(values, list) or not any(isinstance(value, str) for value in values):
                continue
            opened = [self._open(value) if isinstance(value, str) else value for value in values]
            updated = dict(example)
            updated[column] = opened
            return updated
        return example

    def _open(self, value: str) -> PILImage.Image:
        """Open one stored image path, downscaled to :attr:`max_pixels`.

        Args:
            value (str): The path recorded in the dataset.

        Returns:
            PILImage.Image: The opened image.

        Raises:
            ImageNotFoundError: If the path does not resolve to an existing file.
        """
        path = resolve_image_path(value, self._roots, self._image_root)
        with PILImage.open(path) as opened:
            # Ask a JPEG decoder to sample its DCT blocks down before it builds
            # the full raster. Without this, decoding a 20 megapixel scan costs
            # ~58 MB of pixels per page even though we discard most of them below.
            if self._max_pixels is not None:
                opened.draft(None, _draft_size(opened.size, self._max_pixels))
            # Force the pixel data in before the file handle closes, so the image
            # stays usable once this method returns.
            opened.load()
            # draft() only snaps to a JPEG's available power-of-two reductions, so
            # it can land above the ceiling; resize() makes the bound exact.
            if self._max_pixels is not None:
                opened = _resize_to_max_pixels(opened, self._max_pixels)
            return opened

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the wrapped collator.

        Args:
            name (str): Attribute name.

        Returns:
            Any: The attribute of the wrapped collator.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._collator, name)


def wrap_trainer_collator(
    trainer: Any,
    dataset: Dataset,
    image_root: Path | str | None = None,
    dataset_dir: Path | str | None = None,
    max_pixels: int | None = DEFAULT_MAX_PIXELS,
) -> bool:
    """Make a trainer decode reference-mode images lazily, if its dataset uses them.

    Call this *after* the trainer has been constructed: TRL picks and configures
    the collator itself (including the vision collator and its processor), and
    the collator is only read once the dataloader iterates.

    Args:
        trainer (Any): The constructed TRL/HF trainer.
        dataset (Dataset): The dataset whose image column may hold paths.
        image_root (Path | str | None): Replacement root for a corpus that moved.
        dataset_dir (Path | str | None): Dataset directory to read the sidecar from;
            discovered from ``dataset`` when omitted.
        max_pixels (int | None): Largest number of pixels to decode an image at,
            or None to decode at the native resolution of the source file.

    Returns:
        bool: True when the collator was wrapped, False when the dataset already
            embeds its images and nothing needed to change.
    """
    if not is_reference_dataset(dataset):
        return False

    directory = dataset_dir or dataset_output_dir(dataset)
    manifest = read_image_manifest(directory) if directory else None
    roots = manifest.get("roots", []) if manifest else []

    existing = trainer.data_collator
    if isinstance(existing, LazyImageCollator):
        # One wrapper already serves this trainer (e.g. the train dataset); let it
        # resolve this dataset's paths too instead of nesting a second wrapper.
        existing.add_roots(roots)
        return True

    trainer.data_collator = LazyImageCollator(existing, roots, image_root, max_pixels)
    return True


def wrap_trainer_images(
    trainer: Any,
    train_dataset: Dataset,
    eval_dataset: Dataset | None = None,
    image_root: Path | str | None = None,
    max_pixels: int | None = DEFAULT_MAX_PIXELS,
) -> bool:
    """Make a trainer decode reference-mode images lazily for both its datasets.

    The collator is shared by the train and eval dataloaders, so a single wrapper
    covers both; it resolves each row's paths against the manifest of the dataset
    the row came from.

    Args:
        trainer (Any): The constructed TRL/HF trainer.
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset | None): The evaluation dataset, when configured.
        image_root (Path | str | None): Replacement root for a corpus that moved.
        max_pixels (int | None): Largest number of pixels to decode an image at,
            or None to decode at the native resolution of the source file.

    Returns:
        bool: True when the collator was wrapped, False when nothing needed to change.
    """
    wrapped = wrap_trainer_collator(trainer, train_dataset, image_root, max_pixels=max_pixels)
    if eval_dataset is not None and wrap_trainer_collator(trainer, eval_dataset, image_root, max_pixels=max_pixels):
        wrapped = True
    return wrapped
