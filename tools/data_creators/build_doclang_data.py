"""Build SFT training data that teaches a model to emit DocLang natively.

DocLang (https://doclang.ai/) is an AI-native document format designed for LLMs
to consume directly. If a model does not support it natively, we teach it to by
giving it many examples of existing OCR already in DocLang format.

This tool turns existing DocLang documents and their source images into SFT
examples: the model sees a document image and must produce the DocLang
representation as its completion. The completion is the raw DocLang markup, so
the model learns the format itself.

``--input`` accepts one or more corpus roots and is searched recursively for
``.dclg`` files, so documents may be nested at any depth. ``--images`` is
optional and also accepts multiple roots; when it is omitted, images are found
from the corpus layout itself (the document's own directory, or the directory
above a ``page`` directory).

Every document is validated with the official `doclang` package before it is
added as an SFT record. Documents that fail validation, or that have no matching
source image, are skipped with a warning.

Note: DocLang is an open standard (Linux Foundation). Documents use the `.dclg`
extension.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from doclang import ValidationError, validate

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..", "..")))
from utils.discovery import (  # noqa: E402, F401  (find_image is re-exported for callers)
    DiscoveryError,
    discover_doclang,
    find_image,
    resolve_image,
)
from utils.image_loading import (  # noqa: E402
    add_image_mode_argument,
    image_field,
    image_roots,
    save_image_dataset,
)


def build_sft_record(
    doclang_path: Path,
    image_path: Path,
    image_mode: str = "embed",
    roots: Sequence[Path] | None = None,
) -> dict:
    """Build a single SFT record teaching the model to emit DocLang.

    Args:
        doclang_path (Path): Path to an existing DocLang document.
        image_path (Path): Path to the source document image.
        image_mode (str): One of ``utils.image_loading.IMAGE_MODES``; "embed" inlines
            the image bytes, "reference" stores the image path instead.
        roots (Sequence[Path] | None): Corpus roots that stored paths are made relative to.

    Returns:
        dict: An SFT record with prompt, completion (the DocLang markup), and images.
    """
    doclang_text = doclang_path.read_text(encoding="utf-8").strip()

    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Transcribe this historical document into DocLang format."},
            ],
        }
    ]
    completion = [{"role": "assistant", "content": [{"type": "text", "text": doclang_text}]}]
    images = image_field(image_path, image_mode, roots)
    return {"prompt": prompt, "completion": completion, "images": images}


def build_dataset(
    input_dir: Path | Sequence[Path],
    image_dir: Path | Sequence[Path] | None = None,
    validator: Callable[[Path], None] = validate,
    *,
    image_mode: str = "embed",
) -> tuple[list[dict], int]:
    """Build SFT records from the DocLang documents discovered under the input roots.

    Each document is validated before it is added as an SFT record; documents
    that fail validation or have no matching source image are skipped with a
    warning.

    Args:
        input_dir (Path | Sequence[Path]): Corpus root(s); ``.dclg`` files are discovered
            recursively at any depth.
        image_dir (Path | Sequence[Path] | None): Explicit source image directories.
            When omitted, images are located from the corpus layout.
        validator (Callable[[Path], None]): Validation callable invoked per document;
            should raise on invalid input. Defaults to the official `doclang.validate`.
        image_mode (str): One of ``utils.image_loading.IMAGE_MODES``. "embed" (the
            default) stores image bytes in the dataset; "reference" stores paths.

    Returns:
        tuple[list[dict], int]: The list of SFT records and the number of skipped files.

    Raises:
        DiscoveryError: If no DocLang file is found under the input roots.
    """
    records: list[dict] = []
    skipped = 0
    roots = image_roots(input_dir, image_dir)
    for doclang_file in discover_doclang(input_dir):
        try:
            validator(doclang_file)
        except ValidationError as exc:
            print(f"Skipping {doclang_file.name}: invalid DocLang ({exc})")
            skipped += 1
            continue
        image_path = resolve_image(doclang_file, image_dir)
        if image_path is None:
            print(f"Skipping {doclang_file.name}: no source image found")
            skipped += 1
            continue
        records.append(build_sft_record(doclang_file, image_path, image_mode, roots))
    return records, skipped


def main() -> None:
    """Build an SFT dataset from existing DocLang documents."""
    parser = argparse.ArgumentParser(description="Build SFT training data from existing DocLang documents.")
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        type=Path,
        help="Corpus root(s); DocLang (.dclg) files are discovered recursively at any depth.",
    )
    parser.add_argument("--output", required=True, help="Path to save the HF dataset.")
    parser.add_argument(
        "--images",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Optional source image director(ies). When omitted, images are found from the corpus "
            "layout (the document's own directory, or the directory above a page directory)."
        ),
    )
    add_image_mode_argument(parser)
    args = parser.parse_args()

    try:
        records, skipped = build_dataset(args.input, args.images, image_mode=args.image_mode)
    except (DiscoveryError, NotADirectoryError) as exc:
        parser.error(str(exc))

    dataset = save_image_dataset(records, args.output, args.image_mode, image_roots(args.input, args.images))
    print(f"Built {len(dataset)} DocLang SFT examples ({args.image_mode} images) -> {args.output}")
    if skipped:
        print(f"Skipped {skipped} DocLang file(s)")


if __name__ == "__main__":
    main()
