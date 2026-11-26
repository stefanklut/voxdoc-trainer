"""Generate a validated DocLang (``.dclg``) document from a vLLM server.

Two modes:

* ``grammar`` (default) — sends the EBNF grammar so the model emits ``.dclg``
  directly (the primary path).
* ``json_schema`` — constrains the model to JSON matching the DocLang schema,
  then serializes JSON → ``.dclg`` (the fallback path).

Both paths are gated by the official ``doclang`` validator, so the output is
guaranteed valid.

Examples::

    python tools/generate_doclang.py \\
        --base-url http://localhost:8000/v1 --model mymodel \\
        --prompt "Transcribe this document." --out doc.dclg

    python tools/generate_doclang.py \\
        --base-url http://localhost:8000/v1 --model mymodel \\
        --image scan.png --out doc.dclg --mode json_schema

``--image`` may also be a **directory**: every image it contains is generated
in turn and written as ``<out>/<stem>.dclg``, so a whole corpus can be
converted in one run. Images that fail are reported and skipped, and the
process exits non-zero if any of them failed.

    python tools/generate_doclang.py \\
        --base-url http://localhost:8000/v1 --model mymodel \\
        --image scans/ --out doclang/
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import sys
from pathlib import Path

# Allow running as a plain script from the repo root (``python tools/...``).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.doclang_structured.client import (  # noqa: E402
    generate_doclang_grammar,
    generate_doclang_json_schema,
)
from utils.discovery import iter_image_files  # noqa: E402


def _image_block(image: str) -> dict:
    """Build an OpenAI image content block from a URL or a local file path.

    Local files are inlined as base64 data URIs; anything that is not an
    existing file is passed through as a URL.
    """
    path = Path(image)
    if path.is_file():
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        url = f"data:{mime};base64,{b64}"
    else:
        url = image
    return {"type": "image_url", "image_url": {"url": url}}


def build_messages(prompt: str, image: str | None) -> list[dict]:
    """Build the chat messages (text, plus an optional image block)."""
    if image:
        content: str | list[dict] = [
            _image_block(image),
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt
    return [{"role": "user", "content": content}]


def _generate(base_url: str, model: str, messages: list[dict], args: argparse.Namespace) -> str:
    """Run one structured generation and return the validated DocLang markup.

    Args:
        base_url (str): vLLM OpenAI-compatible endpoint.
        model (str): Served model identifier.
        messages (list[dict]): Chat messages for the request.
        args (argparse.Namespace): Parsed CLI options supplying mode, backend,
            retries, temperature and API key.

    Returns:
        str: The validated DocLang document.
    """
    if args.mode == "grammar":
        return generate_doclang_grammar(
            base_url,
            model,
            messages,
            api_key=args.api_key,
            backend=args.backend,
            max_retries=args.max_retries,
            temperature=args.temperature,
        )
    return generate_doclang_json_schema(
        base_url,
        model,
        messages,
        api_key=args.api_key,
        max_retries=args.max_retries,
        temperature=args.temperature,
    )


def _write_out(xml: str, out: str | None) -> None:
    """Write DocLang markup to ``out``, or print it when ``out`` is None."""
    if out is None:
        print(xml)
        return
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml + "\n", encoding="utf-8")


def _run_batch(image_dir: Path, out_dir: Path, args: argparse.Namespace) -> int:
    """Generate one ``.dclg`` per image in ``image_dir`` into ``out_dir``.

    Failures are reported per image and do not abort the batch.

    Args:
        image_dir (Path): Directory of document images.
        out_dir (Path): Directory to write ``<stem>.dclg`` files into.
        args (argparse.Namespace): Parsed CLI options.

    Returns:
        int: The number of images that failed.
    """
    images = iter_image_files(image_dir)
    if not images:
        print(f"No images found in {image_dir}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for index, image in enumerate(images, start=1):
        out_path = out_dir / f"{image.stem}.dclg"
        try:
            xml = _generate(args.base_url, args.model, build_messages(args.prompt, str(image)), args)
        except Exception as exc:  # noqa: BLE001 - one bad image must not abort the batch
            failed += 1
            print(f"[{index}/{len(images)}] {image.name}: FAILED ({exc})", file=sys.stderr)
            continue
        _write_out(xml, str(out_path))
        print(f"[{index}/{len(images)}] {image.name} -> {out_path}", file=sys.stderr)
    return failed


def main() -> None:
    """CLI entry point for structured DocLang generation."""
    parser = argparse.ArgumentParser(
        description="Generate a validated DocLang document from a vLLM server."
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="vLLM OpenAI-compatible endpoint (e.g. http://localhost:8000/v1).",
    )
    parser.add_argument("--model", required=True, help="Served model identifier.")
    parser.add_argument(
        "--prompt",
        default="Transcribe this document as DocLang.",
        help="User prompt.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to a document image, an image URL, or a directory of images (batch mode).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output .dclg path, or a directory for batch mode (defaults to printing on stdout).",
    )
    parser.add_argument(
        "--mode",
        choices=["grammar", "json_schema"],
        default="grammar",
        help="Structured-output path to use.",
    )
    parser.add_argument(
        "--backend",
        default="xgrammar",
        help="Structured-output backend (grammar mode only).",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Attempts if output is invalid."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature."
    )
    parser.add_argument(
        "--api-key", default="EMPTY", help="API key (EMPTY for a local server)."
    )
    args = parser.parse_args()

    if args.image is not None and Path(args.image).is_dir():
        if not args.out:
            parser.error("--out is required when --image is a directory")
        failed = _run_batch(Path(args.image), Path(args.out), args)
        total = len(iter_image_files(Path(args.image)))
        print(f"Batch complete: {total - failed}/{total} document(s) generated", file=sys.stderr)
        if failed:
            sys.exit(1)
        return

    xml = _generate(args.base_url, args.model, build_messages(args.prompt, args.image), args)

    # The client already ran the full validator before returning, so the
    # document is guaranteed valid at this point.
    _write_out(xml, args.out)
    if args.out:
        print(f"Wrote validated DocLang to {args.out}", file=sys.stderr)
    print("Validation: OK (guaranteed by client)", file=sys.stderr)


if __name__ == "__main__":
    main()
