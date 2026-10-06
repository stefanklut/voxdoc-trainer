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
        help="Path to a document image, or an image URL.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output .dclg path (defaults to printing on stdout).",
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

    messages = build_messages(args.prompt, args.image)
    if args.mode == "grammar":
        xml = generate_doclang_grammar(
            args.base_url,
            args.model,
            messages,
            api_key=args.api_key,
            backend=args.backend,
            max_retries=args.max_retries,
            temperature=args.temperature,
        )
    else:
        xml = generate_doclang_json_schema(
            args.base_url,
            args.model,
            messages,
            api_key=args.api_key,
            max_retries=args.max_retries,
            temperature=args.temperature,
        )

    # The client already ran the full validator before returning, so the
    # document is guaranteed valid at this point.
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(xml + "\n", encoding="utf-8")
        print(f"Wrote validated DocLang to {out_path}", file=sys.stderr)
    else:
        print(xml)
    print("Validation: OK (guaranteed by client)", file=sys.stderr)


if __name__ == "__main__":
    main()
