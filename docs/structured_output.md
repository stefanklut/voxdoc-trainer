# Structured DocLang output

Constrain an OpenAI-compatible endpoint (e.g. a local **vLLM** server) to emit
**only valid DocLang** (``.dclg`` XML). The package lives in
`src/doclang_structured/` and is driven by two complementary mechanisms, both
gated by the official `doclang` validator so the returned string is guaranteed
valid.

## Why two paths

DocLang's validity is only partly expressible as a context-free grammar or a
JSON schema. Rules such as table rectangularity, location ordering, field
nesting, and thread consistency are *context-sensitive* — no CFG or JSON schema
can express them. So every generated document is run through the official
`doclang.validate()` (XSD + Schematron) before it is returned, and the two
generation paths are simply two ways of getting the model close enough that
validation (with a bounded number of retries) succeeds.

| Path | Mechanism | Model emits | Best when |
| --- | --- | --- | --- |
| **Grammar** (primary) | EBNF grammar via vLLM structured decoding | `.dclg` directly | the server supports a grammar backend (xgrammar, guidance, outlines, lm-format-enforcer) |
| **JSON schema** (fallback) | Pydantic → JSON schema via `response_format` | JSON, then serialized to `.dclg` | the server only supports JSON-schema constrained decoding |

## The two paths

### 1. Grammar (primary)

`generate_doclang_grammar()` sends the EBNF grammar
(`DOCLANG_GRAMMAR`, GBNF dialect) to the server so the model is constrained,
token by token, to emit well-formed `.dclg`. On **vLLM ≥ 0.29** the grammar is
passed through the request's `extra_body`:

```python
extra_body = {
    "structured_outputs": {"grammar": DOCLANG_GRAMMAR},
    "guided_decoding_backend": "xgrammar",  # or guidance / outlines / lm-format-enforcer / auto
}
```

> The older `guided_decoding` dict is no longer accepted by vLLM ≥ 0.29; the
> grammar now travels in `structured_outputs` with a separate
> `guided_decoding_backend` string.

### 2. JSON schema (fallback)

`generate_doclang_json_schema()` constrains the model to emit JSON matching the
DocLang schema (via `response_format`), parses it into the `DocLang` Pydantic
model, serializes it to `.dclg`, and validates. `strict=False` by default
because the DocLang schema uses a discriminated union, which is outside
OpenAI's strict-mode subset.

## Client API

```python
from src.doclang_structured import (
    generate_doclang_grammar,      # primary
    generate_doclang_json_schema,  # fallback
)

messages = [{"role": "user", "content": "Transcribe this document as DocLang."}]

# Primary: grammar-constrained, emits .dclg directly.
xml = generate_doclang_grammar(
    base_url="http://localhost:8000/v1",
    model="my-model",
    messages=messages,
    backend="xgrammar",
    max_retries=3,
    temperature=0.0,
)

# Fallback: JSON-schema-constrained, then serialized to .dclg.
xml = generate_doclang_json_schema(
    base_url="http://localhost:8000/v1",
    model="my-model",
    messages=messages,
    max_retries=3,
    temperature=0.0,
)
```

Both functions:

- accept `messages` whose `content` is a string **or** a list of content blocks
  (e.g. `{"type": "image_url", "image_url": {"url": ...}}`) for multimodal input;
- retry up to `max_retries` times if the output fails validation;
- return a string **guaranteed** to pass `validate_doclang`;
- raise `DocLangValidationError` if no attempt produces valid DocLang.

### Other public helpers

```python
from src.doclang_structured import (
    DOCLANG_GRAMMAR,            # the GBNF grammar string
    get_grammar,                # -> DOCLANG_GRAMMAR
    DocLang,                    # the root Pydantic model
    ELEMENT_MODELS,             # element name -> model class (57 entries)
    get_doclang_json_schema,    # strict-compatible JSON schema
    response_format,            # -> a response_format spec for the client
    model_to_doclang_xml,       # DocLang -> .dclg string
    doclang_xml_to_model,       # .dclg string -> DocLang
    validate_doclang,           # official validator wrapper (xsd_only= for speed)
    DocLangValidationError,
)
```

## Command line

`tools/generate_doclang.py` wraps the client for one-shot generation:

```bash
# Grammar path (default), text prompt, write to a file.
python tools/generate_doclang.py \
    --base-url http://localhost:8000/v1 --model my-model \
    --prompt "Transcribe this document." --out doc.dclg

# With a document image (local file or URL), JSON-schema fallback.
python tools/generate_doclang.py \
    --base-url http://localhost:8000/v1 --model my-model \
    --image scan.png --out doc.dclg --mode json_schema
```

Local image files are inlined as base64 data URIs; anything that is not an
existing file is passed through as a URL. Omit `--out` to print the document on
stdout. The document is validated by the client before it is written, so the
output is always valid.

## Test suite

The structured-output package is covered by a dedicated test suite under
`tests/`. The centerpiece is a **sync guarantee**: the grammar, the Pydantic
models, and the JSON schema are continuously checked against the XSD bundled in
the installed `doclang` package, so a `doclang` upgrade that changes the schema
fails the build loudly instead of silently producing invalid output.

| File | What it checks |
| --- | --- |
| `tests/test_doclang_sync.py` | **Sync guarantee** — grammar / Pydantic / JSON-schema element + enum inventories match the XSD; the spec version is pinned; the root element matches. Fails with a re-sync pointer on drift. |
| `tests/doclang_xsd.py` | Shared inventory extraction (XSD, grammar, Pydantic, JSON schema) used by the sync tests. |
| `tests/test_doclang_grammar.py` | The grammar compiles; it accepts every valid corpus doc; it rejects every CFG-expressible invalid doc. |
| `tests/test_doclang_corpus.py` | The valid corpus passes full validation and round-trips; the invalid corpus fails full validation; the corpus covers every XSD element; the invalid manifest's mechanisms are consistent. |
| `tests/test_doclang_models.py` | Pydantic model invariants (xref/href exclusivity, location counts, `class` alias, enum domains, pinned version, extra-field rejection). |
| `tests/test_doclang_serialize.py` | model → XML → model round-trip stability; escaping; whitespace `<content>` wrapping; canonical head order; root namespace + version. |
| `tests/test_doclang_fuzz.py` | Hypothesis (fixed seed): random model → serialize is grammar-accepted and XSD-valid; round-trip is idempotent; differential mutation test over corpus docs (tag swap, bogus enum, dropped tag, reordered head, broken table, xref+href) — anything the grammar accepts must be XSD-valid, and grammar rejections of XSD-valid mutants are reported as completeness gaps (without failing). |
| `tests/test_doclang_client.py` | Client control flow (mocked): what is sent on the wire, retry behaviour, error handling, multimodal pass-through. |
| `tests/e2e/test_doclang_structured.py` | Live end-to-end against a real vLLM server (skipped unless `VOXDOC_VLLM_BASE_URL` is set). |

Corpora live in `tests/data/doclang_corpus/` (valid) and
`tests/data/doclang_invalid/` (invalid, with a `manifest.json` recording which
mechanism — grammar, validator, or both — catches each violation). They are
gitignored build artifacts, regenerated by `tests/generators/build_doclang_corpus.py`
and `tests/generators/build_doclang_invalid.py`. `tests/conftest.py` runs these
builders automatically before collection when the corpora are missing, so a fresh
clone works with no manual step.

Run the non-e2e suite:

```bash
python -m pytest tests/ -m "not e2e"
```

Run the live e2e tests against a running server:

```bash
VOXDOC_VLLM_BASE_URL=http://localhost:8000/v1 \
VOXDOC_VLLM_MODEL=my-model \
python -m pytest tests/e2e/test_doclang_structured.py -m e2e -v
```

## Compatibility notes

- **vLLM ≥ 0.29** is required for the grammar path (the `structured_outputs`
  API). The grammar path needs a structured-output backend installed on the
  server (`xgrammar` by default; `guidance`, `outlines`, and
  `lm-format-enforcer` also work).
- The **JSON-schema path** works on any OpenAI-compatible server that supports
  `response_format`, and is the portable fallback.
- The XSD anchor is read from the installed `doclang` package
  (`doclang/doclang.xsd`). The pinned spec version is asserted in
  `tests/test_doclang_sync.py`; bump it there (and re-sync the grammar / models
  / schema) when upgrading `doclang`.
- Full validation (XSD + Schematron) shells out to a JRE and is slow
  (~1–2 s/call). Use `validate_doclang(xml, xsd_only=True)` for the fast
  XSD-only path in tight loops (the fuzz tests do this).
