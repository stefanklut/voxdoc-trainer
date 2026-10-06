# Plan: DocLang Structured Output (EBNF grammar + Pydantic fallback) + Full Test Suite

> **Status: COMPLETE — audited & verified 2026-10-02.** All phases implemented; full non-e2e suite green (**91 passed**, 6 e2e deselected). Per-phase ✅ marks below; post-build deviations in the implementation notes; what was actually run is logged at the end of "Verification".

## TL;DR

Make a model on an **OpenAI-compatible endpoint (vLLM, local)** emit **only valid DocLang (`.dclg` XML)**, via two complementary mechanisms (user decision "Both"):

1. **PRIMARY — EBNF grammar + vLLM `guided_decoding`**: a context-free EBNF grammar for the full DocLang spec constrains the raw token stream → model emits `.dclg` **directly** (no JSON, no serializer).
2. **FALLBACK — Pydantic model + JSON schema**: Pydantic v2 models mirroring DocLang → `model_json_schema()` → standard `response_format={"type":"json_schema"}` (portable) → JSON→`.dclg` serializer; also gives typed parsing of `.dclg`.

Both paths are gated by the official **`doclang.validate()` + retry**, because a CFG/JSON-schema cannot express DocLang's *context-sensitive* rules (table rectangular rule, `x0≤x1 / y0≤y1` location ordering, `xref`/`href` exclusivity).

**NEW (this revision): a full test suite whose centerpiece is a *sync guarantee* — the EBNF grammar and the Pydantic JSON schema are continuously checked against the XSD that ships inside the `doclang` validator package** (inventory match + version guard + behavioral differential testing + corpus coverage). When the `doclang` package is upgraded, the suite fails loudly and points at exactly what to re-sync.

**Scope:** full DocLang spec. **Target endpoint:** local vLLM (xgrammar). **New package:** `src/doclang_structured/` + CLI `tools/generate_doclang.py` + `tests/` suite + `tests/data/` corpus.

> **Implementation notes (post-build — where the build diverged from this plan):**
>
> 1. **vLLM structured-output API.** The installed vLLM (0.29.0) no longer accepts the `guided_decoding` dict referenced above (TL;DR item 1, "Key technical facts", and the `client.py` sketch). The grammar is sent as `extra_body={"structured_outputs": {"grammar": DOCLANG_GRAMMAR}, "guided_decoding_backend": backend}` (backends: `auto`/`xgrammar`/`guidance`/`outlines`/`lm-format-enforcer`). See `src/doclang_structured/client.py` and `docs/structured_output.md`.
> 2. **Actual XSD scope.** The authoritative XSD bundled in the `doclang` package (`doclang/doclang.xsd`, spec version **0.7.3**) defines **57 elements** — a *subset* of the broader spec list in "Key technical facts" (e.g. `track`, `cover`, `frame`, `audio`, `voice`, `chapter`, `hours`/`minutes`/`seconds`/`msecs`, `bdiv`, `lcel` are **not** present). The grammar, Pydantic models, and JSON schema are synced to the 57-element XSD, and the sync tests enforce exactly that inventory.
> 3. **Sync guarantee** is implemented as planned (`tests/doclang_xsd.py` + `tests/test_doclang_sync.py`), pinning `EXPECTED_DOCLANG_SPEC_VERSION = "0.7.3"`.
> 4. **Grammar dialect.** The canonical grammar is **GBNF** (xgrammar/vLLM's native dialect), not the lark∩xgrammar EBNF intersection originally decided. It is unit-tested by a mechanical GBNF→lark rewrite (`grammar.to_lark_grammar()`: `::=`→`:`, `[X]`→`/[X]/`); the same `DOCLANG_GRAMMAR` string is what vLLM consumes. See "Decisions" (updated).
> 5. **Fuzz suite.** `tests/test_doclang_fuzz.py` ships three tests: `test_grammar_soundness` (random model → serialize → grammar-accepted AND XSD-valid; supersedes the planned `test_random_pydantic_docs_are_xsd_valid`), `test_round_trip_preserves_model` (idempotent re-serialization), and `test_grammar_soundness_under_mutation` (the planned differential mutation test: 48 precomputed mutants across all six mutation categories; 0 soundness violations, 3 reported completeness gaps where the grammar is stricter than the XSD about leading text in `list`/`tabular`).
> 6. **Invalid-corpus rejection split.** `test_grammar_rejects_cfg_expressible_invalid` rejects only the 11 CFG-expressible invalid docs (per `manifest.json` `mechanism: "grammar+validator"`); the 9 context-sensitive ones (`mechanism: "validator"`) are rejected by the full validator in `test_invalid_corpus_fails_full_validation`. Together they cover all 20 — a CFG cannot express the context-sensitive rules by design.

---

## Key technical facts (verified)

- DocLang = XML, root `<doclang xmlns="https://www.doclang.ai/ns/v0" version="0.7">`, ext `.dclg`. Each semantic element = optional **element head** (ordered: `label?`, `thread?`, `xref|href?`, `layer?`, 4×`location?`, `caption*`, `description?`, `summary?`, `custom?`) + **body** (mixed raw text + inline formatting + nested semantic elements).
- Element categories (spec Reference): Special (`doclang`,`head`,`page_break`); Semantic (`text`,`heading`,`footnote`,`page_header`,`page_footer`,`field_region`,`list`,`table`,`index`,`formula`,`code`,`picture`,`marker`,`group`,`field_heading`,`field_item`,`key`,`value`,`hint`,`caption`,`track`,`cover`,`frame`,`audio`,`voice`,`chapter`); Property (`label`,`thread`,`xref`,`href`,`description`,`summary`,`custom`,`location`,`layer`); Payload (`src`,`tabular`,`checkbox`,`content`,`hours`,`minutes`,`seconds`,`msecs`); Formatting (`bold`,`italic`,`underline`,`strikethrough`,`superscript`,`subscript`,`handwriting`,`rtl`); Structural (`fcel`,`ecel`,`ched`,`rhed`,`corn`,`srow`,`lcel`,`ucel`,`xcel`,`nl`,`ldiv`,`bdiv`); Doc-head (`default_resolution`).
- Element-specific attributes: `heading@level`, `field_heading@level` (positive int); `list@class` ∈ {unordered,ordered}; `picture@class` ∈ {undefined,chart}; `value@class` ∈ {read_only,fillable}; `checkbox@class` ∈ {unselected,selected}; `layer@value` ∈ {body,background,furniture}; `location@value` int in [0,resolution), `location@resolution` optional; `thread@thread_id`/`xref@thread_id` positive int; `href@uri`/`src@uri` URI; `label@value` open string; `minutes/seconds` ∈ [0,59], `msecs` ∈ [0,999], `hours` ≥0; `default_resolution@width/height` non-neg int.
- `xref`/`href` mutually exclusive. `doclang.validate(path)` = XSD+Schematron (needs JRE — present on this machine); `validate(path, xsd_only=True)` = XSD only (fast, no Java). `validate` takes a **file path** → write to temp dir (reuse `utils/tempdir.py::OptionalTemporaryDirectory`).
- **The `doclang` package bundles the authoritative XSD** (used by `validate`). Its path is discoverable at runtime from the package location (e.g. `Path(doclang.__file__).parent` glob `**/*.xsd`) — this is the anchor for the sync tests.
- **No** `pydantic`/`openai`/structured-output code today. vLLM ~0.29.0 is a dep (OpenAI server supports `guided_decoding` w/ `grammar` EBNF + `backend`).
- `pyproject.toml`: `dependencies` (add `pydantic`,`openai`), `dev` (add `lark`,`hypothesis`), setuptools `packages.find` includes `src*` (new pkg auto-discovered), pytest `e2e` marker exists. `environment.yml` pip section (add `pydantic`,`openai`; `lark`/`hypothesis` are dev-only → pyproject dev is sufficient for `pip install -e .[dev]`, but add to environment.yml pip if the conda env is the test runtime — decide in Phase 0; repo memory: mamba writes pip lines verbatim, bare names only).
- `tests/conftest.py`: simple fixtures; follow existing pytest style; mark live-server tests `@pytest.mark.e2e`.

---

## Architecture & module layout

New package `src/doclang_structured/`:

- `__init__.py` — public exports.
- `grammar.py` — `DOCLANG_GRAMMAR: str` (full-spec EBNF), assembled from per-category rule fragments; `get_grammar() -> str`. **Written in the lark∩xgrammar EBNF intersection** (named rules, `"string"` literals, `/regex/` classes, `| * + ? ()`; NO lookahead, NO empty productions inside repetitions) so the same string is unit-testable with lark AND consumable by vLLM/xgrammar.
- `models.py` — Pydantic v2 models (recursive mixed content) + type aliases + an `ELEMENT_MODELS: dict[str, type[BaseModel]]` registry (element name → class) used by sync tests; `model_rebuild()` for recursion.
- `schema.py` — `get_doclang_json_schema() -> dict`; `_make_strict_compatible(schema)`; `response_format() -> dict`.
- `serialize.py` — `model_to_doclang_xml(model) -> str` (canonical head order, 4×location, XML escaping, `<content>` for whitespace).
- `parse.py` — `doclang_xml_to_model(xml) -> DocLang` (ElementTree → model).
- `validate.py` — `validate_doclang(xml, *, xsd_only=False) -> None`; `DocLangValidationError` (carries `xsd_errors`/`schematron_errors`).
- `client.py` — `generate_doclang_grammar(...)` (primary) + `generate_doclang_json_schema(...)` (fallback); shared `_chat()`; retry loop via `validate_doclang`.

CLI: `tools/generate_doclang.py`.

Test suite (new): `tests/doclang_xsd.py` (XSD inventory helper), `tests/test_doclang_sync.py`, `tests/test_doclang_corpus.py`, `tests/test_doclang_fuzz.py`, `tests/test_doclang_grammar.py`, `tests/test_doclang_models.py`, `tests/test_doclang_serialize.py`, `tests/e2e/test_doclang_structured.py`, `tests/data/doclang_corpus/` (+`manifest.json`), `tests/data/doclang_invalid/`.

---

## Steps

### Phase 0 — Scaffolding & deps ✅

1. Deps: `pyproject.toml` `dependencies` += `pydantic`,`openai`; `dev` += `lark`,`hypothesis`. `environment.yml` pip += `pydantic`,`openai` (+ `lark`,`hypothesis` if conda env is the test runtime).
2. Create `src/doclang_structured/__init__.py` stub.
3. Verify: `mamba run -n voxdoc-trainer python -c "import pydantic, openai, lark, hypothesis, doclang"`.
4. **Locate the bundled XSD**: `mamba run -n voxdoc-trainer python -c "import doclang, pathlib; print(pathlib.Path(doclang.__file__).parent)"` then glob `**/*.xsd` — record the exact path pattern for `tests/doclang_xsd.py::find_xsd()`.

### Phase 1 — EBNF grammar (PRIMARY; largest chunk) *depends on 0* ✅

1. `grammar.py` EBNF (lark∩xgrammar dialect). Rules:
   - `root = doclang_doc`; `doclang_doc = "<doclang" root_attrs "?" head "?" body "*" "</doclang>"`.
   - `head = "<head>" head_item* "</head>"`; `head_item = default_resolution | title | date | language | generated_by | doc_summary` (scope: `default_resolution` + simple reserved elements; rich governance metadata out of scope).
   - `element_head = label? thread? (xref | href)? layer? (location location location location)? caption* description? summary? custom?` (ordered).
   - Semantic rules (open-tag + element_head + body + close-tag): `text`,`heading`(+`level`),`footnote`,`page_header`,`page_footer`,`group`,`list`(+`class`),`table`,`index`,`formula`,`code`,`picture`(+`class`,`src?`,`tabular?`),`field_region`,`field_heading`(+`level`),`field_item`,`key`,`value`(+`class`),`hint`,`marker`,`caption`,`track`,`cover`,`frame`,`audio`,`voice`,`chapter`.
   - Inline formatting: `bold|italic|underline|strikethrough|superscript|subscript|handwriting|rtl`.
   - Property/payload: `label`,`thread`,`xref`,`href`,`description`,`summary`,`custom`,`location`,`layer`,`src`,`tabular`,`checkbox`,`content`,`hours`,`minutes`,`seconds`,`msecs`.
   - Structural (empty): `fcel ecel ched rhed corn srow lcel ucel xcel nl bdiv`; `ldiv` (empty or wraps `marker`).
   - Bodies: `body = (text_chunk | inline_fmt | semantic_el)+`-style (avoid empty-in-repetition); `inline_body`; `table_body = (cell_token | text_chunk | semantic_el | nl)+`; `list_body`; `track_body = cover? cue_block+`; `cue_block = bdiv timestamp_run timestamp_run? chapter? frame? audio? transcript?`.
   - Free text: `text_char = /[^<&]/`; `entity = "&lt;"|"&gt;"|"&amp;"|"&quot;"|"&apos;"|/"&#x[0-9a-fA-F]+;"/|/"&#[0-9]+;"/`; `text_chunk = (text_char | entity)+`. Whitespace-preserving code via `content`.
   - `location = "<location" loc_resolution? " value=\"" int_digits "\"/>"`.
2. Grammar must compile under **lark** (unit) — this is the fast local check; xgrammar acceptance is confirmed by e2e (Phase 9).

### Phase 2 — Pydantic models + JSON schema (FALLBACK) *depends on 0; ∥ 1* ✅

1. `models.py`: Pydantic v2 models mirroring the grammar; type aliases `InlineNode`/`SemanticNode`/`BodyNode`/`TableCellNode`/`ListItemNode`/`TrackCueNode`; `ElementHead` with `xref`/`href` mutual-exclusion `model_validator`; **`ELEMENT_MODELS` registry** (element name → class) for sync tests; `DocLang.model_rebuild()`.
2. `schema.py`: `get_doclang_json_schema() = DocLang.model_json_schema()`; `_make_strict_compatible` (force `additionalProperties: False`, drop/transform unsupported keywords, keep `$defs`/`$ref`); `response_format()`.

### Phase 3 — Serializer + parser (FALLBACK) *depends on 2* ✅

1. `serialize.py::model_to_doclang_xml(model) -> str` (canonical head order; 4×`<location value="N"/>`; escape `& < > "`; `<content>` for code; root `xmlns`+`version`).
2. `parse.py::doclang_xml_to_model(xml) -> DocLang` (ElementTree → models).

### Phase 4 — Validation helper (shared) *depends on 0* ✅

 1. `validate.py::validate_doclang(xml, *, xsd_only=False) -> None`: write `doc.dclg` into `OptionalTemporaryDirectory` (from `utils/tempdir.py`), call `doclang.validate(path, xsd_only=...)`, map `ValidationError` → `DocLangValidationError`.

### Phase 5 — vLLM client (PRIMARY) *depends on 1,4* ✅

 1. `client.py::generate_doclang_grammar(base_url, model, messages, *, api_key="EMPTY", backend="xgrammar", max_retries=3, temperature=0.0) -> str`: `OpenAI(base_url, api_key)`; `chat.completions.create(..., extra_body={"guided_decoding": {"grammar": DOCLANG_GRAMMAR, "backend": backend}})`; return raw `.dclg`; `validate_doclang` + retry. Support image content blocks. Confirm exact `guided_decoding` field names vs installed vLLM 0.29.
 2. `client.py::generate_doclang_json_schema(...)`: `response_format=response_format()`; parse JSON → `DocLang` → `model_to_doclang_xml` → validate + retry.

### Phase 6 — Test data (corpus + invalid) *∥ 1–5* ✅

 1. `tests/data/doclang_corpus/`: vendor ~30–50 **valid** `.dclg` docs covering every element category (from the doclang repo `examples/` + hand-authored for gaps: tables w/ spans, nested lists, forms, tracks, code, formulas, pictures, page breaks, threads, captions, all formatting). Add `manifest.json`: `{filename: {elements: [...], features: [...]}}`.
 2. `tests/data/doclang_invalid/`: hand-authored **invalid** docs, one per violated rule (bad enum value, non-rectangular table, `xref`+`both`, raw text where disallowed, broken head order, `x0>x1`), each with a `README.md` stating the violated rule.

### Phase 7 — XSD inventory helper + SYNC tests (the "up to date with the validator" suite) *depends on 0,1,2,6* ✅

 1. `tests/doclang_xsd.py`:
    - `find_xsd() -> Path` (glob the `doclang` package for `*.xsd`).
    - `xsd_inventory() -> XsdInventory` (stdlib ElementTree): `elements: dict[name, ElementInfo]` (attrs, enum domains, required/optional, default), `root_element`, `spec_version` (declared in the XSD). Collects global + local elements/attributes, resolves `xs:restriction`/`xs:enumeration`.
    - `grammar_inventory() -> GrammarInventory`: regex-extract tag literals (`<name`, `</name>`, `<name .../>`), attribute literals (`attr="`), and enum value sets from `DOCLANG_GRAMMAR`.
    - `pydantic_inventory() -> PydanticInventory`: from `ELEMENT_MODELS` + field introspection (names, `Literal` enum domains).
    - `json_schema_inventory() -> SchemaInventory`: walk `get_doclang_json_schema()` `$defs` (names + `enum` arrays).
 2. `tests/test_doclang_sync.py` (the centerpiece):
    - `test_spec_version_pinned`: `XsdInventory.spec_version == EXPECTED_DOCLANG_SPEC_VERSION` (e.g. `"0.7"`). On mismatch → fail with "doclang package upgraded — re-sync `grammar.py` + `models.py`, then bump the pin." (Loud upgrade guard.)
    - `test_grammar_covers_xsd_elements`: every XSD element appears in `grammar_inventory` (no missing tags).
    - `test_grammar_enum_values_match_xsd`: for every XSD enumerated attribute, the grammar's allowed literal values == the XSD domain (exact set match).
    - `test_pydantic_covers_xsd_elements`: every XSD element has an `ELEMENT_MODELS` entry reachable from `DocLang`.
    - `test_pydantic_enum_values_match_xsd`: `Literal` domains == XSD domains.
    - `test_json_schema_covers_xsd`: every XSD element present in `$defs`; enum arrays match.
    - `test_no_stale_elements`: inverse direction — no grammar/pydantic/schema element absent from the XSD (catches typos / removed elements).

### Phase 8 — Full behavioral test suite *depends on 1,2,3,4,6,7* ✅

 1. `tests/test_doclang_grammar.py`:
    - `test_grammar_compiles_lark` (build `lark.Lark(DOCLANG_GRAMMAR, start="root")`).
    - `test_accepts_valid_samples` (corpus docs parse under lark).
    - `test_rejects_invalid_samples` (`tests/data/doclang_invalid/*` all rejected).
 2. `tests/test_doclang_models.py`: valid model builds; invalid rejected (incl. `xref`+`href`); `get_doclang_json_schema()` shape; strict-compat (all objects `additionalProperties: False`).
 3. `tests/test_doclang_serialize.py`: `model_to_doclang_xml` → `validate_doclang(xsd_only=True)` passes; round-trip `doclang_xml_to_model` → `model_to_doclang_xml` → validate.
 4. `tests/test_doclang_corpus.py`:
    - `test_corpus_valid_per_validator`: every corpus doc passes `doclang.validate` (full, XSD+Schematron; skip Schematron part if no JRE).
    - `test_corpus_accepted_by_grammar`: every corpus doc parses under lark.
    - `test_corpus_parses_by_pydantic`: every corpus doc → `doclang_xml_to_model` succeeds.
    - `test_corpus_covers_all_xsd_elements`: union of `manifest.json` elements ⊇ XSD element set (guarantees the acceptance tests actually exercise the whole spec).
 5. `tests/test_doclang_fuzz.py` (differential / property-based, `hypothesis`, fixed seeds, small N):
    - `test_grammar_soundness_under_mutation`: take corpus docs, apply random mutations (tag swap, enum→invalid value, drop tag, reorder head, break table rectangularity, add `xref`+`href`); for each mutant, if lark **accepts** it then `validate_doclang(xsd_only=True)` **must pass** (soundness: grammar ⊆ XSD-valid). Log `XSD-valid but grammar-rejected` as known completeness gaps (report, don't fail).
    - `test_random_pydantic_docs_are_xsd_valid`: random `DocLang` instances (hypothesis strategies over the models) → `model_to_doclang_xml` → `validate_doclang(xsd_only=True)` passes (soundness of the fallback path).
    - Note: fuzz uses `xsd_only=True` (fast, no JRE); context-sensitive Schematron rules (rectangular) are covered by corpus + e2e full validation.
 6. `tests/e2e/test_doclang_structured.py` (`@pytest.mark.e2e`, skipif no `VOXDOC_VLLM_BASE_URL`): live vLLM — `grammar` mode and `json_schema` mode each return XML that passes **full** `validate_doclang`.

### Phase 9 — CLI *depends on 5* ✅

 1. `tools/generate_doclang.py`: argparse (`--base-url --model --image --prompt --out --mode {grammar,json_schema} --backend --max-retries --temperature`); build `messages` (text + optional image block); dispatch to the client fn; write `.dclg`; print validation status.

### Phase 10 — Docs ✅

 1. `docs/structured_output.md` (both paths, CLI + client API, test-suite overview incl. the sync guarantee, compatibility caveats). Update `docs/usage.md` + `README.md`.

---

## Relevant files

- CREATE `src/doclang_structured/{__init__,grammar,models,schema,serialize,parse,validate,client}.py`
- CREATE `tools/generate_doclang.py`
- CREATE `tests/doclang_xsd.py`, `tests/test_doclang_sync.py`, `tests/test_doclang_corpus.py`, `tests/test_doclang_fuzz.py`, `tests/test_doclang_grammar.py`, `tests/test_doclang_models.py`, `tests/test_doclang_serialize.py`, `tests/e2e/test_doclang_structured.py`
- CREATE `tests/data/doclang_corpus/` (+`manifest.json`), `tests/data/doclang_invalid/` (+`README.md`)
- CREATE `docs/structured_output.md`
- EDIT `pyproject.toml` (deps + dev), `environment.yml` (pip), `docs/usage.md`, `README.md`
- REUSE `utils/tempdir.py::OptionalTemporaryDirectory`, `doclang.validate/ValidationError`, the **XSD bundled in the `doclang` package** (sync anchor), `tests/conftest.py` patterns, `e2e` pytest marker.

## Verification

1. `mamba run -n voxdoc-trainer python -c "import pydantic, openai, lark, hypothesis, doclang"`.
2. **Sync suite**: `mamba run -n voxdoc-trainer python -m pytest tests/test_doclang_sync.py -v` — version pin + element/enum inventory match for grammar, Pydantic, and JSON schema.
3. **Full unit+behavioral**: `mamba run -n voxdoc-trainer python -m pytest tests/ -m "not e2e" -v` — grammar compiles/accepts/rejects, models, serializer, corpus (validity + acceptance + coverage), fuzz soundness.
4. **Upgrade-drill (proves the sync guard works)**: temporarily point `find_xsd()`/the version pin at a mismatched value (or bump `EXPECTED_DOCLANG_SPEC_VERSION`) and confirm `test_spec_version_pinned` + inventory tests **fail** with the re-sync message; revert.
5. Live (vLLM served): `VOXDOC_VLLM_BASE_URL=http://localhost:8000/v1 mamba run -n voxdoc-trainer python -m pytest tests/e2e/test_doclang_structured.py -v` — both modes pass **full** `validate_doclang`.
6. CLI smoke: `mamba run -n voxdoc-trainer python tools/generate_doclang.py --base-url http://localhost:8000/v1 --model <id> --image <doc.png> --prompt "Transcribe this historical document into DocLang format." --out /tmp/out.dclg --mode grammar` then `python -c "from doclang import validate; validate('/tmp/out.dclg')"`.
7. Negative/retry: non-rectangular table caught by `validate_doclang` and triggers retry (stubbed generator unit test).

### Verification log (2026-10-02, post-build audit)

1. ✅ Imports OK (`pydantic`, `openai`, `lark`, `hypothesis`, `doclang`).
2. ✅ Sync suite: 9/9 passed.
3. ✅ Full unit+behavioral: **91 passed**, 6 deselected (e2e), ~34 s.
4. ✅ Upgrade-drill: bumped `EXPECTED_DOCLANG_SPEC_VERSION` to `9.9.9` → `test_spec_version_pinned` failed with the re-sync message; reverted, suite green again.
5. ⏸ Live e2e: no vLLM server running on this machine (tests correctly skip without `VOXDOC_VLLM_BASE_URL`); run when a server is available.
6. ⏸ CLI smoke vs live server: same blocker; `--help` + import path verified.
7. ✅ Negative/retry: `tests/test_doclang_client.py` (stubbed generator, 8 tests) + `non_rectangular_table.dclg` in the invalid corpus.

## Decisions

- **Both paths** (user): EBNF grammar = primary (direct XML on vLLM); Pydantic + JSON schema = portable fallback + typed parsing.
- **Full spec scope** (user): all element categories incl. `field_*` forms and `track`.
- **Target = local vLLM / xgrammar**; `guided_decoding` is a vLLM extension (not portable OpenAI `response_format`) — acceptable.
- **Final gate = `doclang.validate()` + retry** on BOTH paths (CFG/JSON-schema can't enforce context-sensitive rules). Authoritative "only valid DocLang" guarantee.
- **Sync anchor = the XSD bundled in the `doclang` package** (discovered at runtime), NOT a copied file — so the suite tracks package upgrades automatically. Guarded by a pinned `EXPECTED_DOCLANG_SPEC_VERSION`.
- **Grammar dialect = GBNF** (xgrammar/vLLM native), unit-tested via the mechanical `to_lark_grammar()` rewrite to lark EBNF (no lookahead, no empty-in-repetition); one string is production-consumable (xgrammar) and unit-testable (lark); e2e is the xgrammar authority. *(Supersedes the original "lark∩xgrammar EBNF intersection" decision — see implementation note 4.)*
- **Fuzz uses `xsd_only=True`** (fast, no JRE); full XSD+Schematron used for corpus + e2e.
- **Placement** = `src/doclang_structured/` (auto-discovered via `src*`) + `tools/` CLI + `tests/` + `tests/data/`.
- **Deps added**: `pydantic`,`openai` (runtime); `lark`,`hypothesis` (dev).

## Further Considerations

1. **XSD introspection robustness**: the bundled XSD may use `xs:group`/`xs:complexType` refs and local elements. `tests/doclang_xsd.py` must resolve these to a flat element/attribute/enum inventory. *Recommendation: build the inventory in Phase 7 step 16 first, validate it against the known element list (Key technical facts) before writing the sync assertions.*
2. **lark parser algorithm**: mixed-content grammars can be ambiguous/slow under LALR. *Recommendation: use `lark.Lark(..., parser="earley")` for test correctness; if too slow, restructure rules to be LALR-friendly.*
3. **Schema size / decoder limits**: full recursive fallback schema is large; some decoders cap depth/size. *Recommendation: if vLLM/OpenAI reject it, prune rare elements (`track`,`custom`) from the *schema* only (keep grammar full) — the sync test `test_json_schema_covers_xsd` would then need an explicit `SCHEMA_PRUNED_ELEMENTS` allowlist.*
4. **`<head>` metadata scope**: grammar + models support `default_resolution` + simple reserved elements (`title`,`date`,`language`,`generated_by`,`summary`); rich governance metadata out of scope for generation (and excluded from the sync inventory comparison via an allowlist).
