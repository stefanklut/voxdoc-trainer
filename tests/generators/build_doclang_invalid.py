"""Generate the DocLang *invalid* test corpus + ``README.md``.

Each hand-authored document violates exactly one rule. The script verifies
every document is actually **rejected** by the official validator (full
XSD + Schematron) and records whether the EBNF grammar (via lark) also
rejects it — CFG-expressible rules are rejected by both, context-sensitive
rules (rectangularity, numeric ordering, ancestry) only by the validator.

Run:
    mamba run -n voxdoc-trainer python tests/generators/build_doclang_invalid.py
"""

from __future__ import annotations

import json
import os
import sys

# Make the repo root importable so we can reach ``src.doclang_structured`` when
# this file is run as a plain script (``python tests/generators/...``).
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from doclang import validate  # noqa: E402

NS = "https://www.doclang.ai/ns/v0"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "doclang_invalid")

_LARK_PARSER = None


def _grammar_rejects(xml: str) -> bool:
    """Return ``True`` if the EBNF grammar (compiled via lark) rejects ``xml``.

    The mechanism column in the README is derived from this *empirically* so it
    always reflects the grammar's actual behaviour; the hand-assigned hint in
    :data:`DOCS` is cross-checked against it.
    """
    global _LARK_PARSER
    if _LARK_PARSER is None:
        import lark

        from src.doclang_structured.grammar import to_lark_grammar

        _LARK_PARSER = lark.Lark(to_lark_grammar(), start="root", parser="earley")
    try:
        _LARK_PARSER.parse(xml)
        return False
    except Exception:  # noqa: BLE001 - any parse failure means "rejected"
        return True


def doc(*body: str) -> str:
    return f'<doclang xmlns="{NS}" version="0.7">{"".join(body)}</doclang>'


# filename -> (xml, rule, mechanism)
# mechanism: "grammar+validator" (CFG-expressible) or "validator" (context-sensitive)
DOCS: dict[str, tuple[str, str, str]] = {
    "bad_enum_list_class.dclg": (
        doc('<list class="bogus"><ldiv/>x</list>'),
        "list@class must be in {unordered, ordered}; 'bogus' is not allowed",
        "grammar+validator",
    ),
    "bad_enum_checkbox_class.dclg": (
        doc('<list class="unordered"><ldiv><marker><checkbox class="bogus"/></marker></ldiv>x</list>'),
        "checkbox@class must be in {unselected, selected}; 'bogus' is not allowed",
        "grammar+validator",
    ),
    "broken_head_order.dclg": (
        doc('<text><layer value="body"/><thread thread_id="1"/>x</text>'),
        "element head order is label? thread? (xref|href)? layer? ...; layer before thread is out of order",
        "grammar+validator",
    ),
    "xref_and_href.dclg": (
        doc('<text><thread thread_id="1"/>a</text><text><xref thread_id="1"/><href uri="http://x"/>b</text>'),
        "xref and href are mutually exclusive in the element head",
        "grammar+validator",
    ),
    "list_no_ldiv.dclg": (
        doc('<list class="ordered"><text>first</text><ldiv/>second</list>'),
        "a list's first element must be ldiv (after the optional element head); here it is <text>",
        "grammar+validator",
    ),
    "table_no_cell_token.dclg": (
        doc("<table><text>text</text><fcel/>a<ecel/>b</table>"),
        "a table's first element must be a cell token (after the optional element head); here it is <text>",
        "grammar+validator",
    ),
    "raw_text_in_group.dclg": (
        doc("<group>stray<text>a</text></group>"),
        "group is non-mixed: raw text content is not allowed (only child elements)",
        "grammar+validator",
    ),
    "unknown_element.dclg": (
        doc("<bogus>x</bogus>"),
        "'bogus' is not a DocLang element",
        "grammar+validator",
    ),
    "bad_enum_layer_value.dclg": (
        doc('<text><layer value="bogus"/>x</text>'),
        "layer@value must be in {body, background, furniture}; 'bogus' is not allowed",
        "grammar+validator",
    ),
    "non_rectangular_table.dclg": (
        doc("<table><fcel/>a<ecel/>b<nl/><fcel/>c<nl/><fcel/>d<ecel/>e</table>"),
        "rectangular grid: row 2 has 1 cell but row 1 has 2 (context-sensitive)",
        "validator",
    ),
    "location_x0_gt_x1.dclg": (
        doc('<text><location value="100"/><location value="0"/><location value="50"/><location value="50"/>x</text>'),
        "location block must satisfy x0<=x1; here x0=100 > x1=50 (context-sensitive)",
        "validator",
    ),
    "location_out_of_range.dclg": (
        doc(
            '<head><default_resolution width="100" height="100"/></head>'
            '<text><location value="150"/><location value="0"/><location value="150"/><location value="50"/>x</text>'
        ),
        "location value must be in [0, axis_limit); x0=150 >= width=100 (context-sensitive)",
        "validator",
    ),
    "xref_undefined_thread.dclg": (
        doc('<text><xref thread_id="99"/>x</text>'),
        "xref thread_id=99 is not defined by any thread element (context-sensitive)",
        "validator",
    ),
    "field_item_outside_region.dclg": (
        doc("<field_item><key>k</key><value>v</value></field_item>"),
        "field_item must be a descendant of field_region (context-sensitive)",
        "validator",
    ),
    "key_outside_field_item.dclg": (
        doc("<field_region><key>k</key></field_region>"),
        "key must be a descendant of field_item (context-sensitive)",
        "validator",
    ),
    "field_item_two_keys.dclg": (
        doc("<field_region><field_item><key>k1</key><key>k2</key><value>v</value></field_item></field_region>"),
        "a field_item may contain at most one own key (context-sensitive)",
        "validator",
    ),
    "picture_tabular_not_chart.dclg": (
        doc('<picture class="undefined"><tabular><fcel/>a<ecel/>b</tabular></picture>'),
        'tabular is only allowed in a picture with class="chart" (context-sensitive)',
        "validator",
    ),
    "picture_src_not_first.dclg": (
        doc('<picture class="undefined"><text>cap</text><src uri="http://x"/></picture>'),
        "src must be the first element of the picture body (the grammar's picture_body only allows src at the start)",
        "grammar+validator",
    ),
    "thread_mixed_host.dclg": (
        doc(
            '<text><thread thread_id="1"/>a</text><picture class="undefined"><thread thread_id="1"/><src uri="http://x"/></picture>'
        ),
        "the same thread_id must not span different host element types (context-sensitive)",
        "validator",
    ),
    "text_before_head.dclg": (
        doc('<text>text before<layer value="body"/>head</text>'),
        "property elements must precede any non-whitespace text (the grammar matches the element head compactly, right after the open tag)",
        "grammar+validator",
    ),
}


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    failures = 0
    rows = []
    for filename, (xml, rule, expected_mechanism) in sorted(DOCS.items()):
        content = xml + "\n"
        path = os.path.join(OUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        # It MUST fail validation.
        try:
            validate(path)
            failures += 1
            print(f"  UNEXPECTEDLY VALID {filename}")
            continue
        except Exception:  # noqa: BLE001 - expected to be invalid
            pass
        # Derive the mechanism empirically from the grammar's actual behaviour.
        mechanism = "grammar+validator" if _grammar_rejects(content) else "validator"
        if mechanism != expected_mechanism:
            print(f"  WARNING {filename}: grammar behaviour ({mechanism}) != hint ({expected_mechanism})")
        rows.append((filename, rule, mechanism))
        print(f"  invalid-ok {filename}  [{mechanism}]")

    # Write README.md
    readme = [
        "# DocLang invalid test corpus",
        "",
        "Each file violates exactly one rule and is **rejected** by the official",
        "validator (`doclang.validate`, XSD + Schematron).",
        "",
        "| File | Violated rule | Rejected by |",
        "| --- | --- | --- |",
    ]
    for filename, rule, mechanism in rows:
        readme.append(f"| `{filename}` | {rule} | {mechanism} |")
    readme += [
        "",
        "`grammar+validator` = the violation is expressible in the EBNF grammar (rejected by",
        "both lark and the validator). `validator` = the rule is context-sensitive and can",
        "only be enforced by the validator (a CFG cannot express it); the grammar may accept",
        "these, which is a known completeness gap covered by the final validation gate.",
        "",
    ]
    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))

    # Write manifest.json (machine-readable, consumed by the test suite).
    manifest = {filename: {"rule": rule, "mechanism": mechanism} for filename, rule, mechanism in rows}
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"\n{len(rows)} invalid docs written, {failures} unexpectedly valid. README + manifest written.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
