"""Generate the DocLang test corpus (valid ``.dclg`` docs) + ``manifest.json``.

Each document is hand-authored to exercise specific element categories and
features. Every document is validated with the **full** official validator
(``doclang.validate`` = XSD + Schematron) before being written; the script
fails loudly if any document is invalid.

Run:
    mamba run -n voxdoc-trainer python tests/generators/build_doclang_corpus.py
"""

from __future__ import annotations

import json
import os
import sys

from doclang import validate

NS = "https://www.doclang.ai/ns/v0"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "doclang_corpus")


def doc(*body: str) -> str:
    return f'<doclang xmlns="{NS}" version="0.7">{"".join(body)}</doclang>'


# filename -> (xml, elements, features)
DOCS: dict[str, tuple[str, list[str], list[str]]] = {
    "001_text_plain.dclg": (
        doc("<text>Hello world</text>"),
        ["text"],
        ["plain-text"],
    ),
    "002_heading_levels.dclg": (
        doc('<heading level="1">H1</heading>', '<heading level="2">H2</heading>', '<heading level="3">H3</heading>'),
        ["heading"],
        ["heading-levels"],
    ),
    "003_formatting_all.dclg": (
        doc(
            "<text><bold>b</bold> <italic>i</italic> <strikethrough>s</strikethrough> "
            "<underline>u</underline> <superscript>sup</superscript> <subscript>sub</subscript> "
            "<rtl>rtl</rtl> <handwriting>hw</handwriting></text>"
        ),
        ["text", "bold", "italic", "strikethrough", "underline", "superscript", "subscript", "rtl", "handwriting"],
        ["all-formatting"],
    ),
    "004_list_ordered.dclg": (
        doc('<list class="ordered"><ldiv/>one<ldiv/>two<ldiv/>three</list>'),
        ["list", "ldiv"],
        ["ordered-list"],
    ),
    "005_list_unordered_marker.dclg": (
        doc(
            '<list class="unordered"><ldiv><marker>\u2022</marker></ldiv>item one'
            "<ldiv><marker>\u2022</marker></ldiv>item two</list>"
        ),
        ["list", "ldiv", "marker"],
        ["unordered-list", "marker"],
    ),
    "006_list_nested.dclg": (
        doc('<list class="unordered"><ldiv/>outer<list class="ordered"><ldiv/>inner</list></list>'),
        ["list", "ldiv"],
        ["nested-list"],
    ),
    "007_table_2x2.dclg": (
        doc("<table><fcel/>a<ecel/>b<nl/><fcel/>c<ecel/>d</table>"),
        ["table", "fcel", "ecel", "nl"],
        ["table-2x2"],
    ),
    "008_table_3x3.dclg": (
        doc("<table><fcel/>a<lcel/>b<ecel/>c<nl/>" "<fcel/>d<lcel/>e<ecel/>f<nl/>" "<fcel/>g<lcel/>h<ecel/>i</table>"),
        ["table", "fcel", "lcel", "ecel", "nl"],
        ["table-3x3"],
    ),
    "009_table_headers.dclg": (
        doc("<table><corn/><ched/>H1<ecel/>H2<nl/>" "<rhed/>R1<fcel/>a<ecel/>b<nl/>" "<rhed/>R2<fcel/>c<ecel/>d</table>"),
        ["table", "corn", "ched", "rhed", "fcel", "ecel", "nl"],
        ["table-headers", "corner", "row-header", "col-header"],
    ),
    "010_table_spans.dclg": (
        doc("<table><fcel/>a<ecel/>b<nl/>" "<srow/>c<ucel/>d<nl/>" "<xcel/>e<ecel/>f</table>"),
        ["table", "fcel", "ecel", "srow", "ucel", "xcel", "nl"],
        ["table-spans"],
    ),
    "011_index.dclg": (
        doc("<index><fcel/>term<ecel/>1<nl/><fcel/>other<ecel/>2</index>"),
        ["index", "fcel", "ecel", "nl"],
        ["index"],
    ),
    "012_code_content.dclg": (
        doc("<code><content>  def foo():\n    return 1  </content></code>"),
        ["code", "content"],
        ["code", "whitespace-preserve"],
    ),
    "013_formula.dclg": (
        doc("<formula>E = mc^2</formula>"),
        ["formula"],
        ["formula"],
    ),
    "014_picture_src.dclg": (
        doc('<picture class="undefined"><src uri="http://example.com/img.png"/></picture>'),
        ["picture", "src"],
        ["picture-src"],
    ),
    "015_picture_tabular.dclg": (
        doc('<picture class="chart"><src uri="http://example.com/chart.png"/>' "<tabular><fcel/>x<ecel/>y</tabular></picture>"),
        ["picture", "src", "tabular", "fcel", "ecel"],
        ["picture-chart", "tabular"],
    ),
    "016_field_form.dclg": (
        doc(
            '<field_region><field_heading level="1">Personal</field_heading>'
            '<field_item><key>Name</key><value class="fillable">John</value></field_item>'
            '<field_item><key>Age</key><value class="read_only">30</value></field_item>'
            "</field_region>"
        ),
        ["field_region", "field_heading", "field_item", "key", "value"],
        ["form", "field-classes"],
    ),
    "017_group.dclg": (
        doc("<group><text>para 1</text><text>para 2</text></group>"),
        ["group", "text"],
        ["group"],
    ),
    "018_footnote.dclg": (
        doc("<text>main text</text><footnote>note 1</footnote>"),
        ["text", "footnote"],
        ["footnote"],
    ),
    "019_page_header_footer.dclg": (
        doc("<page_header>Header</page_header><text>body</text><page_footer>Footer</page_footer>"),
        ["page_header", "text", "page_footer"],
        ["page-header", "page-footer"],
    ),
    "020_page_break.dclg": (
        doc("<text>page 1</text><page_break/><text>page 2</text>"),
        ["text", "page_break"],
        ["page-break"],
    ),
    "021_caption.dclg": (
        doc('<picture class="undefined"><caption>Figure 1</caption><src uri="http://example.com/img.png"/></picture>'),
        ["picture", "caption", "src"],
        ["caption"],
    ),
    "022_element_head_full.dclg": (
        doc(
            '<text><label value="fig1"/><thread thread_id="1"/><layer value="body"/>'
            '<location value="0"/><location value="0"/><location value="100"/><location value="50"/>'
            "Body text</text>"
        ),
        ["text", "label", "thread", "layer", "location"],
        ["element-head", "locations"],
    ),
    "023_xref.dclg": (
        doc('<text><thread thread_id="1"/>Original</text><text><xref thread_id="1"/>Reference</text>'),
        ["text", "thread", "xref"],
        ["thread", "xref"],
    ),
    "024_href.dclg": (
        doc('<text><href uri="http://example.com"/>Link</text>'),
        ["text", "href"],
        ["href"],
    ),
    "025_description_summary.dclg": (
        doc(
            '<picture class="undefined"><description>A picture description</description>'
            '<summary>A summary</summary><src uri="http://example.com/img.png"/></picture>'
        ),
        ["picture", "description", "summary", "src"],
        ["description", "summary"],
    ),
    "026_custom.dclg": (
        doc("<text><custom/>Text with custom</text>"),
        ["text", "custom"],
        ["custom"],
    ),
    "027_checkbox.dclg": (
        doc(
            '<list class="unordered"><ldiv><marker><checkbox class="selected"/></marker></ldiv>Done'
            '<ldiv><marker><checkbox class="unselected"/></marker></ldiv>Not done</list>'
        ),
        ["list", "ldiv", "marker", "checkbox"],
        ["checkbox", "checkbox-classes"],
    ),
    "028_hint.dclg": (
        doc("<text>Text with <hint><bold>hint</bold></hint> here</text>"),
        ["text", "hint", "bold"],
        ["hint"],
    ),
    "029_head_meta.dclg": (
        doc('<head><default_resolution width="100" height="200"/><meta/></head><text>body</text>'),
        ["head", "default_resolution", "meta", "text"],
        ["head", "default-resolution", "meta"],
    ),
    "030_content_whitespace.dclg": (
        doc("<text>Before<content>  spaced  </content>after</text>"),
        ["text", "content"],
        ["whitespace-preserve"],
    ),
    "031_value_classes.dclg": (
        doc(
            '<field_region><field_item><key>A</key><value class="read_only">ro</value></field_item>'
            '<field_item><key>B</key><value class="fillable">fill</value></field_item></field_region>'
        ),
        ["field_region", "field_item", "key", "value"],
        ["value-classes"],
    ),
    "032_picture_classes.dclg": (
        doc(
            '<picture class="undefined"><src uri="http://example.com/a.png"/></picture>'
            '<picture class="chart"><src uri="http://example.com/b.png"/></picture>'
        ),
        ["picture", "src"],
        ["picture-classes"],
    ),
    "033_mixed_nested.dclg": (
        doc(
            '<text>Start <bold>bold</bold> and <italic>italic</italic> with a <list class="ordered"><ldiv/>item</list> nested</text>'
        ),
        ["text", "bold", "italic", "list", "ldiv"],
        ["mixed-content", "nested-in-text"],
    ),
    "034_full_document.dclg": (
        doc(
            '<head><default_resolution width="200" height="300"/></head>',
            "<page_header>Report</page_header>",
            '<heading level="1">Title</heading>',
            "<text>Intro <bold>bold</bold> <italic>italic</italic> <footnote>fn</footnote></text>",
            '<list class="ordered"><ldiv/>one<ldiv/>two</list>',
            "<table><fcel/>h1<ecel/>h2<nl/><fcel/>a<ecel/>b</table>",
            '<picture class="chart"><caption>Fig</caption><src uri="http://example.com/c.png"/><tabular><fcel/>x<ecel/>y</tabular></picture>',
            '<field_region><field_heading level="1">F</field_heading><field_item><key>k</key><value class="fillable">v</value></field_item></field_region>',
            "<code><content>  x=1  </content></code>",
            "<formula>f=x</formula>",
            "<page_break/>",
            "<page_footer>End</page_footer>",
        ),
        [
            "head",
            "default_resolution",
            "page_header",
            "heading",
            "text",
            "bold",
            "italic",
            "footnote",
            "list",
            "ldiv",
            "table",
            "fcel",
            "ecel",
            "nl",
            "picture",
            "caption",
            "src",
            "tabular",
            "field_region",
            "field_heading",
            "field_item",
            "key",
            "value",
            "code",
            "content",
            "formula",
            "page_break",
            "page_footer",
        ],
        ["full-document"],
    ),
}


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest: dict[str, dict] = {}
    failures = 0
    for filename, (xml, elements, features) in sorted(DOCS.items()):
        path = os.path.join(OUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(xml + "\n")
        try:
            validate(path)  # full: XSD + Schematron
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  INVALID {filename}: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
            continue
        manifest[filename] = {"elements": elements, "features": features}
        print(f"  ok      {filename}")

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"\n{len(manifest)} valid docs written, {failures} invalid. Manifest: {manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
