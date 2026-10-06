"""EBNF grammar for the full DocLang spec (v0.7.3).

The canonical grammar (``DOCLANG_GRAMMAR``) is written in **GBNF** (GGML BNF, the
dialect used by llama.cpp and consumed by **xgrammar** / vLLM ``guided_decoding``):

* Rule definitions use ``name ::= expression``.
* Character classes are bare ``[...]`` (no ``/regex/`` delimiters).
* String literals are ``"..."`` and support the escapes ``\\t`` ``\\n`` ``\\r``.
* Operators: ``|`` (choice), ``*`` (zero or more), ``+`` (one or more),
  ``?`` (optional), ``()`` (grouping).

The same grammar is **unit-tested with lark** via :func:`to_lark_grammar`, which
rewrites the GBNF into lark's EBNF dialect (``name: expression`` and ``/regex/``
terminals). The rewrite is a safe, local transformation because:

* No string literal contains ``[`` or ``]`` (so ``[X]`` → ``/X/`` never touches a
  literal).
* No quantifier is applied directly to a character class — single-character
  terminals (``digit``, ``text_char``, ...) carry the class and the quantifier is
  applied to the rule reference (so ``[X]`` → ``/X/`` never produces a broken
  ``/X/*``).
* ``ws`` is a choice of escaped single-character literals, which both dialects
  interpret identically.

Scope note
----------
The grammar is anchored to the XSD bundled in the installed ``doclang`` package
(v0.7.3). That XSD does **not** define the ``track`` / ``cover`` / ``frame`` /
``audio`` / ``voice`` / ``chapter`` / ``bdiv`` / ``hours`` / ``minutes`` /
``seconds`` / ``msecs`` elements that appear in the broader doclang.ai website
spec; those are therefore **out of scope** here. The sync tests
(``tests/test_doclang_sync.py``) enforce that the grammar's element/enum
inventory matches the bundled XSD exactly, so if the ``doclang`` package is
upgraded to add those elements the suite fails loudly and points at what to
re-sync.

Whitespace model
----------------
The XSD (via lxml) treats whitespace-only text as *ignorable* in **non-mixed**
elements (``doclang``, ``group``, ``custom``, ``ldiv``, ``head``) but as
*content* in **mixed** elements (``text``, ``list``, ``table``, ...). The
grammar mirrors this:

* Mixed bodies use ``text_chunk`` (built on ``text_char ::= [^<&]``, which
  subsumes whitespace) for all text.
* Non-mixed bodies use the unambiguous ``ws* first? rest ws*`` pattern where
  ``rest = item*`` and ``item = ws* element``. This accepts pretty-printed
  documents without introducing ambiguity (the only ambiguous case — an empty
  non-mixed element containing whitespace — is a rare edge case).

The ``element_head`` is matched **compactly** (no internal whitespace) — a sound
subset of the XSD, which is what the model is guided to emit.
"""

from __future__ import annotations

import re

__all__ = ["DOCLANG_GRAMMAR", "get_grammar", "to_lark_grammar"]

# --------------------------------------------------------------------------- #
# Text / token primitives
# --------------------------------------------------------------------------- #
_PRIMITIVES = r"""
text_char ::= [^<&]
entity ::= "&lt;" | "&gt;" | "&amp;" | "&quot;" | "&apos;" | hex_entity | dec_entity
hex_entity ::= "&#x" hex_digit+ ";"
dec_entity ::= "&#" digit+ ";"
hex_digit ::= [0-9a-fA-F]
text_chunk ::= (text_char | entity)+
ws ::= " " | "\t" | "\n" | "\r"
digit ::= [0-9]
d1_9 ::= [1-9]
pos_int ::= d1_9 digit*
nonneg_int ::= "0" | d1_9 digit*
"""

# --------------------------------------------------------------------------- #
# Root document
# --------------------------------------------------------------------------- #
_ROOT = r"""
root ::= doclang_el ws*
doclang_el ::= "<doclang" xmlns_attr version_attr? ">" doclang_body "</doclang>"
xmlns_attr ::= " xmlns=\"" "https://www.doclang.ai/ns/v0" "\""
version_attr ::= " version=\"" version_enum "\""
version_enum ::= "0.7"
doclang_body ::= ws* doclang_first? doclang_rest ws*
doclang_first ::= head_el | top_level_el | page_break_el
doclang_rest ::= doclang_item*
doclang_item ::= ws* (top_level_el | page_break_el)
page_break_el ::= "<page_break/>"
"""

# --------------------------------------------------------------------------- #
# <head> (restricted: the XSD allows xs:any; we support the known simple items)
# --------------------------------------------------------------------------- #
_HEAD = r"""
head_el ::= "<head/>" | "<head>" head_body "</head>"
head_body ::= ws* head_first? head_rest ws*
head_first ::= default_resolution_el | meta_el
head_rest ::= head_item*
head_item ::= ws* (default_resolution_el | meta_el)
default_resolution_el ::= "<default_resolution" dr_width? dr_height? "/>"
dr_width ::= " width=\"" nonneg_int "\""
dr_height ::= " height=\"" nonneg_int "\""
meta_el ::= "<meta/>" | "<meta></meta>"
"""

# --------------------------------------------------------------------------- #
# Element head (compact, ordered): label? thread? (xref|href)? layer?
#   location_block? caption? description? summary? custom?
# --------------------------------------------------------------------------- #
_ELEMENT_HEAD = r"""
element_head ::= label_el? thread_el? xref_href? layer_el? location_block? caption_el? description_el? summary_el? custom_el?
xref_href ::= xref_el | href_el
location_block ::= location_el location_el location_el location_el
label_el ::= "<label" label_value? "/>"
label_value ::= " value=\"" label_str "\""
label_str ::= (label_char | entity)*
label_char ::= [^"&]
thread_el ::= "<thread" thread_id_attr "/>"
thread_id_attr ::= " thread_id=\"" pos_int "\""
xref_el ::= "<xref" thread_id_attr "/>"
href_el ::= "<href" uri_attr "/>"
uri_attr ::= " uri=\"" uri_str "\""
uri_str ::= (uri_char | entity)*
uri_char ::= [^"&<>]
layer_el ::= "<layer" layer_value? "/>"
layer_value ::= " value=\"" layer_enum "\""
layer_enum ::= "body" | "background" | "furniture"
location_el ::= "<location" " value=\"" nonneg_int "\"" loc_resolution? "/>"
loc_resolution ::= " resolution=\"" pos_int "\""
custom_el ::= "<custom/>" | "<custom></custom>"
description_el ::= "<description>" desc_body "</description>"
summary_el ::= "<summary>" desc_body "</summary>"
desc_body ::= (text_chunk | content_el)*
content_el ::= "<content>" text_chunk? "</content>" | "<content/>"
"""

# --------------------------------------------------------------------------- #
# Payload / property elements used in bodies
# --------------------------------------------------------------------------- #
_PAYLOAD = r"""
src_el ::= "<src" uri_attr "/>"
checkbox_el ::= "<checkbox" checkbox_class? "/>"
checkbox_class ::= " class=\"" checkbox_enum "\""
checkbox_enum ::= "unselected" | "selected"
tabular_el ::= "<tabular>" tabular_cell* "</tabular>"
tabular_cell ::= table_sep_el element_head? table_cell_body
"""

# --------------------------------------------------------------------------- #
# Table separators (all empty)
# --------------------------------------------------------------------------- #
_TABLE_SEP = r"""
fcel_el ::= "<fcel/>"
ecel_el ::= "<ecel/>"
ched_el ::= "<ched/>"
rhed_el ::= "<rhed/>"
corn_el ::= "<corn/>"
srow_el ::= "<srow/>"
lcel_el ::= "<lcel/>"
ucel_el ::= "<ucel/>"
xcel_el ::= "<xcel/>"
nl_el ::= "<nl/>"
table_sep_el ::= fcel_el | ecel_el | ched_el | rhed_el | corn_el | srow_el | lcel_el | ucel_el | xcel_el | nl_el
"""

# --------------------------------------------------------------------------- #
# Inline formatting (recursive / nestable)
# --------------------------------------------------------------------------- #
_FORMATTING = r"""
formatting_el ::= bold_el | italic_el | underline_el | strikethrough_el | superscript_el | subscript_el | handwriting_el | rtl_el
bold_el ::= "<bold>" fmt_body "</bold>"
italic_el ::= "<italic>" fmt_body "</italic>"
underline_el ::= "<underline>" fmt_body "</underline>"
strikethrough_el ::= "<strikethrough>" fmt_body "</strikethrough>"
superscript_el ::= "<superscript>" fmt_body "</superscript>"
subscript_el ::= "<subscript>" fmt_body "</subscript>"
handwriting_el ::= "<handwriting>" fmt_body "</handwriting>"
rtl_el ::= "<rtl>" fmt_body "</rtl>"
fmt_body ::= (text_chunk | content_el | formatting_el)*
"""

# --------------------------------------------------------------------------- #
# marker / hint
# --------------------------------------------------------------------------- #
_MARKER_HINT = r"""
marker_el ::= "<marker>" element_head marker_body "</marker>"
marker_body ::= (text_chunk | content_el | formatting_el | checkbox_el)*
hint_el ::= "<hint>" hint_body "</hint>"
hint_body ::= (text_chunk | content_el | formatting_el)*
"""

# --------------------------------------------------------------------------- #
# headless_txt: formatting | marker | hint | checkbox
# --------------------------------------------------------------------------- #
_HEADLESS_TXT = r"""
headless_txt ::= formatting_el | marker_el | hint_el | checkbox_el
"""

# --------------------------------------------------------------------------- #
# Semantic elements with a "semantic" body:
#   element_head + (content | headless_txt | top_level_el)*
# --------------------------------------------------------------------------- #
_SEMANTIC = r"""
text_el ::= "<text>" element_head text_body "</text>"
heading_el ::= "<heading" level_attr? ">" element_head text_body "</heading>"
level_attr ::= " level=\"" pos_int "\""
caption_el ::= "<caption>" element_head text_body "</caption>"
page_header_el ::= "<page_header>" element_head text_body "</page_header>"
page_footer_el ::= "<page_footer>" element_head text_body "</page_footer>"
footnote_el ::= "<footnote>" element_head text_body "</footnote>"
text_body ::= (text_chunk | content_el | headless_txt | top_level_el)*
"""

# --------------------------------------------------------------------------- #
# code / formula: element_head + (content | headless_el)*
# --------------------------------------------------------------------------- #
_CODE_FORMULA = r"""
code_el ::= "<code>" element_head code_body "</code>"
formula_el ::= "<formula>" element_head code_body "</formula>"
code_body ::= (text_chunk | content_el | headless_el)*
"""

# --------------------------------------------------------------------------- #
# list: element_head + list_item* ; list_item = ldiv + element_head? + body*
# --------------------------------------------------------------------------- #
_LIST = r"""
list_el ::= "<list" list_class? ">" element_head list_item* "</list>"
list_class ::= " class=\"" list_enum "\""
list_enum ::= "ordered" | "unordered"
list_item ::= ldiv_el element_head list_item_body
list_item_body ::= (text_chunk | content_el | headless_txt | top_level_el)*
ldiv_el ::= "<ldiv/>" | "<ldiv>" ldiv_body "</ldiv>"
ldiv_body ::= ws* marker_el? ws*
"""

# --------------------------------------------------------------------------- #
# table / index: element_head + table_cell* ;
#   table_cell = table_sep + element_head? + body*
# --------------------------------------------------------------------------- #
_TABLE = r"""
table_el ::= "<table>" element_head table_cell* "</table>"
index_el ::= "<index>" element_head table_cell* "</index>"
table_cell ::= table_sep_el element_head table_cell_body
table_cell_body ::= (text_chunk | content_el | headless_txt | top_level_el)*
"""

# --------------------------------------------------------------------------- #
# group: element_head + top_level_el*  (non-mixed)
# --------------------------------------------------------------------------- #
_GROUP = r"""
group_el ::= "<group>" element_head group_body "</group>"
group_body ::= ws* group_first? group_rest ws*
group_first ::= top_level_el
group_rest ::= group_item*
group_item ::= ws* top_level_el
"""

# --------------------------------------------------------------------------- #
# field_* : element_head + (headless_txt | top_level_el)*  (mixed, no <content>)
# key / value: element_head + (content | headless_txt | top_level_el)*
# --------------------------------------------------------------------------- #
_FIELD = r"""
field_region_el ::= "<field_region>" element_head field_body "</field_region>"
field_heading_el ::= "<field_heading" level_attr? ">" element_head field_body "</field_heading>"
field_item_el ::= "<field_item>" element_head field_body "</field_item>"
field_body ::= (text_chunk | headless_txt | top_level_el)*
key_el ::= "<key>" element_head key_body "</key>"
value_el ::= "<value" value_class? ">" element_head key_body "</value>"
value_class ::= " class=\"" value_enum "\""
value_enum ::= "read_only" | "fillable"
key_body ::= (text_chunk | content_el | headless_txt | top_level_el)*
"""

# --------------------------------------------------------------------------- #
# picture: element_head + src? + tabular? + (content | headless_txt | top_level_el)*
# --------------------------------------------------------------------------- #
_PICTURE = r"""
picture_el ::= "<picture" picture_class? ">" element_head picture_body "</picture>"
picture_class ::= " class=\"" picture_enum "\""
picture_enum ::= "undefined" | "chart"
picture_body ::= src_el? tabular_el? (text_chunk | content_el | headless_txt | top_level_el)*
"""

# --------------------------------------------------------------------------- #
# top_level_cat and headless (referenced by the bodies above)
# --------------------------------------------------------------------------- #
_CATEGORIES = r"""
top_level_el ::= text_el | heading_el | code_el | formula_el | page_header_el | page_footer_el | footnote_el | list_el | group_el | field_region_el | field_heading_el | field_item_el | key_el | value_el | picture_el | table_el | index_el
headless_el ::= headless_txt | code_el | formula_el | picture_el | field_region_el | field_heading_el | field_item_el | key_el | value_el
"""

DOCLANG_GRAMMAR: str = "\n".join(
    [
        _PRIMITIVES,
        _ROOT,
        _HEAD,
        _ELEMENT_HEAD,
        _PAYLOAD,
        _TABLE_SEP,
        _FORMATTING,
        _MARKER_HINT,
        _HEADLESS_TXT,
        _SEMANTIC,
        _CODE_FORMULA,
        _LIST,
        _TABLE,
        _GROUP,
        _FIELD,
        _PICTURE,
        _CATEGORIES,
    ]
)


def get_grammar() -> str:
    """Return the full DocLang grammar as a GBNF string (xgrammar / vLLM)."""
    return DOCLANG_GRAMMAR


def to_lark_grammar() -> str:
    """Return the DocLang grammar rewritten in lark's EBNF dialect.

    Transforms the canonical GBNF into lark syntax:

    * ``name ::= expr``  →  ``name: expr``
    * ``[X]`` char class →  ``/X/`` regex terminal

    The transformation is safe for this grammar because no string literal
    contains ``[``/``]`` and no quantifier is applied directly to a character
    class (see the module docstring).
    """
    g = DOCLANG_GRAMMAR.replace("::=", ":")
    g = re.sub(r"\[([^\]]*)\]", r"/[\1]/", g)
    return g
