"""Filesystem autodiscovery for corpora and their source images.

Document corpora are rarely flat. A typical PAGE XML corpus nests its ground
truth one or more levels down, in a directory named ``page``, with the scanned
images sitting next to (or above) it::

    corpus/
      document_001/
        page/
          document_001_0001.xml
          document_001_0002.xml
        document_001_0001.png
        document_001_0002.png

The helpers here let the data creators point ``--input`` at a corpus root
instead of a single flat directory:

- :func:`discover_page_xml` collects ``.xml`` files from directories named
  ``page`` at any depth. Matching is deliberately keyed on the directory name
  rather than the extension alone, so unrelated ``.xml`` files (ALTO output,
  sidecar XSLT, ``metadata.xml``) are never ingested by accident.
- :func:`discover_doclang` collects ``.dclg`` files recursively. No directory
  convention is needed there: the extension is unambiguous.
- :func:`discover_jsonl` collects ``.jsonl`` files from directories (or passes
  individual files through unchanged).
- :func:`resolve_image` maps a ground-truth document to its source image using
  an ordered chain of strategies (explicit image roots, the PAGE XML
  ``imageFilename`` attribute, then the directory above a ``page`` directory).

:func:`find_image` is the single stem-matching primitive shared by all the
data creators; it used to be duplicated in each of them.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from pathlib import Path

from PIL import Image as PILImage

__all__ = [
    "PAGE_DIR_NAME",
    "DiscoveryError",
    "discover_doclang",
    "discover_jsonl",
    "discover_page_xml",
    "find_image",
    "iter_image_files",
    "iter_page_dirs",
    "page_image_filename",
    "resolve_image",
]

#: Name of the directory that holds PAGE XML ground truth (matched case-insensitively).
PAGE_DIR_NAME = "page"

_DOCLANG_SUFFIX = ".dclg"
_XML_SUFFIX = ".xml"
_JSONL_SUFFIX = ".jsonl"


class DiscoveryError(FileNotFoundError):
    """Raised when a corpus root contains no discoverable input files."""


def _as_roots(roots: Path | str | Iterable[Path | str] | None) -> list[Path]:
    """Normalise a single path or an iterable of paths into a list of existing directories.

    Args:
        roots (Path | str | Iterable[Path | str] | None): One directory or an iterable of them.

    Returns:
        list[Path]: The directories that exist, in the order given.

    Raises:
        NotADirectoryError: If a root is not an existing directory.
    """
    if roots is None:
        return []
    candidates: Sequence[Path | str] = [roots] if isinstance(roots, (str, Path)) else list(roots)
    normalised: list[Path] = []
    for candidate in candidates:
        root = Path(candidate)
        if not root.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {root}")
        normalised.append(root)
    return normalised


def iter_page_dirs(roots: Path | str | Iterable[Path | str]) -> Iterable[Path]:
    """Yield every directory named ``page`` at any depth under the given roots.

    A root that is itself named ``page`` counts. Nested ``page`` directories are
    not descended into, so a page directory is reported exactly once.

    Args:
        roots (Path | str | Iterable[Path | str]): Corpus root director(ies).

    Yields:
        Path: Each ``page`` directory, in depth-first sorted order.

    Raises:
        NotADirectoryError: If a root is not an existing directory.
    """
    for root in _as_roots(roots):
        if root.name.lower() == PAGE_DIR_NAME:
            yield root
            continue
        for dirpath, dirnames, _ in os.walk(root):
            dirnames.sort()
            for name in list(dirnames):
                if name.lower() == PAGE_DIR_NAME:
                    dirnames.remove(name)
                    yield Path(dirpath) / name


def discover_page_xml(roots: Path | str | Iterable[Path | str]) -> list[Path]:
    """Discover PAGE XML files in ``page`` directories under the given roots.

    Args:
        roots (Path | str | Iterable[Path | str]): Corpus root director(ies).

    Returns:
        list[Path]: The ``.xml`` files found inside ``page`` directories, sorted.

    Raises:
        NotADirectoryError: If a root is not an existing directory.
        DiscoveryError: If no ``page`` directory holds any ``.xml`` file.
    """
    files: list[Path] = []
    for page_dir in iter_page_dirs(roots):
        files.extend(sorted(page_dir.glob(f"*{_XML_SUFFIX}")))
    if not files:
        root_list = ", ".join(str(Path(root)) for root in _as_roots(roots))
        raise DiscoveryError(
            f"No PAGE XML files found. Expected .xml files inside a directory named " f"'{PAGE_DIR_NAME}' under: {root_list}"
        )
    return sorted(files)


def discover_doclang(roots: Path | str | Iterable[Path | str]) -> list[Path]:
    """Discover DocLang documents recursively under the given roots.

    Args:
        roots (Path | str | Iterable[Path | str]): Corpus root director(ies).

    Returns:
        list[Path]: The ``.dclg`` files found at any depth, sorted.

    Raises:
        NotADirectoryError: If a root is not an existing directory.
        DiscoveryError: If no ``.dclg`` file exists under the roots.
    """
    files: list[Path] = []
    for root in _as_roots(roots):
        files.extend(sorted(root.rglob(f"*{_DOCLANG_SUFFIX}")))
    if not files:
        root_list = ", ".join(str(root) for root in _as_roots(roots))
        raise DiscoveryError(f"No DocLang files (*{_DOCLANG_SUFFIX}) found under: {root_list}")
    return sorted(files)


def discover_page_and_doclang(roots: Path | str | Iterable[Path | str]) -> list[Path]:
    """Discover PAGE XML and DocLang documents together under the given roots.

    PAGE XML is found only inside ``page`` directories while DocLang is found
    at any depth. Neither kind on its own is required — only their union — so
    a corpus of DocLang documents without ``page`` directories (or the
    reverse) still works.

    Args:
        roots (Path | str | Iterable[Path | str]): Corpus root director(ies).

    Returns:
        list[Path]: The discovered ``.xml`` and ``.dclg`` files, sorted.

    Raises:
        NotADirectoryError: If a root is not an existing directory.
        DiscoveryError: If neither kind of document was found.
    """
    root_list = [str(root) for root in _as_roots(roots)]
    files: list[Path] = []
    for page_dir in iter_page_dirs(root_list):
        files.extend(sorted(page_dir.glob(f"*{_XML_SUFFIX}")))
    for root in root_list:
        files.extend(sorted(Path(root).rglob(f"*{_DOCLANG_SUFFIX}")))
    if not files:
        raise DiscoveryError(
            f"No PAGE XML or DocLang files found. Expected .xml files inside a directory named "
            f"'{PAGE_DIR_NAME}', or .dclg files at any depth, under: {', '.join(root_list)}"
        )
    return sorted(set(files))


def discover_jsonl(roots: Path | str | Iterable[Path | str]) -> list[Path]:
    """Discover raw JSONL inputs, expanding directories into their ``.jsonl`` files.

    Individual file paths are passed through unchanged so that a single-file
    invocation keeps working exactly as before; directories contribute the
    ``.jsonl`` files directly inside them.

    Args:
        roots (Path | str | Iterable[Path | str]): Files and/or directories.

    Returns:
        list[Path]: The JSONL files to read, in the order given (directories sorted).

    Raises:
        FileNotFoundError: If a named file does not exist.
        DiscoveryError: If the inputs resolve to no ``.jsonl`` file at all.
    """
    candidates: Sequence[Path | str] = [roots] if isinstance(roots, (str, Path)) else list(roots)
    files: list[Path] = []
    for candidate in candidates:
        path = Path(candidate)
        if path.is_dir():
            files.extend(sorted(path.glob(f"*{_JSONL_SUFFIX}")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")
    if not files:
        raise DiscoveryError(f"No JSONL files (*{_JSONL_SUFFIX}) found in: {', '.join(str(Path(c)) for c in candidates)}")
    return files


def iter_image_files(directory: Path | str) -> list[Path]:
    """List the image files directly inside a directory, in any PIL-readable format.

    Args:
        directory (Path | str): Directory to list.

    Returns:
        list[Path]: Image files sorted by name.

    Raises:
        NotADirectoryError: If the path is not an existing directory.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {root}")
    extensions = {suffix.lower() for suffix in PILImage.registered_extensions()}
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def find_image(image_dir: Path, stem: str) -> Path | None:
    """Find a source image for a document in any format PIL can open.

    Args:
        image_dir (Path): Directory to search for the source document image.
        stem (str): Document stem to match against image filenames.

    Returns:
        Path | None: The first matching image path, or None if no supported image exists.
    """
    for extension in sorted(PILImage.registered_extensions()):
        candidate = image_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def page_image_filename(page_xml_path: Path) -> str | None:
    """Read the ``imageFilename`` attribute of a PAGE XML document.

    The PAGE format declares the name of the image a page was captured from on
    its ``Page`` element, which is the link the spec intends between ground
    truth and pixels. Namespace prefixes are ignored.

    Args:
        page_xml_path (Path): Path to a PAGE XML file.

    Returns:
        str | None: The declared image filename, or None when absent or unreadable.
    """
    try:
        root = ET.parse(page_xml_path).getroot()
    except (ET.ParseError, OSError):
        return None
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "Page":
            declared = element.get("imageFilename")
            if declared:
                return declared
    return None


def _layout_dirs(doc_path: Path) -> list[Path]:
    """Return the corpus-layout directories that may hold a document's image.

    The document's own directory, plus the directory above it when the document
    sits in a ``page`` directory (the common ``document/page/x.xml`` +
    ``document/x.png`` layout).

    Args:
        doc_path (Path): Path to a ground-truth document.

    Returns:
        list[Path]: Candidate directories, nearest first.
    """
    dirs = [doc_path.parent]
    if doc_path.parent.name.lower() == PAGE_DIR_NAME:
        dirs.append(doc_path.parent.parent)
    return dirs


def resolve_image(doc_path: Path, image_roots: Path | str | Iterable[Path | str] | None = None) -> Path | None:
    """Locate the source image for a ground-truth document.

    Strategies are tried in priority order and the first hit wins:

    1. ``<stem>.<ext>`` in each explicit ``image_roots`` directory.
    2. The PAGE XML ``imageFilename`` attribute, resolved against the document's
       own directory, the directory above a ``page`` directory, and each
       explicit image root.
    3. ``<stem>.<ext>`` in the document's own directory and, when the document
       lives in a ``page`` directory, in the directory above it.

    Args:
        doc_path (Path): Path to a PAGE XML (``.xml``) or DocLang (``.dclg``) document.
        image_roots (Path | str | Iterable[Path | str] | None): Explicit image directories, if any.

    Returns:
        Path | None: The source image path, or None when no strategy matched.
    """
    roots = _as_roots(image_roots)
    layout_dirs = _layout_dirs(doc_path)

    for root in roots:
        hit = find_image(root, doc_path.stem)
        if hit is not None:
            return hit

    if doc_path.suffix.lower() == _XML_SUFFIX:
        declared = page_image_filename(doc_path)
        if declared:
            for base in [*layout_dirs, *roots]:
                candidate = base / declared
                if candidate.is_file():
                    return candidate

    for base in layout_dirs:
        hit = find_image(base, doc_path.stem)
        if hit is not None:
            return hit
    return None
