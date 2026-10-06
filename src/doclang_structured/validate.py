"""Validation helper wrapping the official ``doclang`` validator.

The ``doclang`` package's :func:`doclang.validate` takes a **file path** and runs
the bundled reference XSD (always) plus Schematron rules (unless ``xsd_only``).
This module adapts it to operate on XML **strings** by writing to a temporary
``.dclg`` file, and re-raises failures as :class:`DocLangValidationError` which
carries the underlying XSD / Schematron error lists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from doclang import ValidationError as _DocLangValidationError
from doclang import validate as _doclang_validate

from utils.tempdir import OptionalTemporaryDirectory

__all__ = ["DocLangValidationError", "validate_doclang"]


class DocLangValidationError(Exception):
    """Raised when DocLang XML fails validation.

    Attributes:
        xsd_errors: XSD validation errors (empty when none / ``xsd_only=False``
            and only Schematron failed).
        schematron_errors: Schematron violations (empty when ``xsd_only=True``
            or when none).
    """

    def __init__(
        self,
        message: str,
        *,
        xsd_errors: list[dict[str, Any]] | None = None,
        schematron_errors: list[dict[str, Any]] | None = None,
    ) -> None:
        self.xsd_errors: list[dict[str, Any]] = xsd_errors or []
        self.schematron_errors: list[dict[str, Any]] = schematron_errors or []
        super().__init__(message)


def validate_doclang(xml: str, *, xsd_only: bool = False) -> None:
    """Validate a DocLang XML string with the official ``doclang`` validator.

    Args:
        xml: The ``.dclg`` XML content to validate.
        xsd_only: When True, run XSD validation only (fast, no JRE required).
            When False (default), run XSD + Schematron (requires the
            ``doclang[schematron-saxon]`` extra and a JRE).

    Raises:
        DocLangValidationError: When validation fails; carries ``xsd_errors``
            and ``schematron_errors``.
    """
    with OptionalTemporaryDirectory(prefix="doclang_validate_") as tmp:
        path = Path(tmp) / "doc.dclg"
        path.write_text(xml, encoding="utf-8")
        try:
            _doclang_validate(path, xsd_only=xsd_only)
        except _DocLangValidationError as exc:
            raise DocLangValidationError(
                str(exc),
                xsd_errors=exc.xsd_errors,
                schematron_errors=exc.schematron_errors,
            ) from exc
