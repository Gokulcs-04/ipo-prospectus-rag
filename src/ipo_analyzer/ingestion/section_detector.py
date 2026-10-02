from __future__ import annotations

import re

from ipo_analyzer.schemas.document import Document, DocumentPage


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

SECTION_PATTERN = re.compile(
    r"^SECTION\s+([IVXLC]+)\s*[:\-–—]\s*(.+?)$",
    flags=re.IGNORECASE,
)

NESTED_SECTION_PATTERN = re.compile(
    r"^Section\s+\d+\s*[:\-–—]\s*(.+?)$",
    flags=re.IGNORECASE,
)

NUMBERED_HEADING_PATTERN = re.compile(
    r"^\d+[\.\)]\s+"
)


# Lines that are commonly administrative metadata rather than headings.
EXCLUDED_SUBSECTION_LINES = {
    "CIN",
    "PAN",
    "TAN",
    "DIN",
    "UDIN",
    "GST",
    "ISBN",
}


def _clean_line(line: str) -> str:
    """Normalize whitespace while preserving the textual content."""
    return re.sub(r"\s+", " ", line).strip()


def _contains_table_of_contents(lines: list[str]) -> bool:
    """
    Detect whether the page appears to be a table-of-contents page.

    Some prospectuses use "TABLE OF CONTENTS", while others simply
    use "CONTENTS".
    """
    first_lines = lines[:20]

    for line in first_lines:
        normalized = re.sub(
            r"[^A-Z ]",
            "",
            line.upper(),
        ).strip()

        if normalized in {
            "TABLE OF CONTENTS",
            "CONTENTS",
        }:
            return True

    return False

def _normalize_heading(text: str) -> str:
    """
    Remove harmless leading/trailing punctuation introduced by
    PDF extraction or OCR while preserving the actual heading text.
    """
    return text.strip(
        " \t\r\n"
        "-–—"
        "•·"
        ":;,. "
    ).strip()


def _parse_major_section(line: str) -> str | None:
    """
    Return the major section title when the line matches the
    explicit SECTION heading pattern.
    """
    match = SECTION_PATTERN.match(line)

    if not match:
        return None

    title = match.group(2).strip()

    return _normalize_heading(title)

def _parse_nested_section(line: str) -> str | None:
    match = NESTED_SECTION_PATTERN.match(line)
    if not match:
        return None

    title = match.group(1).strip()
    return _normalize_heading(title)

def _is_uppercase_heading(
    line: str,
) -> bool:
    """
    Identify a conservative subsection-heading candidate.

    This intentionally rejects:
    - empty lines
    - numbered list items
    - very long paragraph-like text
    - lines dominated by numbers
    - common administrative identifiers
    """

    if not line:
        return False

    normalized = _normalize_heading(line)

    if len(normalized) < 3 or len(normalized) > 100:
        return False

    # Do not treat numbered list items as headings.
    if NUMBERED_HEADING_PATTERN.match(normalized):
        return False

    # Ignore common identifier-only lines.
    if normalized.upper() in EXCLUDED_SUBSECTION_LINES:
        return False

    # Ignore lines containing obvious sentence punctuation.
    if any(
        punctuation in normalized
        for punctuation in [".", ",", ";"]
    ):
        return False

    letters = [
        character
        for character in normalized
        if character.isalpha()
    ]

    if len(letters) < 3:
        return False

    # Require the alphabetic characters to be predominantly uppercase.
    uppercase_ratio = sum(
        character.isupper()
        for character in letters
    ) / len(letters)

    if uppercase_ratio < 0.90:
        return False

    # Reject lines overwhelmingly made up of digits.
    digits = sum(
        character.isdigit()
        for character in normalized
    )

    if digits > len(normalized) * 0.40:
        return False

    # Avoid treating short standalone acronyms such as "CAGR" as headings.
    words = normalized.split()

    if len(words) == 1 and len(normalized) < 6:
        return False

    return True


def _find_major_section_on_page(page: DocumentPage) -> str | None:
    lines = [
        _clean_line(line)
        for line in page.text.splitlines()
        if _clean_line(line)
    ]

    if _contains_table_of_contents(lines):
        return None

    for line in lines:
        major_section = _parse_major_section(line)

        if major_section is not None:
            return major_section

    return None
def _find_subsection_on_page(
    page: DocumentPage,
    current_section: str | None,
) -> str | None:
    if current_section is None:
        return None

    lines = [
        _clean_line(line)
        for line in page.text.splitlines()
        if _clean_line(line)
    ]

    if _contains_table_of_contents(lines):
        return None

    for line in lines:
        # Major "SECTION ..." headings belong to the section field.
        if _parse_major_section(line) is not None:
            continue

        # Nested "Section 2/3/4: ..." headings belong to the subsection field.
        nested_section = _parse_nested_section(line)
        if nested_section is not None:
            return nested_section

        if _is_uppercase_heading(line):
            return _normalize_heading(line)

    return None

def detect_sections(
    document: Document,
) -> Document:
    """
    Assign section and subsection metadata to document pages.

    Major sections are detected from explicit "SECTION ..." headings.
    Once detected, the current major section is propagated to subsequent
    pages until another major section is encountered.

    Subsections are detected conservatively from standalone uppercase
    headings within the current major section.

    Parameters
    ----------
    document:
        Processed document containing page-level text.

    Returns
    -------
    Document
        The same document with section/subsection metadata assigned.
    """

    current_section: str | None = None
    current_subsection: str | None = None

    for page in document.pages:
        major_section = _find_major_section_on_page(page)

        if major_section is not None:
            current_section = major_section
            current_subsection = None

        page.section = current_section

        subsection = None

        if current_section is not None:
            subsection = _find_subsection_on_page(
                page,
                current_section=current_section,
            )

        if subsection is not None:
            current_subsection = subsection

        page.subsection = current_subsection

    return document