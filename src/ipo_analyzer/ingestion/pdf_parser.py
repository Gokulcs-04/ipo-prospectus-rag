from __future__ import annotations

from pathlib import Path
from typing import Literal

import pymupdf

from ipo_analyzer.schemas.document import Document, DocumentPage


DocumentType = Literal[
    "RHP",
    "DRHP",
    "FINAL_PROSPECTUS",
]


def _count_words(text: str) -> int:
    """Return a simple whitespace-based word count."""
    return len(text.split())


def _assess_text_quality(
    text: str,
    *,
    character_count: int,
    word_count: int,
    image_count: int,
) -> Literal["normal", "suspicious"]:
    """
    Perform a conservative first-pass extraction-quality check.

    The check uses multiple signals:
    - completely empty extraction
    - malformed/replacement characters
    - image page with very little meaningful text
    - extremely small text extraction

    Whitespace is excluded when measuring meaningful characters.
    """

    stripped_text = text.strip()

    # No extracted text at all.
    if not stripped_text:
        return "suspicious"

    # Count characters after removing whitespace.
    meaningful_character_count = len(
        "".join(text.split())
    )

    # Detect possible malformed extraction.
    replacement_count = text.count("\ufffd")
    replacement_ratio = replacement_count / max(
        meaningful_character_count,
        1,
    )

    if replacement_ratio > 0.005:
        return "suspicious"

    # An image-containing page with almost no meaningful
    # extracted text is a strong OCR candidate.
    if (
        image_count > 0
        and meaningful_character_count < 100
        and word_count < 20
    ):
        return "suspicious"

    # Completely tiny text extraction.
    if meaningful_character_count < 30 and word_count < 5:
        return "suspicious"

    return "normal"


def parse_pdf(
    pdf_path: str | Path,
    *,
    document_id: str,
    company_name: str,
    document_type: DocumentType,
) -> Document:
    """
    Parse an IPO prospectus PDF using PyMuPDF.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.

    document_id:
        Unique identifier assigned to the source document.

    company_name:
        Name of the issuing company.

    document_type:
        Type of IPO document:
        RHP, DRHP, or FINAL_PROSPECTUS.

    Returns
    -------
    Document
        Parsed document containing one DocumentPage per PDF page.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"PDF path is not a file: {path}"
        )

    try:
        pdf_document = pymupdf.open(str(path))
    except Exception as exc:
        raise ValueError(
            f"Unable to open PDF: {path}"
        ) from exc

    pages: list[DocumentPage] = []
    page_count = pdf_document.page_count

    try:
        for page_index in range(page_count):
            pdf_page = pdf_document[page_index]

            # Extract native PDF text.
            #
            # PyMuPDF internally uses zero-based page indices.
            # Our project uses 1-indexed page numbers.
            text = pdf_page.get_text(
                "text",
                sort=True,
            )

            character_count = len(text)
            word_count = _count_words(text)

            # Count image references on the page.
            image_count = len(
                pdf_page.get_images(full=False)
            )

            quality_flag = _assess_text_quality(
                text,
                character_count=character_count,
                word_count=word_count,
                image_count=image_count,
            )

            page = DocumentPage(
                document_id=document_id,
                page_number=page_index + 1,
                text=text,
                extraction_method="pymupdf",
                character_count=character_count,
                word_count=word_count,
                image_count=image_count,
                quality_flag=quality_flag,
                section=None,
                subsection=None,
            )

            pages.append(page)

    finally:
        pdf_document.close()

    return Document(
        document_id=document_id,
        company_name=company_name,
        document_type=document_type,
        source_file=path.name,
        page_count=page_count,
        pages=pages,
    )