from __future__ import annotations

from pathlib import Path
from typing import Literal

import pymupdf

from ipo_analyzer.ingestion.ocr import _ocr_page_from_document
from ipo_analyzer.ingestion.pdf_parser import parse_pdf
from ipo_analyzer.schemas.document import Document, DocumentPage


DocumentType = Literal[
    "RHP",
    "DRHP",
    "FINAL_PROSPECTUS",
]


def _meaningful_character_count(text: str) -> int:
    """Count non-whitespace characters in extracted text."""
    return len("".join(text.split()))


def _should_replace_with_ocr(
    original_page: DocumentPage,
    ocr_page_result: DocumentPage,
) -> bool:
    """
    Decide whether OCR produced a better page representation.

    OCR is preferred when:
    - it produced a normal-quality result, and
    - it contains more meaningful text than the original extraction.

    This prevents an unsuccessful OCR attempt from blindly replacing
    an already usable page.
    """

    if ocr_page_result.quality_flag != "normal":
        return False

    original_meaningful_chars = _meaningful_character_count(
        original_page.text
    )

    ocr_meaningful_chars = _meaningful_character_count(
        ocr_page_result.text
    )

    return ocr_meaningful_chars > original_meaningful_chars


def process_document(
    pdf_path: str | Path,
    *,
    document_id: str,
    company_name: str,
    document_type: DocumentType,
) -> Document:
    """
    Parse and process an IPO prospectus.

    PyMuPDF is used as the primary extraction method.
    Pages flagged as suspicious are passed through the OCR fallback.
    """

    document = parse_pdf(
        pdf_path,
        document_id=document_id,
        company_name=company_name,
        document_type=document_type,
    )

    processed_pages: list[DocumentPage] = list(document.pages)

    suspicious_pages = [
        page
        for page in document.pages
        if page.quality_flag == "suspicious"
    ]

    # No OCR required.
    if not suspicious_pages:
        return document

    path = Path(pdf_path)

    try:
        pdf_document = pymupdf.open(str(path))
    except Exception as exc:
        raise ValueError(
            f"Unable to open PDF for OCR: {path}"
        ) from exc

    try:
        total_ocr_pages = len(suspicious_pages)

        for ocr_index, original_page in enumerate(
            suspicious_pages,
            start=1,
        ):
            print(
                f"OCR processing page "
                f"{original_page.page_number} "
                f"({ocr_index}/{total_ocr_pages})...",
                flush=True,
            )

            ocr_result = _ocr_page_from_document(
                pdf_document,
                original_page.page_number,
                document_id=document.document_id,
            )

            if _should_replace_with_ocr(
                original_page,
                ocr_result,
            ):
                processed_pages[
                    original_page.page_number - 1
                ] = ocr_result

    finally:
        pdf_document.close()

    document.pages = processed_pages

    return document