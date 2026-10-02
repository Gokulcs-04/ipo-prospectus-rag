from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

import pymupdf
import pytesseract
from PIL import Image

from ipo_analyzer.schemas.document import DocumentPage


def _count_words(text: str) -> int:
    """Return a simple whitespace-based word count."""
    return len(text.split())


def _assess_ocr_quality(
    text: str,
    *,
    character_count: int,
    word_count: int,
) -> Literal["normal", "suspicious"]:
    """
    Perform a basic quality check on OCR output.

    OCR output is considered suspicious when very little meaningful
    text is produced.
    """

    if not text.strip():
        return "suspicious"

    meaningful_character_count = len(
        "".join(text.split())
    )

    if meaningful_character_count < 30 and word_count < 5:
        return "suspicious"

    return "normal"


def _ocr_page_from_document(
    pdf_document: pymupdf.Document,
    page_number: int,
    *,
    document_id: str,
) -> DocumentPage:
    """
    OCR one page using an already-open PyMuPDF document.

    This internal helper avoids repeatedly opening and closing the
    same PDF when multiple pages require OCR.
    """

    page_index = page_number - 1

    if page_number < 1:
        raise ValueError(
            "page_number must be 1 or greater."
        )

    if page_index >= pdf_document.page_count:
        raise ValueError(
            f"Page {page_number} does not exist. "
            f"PDF contains {pdf_document.page_count} pages."
        )

    pdf_page = pdf_document[page_index]

    # Render the PDF page at approximately 200 DPI.
    zoom = 200 / 72
    matrix = pymupdf.Matrix(zoom, zoom)

    pixmap = pdf_page.get_pixmap(
        matrix=matrix,
        colorspace=pymupdf.csRGB,
        alpha=False,
    )

    try:
        image_bytes = pixmap.tobytes("png")

        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()

            # Tesseract performs the actual OCR.
            text = pytesseract.image_to_string(
                image,
                lang="eng",
                config="--psm 3",
            )
    finally:
        pixmap = None

    character_count = len(text)
    word_count = _count_words(text)

    # Preserve the PDF's image references as metadata.
    image_count = len(
        pdf_page.get_images(full=False)
    )

    quality_flag = _assess_ocr_quality(
        text,
        character_count=character_count,
        word_count=word_count,
    )

    return DocumentPage(
        document_id=document_id,
        page_number=page_number,
        text=text,
        extraction_method="ocr",
        character_count=character_count,
        word_count=word_count,
        image_count=image_count,
        quality_flag=quality_flag,
        section=None,
        subsection=None,
    )


def ocr_page(
    pdf_path: str | Path,
    page_number: int,
    *,
    document_id: str,
) -> DocumentPage:
    """
    Extract text from one PDF page using OCR.

    This remains the public page-level OCR interface.
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

    if page_number < 1:
        raise ValueError(
            "page_number must be 1 or greater."
        )

    try:
        pdf_document = pymupdf.open(str(path))
    except Exception as exc:
        raise ValueError(
            f"Unable to open PDF: {path}"
        ) from exc

    try:
        return _ocr_page_from_document(
            pdf_document,
            page_number,
            document_id=document_id,
        )
    finally:
        pdf_document.close()