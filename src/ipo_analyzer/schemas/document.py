from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    """
    Represents a single page of an IPO prospectus.

    This is the main interface between the ingestion stage
    and downstream document-processing stages.
    """

    document_id: str = Field(
        ...,
        description="Unique identifier of the source document."
    )

    page_number: int = Field(
        ...,
        ge=1,
        description="1-indexed PDF page number."
    )

    text: str = Field(
        default="",
        description="Text extracted from the page."
    )

    extraction_method: Literal[
        "pymupdf",
        "ocr",
    ] = Field(
        default="pymupdf",
        description="Method used to extract the page text."
    )

    character_count: int = Field(
        default=0,
        ge=0,
        description="Number of extracted characters."
    )

    word_count: int = Field(
        default=0,
        ge=0,
        description="Number of extracted words."
    )

    image_count: int = Field(
        default=0,
        ge=0,
        description="Number of images detected on the page."
    )

    quality_flag: Literal[
        "normal",
        "suspicious",
    ] = Field(
        default="normal",
        description="Basic extraction-quality indicator."
    )

    section: str | None = Field(
        default=None,
        description="Section assigned during section detection."
    )

    subsection: str | None = Field(
        default=None,
        description="Optional subsection assigned during section detection."
    )


class Document(BaseModel):
    """
    Represents one complete IPO prospectus.
    """

    document_id: str = Field(
        ...,
        description="Unique identifier of the document."
    )

    company_name: str = Field(
        ...,
        description="Name of the issuing company."
    )

    document_type: Literal[
        "RHP",
        "DRHP",
        "FINAL_PROSPECTUS",
    ] = Field(
        ...,
        description="Type of IPO offer document."
    )

    source_file: str = Field(
        ...,
        description="Original PDF filename."
    )

    page_count: int = Field(
        ...,
        ge=0,
        description="Total number of pages in the PDF."
    )

    pages: list[DocumentPage] = Field(
        default_factory=list,
        description="Pages belonging to this document."
    )