from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """
    Represents a semantic chunk derived from one or more
    pages of an IPO prospectus.

    This is the interface between the chunking stage
    and downstream embedding/retrieval stages.
    """

    chunk_id: str = Field(
        ...,
        description="Unique identifier for the chunk."
    )

    document_id: str = Field(
        ...,
        description="Identifier of the source document."
    )

    content: str = Field(
        ...,
        min_length=1,
        description="Text content of the chunk."
    )

    page_start: int = Field(
        ...,
        ge=1,
        description="First PDF page covered by the chunk."
    )

    page_end: int = Field(
        ...,
        ge=1,
        description="Last PDF page covered by the chunk."
    )

    pages: list[int] = Field(
        ...,
        min_length=1,
        description="All PDF pages contributing to the chunk."
    )

    section: str | None = Field(
        default=None,
        description="Section containing the chunk."
    )

    subsection: str | None = Field(
        default=None,
        description="Optional subsection containing the chunk."
    )

    chunk_index: int = Field(
        ...,
        ge=0,
        description="Zero-based position of the chunk within the document."
    )