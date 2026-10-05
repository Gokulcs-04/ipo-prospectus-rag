"""
Chunk metadata: heading context, page provenance, chunk_id / chunk_index.

The chunker (semantic_chunker.py) produces RawChunk objects; this module turns them into
DocumentChunk objects without changing the schema.

Rules
  - page_start / page_end / pages are the ORIGINAL 1-indexed PDF page numbers (never printed page numbers)
  - section / subsection are inherited from the page-level section detector (never re-detected here)
  - chunk_index is zero-based, sequential and independent per document_id
  - chunk_id != chunk_index:  "<document_id>_chunk_<index:04d>"
  - content is never edited; only heading context (section/subsection names) is prepended when missing
"""

import re
from dataclasses import dataclass

from ipo_analyzer.schemas.chunk import DocumentChunk


@dataclass
class RawChunk:
    kind: str                        # text | table
    section: str | None
    subsection: str | None
    body: str
    pages: set[int]


def _norm_key(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def add_heading_context(rc: RawChunk, prefix_headings: bool = True) -> str:
    """Prepend section/subsection names so the chunk is understandable alone.
    Skipped for a heading that already appears at the start of the chunk (no duplicates)."""
    if not prefix_headings:
        return rc.body
    head = _norm_key(rc.body[:400])
    add: list[str] = []
    for h in (rc.section, rc.subsection):
        if h and _norm_key(h) not in head and _norm_key(h) not in {_norm_key(a) for a in add}:
            add.append(h.strip())
    return "\n".join(add + [rc.body]) if add else rc.body


def build_document_chunks(document_id: str, raw: list[RawChunk], prefix_headings: bool = True) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for rc in raw:
        content = add_heading_context(rc, prefix_headings).strip()
        if not content or not rc.pages:
            continue
        idx = len(chunks)
        pages = sorted(rc.pages)
        chunks.append(DocumentChunk(
            chunk_id=f"{document_id}_chunk_{idx:04d}",
            document_id=document_id,
            content=content,
            page_start=pages[0],
            page_end=pages[-1],
            pages=pages,
            section=rc.section,
            subsection=rc.subsection,
            chunk_index=idx,
        ))
    return chunks
