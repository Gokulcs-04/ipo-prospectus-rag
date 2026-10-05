from __future__ import annotations

import numpy as np
import pytest

from ipo_analyzer.retrieval.lexical_store import BM25LexicalStore, tokenize
from ipo_analyzer.schemas.chunk import DocumentChunk


def make_chunk(chunk_id: str, index: int, content: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=content,
        page_start=index + 1,
        page_end=index + 1,
        pages=[index + 1],
        section="SECTION",
        subsection="SUBSECTION",
        chunk_index=index,
    )


def test_tokenizer_keeps_exact_phrase_as_bigram() -> None:
    tokens = tokenize("What is the size of the Fresh Issue?")
    assert "fresh" in tokens
    assert "issue" in tokens
    assert "fresh__issue" in tokens
    assert "what" not in tokens
    assert "the" not in tokens


def test_exact_prospectus_terms_rank_matching_chunk_first() -> None:
    chunks = [
        make_chunk("c1", 0, "The company describes ordinary equity shares."),
        make_chunk("c2", 1, "The Fresh Issue aggregates up to 1,200,000 Equity Shares."),
        make_chunk("c3", 2, "The objects of the Offer are described here."),
    ]
    store = BM25LexicalStore(chunks)

    scores, ids = store.search("What is the size of the Fresh Issue?", top_k=3)

    assert ids[0][0] == "c2"
    assert scores.shape == (1, 1)
    assert scores[0][0] > 0.0


def test_total_assets_term_is_retrievable() -> None:
    chunks = [
        make_chunk("c1", 0, "The company had total revenue and profit."),
        make_chunk("c2", 1, "Total Assets as at March 31, 2025 were Rs. 5,000 million."),
    ]
    store = BM25LexicalStore(chunks)

    _, ids = store.search("What were the company's total assets as at March 31, 2025?", top_k=2)

    assert ids[0][0] == "c2"


def test_validation() -> None:
    chunks = [make_chunk("c1", 0, "Revenue from operations")]
    store = BM25LexicalStore(chunks)

    with pytest.raises(ValueError, match="query must not be empty"):
        store.search("   ")

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        store.search("revenue", top_k=0)

    empty = BM25LexicalStore([])
    with pytest.raises(ValueError, match="empty lexical index"):
        empty.search("revenue")
