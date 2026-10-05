from __future__ import annotations

import numpy as np
import pytest

from ipo_analyzer.retrieval.retriever import RetrievalResult, Retriever
from ipo_analyzer.schemas.chunk import DocumentChunk


class FakeEmbeddingModel:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.calls: list[str] = []
        self.vector = np.asarray(vector or [1.0, 0.0], dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return self.vector


class FakeVectorStore:
    def __init__(self, rows: list[tuple[float, str]]) -> None:
        self.rows = rows
        self.calls: list[tuple[np.ndarray, int]] = []
        self.size = len(rows)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        self.calls.append((query_embedding.copy(), top_k))
        rows = self.rows[: min(top_k, self.size)]
        scores = np.asarray([[score for score, _ in rows]], dtype=np.float32)
        ids = [[chunk_id for _, chunk_id in rows]]
        return scores, ids


class EmptyVectorStore(FakeVectorStore):
    def __init__(self) -> None:
        super().__init__([])

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        raise ValueError("Cannot search an empty FAISS index.")


def make_chunk(
    chunk_id: str,
    index: int,
    *,
    pages: list[int] | None = None,
) -> DocumentChunk:
    pages = pages or [index + 1]
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=f"content for {chunk_id}",
        page_start=pages[0],
        page_end=pages[-1],
        pages=pages,
        section="FINANCIAL INFORMATION",
        subsection="RESULTS OF OPERATIONS",
        chunk_index=index,
    )


def test_query_embedding_is_generated() -> None:
    chunks = [make_chunk("c1", 0)]
    embedding = FakeEmbeddingModel([0.2, 0.8])
    store = FakeVectorStore([(0.9, "c1")])
    retriever = Retriever(embedding, store, chunks)

    results = retriever.retrieve("revenue", top_k=1)

    assert embedding.calls == ["revenue"]
    assert np.array_equal(store.calls[0][0], np.asarray([0.2, 0.8], dtype=np.float32))
    assert store.calls[0][1] == 1
    assert results[0].chunk.chunk_id == "c1"


def test_top_k_results_are_returned_in_similarity_order() -> None:
    chunks = [make_chunk("c1", 0), make_chunk("c2", 1), make_chunk("c3", 2)]
    store = FakeVectorStore([(0.91, "c2"), (0.72, "c1"), (0.51, "c3")])
    retriever = Retriever(FakeEmbeddingModel(), store, chunks)

    results = retriever.retrieve("query", top_k=2)

    assert [r.chunk.chunk_id for r in results] == ["c2", "c1"]
    assert [r.score for r in results] == pytest.approx([0.91, 0.72])


def test_chunk_ids_map_to_original_document_chunks() -> None:
    c1 = make_chunk("c1", 0, pages=[10, 11])
    c2 = make_chunk("c2", 1, pages=[20])
    retriever = Retriever(
        FakeEmbeddingModel(),
        FakeVectorStore([(0.9, "c2")]),
        [c1, c2],
    )

    result = retriever.retrieve("q", top_k=1)[0]

    assert result.chunk is c2


def test_metadata_is_preserved() -> None:
    chunk = DocumentChunk(
        chunk_id="c1",
        document_id="doc-ellenbarrie",
        content="Financial information",
        page_start=310,
        page_end=312,
        pages=[310, 311, 312],
        section="FINANCIAL INFORMATION",
        subsection="RESULTS OF OPERATIONS",
        chunk_index=802,
    )
    result = Retriever(
        FakeEmbeddingModel(),
        FakeVectorStore([(0.88, "c1")]),
        [chunk],
    ).retrieve("revenue", top_k=1)[0]

    assert result.chunk.chunk_id == "c1"
    assert result.chunk.document_id == "doc-ellenbarrie"
    assert result.chunk.content == "Financial information"
    assert result.chunk.page_start == 310
    assert result.chunk.page_end == 312
    assert result.chunk.pages == [310, 311, 312]
    assert result.chunk.section == "FINANCIAL INFORMATION"
    assert result.chunk.subsection == "RESULTS OF OPERATIONS"
    assert result.chunk.chunk_index == 802
    assert result.score == pytest.approx(0.88)


def test_top_k_larger_than_corpus_is_safe() -> None:
    chunks = [make_chunk("c1", 0), make_chunk("c2", 1)]
    store = FakeVectorStore([(0.9, "c1"), (0.8, "c2")])
    results = Retriever(FakeEmbeddingModel(), store, chunks).retrieve("q", top_k=10)

    assert len(results) == 2
    assert store.calls[0][1] == 10


def test_missing_chunk_id_raises_key_error() -> None:
    retriever = Retriever(
        FakeEmbeddingModel(),
        FakeVectorStore([(0.9, "missing")]),
        [make_chunk("c1", 0)],
    )

    with pytest.raises(KeyError, match="unknown chunk_id: missing"):
        retriever.retrieve("q", top_k=1)


def test_empty_query_is_rejected() -> None:
    retriever = Retriever(FakeEmbeddingModel(), FakeVectorStore([]), [make_chunk("c1", 0)])

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.retrieve("   ")


def test_invalid_top_k_is_rejected() -> None:
    retriever = Retriever(FakeEmbeddingModel(), FakeVectorStore([]), [make_chunk("c1", 0)])

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.retrieve("q", top_k=0)


def test_empty_vector_store_error_is_propagated() -> None:
    retriever = Retriever(FakeEmbeddingModel(), EmptyVectorStore(), [make_chunk("c1", 0)])

    with pytest.raises(ValueError, match="empty FAISS index"):
        retriever.retrieve("q", top_k=1)


def test_duplicate_chunk_ids_are_rejected_at_construction() -> None:
    chunks = [make_chunk("c1", 0), make_chunk("c1", 1)]

    with pytest.raises(ValueError, match="Duplicate chunk_id: c1"):
        Retriever(FakeEmbeddingModel(), FakeVectorStore([(0.9, "c1")]), chunks)
