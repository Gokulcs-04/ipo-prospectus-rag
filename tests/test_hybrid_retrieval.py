from __future__ import annotations

import numpy as np

from ipo_analyzer.retrieval.hybrid_retriever import HybridRetriever
from ipo_analyzer.schemas.chunk import DocumentChunk


class FakeEmbeddingModel:
    def encode_single(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


class FakeVectorStore:
    size = 3

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        rows = [
            (0.95, "c2"),
            (0.80, "c3"),
            (0.70, "c1"),
        ][:top_k]
        return (
            np.asarray([[score for score, _ in rows]], dtype=np.float32),
            [[chunk_id for _, chunk_id in rows]],
        )


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


def test_hybrid_retriever_preserves_retrieval_contract_and_fuses_rankings() -> None:
    chunks = [
        make_chunk("c1", 0, "Fresh Issue aggregates up to 1,200,000 shares."),
        make_chunk("c2", 1, "Company financial highlights."),
        make_chunk("c3", 2, "Other business information."),
    ]

    retriever = HybridRetriever(
        FakeEmbeddingModel(),
        FakeVectorStore(),
        chunks,
        candidate_k=3,
    )

    results = retriever.retrieve("What is the size of the Fresh Issue?", top_k=2)

    # c2 is dense-rank 1, while c1 is lexical-rank 1 and dense-rank 3.
    # RRF therefore rewards c1 for appearing in both retrieval arms.
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].chunk.document_id == "doc-1"
    assert results[0].score > results[1].score


def test_metadata_is_preserved() -> None:
    chunks = [
        make_chunk("c1", 0, "Total Assets are disclosed here."),
        make_chunk("c2", 1, "Unrelated text."),
        make_chunk("c3", 2, "More unrelated text."),
    ]
    retriever = HybridRetriever(
        FakeEmbeddingModel(),
        FakeVectorStore(),
        chunks,
        candidate_k=3,
    )

    result = retriever.retrieve("total assets", top_k=1)[0]

    assert result.chunk.chunk_id == "c1"
    assert result.chunk.pages == [1]
    assert result.chunk.section == "SECTION"
