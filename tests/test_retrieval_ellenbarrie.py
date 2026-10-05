from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ipo_analyzer.chunking.semantic_chunker import chunk_document
from ipo_analyzer.retrieval.retriever import Retriever
from ipo_analyzer.schemas.document import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ELLENBARRIE_JSON = PROJECT_ROOT / "data" / "processed" / "ellenbarrie_rhp.json"


class ReplayEmbeddingModel:
    """Test-only embedding stub: records the natural-language query."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def encode_single(self, text: str) -> np.ndarray:
        self.queries.append(text)
        return np.asarray([1.0, 0.0], dtype=np.float32)


class ReplayVectorStore:
    """Replay IDs/scores from the previously recorded Ellenbarrie baseline run.

    This intentionally does not alter production ranking. It tests that the new
    retriever maps real, previously observed vector-store IDs to the real
    DocumentChunk objects produced from the Ellenbarrie prospectus.
    """

    BASELINE = {
        "What was the company's revenue from operations in Fiscal 2025?": (
            [(0.5836, "ellenbarrie_rhp_chunk_0802")]
        ),
        "Who are the promoters of the company?": (
            [(0.5660, "ellenbarrie_rhp_chunk_0622")]
        ),
        "Who is the Chief Financial Officer of the company?": (
            [(0.6026, "ellenbarrie_rhp_chunk_1056")]
        ),
        "What is the size of the Fresh Issue?": (
            [(0.2585, "ellenbarrie_rhp_chunk_0392")]
        ),
        "What were the company's total assets as at March 31, 2025?": (
            [(0.5688, "ellenbarrie_rhp_chunk_0217")]
        ),
        "What are the objects of the Offer?": (
            [(0.5935, "ellenbarrie_rhp_chunk_0213")]
        ),
    }

    def __init__(self, query: str) -> None:
        self.query = query
        self.size = 1076

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        rows = self.BASELINE[self.query][:top_k]
        scores = np.asarray([[score for score, _ in rows]], dtype=np.float32)
        ids = [[chunk_id for _, chunk_id in rows]
               ]
        return scores, ids


def test_real_ellenbarrie_chunk_mapping_for_baseline_queries() -> None:
    data = json.loads(ELLENBARRIE_JSON.read_text(encoding="utf-8"))
    document = Document.model_validate(data)
    chunks = chunk_document(document)

    assert len(document.pages) == 417
    assert len(chunks) == 1076

    cases = [
        (
            "What was the company's revenue from operations in Fiscal 2025?",
            "ellenbarrie_rhp_chunk_0802",
            "revenue from operations",
        ),
        (
            "Who are the promoters of the company?",
            "ellenbarrie_rhp_chunk_0622",
            "promoter",
        ),
        (
            "Who is the Chief Financial Officer of the company?",
            "ellenbarrie_rhp_chunk_1056",
            "chief financial office",
        ),
        (
            "What were the company's total assets as at March 31, 2025?",
            "ellenbarrie_rhp_chunk_0217",
            "summary of financial information",
        ),
        (
            "What are the objects of the Offer?",
            "ellenbarrie_rhp_chunk_0213",
            "offer",
        ),
    ]

    chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}

    for query, expected_id, expected_text in cases:
        embedding = ReplayEmbeddingModel()
        result = Retriever(
            embedding,
            ReplayVectorStore(query),
            chunks,
        ).retrieve(query, top_k=1)[0]

        assert result.chunk is chunk_lookup[expected_id]
        assert expected_text in result.chunk.content.lower()
        assert result.chunk.document_id == "ellenbarrie_rhp"
        assert result.chunk.page_start >= 1
        assert result.chunk.pages
        assert result.score > 0
        assert embedding.queries == [query]
