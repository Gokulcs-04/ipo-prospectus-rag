from __future__ import annotations

from collections import defaultdict
from typing import Protocol, Sequence

import numpy as np

from ipo_analyzer.retrieval.lexical_store import BM25LexicalStore
from ipo_analyzer.retrieval.retriever import RetrievalResult, Retriever
from ipo_analyzer.schemas.chunk import DocumentChunk


class LexicalStoreProtocol(Protocol):
    @property
    def size(self) -> int:
        ...

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[np.ndarray, list[list[str]]]:
        ...


class HybridRetriever:
    """
    Hybrid dense + lexical retriever using Reciprocal Rank Fusion (RRF).

    Public retrieval contract intentionally matches Retriever:
        retrieve(query: str, top_k: int = 5) -> list[RetrievalResult]

    Dense retrieval remains owned by the existing Retriever/FAISS stack.
    Lexical retrieval is an independent BM25 index. RRF combines rankings
    rather than raw cosine/BM25 scores because those score scales differ.
    """

    def __init__(
        self,
        embedding_model,
        vector_store,
        chunks: Sequence[DocumentChunk],
        lexical_store: LexicalStoreProtocol | None = None,
        *,
        rrf_k: int = 60,
        candidate_k: int = 50,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0.")
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1.")

        self.dense_retriever = Retriever(embedding_model, vector_store, chunks)
        self.chunk_lookup = self.dense_retriever.chunk_lookup
        self.lexical_store = lexical_store or BM25LexicalStore(chunks)
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

        if self.lexical_store.size != len(self.chunk_lookup):
            raise ValueError(
                "Lexical store size must match the number of DocumentChunks."
            )

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve top-k chunks using dense + lexical rank fusion."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not query.strip():
            raise ValueError("query must not be empty.")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        candidate_k = max(top_k, self.candidate_k)
        dense_results = self.dense_retriever.retrieve(query, top_k=candidate_k)
        _, lexical_ids = self.lexical_store.search(query, top_k=candidate_k)
        lexical_ids = lexical_ids[0]

        # RRF score for a result at 1-indexed rank r is 1 / (rrf_k + r).
        fused_scores: defaultdict[str, float] = defaultdict(float)
        best_rank: dict[str, int] = {}

        for rank, result in enumerate(dense_results, start=1):
            chunk_id = result.chunk.chunk_id
            fused_scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            best_rank[chunk_id] = min(best_rank.get(chunk_id, rank), rank)

        for rank, chunk_id in enumerate(lexical_ids, start=1):
            fused_scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            best_rank[chunk_id] = min(best_rank.get(chunk_id, rank), rank)

        ranked_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (
                -fused_scores[chunk_id],
                best_rank[chunk_id],
                self.chunk_lookup[chunk_id].chunk_index,
            ),
        )[:top_k]

        return [
            RetrievalResult(
                chunk=self.chunk_lookup[chunk_id],
                # For HybridRetriever, score is the fused RRF score and should
                # be treated as a ranking score rather than a cosine similarity.
                score=fused_scores[chunk_id],
            )
            for chunk_id in ranked_ids
        ]
