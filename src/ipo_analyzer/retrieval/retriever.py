from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from ipo_analyzer.schemas.chunk import DocumentChunk


class EmbeddingModelProtocol(Protocol):
    """Minimal embedding interface required by the retriever."""

    def encode_single(self, text: str) -> np.ndarray:
        ...


class VectorStoreProtocol(Protocol):
    """Minimal vector-store interface required by the retriever."""

    @property
    def size(self) -> int:
        ...

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> tuple[np.ndarray, list[list[str]]]:
        ...


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A retrieved chunk together with its vector-similarity score."""

    chunk: DocumentChunk
    score: float


class Retriever:
    """
    Single-document retrieval layer.

    Responsibilities:
        query -> embedding -> vector search -> chunk IDs -> DocumentChunks

    The retriever does not create or modify chunk metadata and does not
    generate natural-language answers.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModelProtocol,
        vector_store: VectorStoreProtocol,
        chunks: Sequence[DocumentChunk],
    ) -> None:
        if embedding_model is None:
            raise ValueError("embedding_model must not be None.")
        if vector_store is None:
            raise ValueError("vector_store must not be None.")
        if chunks is None:
            raise ValueError("chunks must not be None.")

        chunk_lookup: dict[str, DocumentChunk] = {}
        for chunk in chunks:
            if not isinstance(chunk, DocumentChunk):
                raise TypeError("chunks must contain DocumentChunk objects.")
            if chunk.chunk_id in chunk_lookup:
                raise ValueError(f"Duplicate chunk_id: {chunk.chunk_id}")
            chunk_lookup[chunk.chunk_id] = chunk

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.chunk_lookup = chunk_lookup

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        Retrieve the top-k most similar chunks for a single query.

        Args:
            query: Natural-language user query.
            top_k: Maximum number of results to return. The vector store may
                return fewer results when the corpus is smaller than top_k.

        Raises:
            TypeError: If query is not a string or top_k is not an integer.
            ValueError: If query is empty or top_k is less than 1.
            KeyError: If the vector store returns an unknown chunk ID.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not query.strip():
            raise ValueError("query must not be empty.")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        # Let the vector store own empty-index behavior and corpus-size capping.
        query_embedding = np.asarray(
            self.embedding_model.encode_single(query),
            dtype=np.float32,
        )

        if query_embedding.ndim != 1:
            raise ValueError(
                "EmbeddingModel.encode_single() must return a 1-dimensional vector."
            )

        scores, result_ids = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        if len(result_ids) != 1:
            raise ValueError(
                "Vector store search must return exactly one result row for a single query."
            )
        if np.asarray(scores).ndim != 2 or np.asarray(scores).shape[0] != 1:
            raise ValueError(
                "Vector store scores must be a 2-dimensional array with one query row."
            )

        ids = result_ids[0]
        row_scores = np.asarray(scores)[0]

        if len(ids) != len(row_scores):
            raise ValueError(
                "Vector store returned a different number of IDs and scores."
            )

        results: list[RetrievalResult] = []
        for chunk_id, score in zip(ids, row_scores):
            try:
                chunk = self.chunk_lookup[chunk_id]
            except KeyError as exc:
                raise KeyError(
                    f"Vector store returned unknown chunk_id: {chunk_id}"
                ) from exc

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                )
            )

        return results
