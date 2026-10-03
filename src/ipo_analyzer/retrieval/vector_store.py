from __future__ import annotations

from pathlib import Path
from typing import Sequence

import faiss
import numpy as np


class FAISSVectorStore:
    """
    Stores embedding vectors using FAISS and performs
    similarity-based vector search.
    """

    def __init__(self, dimension: int) -> None:
        """
        Create an empty FAISS vector store.

        Args:
            dimension: Dimensionality of each embedding vector.
        """

        self.dimension = dimension

        # Inner-product similarity is used because the
        # embedding vectors are normalized.
        self.index = faiss.IndexFlatIP(dimension)

        # Maps FAISS vector positions to application-level IDs.
        self.ids: list[str] = []

    def add(
        self,
        embeddings: np.ndarray,
        ids: Sequence[str],
    ) -> None:
        """
        Add embedding vectors to the FAISS index.

        Args:
            embeddings: Array of shape
                (number_of_vectors, embedding_dimension).
            ids: IDs corresponding to each embedding vector.
        """

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2-dimensional array."
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"but received {vectors.shape[1]}."
            )

        if len(ids) != len(vectors):
            raise ValueError(
                "The number of IDs must match the number "
                "of embedding vectors."
            )

        self.index.add(vectors)
        self.ids.extend(ids)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> tuple[np.ndarray, list[list[str]]]:
        """
        Search the FAISS index for the most similar vectors.

        Args:
            query_embedding: Query vector or batch of query vectors.
            top_k: Number of nearest results to return.

        Returns:
            A tuple containing:
            - similarity scores
            - corresponding vector IDs
        """

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.ndim != 2:
            raise ValueError(
                "Query embedding must be a 1D or 2D array."
            )

        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query dimension {self.dimension}, "
                f"but received {query.shape[1]}."
            )

        if self.index.ntotal == 0:
            raise ValueError(
                "Cannot search an empty FAISS index."
            )

        top_k = min(top_k, self.index.ntotal)

        scores, positions = self.index.search(query, top_k)

        result_ids = [
            [
                self.ids[position]
                for position in row
                if position >= 0
            ]
            for row in positions
        ]

        return scores, result_ids

    @property
    def size(self) -> int:
        """Return the number of vectors stored in FAISS."""

        return self.index.ntotal

    def save(self, directory: str | Path) -> None:
        """
        Save the FAISS index and vector ID mapping to disk.
        """

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(directory / "index.faiss"),
        )

        np.save(
            directory / "ids.npy",
            np.array(self.ids, dtype=object),
            allow_pickle=True,
        )

    @classmethod
    def load(cls, directory: str | Path) -> "FAISSVectorStore":
        """
        Load a previously saved FAISS vector store from disk.
        """

        directory = Path(directory)

        index_path = directory / "index.faiss"
        ids_path = directory / "ids.npy"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not ids_path.exists():
            raise FileNotFoundError(
                f"Vector ID mapping not found: {ids_path}"
            )

        index = faiss.read_index(str(index_path))

        ids = np.load(
            ids_path,
            allow_pickle=True,
        ).tolist()

        store = cls(index.d)
        store.index = index
        store.ids = ids

        return store