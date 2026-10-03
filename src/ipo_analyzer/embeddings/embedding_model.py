from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Converts text into dense semantic embeddings using
    a pretrained Sentence Transformer model.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """
        Convert a sequence of texts into embedding vectors.

        Returns:
            A NumPy array of shape:
            (number_of_texts, embedding_dimension)
        """

        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """
        Convert a single text into one embedding vector.
        """

        embedding = self.encode([text])

        return embedding[0]

    @property
    def dimension(self) -> int:
        """
        Return the dimensionality of the embedding vectors.
        """

        return self.model.get_embedding_dimension()