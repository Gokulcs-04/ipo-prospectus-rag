from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np

from ipo_analyzer.schemas.chunk import DocumentChunk


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Kept intentionally small. The goal is to remove high-frequency function words
# without introducing an NLP dependency or removing domain terminology.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
    "did", "do", "does", "for", "from", "has", "have", "how", "in", "is",
    "it", "may", "of", "on", "or", "the", "this", "that", "these", "those",
    "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "would",
}


def tokenize(text: str) -> list[str]:
    """
    Convert text into lexical retrieval tokens.

    The tokenizer keeps unigrams and adjacent-token bigrams. Bigrams are useful
    for exact prospectus phrases such as ``fresh issue`` and ``total assets``
    without hard-coding any document-specific terminology.
    """
    if not text:
        return []

    words = [
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if token not in _STOPWORDS
    ]

    bigrams = [f"{left}__{right}" for left, right in zip(words, words[1:])]
    return words + bigrams


class BM25LexicalStore:
    """
    Lightweight BM25 lexical index over DocumentChunk content.

    The index is intentionally dependency-free so the hybrid retriever can be
    added without changing the current requirements.txt or FAISS interface.
    """

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if chunks is None:
            raise ValueError("chunks must not be None.")
        if k1 <= 0:
            raise ValueError("k1 must be greater than 0.")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self.k1 = float(k1)
        self.b = float(b)
        self.chunks = list(chunks)

        seen_ids: set[str] = set()
        for chunk in self.chunks:
            if not isinstance(chunk, DocumentChunk):
                raise TypeError("chunks must contain DocumentChunk objects.")
            if chunk.chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id: {chunk.chunk_id}")
            seen_ids.add(chunk.chunk_id)

        self.ids = [chunk.chunk_id for chunk in self.chunks]
        self._documents: list[list[str]] = []
        self._term_frequencies: list[Counter[str]] = []
        self._postings: dict[str, list[int]] = defaultdict(list)
        self._document_frequencies: Counter[str] = Counter()
        self._document_lengths: list[int] = []

        for doc_idx, chunk in enumerate(self.chunks):
            # Section metadata is searchable as well, but DocumentChunk itself
            # remains unchanged.
            searchable_text = " ".join(
                part
                for part in (chunk.section, chunk.subsection, chunk.content)
                if part
            )
            tokens = tokenize(searchable_text)
            frequencies = Counter(tokens)

            self._documents.append(tokens)
            self._term_frequencies.append(frequencies)
            self._document_lengths.append(len(tokens))

            for term in frequencies:
                self._document_frequencies[term] += 1
                self._postings[term].append(doc_idx)

        self._document_count = len(self.chunks)
        total_length = sum(self._document_lengths)
        self._average_document_length = (
            total_length / self._document_count
            if self._document_count
            else 0.0
        )

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""
        return self._document_count

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[np.ndarray, list[list[str]]]:
        """
        Search the lexical index and return BM25 scores plus chunk IDs.

        The return shape mirrors the current vector-store search contract:
        scores are shape ``(1, k)`` and IDs are ``[[...]]`` for one query.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not query.strip():
            raise ValueError("query must not be empty.")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not self._document_count:
            raise ValueError("Cannot search an empty lexical index.")

        query_terms = tokenize(query)
        if not query_terms:
            return np.empty((1, 0), dtype=np.float32), [[]]

        scores = np.zeros(self._document_count, dtype=np.float32)
        average_length = self._average_document_length or 1.0

        # Score only documents containing at least one query term.
        for term in query_terms:
            document_frequency = self._document_frequencies.get(term, 0)
            if document_frequency == 0:
                continue

            # Standard BM25+ style positive IDF form.
            idf = math.log(
                1.0
                + (self._document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )

            for doc_idx in self._postings[term]:
                term_frequency = self._term_frequencies[doc_idx][term]
                document_length = self._document_lengths[doc_idx]
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * document_length / average_length
                )
                scores[doc_idx] += (
                    idf
                    * term_frequency
                    * (self.k1 + 1.0)
                    / denominator
                )

        candidate_indices = np.flatnonzero(scores > 0.0).tolist()
        candidate_indices.sort(key=lambda i: (-float(scores[i]), i))
        candidate_indices = candidate_indices[: min(top_k, self._document_count)]

        result_scores = np.asarray(
            [[float(scores[i]) for i in candidate_indices]],
            dtype=np.float32,
        )
        result_ids = [[self.ids[i] for i in candidate_indices]]
        return result_scores, result_ids
