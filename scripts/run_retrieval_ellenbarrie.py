from __future__ import annotations

import argparse
import json
from pathlib import Path

from ipo_analyzer.chunking.semantic_chunker import chunk_document
from ipo_analyzer.embeddings.embedding_model import EmbeddingModel
from ipo_analyzer.retrieval.retriever import Retriever
from ipo_analyzer.retrieval.vector_store import FAISSVectorStore
from ipo_analyzer.schemas.document import Document


QUERIES = [
    "What was the company's revenue from operations in Fiscal 2025?",
    "Who are the promoters of the company?",
    "Who is the Chief Financial Officer of the company?",
    "What were the company's total assets as at March 31, 2025?",
    "What are the objects of the Offer?",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Retriever against the Ellenbarrie RHP using an existing FAISS index."
    )
    parser.add_argument("--document-json", required=True, type=Path)
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    data = json.loads(args.document_json.read_text(encoding="utf-8"))
    document = Document.model_validate(data)
    chunks = chunk_document(document)

    embedding_model = EmbeddingModel()
    vector_store = FAISSVectorStore.load(args.index_dir)
    retriever = Retriever(embedding_model, vector_store, chunks)

    print(f"Document: {document.company_name}")
    print(f"Pages: {document.page_count}")
    print(f"Chunks: {len(chunks)}")
    print(f"Vector count: {vector_store.size}")
    print(f"Embedding dimension: {embedding_model.dimension}")

    for query in QUERIES:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        for rank, result in enumerate(retriever.retrieve(query, top_k=args.top_k), start=1):
            chunk = result.chunk
            print(
                f"{rank}. score={result.score:.4f} | "
                f"chunk_id={chunk.chunk_id} | "
                f"pages={chunk.pages} | "
                f"section={chunk.section} | "
                f"subsection={chunk.subsection}"
            )
            print(chunk.content[:500].replace("\n", " "))


if __name__ == "__main__":
    main()
