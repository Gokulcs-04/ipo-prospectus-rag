from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ipo_analyzer.chunking.semantic_chunker import chunk_document
from ipo_analyzer.embeddings.embedding_model import EmbeddingModel
from ipo_analyzer.retrieval.retriever import Retriever
from ipo_analyzer.retrieval.vector_store import FAISSVectorStore
from ipo_analyzer.schemas.document import Document


def load_gold(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def recall_hit(retrieved_ids: list[str], relevant_ids: set[str]) -> bool:
    return bool(set(retrieved_ids) & relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate single-document retrieval on a gold-standard JSON file."
    )
    parser.add_argument("--document-json", required=True, type=Path)
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument(
        "--gold-standard",
        type=Path,
        default=Path("eval/gold_standard/retrieval_ellenbarrie.json"),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--save-report", type=Path)
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    document = Document.model_validate(
        json.loads(args.document_json.read_text(encoding="utf-8"))
    )
    gold = load_gold(args.gold_standard)

    if gold.get("document_id") != document.document_id:
        raise ValueError(
            f"Gold-standard document_id {gold.get('document_id')!r} does not match "
            f"document document_id {document.document_id!r}."
        )

    chunks = chunk_document(document)
    chunk_ids = {chunk.chunk_id for chunk in chunks}

    # Detect a stale/incompatible gold file before running retrieval.
    unknown_gold_ids = sorted(
        {
            chunk_id
            for case in gold["cases"]
            for chunk_id in case["relevant_chunk_ids"]
            if chunk_id not in chunk_ids
        }
    )
    if unknown_gold_ids:
        raise ValueError(
            "Gold-standard contains chunk IDs that are not present in the current "
            f"chunking output: {unknown_gold_ids}"
        )

    embedding_model = EmbeddingModel()
    vector_store = FAISSVectorStore.load(args.index_dir)

    if vector_store.size != len(chunks):
        raise ValueError(
            f"Vector/chunk count mismatch: FAISS has {vector_store.size} vectors, "
            f"but current chunking produced {len(chunks)} chunks."
        )

    retriever = Retriever(embedding_model, vector_store, chunks)

    metric_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    reciprocal_ranks: list[float] = []
    case_reports: list[dict[str, Any]] = []

    print(f"Document: {document.company_name}")
    print(f"Pages: {document.page_count}")
    print(f"Chunks: {len(chunks)}")
    print(f"Vectors: {vector_store.size}")
    print(f"Embedding dimension: {embedding_model.dimension}")
    print(f"Gold cases: {len(gold['cases'])}")

    for case in gold["cases"]:
        relevant_ids = set(case["relevant_chunk_ids"])
        results = retriever.retrieve(case["query"], top_k=args.top_k)
        retrieved_ids = [result.chunk.chunk_id for result in results]

        case_recall: dict[str, bool] = {}
        for k in (1, 3, 5, 10):
            hit = recall_hit(retrieved_ids[:k], relevant_ids)
            case_recall[f"recall@{k}"] = hit
            if hit:
                metric_hits[k] += 1

        rr = reciprocal_rank(retrieved_ids, relevant_ids)
        reciprocal_ranks.append(rr)

        first_hit_rank = next(
            (rank for rank, chunk_id in enumerate(retrieved_ids, start=1) if chunk_id in relevant_ids),
            None,
        )

        case_reports.append(
            {
                "id": case["id"],
                "query": case["query"],
                "relevant_chunk_ids": sorted(relevant_ids),
                "retrieved_chunk_ids": retrieved_ids,
                "first_relevant_rank": first_hit_rank,
                **case_recall,
                "reciprocal_rank": rr,
            }
        )

        print("\n" + "=" * 88)
        print(f"QUERY: {case['query']}")
        print(f"Gold chunks: {sorted(relevant_ids)}")
        print(f"First relevant rank: {first_hit_rank}")
        for rank, result in enumerate(results, start=1):
            chunk = result.chunk
            marker = " <-- GOLD" if chunk.chunk_id in relevant_ids else ""
            print(
                f"{rank}. score={result.score:.4f} | "
                f"chunk_id={chunk.chunk_id} | "
                f"pages={chunk.pages} | "
                f"section={chunk.section} | "
                f"subsection={chunk.subsection}{marker}"
            )

    n = len(gold["cases"])
    metrics = {
        f"Recall@{k}": metric_hits[k] / n if n else 0.0
        for k in (1, 3, 5, 10)
    }
    metrics["MRR"] = sum(reciprocal_ranks) / n if n else 0.0

    print("\n" + "=" * 88)
    print("RETRIEVAL BASELINE SUMMARY")
    print("=" * 88)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f} ({value * 100:.2f}%)")

    report = {
        "document_id": document.document_id,
        "document_name": document.company_name,
        "top_k_requested": args.top_k,
        "num_cases": n,
        "metrics": metrics,
        "cases": case_reports,
    }

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report: {args.save_report}")


if __name__ == "__main__":
    main()
