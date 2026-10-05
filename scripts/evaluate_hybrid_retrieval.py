from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ipo_analyzer.evaluation.retrieval_evaluator import CUTOFFS


def load_gold(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("Gold standard must contain a JSON object.")

    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Gold standard must contain a non-empty 'cases' list.")

    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each gold-standard case must be a JSON object.")
        if not isinstance(case.get("id"), str) or not case["id"].strip():
            raise ValueError("Each case must have a non-empty string 'id'.")
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError(f"Case {case.get('id')!r} must have a non-empty 'query'.")
        ids = case.get("relevant_chunk_ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
            raise ValueError(
                f"Case {case.get('id')!r} must have a non-empty list of string 'relevant_chunk_ids'."
            )

    return data

def print_method_results(
    method_name: str,
    cases: list[dict[str, Any]],
    rankings: dict[str, list[tuple[str, float]]],
    *,
    top_k: int,
) -> dict[str, Any]:
    print("\n" + "=" * 88)
    print(method_name)
    print("=" * 88)

    from ipo_analyzer.evaluation.retrieval_evaluator import evaluate_rankings, first_relevant_rank

    id_rankings: dict[str, list[str]] = {}

    for case in cases:
        case_id = case["id"]
        relevant = set(case["relevant_chunk_ids"])
        results = rankings[case_id][:top_k]
        ids = [chunk_id for chunk_id, _ in results]
        id_rankings[case_id] = ids
        rank = first_relevant_rank(ids, relevant)

        print(f"\nQUERY: {case['query']}")
        print(f"Gold chunks: {case['relevant_chunk_ids']}")
        print(f"First relevant rank: {rank if rank is not None else 'None'}")

        for position, (chunk_id, score) in enumerate(results, start=1):
            marker = " <-- GOLD" if chunk_id in relevant else ""
            print(f"{position}. score={score:.6f} | chunk_id={chunk_id}{marker}")

    evaluation = evaluate_rankings(cases, id_rankings, top_k=top_k)

    print("\n" + "-" * 88)
    print(f"{method_name} SUMMARY")
    print("-" * 88)
    for cutoff in CUTOFFS:
        value = evaluation["recall_at"][cutoff]
        print(f"Recall@{cutoff}: {value:.4f} ({value * 100:.2f}%)")
    print(f"MRR: {evaluation['mrr']:.4f} ({evaluation['mrr'] * 100:.2f}%)")

    return evaluation


def print_comparison(dense: dict[str, Any], hybrid: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print("DENSE vs HYBRID COMPARISON")
    print("=" * 88)
    print(
        f"{'Metric':<14} {'Dense':>12} {'Hybrid':>12} {'Change':>12}"
    )
    print("-" * 54)

    for cutoff in CUTOFFS:
        dense_value = dense["recall_at"][cutoff]
        hybrid_value = hybrid["recall_at"][cutoff]
        change = hybrid_value - dense_value
        print(
            f"Recall@{cutoff:<7} {dense_value * 100:>10.2f}% "
            f"{hybrid_value * 100:>10.2f}% {change * 100:>+10.2f} pp"
        )

    dense_mrr = dense["mrr"]
    hybrid_mrr = hybrid["mrr"]
    print(
        f"{'MRR':<14} {dense_mrr * 100:>10.2f}% "
        f"{hybrid_mrr * 100:>10.2f}% {(hybrid_mrr - dense_mrr) * 100:>+10.2f} pp"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate dense FAISS retrieval and hybrid dense+BM25 RRF retrieval "
            "against the same gold-standard questions."
        )
    )
    parser.add_argument("--document-json", required=True, type=Path)
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument("--gold-standard", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for saving the evaluation results.",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")
    if args.candidate_k < 1:
        parser.error("--candidate-k must be at least 1.")
    if args.rrf_k < 1:
        parser.error("--rrf-k must be greater than 0.")

    gold = load_gold(args.gold_standard)
    cases = gold["cases"]

    # Imports are kept inside main so the metric helpers can be unit-tested
    # without loading the embedding model.
    from ipo_analyzer.chunking.semantic_chunker import chunk_document
    from ipo_analyzer.embeddings.embedding_model import EmbeddingModel
    from ipo_analyzer.retrieval.hybrid_retriever import HybridRetriever
    from ipo_analyzer.retrieval.retriever import Retriever
    from ipo_analyzer.retrieval.vector_store import FAISSVectorStore
    from ipo_analyzer.schemas.document import Document

    data = json.loads(args.document_json.read_text(encoding="utf-8"))
    document = Document.model_validate(data)
    chunks = chunk_document(document)

    embedding_model = EmbeddingModel()
    vector_store = FAISSVectorStore.load(args.index_dir)

    dense_retriever = Retriever(
        embedding_model,
        vector_store,
        chunks,
    )
    hybrid_retriever = HybridRetriever(
        embedding_model,
        vector_store,
        chunks,
        rrf_k=args.rrf_k,
        candidate_k=args.candidate_k,
    )

    print(f"Document: {document.company_name}")
    print(f"Pages: {document.page_count}")
    print(f"Chunks: {len(chunks)}")
    print(f"Dense vectors: {vector_store.size}")
    print(f"Embedding dimension: {embedding_model.dimension}")
    print(f"Lexical chunks: {hybrid_retriever.lexical_store.size}")
    print(f"Gold cases: {len(cases)}")
    print(f"Top-k: {args.top_k}")
    print(f"Hybrid candidate-k: {args.candidate_k}")
    print(f"RRF k: {args.rrf_k}")

    dense_rankings: dict[str, list[tuple[str, float]]] = {}
    hybrid_rankings: dict[str, list[tuple[str, float]]] = {}

    for case in cases:
        query = case["query"]
        dense_rankings[case["id"]] = [
            (result.chunk.chunk_id, result.score)
            for result in dense_retriever.retrieve(query, top_k=args.top_k)
        ]
        hybrid_rankings[case["id"]] = [
            (result.chunk.chunk_id, result.score)
            for result in hybrid_retriever.retrieve(query, top_k=args.top_k)
        ]

    dense_eval = print_method_results(
        "DENSE BASELINE",
        cases,
        dense_rankings,
        top_k=args.top_k,
    )
    hybrid_eval = print_method_results(
        "HYBRID DENSE + BM25 RRF",
        cases,
        hybrid_rankings,
        top_k=args.top_k,
    )
    print_comparison(dense_eval, hybrid_eval)

    output = {
        "document_id": document.document_id,
        "document_name": document.company_name,
        "evaluation_type": "dense_vs_hybrid_retrieval",
        "gold_standard": str(args.gold_standard),
        "parameters": {
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "rrf_k": args.rrf_k,
        },
        "dataset": {
            "pages": document.page_count,
            "chunks": len(chunks),
            "dense_vectors": vector_store.size,
            "embedding_dimension": embedding_model.dimension,
            "lexical_chunks": hybrid_retriever.lexical_store.size,
            "gold_cases": len(cases),
        },
        "dense": dense_eval,
        "hybrid": hybrid_eval,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved evaluation results to: {args.output}")


if __name__ == "__main__":
    main()
