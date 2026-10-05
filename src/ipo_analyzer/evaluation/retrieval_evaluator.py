from __future__ import annotations

from typing import Any

CUTOFFS = (1, 3, 5, 10)


def first_relevant_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> int | None:
    """Return the 1-indexed rank of the first relevant result, or None."""
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return rank
    return None


def evaluate_rankings(
    cases: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Calculate Recall@1/3/5/10 and MRR from ranked chunk IDs."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not cases:
        raise ValueError("cases must not be empty.")

    case_results: list[dict[str, Any]] = []
    reciprocal_sum = 0.0

    for case in cases:
        case_id = case["id"]
        relevant_ids = set(case["relevant_chunk_ids"])
        retrieved_ids = rankings[case_id][:top_k]
        rank = first_relevant_rank(retrieved_ids, relevant_ids)

        if rank is not None:
            reciprocal_sum += 1.0 / rank

        case_results.append(
            {
                "id": case_id,
                "query": case["query"],
                "relevant_chunk_ids": case["relevant_chunk_ids"],
                "retrieved_chunk_ids": retrieved_ids,
                "first_relevant_rank": rank,
            }
        )

    n_cases = len(cases)
    recall_at: dict[int, float] = {}
    for cutoff in CUTOFFS:
        hits = 0
        for case in cases:
            retrieved_ids = rankings[case["id"]][:top_k]
            rank = first_relevant_rank(
                retrieved_ids,
                set(case["relevant_chunk_ids"]),
            )
            if rank is not None and rank <= cutoff:
                hits += 1
        recall_at[cutoff] = hits / n_cases

    return {
        "recall_at": recall_at,
        "mrr": reciprocal_sum / n_cases,
        "cases": case_results,
    }
