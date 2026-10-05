from __future__ import annotations

import pytest

from ipo_analyzer.evaluation.retrieval_evaluator import (
    evaluate_rankings,
    first_relevant_rank,
)


def test_first_relevant_rank() -> None:
    assert first_relevant_rank(["a", "b", "c"], {"b"}) == 2
    assert first_relevant_rank(["a", "b", "c"], {"x"}) is None


def test_metrics_with_multiple_relevant_chunks() -> None:
    cases = [
        {
            "id": "case1",
            "query": "q1",
            "relevant_chunk_ids": ["b", "d"],
        },
        {
            "id": "case2",
            "query": "q2",
            "relevant_chunk_ids": ["z"],
        },
    ]
    rankings = {
        "case1": ["a", "b", "c", "d"],
        "case2": ["x", "y", "z"],
    }

    result = evaluate_rankings(cases, rankings, top_k=4)

    assert result["recall_at"][1] == pytest.approx(0.0)
    assert result["recall_at"][3] == pytest.approx(1.0)
    assert result["recall_at"][5] == pytest.approx(1.0)
    assert result["recall_at"][10] == pytest.approx(1.0)
    assert result["mrr"] == pytest.approx((1 / 2 + 1 / 3) / 2)


def test_metrics_use_first_relevant_rank_for_mrr() -> None:
    cases = [
        {
            "id": "case1",
            "query": "q1",
            "relevant_chunk_ids": ["c", "e"],
        }
    ]
    rankings = {"case1": ["a", "b", "c", "d", "e"]}

    result = evaluate_rankings(cases, rankings, top_k=5)

    assert result["mrr"] == pytest.approx(1 / 3)
