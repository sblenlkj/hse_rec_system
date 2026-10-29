"""Evaluation metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def recall_at_k(
    ground_truth: Mapping[int, Sequence[int]],
    recommendations: Mapping[int, Sequence[int]],
    k: int = 10,
) -> float:
    """Return macro Recall@k over users with non-empty ground truth.

    Users without ground truth are excluded. Duplicate recommendations are an
    error rather than silently receiving extra credit.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    recalls: list[float] = []
    for user, relevant_items in ground_truth.items():
        relevant = set(relevant_items)
        if not relevant:
            continue
        predicted = list(recommendations.get(user, ()))[:k]
        if len(predicted) != len(set(predicted)):
            raise ValueError(f"duplicate recommendations for user {user}")
        recalls.append(len(relevant.intersection(predicted)) / len(relevant))
    return sum(recalls) / len(recalls) if recalls else 0.0
