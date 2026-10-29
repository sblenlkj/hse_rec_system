"""Candidate union, leakage-safe feature assembly, and reranking helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from recsys.features import FeatureBundle
from recsys.split import ITEM, USER


SOURCE_LIMITS = {"sasrec": 200, "als": 200, "knn": 200, "popularity": 50}


def _pd() -> Any:
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "two-stage dependencies are missing; install with `uv sync --extra ranking`"
        ) from error
    return pd


def scored_candidates_frame(
    candidates: Mapping[int, tuple[np.ndarray, np.ndarray]], source: str
) -> Any:
    """Convert a scored recommender result into one deterministic source table."""
    pd = _pd()
    rows: list[tuple[int, int, float, int]] = []
    for user in sorted(candidates):
        items, scores = candidates[user]
        rows.extend(
            (int(user), int(item), float(score), rank)
            for rank, (item, score) in enumerate(zip(items, scores, strict=True), 1)
        )
    return pd.DataFrame(
        rows,
        columns=["user_id", "item_id", f"{source}_score", f"{source}_rank"],
    )


def ranked_candidates_frame(
    candidates: Mapping[int, Sequence[int]], source: str
) -> Any:
    """Convert a rank-only recommender result into one source table."""
    pd = _pd()
    rows = [
        (int(user), int(item), rank)
        for user in sorted(candidates)
        for rank, item in enumerate(candidates[user], 1)
    ]
    return pd.DataFrame(
        rows, columns=["user_id", "item_id", f"{source}_rank"]
    )


def build_candidate_union(source_frames: Mapping[str, Any]) -> Any:
    """Outer-join sources and apply documented stable missing-value handling.

    Missing scores are set to zero after per-user/source min-max normalization;
    missing ranks are set to ``source_limit + 1``. Source flags make these
    sentinels unambiguous to a learned ranker.
    """
    pd = _pd()
    expected = set(SOURCE_LIMITS)
    if set(source_frames) != expected:
        raise ValueError(f"source frames must be exactly {sorted(expected)}")
    keys = ["user_id", "item_id"]
    table = pd.concat([frame[keys] for frame in source_frames.values()]).drop_duplicates()
    for source in ("sasrec", "als", "knn", "popularity"):
        frame = source_frames[source]
        if bool(frame.duplicated(keys).any()):
            raise ValueError(f"{source} contains duplicate user-item pairs")
        table = table.merge(frame, on=keys, how="left", validate="one_to_one")
        rank_column = f"{source}_rank"
        table[f"from_{source}"] = table[rank_column].notna().astype(np.int8)
        if source != "popularity":
            score_column = f"{source}_score"
            available = table[score_column].notna()
            per_user = table[score_column].where(available).groupby(table["user_id"])
            minimum = per_user.transform("min")
            maximum = per_user.transform("max")
            denominator = maximum - minimum
            normalized = np.divide(
                table[score_column] - minimum,
                denominator,
                out=np.zeros(len(table), dtype=np.float64),
                where=available & denominator.gt(0),
            )
            table[f"{source}_score_norm"] = normalized.astype(np.float32)
            table[score_column] = table[score_column].fillna(0.0).astype(np.float32)
        table[rank_column] = (
            table[rank_column].fillna(SOURCE_LIMITS[source] + 1).astype(np.int16)
        )
    table["n_sources"] = table[
        [f"from_{source}" for source in SOURCE_LIMITS]
    ].sum(axis=1).astype(np.int8)
    return table.sort_values(keys, kind="stable").reset_index(drop=True)


def label_candidates(table: Any, target_events: np.ndarray) -> np.ndarray:
    """Return binary labels without adding target-derived feature columns."""
    pd = _pd()
    target = pd.DataFrame(
        np.unique(target_events[:, [USER, ITEM]], axis=0),
        columns=["user_id", "item_id"],
    )
    positives = target.assign(label=np.int8(1))
    labeled = table[["user_id", "item_id"]].merge(
        positives, on=["user_id", "item_id"], how="left", validate="many_to_one"
    )
    return labeled["label"].fillna(0).to_numpy(dtype=np.int8)


def candidate_recall(
    table: Any,
    ground_truth: Mapping[int, Sequence[int]],
    mask: Any | None = None,
) -> float:
    """Macro recall of a candidate pool over users with non-empty targets."""
    selected = table if mask is None else table[mask]
    found = {
        int(user): set(group["item_id"].astype(int))
        for user, group in selected.groupby("user_id", sort=False)
    }
    values = [
        len(set(items).intersection(found.get(int(user), set()))) / len(set(items))
        for user, items in ground_truth.items()
        if items
    ]
    return float(np.mean(values)) if values else 0.0


def candidate_diagnostics(table: Any, ground_truth: Mapping[int, Sequence[int]]) -> dict[str, Any]:
    """Measure individual sources, incremental unions, and pool sizes."""
    sasrec = table["from_sasrec"].eq(1)
    als = table["from_als"].eq(1)
    knn = table["from_knn"].eq(1)
    popularity = table["from_popularity"].eq(1)
    metrics = {
        "sasrec_recall_at_50": candidate_recall(
            table, ground_truth, sasrec & table["sasrec_rank"].le(50)
        ),
        "sasrec_recall_at_100": candidate_recall(
            table, ground_truth, sasrec & table["sasrec_rank"].le(100)
        ),
        "sasrec_recall_at_200": candidate_recall(table, ground_truth, sasrec),
        "als_recall_at_100": candidate_recall(
            table, ground_truth, als & table["als_rank"].le(100)
        ),
        "als_recall_at_200": candidate_recall(table, ground_truth, als),
        "knn_recall_at_100": candidate_recall(
            table, ground_truth, knn & table["knn_rank"].le(100)
        ),
        "knn_recall_at_200": candidate_recall(table, ground_truth, knn),
        "sasrec_als_union_recall": candidate_recall(table, ground_truth, sasrec | als),
        "sasrec_als_knn_union_recall": candidate_recall(
            table, ground_truth, sasrec | als | knn
        ),
        "all_sources_union_recall": candidate_recall(
            table, ground_truth, sasrec | als | knn | popularity
        ),
    }
    pd = _pd()
    target_pairs = pd.DataFrame(
        [(int(user), int(item)) for user, items in ground_truth.items() for item in set(items)],
        columns=["user_id", "item_id"],
    )
    hit_sources = target_pairs.merge(
        table[
            ["user_id", "item_id", "from_sasrec", "from_als", "from_knn", "from_popularity"]
        ],
        on=["user_id", "item_id"],
        how="left",
        validate="one_to_one",
    ).fillna(0)
    metrics["incremental_relevant_pairs_beyond_sasrec"] = {
        source: int(
            (hit_sources[f"from_{source}"].eq(1) & hit_sources["from_sasrec"].eq(0)).sum()
        )
        for source in ("als", "knn", "popularity")
    }
    pool_sizes = table.groupby("user_id").size()
    metrics["pool_size"] = {
        "min": int(pool_sizes.min()),
        "median": float(pool_sizes.median()),
        "max": int(pool_sizes.max()),
        "rows": int(len(table)),
    }
    return metrics


def load_user_demographics(path: Path, n_users: int) -> tuple[np.ndarray, np.ndarray]:
    """Load item-indexed age bucket and binary gender features."""
    pd = _pd()
    raw = pd.read_csv(path)
    age = np.zeros(n_users, dtype=np.float32)
    gender = np.zeros(n_users, dtype=np.float32)
    users = raw["user_id"].to_numpy(dtype=np.int64)
    age[users] = raw["age"].to_numpy(dtype=np.float32)
    gender[users] = raw["gender"].eq("M").to_numpy(dtype=np.float32)
    return age, gender


def attach_ranking_features(
    table: Any,
    bundle: FeatureBundle,
    age: np.ndarray,
    gender: np.ndarray,
) -> tuple[Any, list[str]]:
    """Attach compact train-only user, item, demographic, and affinity features."""
    output = table.copy()
    users = output["user_id"].to_numpy(dtype=np.int64)
    items = output["item_id"].to_numpy(dtype=np.int64)

    user_columns = {
        "user_interaction_count": 0,
        "user_mean_rating": 1,
        "user_std_rating": 2,
        "user_fraction_positive": 4,
    }
    item_columns = {
        "item_interaction_count": 0,
        "item_mean_rating": 1,
        "item_std_rating": 2,
        "item_fraction_positive": 4,
    }
    for name, column in user_columns.items():
        output[name] = bundle.user_features[users, column]
    output["user_age"] = age[users]
    output["user_gender_m"] = gender[users]
    for name, column in item_columns.items():
        output[name] = bundle.item_features[items, column]
    for genre in range(bundle.item_genres.shape[1]):
        output[f"genre_{genre}"] = bundle.item_genres[items, genre]

    item_norms = np.linalg.norm(bundle.item_genres, axis=1, keepdims=True)
    normalized_items = np.divide(
        bundle.item_genres,
        item_norms,
        out=np.zeros_like(bundle.item_genres),
        where=item_norms != 0,
    )
    output["positive_genre_affinity"] = np.einsum(
        "ij,ij->i", bundle.user_genre_positive[users], normalized_items[items]
    )
    output["rating_weighted_genre_affinity"] = np.einsum(
        "ij,ij->i", bundle.user_genre_rating_weighted[users], normalized_items[items]
    )

    source_features = [
        "sasrec_score", "sasrec_score_norm", "sasrec_rank", "from_sasrec",
        "als_score", "als_score_norm", "als_rank", "from_als",
        "knn_score", "knn_score_norm", "knn_rank", "from_knn",
        "popularity_rank", "from_popularity", "n_sources",
    ]
    feature_names = (
        source_features
        + list(user_columns)
        + ["user_age", "user_gender_m"]
        + list(item_columns)
        + [f"genre_{genre}" for genre in range(bundle.item_genres.shape[1])]
        + ["positive_genre_affinity", "rating_weighted_genre_affinity"]
    )
    return output, feature_names


def recommendations_from_scores(
    table: Any, scores: np.ndarray, k: int = 10
) -> dict[int, list[int]]:
    """Return deterministic top-k per user with item ID as final tie-break."""
    scored = table[["user_id", "item_id"]].copy()
    scored["score"] = np.asarray(scores, dtype=np.float64)
    scored = scored.sort_values(
        ["user_id", "score", "item_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    top = scored.groupby("user_id", sort=False).head(k)
    return {
        int(user): group["item_id"].astype(int).tolist()
        for user, group in top.groupby("user_id", sort=False)
    }
