"""One-command validation, model training, and submission pipeline."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from recsys.baselines import PopularityRecommender
from recsys.metrics import recall_at_k
from recsys.models import ItemKNNRecommender
from recsys.split import ITEM, USER, chronological_user_split
from recsys.submission import read_submission_users, write_submission


def load_events(path: Path) -> np.ndarray:
    with path.open() as source:
        header = source.readline().strip()
    if header != "user_id,item_id,rating,timestamp":
        raise ValueError(f"unexpected events header in {path}: {header}")
    events = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"unexpected events schema in {path}")
    sanity_check_events(events)
    return events


def sanity_check_events(events: np.ndarray) -> None:
    """Check assumptions required by the split and implicit-feedback models."""
    if len(events) == 0:
        raise ValueError("events must not be empty")
    if events[:, 2].min() < 1 or events[:, 2].max() > 5:
        raise ValueError("ratings must be in the inclusive range 1..5")
    if np.any(events[:, [USER, ITEM, 3]] < 0):
        raise ValueError("user_id, item_id, and timestamp must be non-negative")

    item_base = int(events[:, ITEM].max()) + 1
    pairs = events[:, USER] * item_base + events[:, ITEM]
    if len(np.unique(pairs)) != len(events):
        raise ValueError("duplicate user-item interactions are not supported")
    timestamp_base = int(events[:, 3].max()) + 1
    user_timestamps = events[:, USER] * timestamp_base + events[:, 3]
    if len(np.unique(user_timestamps)) != len(events):
        raise ValueError("timestamps must be unique within each user history")


def make_ground_truth(validation: np.ndarray) -> dict[int, list[int]]:
    ground_truth: defaultdict[int, list[int]] = defaultdict(list)
    for user, item in validation[:, [USER, ITEM]]:
        ground_truth[int(user)].append(int(item))
    return dict(ground_truth)


def run(args: argparse.Namespace) -> dict[str, object]:
    total_start = time.perf_counter()
    events = load_events(args.data_dir / "events.csv")
    sample_users = read_submission_users(args.data_dir / "submission_sample.csv")
    n_users = max(int(events[:, USER].max()) + 1, max(sample_users) + 1)
    n_items = int(events[:, ITEM].max()) + 1
    train, validation = chronological_user_split(events)
    users = np.array(sorted(set(validation[:, USER].tolist())), dtype=np.int64)
    ground_truth = make_ground_truth(validation)

    results: dict[str, object] = {
        "data": {
            "events": int(len(events)),
            "users": int(len(np.unique(events[:, USER]))),
            "items": int(len(np.unique(events[:, ITEM]))),
            "train_events": int(len(train)),
            "validation_events": int(len(validation)),
        },
        "experiments": {},
    }

    for name, model in (
        ("global_popularity", PopularityRecommender()),
        ("positive_popularity_rating_ge_4", PopularityRecommender(minimum_rating=4)),
    ):
        start = time.perf_counter()
        model.fit(train, n_items=n_items)
        recommendations = model.recommend(users, k=10)
        results["experiments"][name] = {
            "recall_at_10": recall_at_k(ground_truth, recommendations),
            "runtime_seconds": time.perf_counter() - start,
        }

    start = time.perf_counter()
    item_knn = ItemKNNRecommender(args.neighbors, args.batch_size)
    item_knn.fit(train, n_users=n_users, n_items=n_items)
    recommendations = item_knn.recommend(users, k=10)
    results["experiments"]["itemknn_cosine"] = {
        "recall_at_10": recall_at_k(ground_truth, recommendations),
        "runtime_seconds": time.perf_counter() - start,
        "neighbors": args.neighbors,
    }

    # Validation is no longer used below: the final model is deliberately refit
    # on all available events before producing competition recommendations.
    start = time.perf_counter()
    final_model = ItemKNNRecommender(args.neighbors, args.batch_size)
    final_model.fit(events, n_users=n_users, n_items=n_items)
    final_recommendations = final_model.recommend(np.asarray(sample_users), k=10)
    write_submission(args.output, sample_users, final_recommendations, k=10)
    results["submission"] = {
        "path": str(args.output),
        "users": len(sample_users),
        "runtime_seconds": time.perf_counter() - start,
        "model": "itemknn_cosine_refit_on_all_events",
        "seen_filtering": "all observed events are excluded per user",
    }
    results["total_runtime_seconds"] = time.perf_counter() - total_start

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path(".artifacts/submission_itemknn.csv")
    )
    parser.add_argument(
        "--results", type=Path, default=Path(".artifacts/validation_results.json")
    )
    parser.add_argument("--neighbors", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
