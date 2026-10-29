"""Task 03: ALS on chronological and diagnostic masked validation splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import implicit
import numpy as np

from recsys.als_models import ImplicitALSRecommender
from recsys.baselines import PopularityRecommender
from recsys.metrics import recall_at_k
from recsys.models import ItemKNNRecommender
from recsys.pipeline import load_events, make_ground_truth
from recsys.split import ITEM, USER, chronological_user_split, masked_user_split
from recsys.submission import read_submission_users, write_submission


TASK_01_ITEMKNN_RECALL = 0.05581652878565533
TASK_02_BEST_RECALL = 0.05731585703373961


def _evaluate_model(
    model: object,
    train: np.ndarray,
    validation: np.ndarray,
    n_users: int,
    n_items: int,
    protocol_description: str,
) -> dict[str, object]:
    users = np.unique(validation[:, USER]).astype(np.int64, copy=False)
    ground_truth = make_ground_truth(validation)
    start = time.perf_counter()
    model.fit(train, n_users=n_users, n_items=n_items)
    recommendations = model.recommend(users, k=10)
    return {
        "recall_at_10": recall_at_k(ground_truth, recommendations),
        "runtime_seconds": time.perf_counter() - start,
        "validation_protocol": protocol_description,
        "train_events": int(len(train)),
        "validation_events": int(len(validation)),
        "validation_users": int(len(users)),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    total_start = time.perf_counter()
    events = load_events(args.data_dir / "events.csv")
    sample_users = read_submission_users(args.data_dir / "submission_sample.csv")
    n_users = max(int(events[:, USER].max()) + 1, max(sample_users) + 1)
    n_items = int(events[:, ITEM].max()) + 1
    protocols = {
        "chronological": (
            "chronological 90/10 per user; train-only fitting; "
            "seen filtering by train"
        ),
        "masked": (
            f"deterministic masked 10% per user (seed={args.seed}); "
            "train-only fitting; seen filtering by train"
        ),
    }
    splits = {
        "chronological": chronological_user_split(events),
        "masked": masked_user_split(events, seed=args.seed),
    }

    results: dict[str, object] = {
        "task_id": "003",
        "model_family": "implicit ALS + validation protocol diagnostic",
        "library_versions": {"implicit": implicit.__version__},
        "data": {
            "events": int(len(events)),
            "users": int(len(np.unique(events[:, USER]))),
            "items": int(len(np.unique(events[:, ITEM]))),
        },
        "protocols": protocols,
        "experiments": {
            "popularity": {"hyperparameters": {"feedback": "interaction count"}},
            "itemknn_cosine": {
                "hyperparameters": {"neighbors": args.neighbors, "feedback": "binary"}
            },
            "als_binary": {
                "hyperparameters": _als_hyperparameters(args, "binary")
            },
            "als_rating_weighted": {
                "hyperparameters": _als_hyperparameters(args, "rating")
            },
        },
    }

    for protocol, (train, validation) in splits.items():
        models = {
            "popularity": PopularityRecommender(),
            "itemknn_cosine": ItemKNNRecommender(args.neighbors, args.batch_size),
            "als_binary": _make_als(args, "binary"),
            "als_rating_weighted": _make_als(args, "rating"),
        }
        for name, model in models.items():
            results["experiments"][name][protocol] = _evaluate_model(
                model, train, validation, n_users, n_items, protocols[protocol]
            )

    for experiment in results["experiments"].values():
        experiment["chronological_recall_at_10"] = experiment["chronological"][
            "recall_at_10"
        ]
        experiment["masked_recall_at_10"] = experiment["masked"]["recall_at_10"]

    chronological_ranking = sorted(
        results["experiments"],
        key=lambda name: results["experiments"][name]["chronological"]["recall_at_10"],
        reverse=True,
    )
    masked_ranking = sorted(
        results["experiments"],
        key=lambda name: results["experiments"][name]["masked"]["recall_at_10"],
        reverse=True,
    )
    als_names = ("als_binary", "als_rating_weighted")
    best_als = max(
        als_names,
        key=lambda name: results["experiments"][name]["chronological"]["recall_at_10"],
    )
    best_als_recall = results["experiments"][best_als]["chronological"]["recall_at_10"]
    results["comparison"] = {
        "chronological_ranking": chronological_ranking,
        "masked_ranking": masked_ranking,
        "rankings_agree": chronological_ranking == masked_ranking,
        "best_als": best_als,
        "best_als_chronological_recall_at_10": best_als_recall,
        "task_01_itemknn_recall_at_10": TASK_01_ITEMKNN_RECALL,
        "difference_vs_task_01_itemknn": best_als_recall - TASK_01_ITEMKNN_RECALL,
        "task_02_best_hybrid_recall_at_10": TASK_02_BEST_RECALL,
        "difference_vs_task_02_best": best_als_recall - TASK_02_BEST_RECALL,
    }

    submission_start = time.perf_counter()
    final_model = _make_als(
        args, "binary" if best_als == "als_binary" else "rating"
    ).fit(events, n_users=n_users, n_items=n_items)
    final_recommendations = final_model.recommend(
        np.asarray(sample_users, dtype=np.int64), k=10
    )
    write_submission(args.submission, sample_users, final_recommendations, k=10)
    results["submission"] = {
        "created": True,
        "path": str(args.submission),
        "model": best_als,
        "refit_source": "all events after local comparison",
        "runtime_seconds": time.perf_counter() - submission_start,
        "diagnostic_note": (
            "candidate created even if ALS is below the current local best; "
            "Kaggle upload remains manual"
        ),
        "kaggle_recall_at_10": None,
    }
    results["total_runtime_seconds"] = time.perf_counter() - total_start

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return results


def _als_hyperparameters(args: argparse.Namespace, weighting: str) -> dict[str, object]:
    return {
        "library": f"implicit=={implicit.__version__}",
        "factors": args.factors,
        "regularization": args.regularization,
        "iterations": args.iterations,
        "weighting": weighting,
        "confidence": "1.0 per interaction" if weighting == "binary" else "raw rating 1..5",
        "num_threads": args.num_threads,
        "random_state": args.seed,
    }


def _make_als(args: argparse.Namespace, weighting: str) -> ImplicitALSRecommender:
    return ImplicitALSRecommender(
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        weighting=weighting,
        num_threads=args.num_threads,
        random_state=args.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--results", type=Path, default=Path(".artifacts/003_als_validation_diagnostic.json")
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path(".artifacts/submissions/3_submission_als.csv"),
    )
    parser.add_argument("--neighbors", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
