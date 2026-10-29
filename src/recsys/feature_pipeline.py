"""Task 02: leakage-safe features, library ItemKNN, content, and hybrid evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import implicit
import numpy as np

from recsys.features import (
    assert_features_fit_on_train,
    build_train_features,
    load_item_genres,
)
from recsys.library_models import (
    GenreProfileRecommender,
    ImplicitItemKNNRecommender,
    hybrid_recommend,
)
from recsys.metrics import recall_at_k
from recsys.models import ItemKNNRecommender
from recsys.pipeline import load_events, make_ground_truth
from recsys.split import ITEM, USER, chronological_user_split
from recsys.submission import read_submission_users, write_submission


REFERENCE_RECALL = 0.055817
VALIDATION_PROTOCOL = "chronological 90/10 per user; all fitting and features use train only"


def run(args: argparse.Namespace) -> dict[str, object]:
    total_start = time.perf_counter()
    events = load_events(args.data_dir / "events.csv")
    item_genres = load_item_genres(args.data_dir / "item_features.csv")
    sample_users = read_submission_users(args.data_dir / "submission_sample.csv")
    n_users = max(int(events[:, USER].max()) + 1, max(sample_users) + 1)
    n_items = max(int(events[:, ITEM].max()) + 1, len(item_genres))
    train, validation = chronological_user_split(events)
    users = np.array(sorted(np.unique(validation[:, USER]).tolist()), dtype=np.int64)
    ground_truth = make_ground_truth(validation)

    results: dict[str, object] = {
        "task_id": "002",
        "model_family": "feature engineering + library-based collaborative filtering",
        "validation_protocol": VALIDATION_PROTOCOL,
        "library_versions": {"implicit": implicit.__version__},
        "data": {
            "events": int(len(events)),
            "train_events": int(len(train)),
            "validation_events": int(len(validation)),
            "users": int(len(users)),
            "items_in_feature_catalog": int(n_items),
        },
        "features": {
            "fit_source": "train partition only",
            "user_aggregates": [
                "interaction_count",
                "mean_rating",
                "std_rating",
                "positive_count_rating_ge_4",
                "fraction_positive",
                "rating_1..5_distribution",
            ],
            "item_aggregates": [
                "interaction_count",
                "mean_rating",
                "std_rating",
                "positive_count_rating_ge_4",
                "fraction_positive",
                "18 raw genre indicators",
            ],
            "user_genre_profiles": ["rating_weighted_l2", "positive_only_l2"],
            "profile_used_for_ranking": "positive_only_l2",
        },
        "experiments": {},
    }

    start = time.perf_counter()
    features = build_train_features(train, item_genres, n_users=n_users, n_items=n_items)
    assert_features_fit_on_train(features, train, validation)
    results["feature_build"] = {
        "runtime_seconds": time.perf_counter() - start,
        "source_event_count": features.source_event_count,
        "source_fingerprint_sha256": features.source_fingerprint,
        "leakage_check": "passed: fingerprint matches exact train partition",
    }

    start = time.perf_counter()
    custom = ItemKNNRecommender(args.neighbors, args.batch_size)
    custom.fit(train, n_users=n_users, n_items=n_items)
    custom_recommendations = custom.recommend(users, k=10)
    results["experiments"]["custom_itemknn_reference_rerun"] = _experiment(
        recall_at_k(ground_truth, custom_recommendations),
        time.perf_counter() - start,
        {"neighbors": args.neighbors, "feedback": "binary"},
    )

    start = time.perf_counter()
    library_knn = ImplicitItemKNNRecommender(args.neighbors, args.num_threads)
    library_knn.fit(train, n_users=n_users, n_items=n_items)
    knn_candidates = library_knn.recommend_with_scores(users, k=args.candidate_count)
    library_recommendations = {
        user: items[:10].astype(int).tolist()
        for user, (items, _) in knn_candidates.items()
    }
    results["experiments"]["implicit_cosine_itemknn"] = _experiment(
        recall_at_k(ground_truth, library_recommendations),
        time.perf_counter() - start,
        {
            "library": f"implicit=={implicit.__version__}",
            "neighbors": args.neighbors,
            "candidate_count": args.candidate_count,
            "feedback": "binary",
            "num_threads": args.num_threads,
        },
    )

    start = time.perf_counter()
    content = GenreProfileRecommender(profile="positive", batch_size=args.batch_size)
    content.fit(train, features)
    content_recommendations = content.recommend(users, k=10)
    results["experiments"]["genre_content_positive_profile"] = _experiment(
        recall_at_k(ground_truth, content_recommendations),
        time.perf_counter() - start,
        {"similarity": "cosine", "profile": "rating >= 4 interactions"},
    )

    alpha_recommendations: dict[float, dict[int, list[int]]] = {}
    for alpha in args.alphas:
        start = time.perf_counter()
        recommendations = hybrid_recommend(
            knn_candidates, users, features, alpha=alpha, k=10
        )
        alpha_recommendations[alpha] = recommendations
        results["experiments"][f"implicit_itemknn_genre_alpha_{alpha:g}"] = _experiment(
            recall_at_k(ground_truth, recommendations),
            time.perf_counter() - start,
            {
                "formula": "minmax(knn_score) + alpha * minmax(genre_cosine)",
                "alpha": alpha,
                "candidate_count": args.candidate_count,
                "genre_profile": "positive_only_l2",
            },
        )

    new_models = {
        "implicit_cosine_itemknn": library_recommendations,
        "genre_content_positive_profile": content_recommendations,
        **{
            f"implicit_itemknn_genre_alpha_{alpha:g}": recommendations
            for alpha, recommendations in alpha_recommendations.items()
        },
    }
    best_name = max(
        new_models,
        key=lambda name: results["experiments"][name]["recall_at_10"],
    )
    best_recall = results["experiments"][best_name]["recall_at_10"]
    improvement = best_recall - REFERENCE_RECALL
    results["comparison"] = {
        "reference_model": "Task 01 custom cosine ItemKNN",
        "reference_recall_at_10": REFERENCE_RECALL,
        "best_new_model": best_name,
        "best_new_recall_at_10": best_recall,
        "absolute_improvement": improvement,
        "submission_threshold_absolute": args.minimum_improvement,
    }

    if improvement >= args.minimum_improvement:
        submission_start = time.perf_counter()
        full_features = build_train_features(
            events, item_genres, n_users=n_users, n_items=n_items
        )
        full_users = np.asarray(sample_users, dtype=np.int64)
        if best_name == "implicit_cosine_itemknn":
            full_model = ImplicitItemKNNRecommender(args.neighbors, args.num_threads)
            full_model.fit(events, n_users=n_users, n_items=n_items)
            final_recommendations = full_model.recommend(full_users, k=10)
        elif best_name == "genre_content_positive_profile":
            full_model = GenreProfileRecommender(
                profile="positive", batch_size=args.batch_size
            ).fit(events, full_features)
            final_recommendations = full_model.recommend(full_users, k=10)
        else:
            alpha = float(best_name.rsplit("_", 1)[1])
            full_model = ImplicitItemKNNRecommender(args.neighbors, args.num_threads)
            full_model.fit(events, n_users=n_users, n_items=n_items)
            full_candidates = full_model.recommend_with_scores(
                full_users, k=args.candidate_count
            )
            final_recommendations = hybrid_recommend(
                full_candidates, full_users, full_features, alpha=alpha, k=10
            )
        write_submission(args.submission, sample_users, final_recommendations, k=10)
        results["submission"] = {
            "created": True,
            "path": str(args.submission),
            "model": best_name,
            "refit_source": "all events after local model selection",
            "runtime_seconds": time.perf_counter() - submission_start,
            "kaggle_recall_at_10": None,
        }
    else:
        results["submission"] = {
            "created": False,
            "reason": "best new local score did not exceed the reference by the configured material threshold",
            "kaggle_recall_at_10": None,
        }

    results["total_runtime_seconds"] = time.perf_counter() - total_start
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return results


def _experiment(recall: float, runtime: float, hyperparameters: dict[str, object]) -> dict:
    return {
        "recall_at_10": recall,
        "runtime_seconds": runtime,
        "hyperparameters": hyperparameters,
        "validation_protocol": VALIDATION_PROTOCOL,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(".artifacts/002_feature_engineering_library_cf.json"),
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path(".artifacts/submissions/2_submission_library_genre_hybrid.csv"),
    )
    parser.add_argument("--neighbors", type=int, default=200)
    parser.add_argument("--candidate-count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--minimum-improvement", type=float, default=0.0005)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
