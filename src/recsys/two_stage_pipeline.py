"""Task 05: leakage-safe 80/10/10 candidate union and CatBoost reranking."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import implicit
import numpy as np

from recsys.als_models import ImplicitALSRecommender
from recsys.baselines import PopularityRecommender
from recsys.features import (
    assert_features_fit_on_train,
    build_train_features,
    load_item_genres,
)
from recsys.library_models import ImplicitItemKNNRecommender
from recsys.metrics import recall_at_k
from recsys.pipeline import load_events, make_ground_truth
from recsys.sasrec_data import build_item_catalog, build_user_sequences
from recsys.sasrec_pipeline import (
    _assert_candidates,
    _fit_full_model,
    _frames,
    _metadata,
    _predict_candidates,
    _prepare_runtime,
    _schema,
    _write_parquet_frames,
)
from recsys.split import ITEM, USER, temporal_ranker_split
from recsys.submission import read_submission_users, write_submission
from recsys.two_stage import (
    attach_ranking_features,
    build_candidate_union,
    candidate_diagnostics,
    label_candidates,
    load_user_demographics,
    ranked_candidates_frame,
    recommendations_from_scores,
    scored_candidates_frame,
)


SASREC_REFERENCE_RECALL = 0.1303846090744553
PROTOCOL = (
    "per-user temporal 80/10/10: generators/features fit on first 80%, "
    "ranker labels from next 10%; frozen ranker evaluated with generators/features "
    "refit on first 90% against final 10%"
)


def _ranking_imports() -> tuple[Any, Any, Any]:
    try:
        import catboost
        from catboost import CatBoostRanker, Pool
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "two-stage dependencies are missing; install with "
            "`uv sync --extra sasrec --extra ranking`"
        ) from error
    return catboost, CatBoostRanker, Pool


def _sasrec_for_history(
    train_events: np.ndarray,
    prefix: str,
    checkpoint: Path,
    args: argparse.Namespace,
    neural: tuple[dict[str, Any], str, str, Any],
) -> tuple[Any, float]:
    imports, _, accelerator, torch_device = neural
    catalog = build_item_catalog(train_events)
    sequences = build_user_sequences(train_events, catalog)
    train_frame, _ = _frames(imports, sequences)
    paths = _write_parquet_frames(
        train_frame, args.transformer_data_dir, prefix
    )
    schema = _schema(imports, len(catalog), args.embedding_dim)
    metadata = _metadata(
        len(catalog), args.max_sequence_length, max(map(len, sequences.values()))
    )
    model, runtime = _fit_full_model(
        imports,
        paths,
        metadata,
        schema,
        args,
        accelerator,
        torch_device,
        args.sasrec_epochs,
        checkpoint,
    )
    candidates = _predict_candidates(
        imports,
        model,
        paths,
        metadata,
        schema,
        catalog,
        args,
        accelerator,
        torch_device,
        top_k=args.sasrec_candidates,
    )
    _assert_candidates(candidates, sequences, catalog, args.sasrec_candidates)
    return candidates, runtime


def _load_or_fit_full_sasrec(
    events: np.ndarray,
    args: argparse.Namespace,
    neural: tuple[dict[str, Any], str, str, Any],
) -> tuple[Any, float, str]:
    import pandas as pandas

    if args.reuse_task04_full_sasrec and args.task04_full_candidates.exists():
        candidates = pandas.read_csv(args.task04_full_candidates)
        catalog = build_item_catalog(events)
        sequences = build_user_sequences(events, catalog)
        _assert_candidates(candidates, sequences, catalog, args.sasrec_candidates)
        return candidates, 0.0, "reused Task 04 fixed-10-epoch full-data candidates"
    candidates, runtime = _sasrec_for_history(
        events, "two_stage_full", args.full_sasrec_checkpoint, args, neural
    )
    return candidates, runtime, "fitted in Task 05"


def _classical_frames(
    train_events: np.ndarray,
    users: np.ndarray,
    n_users: int,
    n_items: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, float]]:
    runtimes: dict[str, float] = {}
    start = time.perf_counter()
    als = ImplicitALSRecommender(
        factors=args.als_factors,
        regularization=args.als_regularization,
        iterations=args.als_iterations,
        weighting="rating",
        num_threads=args.num_threads,
        random_state=args.seed,
    ).fit(train_events, n_users=n_users, n_items=n_items)
    als_candidates = als.recommend_with_scores(users, k=args.als_candidates)
    runtimes["als"] = time.perf_counter() - start

    start = time.perf_counter()
    knn = ImplicitItemKNNRecommender(
        n_neighbors=args.knn_neighbors, num_threads=args.num_threads
    ).fit(train_events, n_users=n_users, n_items=n_items)
    knn_candidates = knn.recommend_with_scores(users, k=args.knn_candidates)
    runtimes["knn"] = time.perf_counter() - start

    start = time.perf_counter()
    popularity = PopularityRecommender().fit(
        train_events, n_users=n_users, n_items=n_items
    )
    popularity_candidates = popularity.recommend(users, k=args.popularity_candidates)
    runtimes["popularity"] = time.perf_counter() - start
    return {
        "als": scored_candidates_frame(als_candidates, "als"),
        "knn": scored_candidates_frame(knn_candidates, "knn"),
        "popularity": ranked_candidates_frame(popularity_candidates, "popularity"),
    }, runtimes


def _build_union(
    sasrec_candidates: Any,
    train_events: np.ndarray,
    users: np.ndarray,
    n_users: int,
    n_items: int,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, float]]:
    classical, runtimes = _classical_frames(
        train_events, users, n_users, n_items, args
    )
    sasrec = sasrec_candidates[
        ["user_id", "item_id", "sasrec_score", "sasrec_rank"]
    ].copy()
    table = build_candidate_union({"sasrec": sasrec, **classical})
    return table, runtimes


def _source_recommendations(table: Any, source: str, k: int = 10) -> dict[int, list[int]]:
    selected = table[table[f"from_{source}"].eq(1)]
    return recommendations_from_scores(
        selected, -selected[f"{source}_rank"].to_numpy(dtype=np.float64), k=k
    )


def _baseline_scores(table: Any) -> dict[str, tuple[Any, np.ndarray]]:
    score_mask = table["from_sasrec"].eq(1) | table["from_als"].eq(1)
    score_table = table[score_mask]
    score_blend = (
        score_table["sasrec_score_norm"].to_numpy(dtype=np.float64)
        + score_table["als_score_norm"].to_numpy(dtype=np.float64)
    )
    rank_mask = score_mask | table["from_knn"].eq(1)
    rank_table = table[rank_mask]
    rank_blend = sum(
        rank_table[f"from_{source}"].to_numpy(dtype=np.float64)
        / (10.0 + rank_table[f"{source}_rank"].to_numpy(dtype=np.float64))
        for source in ("sasrec", "als", "knn")
    )
    return {
        "sasrec_als_normalized_score_blend": (score_table, score_blend),
        "sasrec_als_knn_reciprocal_rank_blend": (rank_table, rank_blend),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    total_start = time.perf_counter()
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    catboost, CatBoostRanker, Pool = _ranking_imports()
    events = load_events(args.data_dir / "events.csv")
    sample_users = read_submission_users(args.data_dir / "submission_sample.csv")
    base_train, ranker_target, final_validation = temporal_ranker_split(events)
    train_90 = np.concatenate([base_train, ranker_target])
    users = np.unique(events[:, USER]).astype(np.int64, copy=False)
    n_users = max(int(events[:, USER].max()) + 1, max(sample_users) + 1)
    item_genres = load_item_genres(args.data_dir / "item_features.csv")
    n_items = max(int(events[:, ITEM].max()) + 1, len(item_genres))
    age, gender = load_user_demographics(
        args.data_dir / "user_features.csv", n_users
    )

    neural = _prepare_runtime(args)
    device_name = neural[1]
    sasrec_base, sasrec_base_runtime = _sasrec_for_history(
        base_train,
        "two_stage_base80",
        args.base_sasrec_checkpoint,
        args,
        neural,
    )
    sasrec_final, sasrec_final_runtime = _sasrec_for_history(
        train_90,
        "two_stage_train90",
        args.validation_sasrec_checkpoint,
        args,
        neural,
    )

    ranker_table, base_classical_runtime = _build_union(
        sasrec_base, base_train, users, n_users, n_items, args
    )
    final_table, final_classical_runtime = _build_union(
        sasrec_final, train_90, users, n_users, n_items, args
    )
    ranker_ground_truth = make_ground_truth(ranker_target)
    final_ground_truth = make_ground_truth(final_validation)
    candidate_results = {
        "task_id": "005-candidates",
        "protocol": PROTOCOL,
        "ranker_training_target": candidate_diagnostics(
            ranker_table, ranker_ground_truth
        ),
        "final_validation": candidate_diagnostics(final_table, final_ground_truth),
    }
    args.candidate_recall.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_recall.write_text(json.dumps(candidate_results, indent=2) + "\n")

    base_features = build_train_features(
        base_train, item_genres, n_users=n_users, n_items=n_items
    )
    assert_features_fit_on_train(
        base_features, base_train, np.concatenate([ranker_target, final_validation])
    )
    ranker_table, feature_names = attach_ranking_features(
        ranker_table, base_features, age, gender
    )
    labels = label_candidates(ranker_table, ranker_target)
    ranker_table["label"] = labels

    final_features = build_train_features(
        train_90, item_genres, n_users=n_users, n_items=n_items
    )
    assert_features_fit_on_train(final_features, train_90, final_validation)
    final_table, final_feature_names = attach_ranking_features(
        final_table, final_features, age, gender
    )
    if feature_names != final_feature_names:
        raise AssertionError("ranker train and validation feature schemas differ")

    args.ranker_candidates.parent.mkdir(parents=True, exist_ok=True)
    ranker_table.to_parquet(args.ranker_candidates, index=False)
    final_table.to_parquet(args.validation_candidates, index=False)

    train_pool = Pool(
        ranker_table[feature_names],
        label=labels,
        group_id=ranker_table["user_id"].to_numpy(dtype=np.int64),
        feature_names=feature_names,
    )
    ranker = CatBoostRanker(
        loss_function="YetiRank",
        iterations=args.ranker_iterations,
        depth=args.ranker_depth,
        learning_rate=args.ranker_learning_rate,
        l2_leaf_reg=args.ranker_l2,
        random_seed=args.seed,
        thread_count=args.ranker_threads,
        allow_writing_files=False,
        verbose=args.ranker_verbose,
    )
    ranker_start = time.perf_counter()
    ranker.fit(train_pool)
    ranker_runtime = time.perf_counter() - ranker_start
    args.ranker_model.parent.mkdir(parents=True, exist_ok=True)
    ranker.save_model(str(args.ranker_model))

    experiments: dict[str, dict[str, float]] = {}
    sasrec_recommendations = _source_recommendations(final_table, "sasrec")
    experiments["sasrec_single_stage"] = {
        "recall_at_10": recall_at_k(final_ground_truth, sasrec_recommendations)
    }
    for name, (baseline_table, scores) in _baseline_scores(final_table).items():
        recommendations = recommendations_from_scores(baseline_table, scores, k=10)
        experiments[name] = {
            "recall_at_10": recall_at_k(final_ground_truth, recommendations)
        }
    final_scores = ranker.predict(final_table[feature_names])
    learned_recommendations = recommendations_from_scores(
        final_table, final_scores, k=10
    )
    experiments["catboost_yetirank"] = {
        "recall_at_10": recall_at_k(final_ground_truth, learned_recommendations)
    }

    full_sasrec, full_sasrec_runtime, full_sasrec_source = _load_or_fit_full_sasrec(
        events, args, neural
    )
    full_table, full_classical_runtime = _build_union(
        full_sasrec,
        events,
        np.asarray(sample_users, dtype=np.int64),
        n_users,
        n_items,
        args,
    )
    full_features = build_train_features(
        events, item_genres, n_users=n_users, n_items=n_items
    )
    full_table, full_feature_names = attach_ranking_features(
        full_table, full_features, age, gender
    )
    if feature_names != full_feature_names:
        raise AssertionError("full inference feature schema differs from ranker training")
    full_scores = ranker.predict(full_table[feature_names])
    final_recommendations = recommendations_from_scores(full_table, full_scores, k=10)
    write_submission(args.submission, sample_users, final_recommendations, k=10)
    args.full_candidates.parent.mkdir(parents=True, exist_ok=True)
    full_table.to_parquet(args.full_candidates, index=False)

    best_name = max(experiments, key=lambda name: experiments[name]["recall_at_10"])
    result = {
        "task_id": "005",
        "model": "candidate union + CatBoostRanker",
        "protocol": PROTOCOL,
        "library_versions": {
            "catboost": catboost.__version__,
            "implicit": implicit.__version__,
            "replay": neural[0]["replay"].__version__,
            "torch": neural[0]["torch"].__version__,
        },
        "device": device_name,
        "data": {
            "events": int(len(events)),
            "base_train": int(len(base_train)),
            "ranker_train_target": int(len(ranker_target)),
            "train_90": int(len(train_90)),
            "final_validation": int(len(final_validation)),
            "users": int(len(users)),
        },
        "candidate_recall_path": str(args.candidate_recall),
        "candidate_diagnostics": candidate_results,
        "ranker": {
            "objective": "YetiRank",
            "iterations": args.ranker_iterations,
            "depth": args.ranker_depth,
            "learning_rate": args.ranker_learning_rate,
            "l2_leaf_reg": args.ranker_l2,
            "seed": args.seed,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "training_rows": int(len(ranker_table)),
            "positive_rows": int(labels.sum()),
            "runtime_seconds": ranker_runtime,
            "model_path": str(args.ranker_model),
        },
        "experiments": experiments,
        "comparison": {
            "task04_sasrec_reference_recall_at_10": SASREC_REFERENCE_RECALL,
            "same_final_holdout_sasrec_recall_at_10": experiments[
                "sasrec_single_stage"
            ]["recall_at_10"],
            "catboost_recall_at_10": experiments["catboost_yetirank"]["recall_at_10"],
            "catboost_difference_vs_task04_reference": experiments[
                "catboost_yetirank"
            ]["recall_at_10"]
            - SASREC_REFERENCE_RECALL,
            "catboost_difference_vs_same_holdout_sasrec": experiments[
                "catboost_yetirank"
            ]["recall_at_10"]
            - experiments["sasrec_single_stage"]["recall_at_10"],
            "best_experiment": best_name,
            "best_recall_at_10": experiments[best_name]["recall_at_10"],
        },
        "runtimes": {
            "sasrec_base80_seconds": sasrec_base_runtime,
            "sasrec_train90_seconds": sasrec_final_runtime,
            "sasrec_full_seconds": full_sasrec_runtime,
            "base_classical_seconds": base_classical_runtime,
            "final_classical_seconds": final_classical_runtime,
            "full_classical_seconds": full_classical_runtime,
            "total_seconds": time.perf_counter() - total_start,
        },
        "artifacts": {
            "ranker_train_candidates": str(args.ranker_candidates),
            "validation_candidates": str(args.validation_candidates),
            "full_candidates": str(args.full_candidates),
            "base_sasrec_checkpoint": str(args.base_sasrec_checkpoint),
            "validation_sasrec_checkpoint": str(args.validation_sasrec_checkpoint),
            "full_sasrec_source": full_sasrec_source,
        },
        "submission": {
            "path": str(args.submission),
            "users": len(sample_users),
            "ranking_model": "frozen CatBoostRanker trained on 80->10 temporal task",
            "candidate_generators": "refit on all observed events",
            "kaggle_recall_at_10": None,
        },
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path(".artifacts"))
    parser.add_argument(
        "--transformer-data-dir",
        type=Path,
        default=Path(".artifacts/transformer_data"),
    )
    parser.add_argument(
        "--candidate-recall", type=Path, default=Path(".artifacts/005_candidate_recall.json")
    )
    parser.add_argument(
        "--results", type=Path, default=Path(".artifacts/005_two_stage_validation.json")
    )
    parser.add_argument(
        "--ranker-model", type=Path, default=Path(".artifacts/models/5_two_stage_ranker.cbm")
    )
    parser.add_argument(
        "--ranker-candidates",
        type=Path,
        default=Path(".artifacts/candidates/5_ranker_train_candidates.parquet"),
    )
    parser.add_argument(
        "--validation-candidates",
        type=Path,
        default=Path(".artifacts/candidates/5_final_validation_candidates.parquet"),
    )
    parser.add_argument(
        "--full-candidates",
        type=Path,
        default=Path(".artifacts/candidates/5_full_candidates.parquet"),
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path(".artifacts/submissions/5_submission_two_stage.csv"),
    )
    parser.add_argument(
        "--base-sasrec-checkpoint",
        type=Path,
        default=Path(".artifacts/models/5_sasrec_base80.ckpt"),
    )
    parser.add_argument(
        "--validation-sasrec-checkpoint",
        type=Path,
        default=Path(".artifacts/models/5_sasrec_train90.ckpt"),
    )
    parser.add_argument(
        "--full-sasrec-checkpoint",
        type=Path,
        default=Path(".artifacts/models/5_sasrec_full.ckpt"),
    )
    parser.add_argument(
        "--task04-full-candidates",
        type=Path,
        default=Path(".artifacts/candidates/4_sasrec_full_top200.csv"),
    )
    parser.add_argument(
        "--reuse-task04-full-sasrec",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-sequence-length", type=int, default=50)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--sasrec-epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrow-threads", type=int, default=4)
    parser.add_argument("--sasrec-candidates", type=int, default=200)
    parser.add_argument("--als-candidates", type=int, default=200)
    parser.add_argument("--knn-candidates", type=int, default=200)
    parser.add_argument("--popularity-candidates", type=int, default=50)
    parser.add_argument("--als-factors", type=int, default=64)
    parser.add_argument("--als-regularization", type=float, default=0.01)
    parser.add_argument("--als-iterations", type=int, default=15)
    parser.add_argument("--knn-neighbors", type=int, default=200)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--ranker-iterations", type=int, default=300)
    parser.add_argument("--ranker-depth", type=int, default=7)
    parser.add_argument("--ranker-learning-rate", type=float, default=0.05)
    parser.add_argument("--ranker-l2", type=float, default=5.0)
    parser.add_argument("--ranker-threads", type=int, default=4)
    parser.add_argument("--ranker-verbose", type=int, default=50)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
