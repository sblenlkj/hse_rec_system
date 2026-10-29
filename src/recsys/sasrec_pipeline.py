"""Task 04: RePlay SASRec smoke test, validation, candidates, and submission."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any
import zipfile

import numpy as np

from recsys.metrics import recall_at_k
from recsys.pipeline import load_events, make_ground_truth
from recsys.sasrec_data import (
    assert_train_only_sequences,
    build_item_catalog,
    build_user_sequences,
    encode_items,
)
from recsys.split import chronological_user_split
from recsys.submission import read_submission_users, write_submission


ALS_REFERENCE_RECALL = 0.06704557877787688
VALIDATION_PROTOCOL = (
    "chronological 90/10 per user; SASRec encoder, sequences, fitting, "
    "candidate generation, and seen filtering use train only"
)


def _neural_imports() -> dict[str, Any]:
    try:
        import lightning as L
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from lightning.pytorch.loggers import CSVLogger
        import pandas as pd
        import replay
        from replay.data import FeatureHint, FeatureType
        from replay.data.nn import ParquetModule, TensorFeatureInfo, TensorSchema
        from replay.nn.lightning import LightningModule
        from replay.nn.lightning.callback import (
            ComputeMetricsCallback,
            PandasTopItemsCallback,
        )
        from replay.nn.lightning.optimizer import OptimizerFactory
        from replay.nn.lightning.postprocessor import SeenItemsFilter
        from replay.nn.sequential import SasRec
        from replay.nn.transform.template import make_default_sasrec_transforms
        import torch
    except ImportError as error:
        raise RuntimeError(
            "SASRec dependencies are missing. Install with "
            "`uv sync --extra sasrec` or `pip install -e '.[sasrec]'`."
        ) from error
    return locals()


def _select_device(torch: Any, requested: str) -> tuple[str, str]:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    accelerator = "gpu" if requested == "cuda" else requested
    return requested, accelerator


def _schema(imports: dict[str, Any], item_count: int, embedding_dim: int) -> Any:
    return imports["TensorSchema"](
        [
            imports["TensorFeatureInfo"](
                name="item_id",
                is_seq=True,
                padding_value=item_count,
                cardinality=item_count,
                embedding_dim=embedding_dim,
                feature_type=imports["FeatureType"].CATEGORICAL,
                feature_hint=imports["FeatureHint"].ITEM_ID,
            )
        ]
    )


def _make_model(imports: dict[str, Any], schema: Any, args: argparse.Namespace) -> Any:
    sasrec = imports["SasRec"].from_params(
        schema=schema,
        embedding_dim=args.embedding_dim,
        max_sequence_length=args.max_sequence_length,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    )
    optimizer = imports["OptimizerFactory"](
        optimizer="adam", learning_rate=args.learning_rate
    )
    return imports["LightningModule"](sasrec, optimizer_factory=optimizer)


def _frames(
    imports: dict[str, Any],
    sequences: dict[int, list[int]],
    ground_truth_events: np.ndarray | None = None,
    catalog: np.ndarray | None = None,
) -> tuple[Any, Any | None]:
    pd = imports["pd"]
    users = sorted(sequences)
    train_frame = pd.DataFrame(
        {"user_id": users, "item_id": [sequences[user] for user in users]}
    )
    if ground_truth_events is None:
        return train_frame, None
    if catalog is None:
        raise ValueError("catalog is required for validation ground truth")

    raw_ground_truth = make_ground_truth(ground_truth_events)
    encoded_ground_truth: list[list[int]] = []
    for user in users:
        raw_items = np.asarray(raw_ground_truth.get(user, []), dtype=np.int64)
        known = encode_items(raw_items, catalog)
        encoded_ground_truth.append([int(item) for item in known if item >= 0])
    validation_frame = pd.DataFrame(
        {
            "user_id": users,
            "item_id": [sequences[user] for user in users],
            "seen_ids": [sequences[user] for user in users],
            "ground_truth": encoded_ground_truth,
        }
    )
    return train_frame, validation_frame


def _write_parquet_frames(
    train_frame: Any,
    output_dir: Path,
    prefix: str,
    validation_frame: Any | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"train": output_dir / f"{prefix}_train.parquet"}
    train_frame.to_parquet(paths["train"], index=False)
    if validation_frame is not None:
        paths["validation"] = output_dir / f"{prefix}_validation.parquet"
        paths["predict"] = output_dir / f"{prefix}_predict.parquet"
        validation_frame.to_parquet(paths["validation"], index=False)
        validation_frame.drop(columns=["ground_truth"]).to_parquet(
            paths["predict"], index=False
        )
    else:
        paths["predict"] = output_dir / f"{prefix}_predict.parquet"
        prediction_frame = train_frame.assign(seen_ids=train_frame["item_id"])
        prediction_frame.to_parquet(paths["predict"], index=False)
    return paths


def _metadata(
    item_count: int,
    max_sequence_length: int,
    max_seen_length: int,
    max_ground_truth_length: int | None = None,
) -> dict[str, dict[str, dict[str, int]]]:
    metadata: dict[str, dict[str, dict[str, int]]] = {
        "train": {
            "item_id": {"shape": max_sequence_length + 1, "padding": item_count}
        },
        "predict": {
            "user_id": {},
            "item_id": {"shape": max_sequence_length, "padding": item_count},
            "seen_ids": {"shape": max_seen_length, "padding": -1},
        },
    }
    if max_ground_truth_length is not None:
        metadata["validate"] = {
            "item_id": {"shape": max_sequence_length, "padding": item_count},
            "seen_ids": {"shape": max_seen_length, "padding": -1},
            "ground_truth": {"shape": max_ground_truth_length, "padding": -1},
        }
    return metadata


def _data_module(
    imports: dict[str, Any],
    paths: dict[str, Path],
    metadata: dict[str, Any],
    schema: Any,
    batch_size: int,
    torch_device: Any,
    stages: tuple[str, ...],
) -> Any:
    config = {stage: {"device": torch_device} for stage in stages}
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "metadata": {stage: metadata[stage] for stage in stages},
        "transforms": imports["make_default_sasrec_transforms"](schema),
        "config": config,
    }
    if "train" in stages:
        kwargs["train_path"] = str(paths["train"])
    if "validate" in stages:
        kwargs["validate_path"] = str(paths["validation"])
    if "predict" in stages:
        kwargs["predict_path"] = str(paths["predict"])
    return imports["ParquetModule"](**kwargs)


def _train_validation_model(
    imports: dict[str, Any],
    paths: dict[str, Path],
    metadata: dict[str, Any],
    schema: Any,
    args: argparse.Namespace,
    accelerator: str,
    torch_device: Any,
    checkpoint_dir: Path,
    smoke: bool,
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    item_count = schema.item_id_features.item().cardinality
    seen_filter = imports["SeenItemsFilter"](item_count=item_count)
    metrics_callback = imports["ComputeMetricsCallback"](
        metrics=["recall"],
        ks=[10],
        item_count=item_count,
        postprocessors=[seen_filter],
        verbose=False,
    )
    checkpoint_callback = imports["ModelCheckpoint"](
        dirpath=checkpoint_dir,
        filename=("4_sasrec_smoke" if smoke else "4_sasrec_validation_best"),
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor="recall@10",
        mode="max",
    )
    callbacks: list[Any] = [checkpoint_callback, metrics_callback]
    if not smoke:
        callbacks.append(
            imports["EarlyStopping"](
                monitor="recall@10", mode="max", patience=args.early_stopping_patience
            )
        )
    logger = imports["CSVLogger"](
        save_dir=str(args.artifacts_dir / "logs"),
        name="sasrec_smoke" if smoke else "sasrec_validation",
    )
    module = _data_module(
        imports,
        paths,
        metadata,
        schema,
        args.batch_size,
        torch_device,
        ("train", "validate"),
    )
    trainer = imports["L"].Trainer(
        max_epochs=1 if smoke else args.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        num_sanity_val_steps=0,
        limit_train_batches=args.smoke_batches if smoke else 1.0,
        limit_val_batches=args.smoke_validation_batches if smoke else 1.0,
        enable_progress_bar=True,
    )
    model = _make_model(imports, schema, args)
    start = time.perf_counter()
    trainer.fit(model, datamodule=module)
    runtime = time.perf_counter() - start
    metrics_history = metrics_callback.get_metrics()
    best_epoch = max(
        metrics_history,
        key=lambda epoch: metrics_history[epoch]["recall@10"],
    )
    return {
        "checkpoint": checkpoint_callback.best_model_path,
        "runtime_seconds": runtime,
        "best_epoch": int(best_epoch),
        "callback_recall_at_10": float(
            metrics_history[best_epoch]["recall@10"]
        ),
        "metrics_history": metrics_history,
    }


def _load_checkpoint_model(
    imports: dict[str, Any], schema: Any, args: argparse.Namespace, checkpoint: str
) -> Any:
    sasrec = imports["SasRec"].from_params(
        schema=schema,
        embedding_dim=args.embedding_dim,
        max_sequence_length=args.max_sequence_length,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    )
    model = imports["LightningModule"].load_from_checkpoint(
        checkpoint, model=sasrec, weights_only=False
    )
    model.eval()
    return model


def _predict_candidates(
    imports: dict[str, Any],
    model: Any,
    paths: dict[str, Path],
    metadata: dict[str, Any],
    schema: Any,
    catalog: np.ndarray,
    args: argparse.Namespace,
    accelerator: str,
    torch_device: Any,
    top_k: int,
    limit_batches: int | float = 1.0,
) -> Any:
    callback = imports["PandasTopItemsCallback"](
        top_k=top_k,
        query_column="user_id",
        item_column="item_id",
        rating_column="sasrec_score",
        postprocessors=[imports["SeenItemsFilter"](item_count=len(catalog))],
    )
    module = _data_module(
        imports,
        paths,
        metadata,
        schema,
        args.batch_size,
        torch_device,
        ("predict",),
    )
    trainer = imports["L"].Trainer(
        accelerator=accelerator,
        devices=1,
        callbacks=[callback],
        logger=False,
        inference_mode=True,
        limit_predict_batches=limit_batches,
        enable_progress_bar=True,
    )
    trainer.predict(model, datamodule=module, return_predictions=False)
    candidates = callback.get_result().copy()
    candidates["user_id"] = candidates["user_id"].astype("int64")
    encoded = candidates["item_id"].astype("int64").to_numpy()
    candidates["item_id"] = catalog[encoded]
    candidates["sasrec_score"] = candidates["sasrec_score"].astype("float64")
    candidates = candidates.sort_values(
        ["user_id", "sasrec_score", "item_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    candidates["sasrec_rank"] = (
        candidates.groupby("user_id", sort=False).cumcount() + 1
    )
    return candidates.reset_index(drop=True)


def _recommendation_dict(candidates: Any, k: int) -> dict[int, list[int]]:
    top = candidates[candidates["sasrec_rank"] <= k]
    return {
        int(user): group["item_id"].astype(int).tolist()
        for user, group in top.groupby("user_id", sort=False)
    }


def _assert_candidates(
    candidates: Any,
    sequences: dict[int, list[int]],
    catalog: np.ndarray,
    top_k: int,
) -> None:
    counts = candidates.groupby("user_id").size()
    if len(counts) == 0 or not bool((counts == top_k).all()):
        raise AssertionError(f"each predicted user must have exactly {top_k} candidates")
    if bool(candidates.duplicated(["user_id", "item_id"]).any()):
        raise AssertionError("SASRec candidates contain duplicate user-item pairs")
    raw_seen = {
        user: {int(catalog[item]) for item in encoded_items}
        for user, encoded_items in sequences.items()
    }
    for user, group in candidates.groupby("user_id", sort=False):
        if raw_seen[int(user)].intersection(group["item_id"].astype(int)):
            raise AssertionError(f"seen item recommended for user {int(user)}")


def _fit_full_model(
    imports: dict[str, Any],
    paths: dict[str, Path],
    metadata: dict[str, Any],
    schema: Any,
    args: argparse.Namespace,
    accelerator: str,
    torch_device: Any,
    epochs: int,
    checkpoint_path: Path,
) -> tuple[Any, float]:
    module = _data_module(
        imports,
        paths,
        metadata,
        schema,
        args.batch_size,
        torch_device,
        ("train",),
    )
    model = _make_model(imports, schema, args)
    trainer = imports["L"].Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        logger=imports["CSVLogger"](
            save_dir=str(args.artifacts_dir / "logs"), name="sasrec_full"
        ),
        deterministic=True,
        limit_val_batches=0,
        enable_progress_bar=True,
    )
    start = time.perf_counter()
    trainer.fit(model, datamodule=module)
    runtime = time.perf_counter() - start
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(checkpoint_path)
    model.eval()
    return model, runtime


def _hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_sequence_length": args.max_sequence_length,
        "embedding_dim": args.embedding_dim,
        "num_blocks": args.num_blocks,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "max_epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "seed": args.seed,
        "feedback": "ordered item ids; ratings are not model inputs",
    }


def _prepare_runtime(args: argparse.Namespace) -> tuple[dict[str, Any], str, str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("ARROW_NUM_THREADS", str(args.arrow_threads))
    os.environ.setdefault("ARROW_IO_THREADS", str(args.arrow_threads))
    imports = _neural_imports()
    torch = imports["torch"]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    imports["L"].seed_everything(args.seed, workers=True)
    device_name, accelerator = _select_device(torch, args.device)
    torch_device = torch.device(device_name)
    return imports, device_name, accelerator, torch_device


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    imports, device_name, accelerator, torch_device = _prepare_runtime(args)
    events = load_events(args.data_dir / "events.csv")
    train, validation = chronological_user_split(events)
    catalog = build_item_catalog(train)
    sequences = build_user_sequences(train, catalog)
    assert_train_only_sequences(sequences, train, validation, catalog)
    train_frame, validation_frame = _frames(
        imports, sequences, validation, catalog
    )
    paths = _write_parquet_frames(
        train_frame,
        args.transformer_data_dir,
        "smoke_chronological",
        validation_frame,
    )
    schema = _schema(imports, len(catalog), args.embedding_dim)
    metadata = _metadata(
        len(catalog),
        args.max_sequence_length,
        max(map(len, sequences.values())),
        max(max(map(len, validation_frame["ground_truth"])), 1),
    )
    outcome = _train_validation_model(
        imports,
        paths,
        metadata,
        schema,
        args,
        accelerator,
        torch_device,
        Path("/tmp/hse_recsys_sasrec_smoke"),
        smoke=True,
    )
    model = _load_checkpoint_model(
        imports, schema, args, outcome["checkpoint"]
    )
    smoke_candidates = _predict_candidates(
        imports,
        model,
        paths,
        metadata,
        schema,
        catalog,
        args,
        accelerator,
        torch_device,
        top_k=10,
        limit_batches=1,
    )
    smoke_users = smoke_candidates["user_id"].nunique()
    train_batches = math.ceil(len(sequences) / args.batch_size)
    batches_run = min(args.smoke_batches, train_batches)
    estimated_epoch_seconds = outcome["runtime_seconds"] * train_batches / batches_run
    result = {
        "task_id": "004-smoke",
        "library_versions": {
            "replay": imports["replay"].__version__,
            "torch": imports["torch"].__version__,
            "lightning": imports["L"].__version__,
        },
        "device": device_name,
        "mps_built": bool(imports["torch"].backends.mps.is_built()),
        "mps_available": bool(imports["torch"].backends.mps.is_available()),
        "train_events": int(len(train)),
        "validation_events": int(len(validation)),
        "train_users": int(len(sequences)),
        "item_vocabulary_train_only": int(len(catalog)),
        "smoke_train_batches": batches_run,
        "smoke_validation_batches": args.smoke_validation_batches,
        "runtime_seconds": outcome["runtime_seconds"],
        "estimated_epoch_seconds": estimated_epoch_seconds,
        "estimated_full_validation_training_seconds": (
            estimated_epoch_seconds * args.epochs
        ),
        "technically_valid_recommendation_users": int(smoke_users),
        "checkpoint": outcome["checkpoint"],
        "hyperparameters": _hyperparameters(args),
        "decision_threshold_seconds": args.local_training_threshold_minutes * 60,
    }
    result["recommended_execution"] = (
        "local"
        if result["estimated_full_validation_training_seconds"]
        <= result["decision_threshold_seconds"]
        else "colab"
    )
    args.smoke_results.parent.mkdir(parents=True, exist_ok=True)
    args.smoke_results.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    total_start = time.perf_counter()
    imports, device_name, accelerator, torch_device = _prepare_runtime(args)
    events = load_events(args.data_dir / "events.csv")
    sample_users = read_submission_users(args.data_dir / "submission_sample.csv")
    train, validation = chronological_user_split(events)

    catalog = build_item_catalog(train)
    sequences = build_user_sequences(train, catalog)
    assert_train_only_sequences(sequences, train, validation, catalog)
    train_frame, validation_frame = _frames(
        imports, sequences, validation, catalog
    )
    paths = _write_parquet_frames(
        train_frame,
        args.transformer_data_dir,
        "chronological",
        validation_frame,
    )
    schema = _schema(imports, len(catalog), args.embedding_dim)
    metadata = _metadata(
        len(catalog),
        args.max_sequence_length,
        max(map(len, sequences.values())),
        max(max(map(len, validation_frame["ground_truth"])), 1),
    )
    validation_outcome = _train_validation_model(
        imports,
        paths,
        metadata,
        schema,
        args,
        accelerator,
        torch_device,
        args.models_dir,
        smoke=False,
    )
    validation_model = _load_checkpoint_model(
        imports, schema, args, validation_outcome["checkpoint"]
    )
    validation_candidates = _predict_candidates(
        imports,
        validation_model,
        paths,
        metadata,
        schema,
        catalog,
        args,
        accelerator,
        torch_device,
        top_k=args.candidate_count,
    )
    _assert_candidates(
        validation_candidates, sequences, catalog, args.candidate_count
    )
    validation_recommendations = _recommendation_dict(validation_candidates, 10)
    local_recall = recall_at_k(
        make_ground_truth(validation), validation_recommendations, k=10
    )
    args.validation_candidates.parent.mkdir(parents=True, exist_ok=True)
    validation_candidates.to_csv(args.validation_candidates, index=False)

    full_catalog = build_item_catalog(events)
    full_sequences = build_user_sequences(events, full_catalog)
    full_train_frame, _ = _frames(imports, full_sequences)
    full_paths = _write_parquet_frames(
        full_train_frame, args.transformer_data_dir, "full"
    )
    full_schema = _schema(imports, len(full_catalog), args.embedding_dim)
    full_metadata = _metadata(
        len(full_catalog),
        args.max_sequence_length,
        max(map(len, full_sequences.values())),
    )
    full_epochs = validation_outcome["best_epoch"] + 1
    full_model, full_training_runtime = _fit_full_model(
        imports,
        full_paths,
        full_metadata,
        full_schema,
        args,
        accelerator,
        torch_device,
        full_epochs,
        args.full_checkpoint,
    )
    full_candidates = _predict_candidates(
        imports,
        full_model,
        full_paths,
        full_metadata,
        full_schema,
        full_catalog,
        args,
        accelerator,
        torch_device,
        top_k=args.candidate_count,
    )
    _assert_candidates(
        full_candidates, full_sequences, full_catalog, args.candidate_count
    )
    args.full_candidates.parent.mkdir(parents=True, exist_ok=True)
    full_candidates.to_csv(args.full_candidates, index=False)
    full_recommendations = _recommendation_dict(full_candidates, 10)
    write_submission(args.submission, sample_users, full_recommendations, k=10)

    smoke_result = None
    if args.smoke_results.exists():
        smoke_result = json.loads(args.smoke_results.read_text())
    result: dict[str, Any] = {
        "task_id": "004",
        "model": "RePlay SasRec",
        "library_versions": {
            "replay": imports["replay"].__version__,
            "torch": imports["torch"].__version__,
            "lightning": imports["L"].__version__,
        },
        "validation_protocol": VALIDATION_PROTOCOL,
        "device": device_name,
        "data": {
            "events": int(len(events)),
            "train_events": int(len(train)),
            "validation_events": int(len(validation)),
            "users": int(len(sequences)),
            "train_item_vocabulary": int(len(catalog)),
            "full_item_vocabulary": int(len(full_catalog)),
        },
        "hyperparameters": _hyperparameters(args),
        "local_smoke": smoke_result,
        "validation": {
            "recall_at_10": local_recall,
            "replay_known_item_recall_at_10": validation_outcome[
                "callback_recall_at_10"
            ],
            "best_epoch_zero_based": validation_outcome["best_epoch"],
            "runtime_seconds": validation_outcome["runtime_seconds"],
            "checkpoint": validation_outcome["checkpoint"],
            "candidate_path": str(args.validation_candidates),
            "candidate_count_per_user": args.candidate_count,
        },
        "comparison": {
            "als_reference_recall_at_10": ALS_REFERENCE_RECALL,
            "difference_vs_als": local_recall - ALS_REFERENCE_RECALL,
        },
        "full_refit": {
            "epochs": full_epochs,
            "runtime_seconds": full_training_runtime,
            "checkpoint": str(args.full_checkpoint),
            "candidate_path": str(args.full_candidates),
        },
        "submission": {
            "path": str(args.submission),
            "users": len(sample_users),
            "model": "SASRec refit on full events",
            "seen_filtering": "all full-history items excluded",
            "kaggle_recall_at_10": None,
        },
        "artifact_bundle": str(args.bundle),
        "total_runtime_seconds": time.perf_counter() - total_start,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, indent=2) + "\n")
    _write_bundle(args)
    print(json.dumps(result, indent=2))
    return result


def _write_bundle(args: argparse.Namespace) -> None:
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    files = [
        args.full_checkpoint,
        args.results,
        args.validation_candidates,
        args.full_candidates,
        args.submission,
    ]
    with zipfile.ZipFile(args.bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path(".artifacts"))
    parser.add_argument(
        "--transformer-data-dir",
        type=Path,
        default=Path(".artifacts/transformer_data"),
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path(".artifacts/models")
    )
    parser.add_argument(
        "--full-checkpoint",
        type=Path,
        default=Path(".artifacts/models/4_sasrec_checkpoint.ckpt"),
    )
    parser.add_argument(
        "--smoke-results",
        type=Path,
        default=Path(".artifacts/004_sasrec_smoke.json"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(".artifacts/004_sasrec_validation.json"),
    )
    parser.add_argument(
        "--validation-candidates",
        type=Path,
        default=Path(".artifacts/candidates/4_sasrec_validation_top200.csv"),
    )
    parser.add_argument(
        "--full-candidates",
        type=Path,
        default=Path(".artifacts/candidates/4_sasrec_full_top200.csv"),
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path(".artifacts/submissions/4_submission_sasrec.csv"),
    )
    parser.add_argument(
        "--bundle", type=Path, default=Path(".artifacts/4_sasrec_artifacts.zip")
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-sequence-length", type=int, default=50)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--candidate-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrow-threads", type=int, default=4)
    parser.add_argument("--smoke-batches", type=int, default=8)
    parser.add_argument("--smoke-validation-batches", type=int, default=4)
    parser.add_argument("--local-training-threshold-minutes", type=float, default=45.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
