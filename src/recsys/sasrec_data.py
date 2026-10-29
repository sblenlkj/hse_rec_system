"""Leakage-safe sequence preparation helpers for SASRec experiments."""

from __future__ import annotations

import numpy as np

from recsys.split import ITEM, TIMESTAMP, USER


def build_item_catalog(train_events: np.ndarray) -> np.ndarray:
    """Return the sorted train-only item vocabulary used by the encoder."""
    return np.unique(train_events[:, ITEM]).astype(np.int64, copy=False)


def encode_items(items: np.ndarray, catalog: np.ndarray) -> np.ndarray:
    """Encode raw item ids with a train-fitted catalog; unknown ids become -1."""
    raw_items = np.asarray(items, dtype=np.int64)
    positions = np.searchsorted(catalog, raw_items)
    encoded = np.full(raw_items.shape, -1, dtype=np.int64)
    within = positions < len(catalog)
    matched = np.zeros(raw_items.shape, dtype=bool)
    matched[within] = catalog[positions[within]] == raw_items[within]
    encoded[matched] = positions[matched]
    return encoded


def build_user_sequences(
    events: np.ndarray, catalog: np.ndarray
) -> dict[int, list[int]]:
    """Build timestamp-ordered encoded item sequences from the supplied events."""
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must have four columns")
    order = np.lexsort((events[:, TIMESTAMP], events[:, USER]))
    ordered = events[order]
    encoded_items = encode_items(ordered[:, ITEM], catalog)
    if np.any(encoded_items < 0):
        raise ValueError("sequence source contains items outside the fitted catalog")

    sequences: dict[int, list[int]] = {}
    for raw_user, encoded_item in zip(
        ordered[:, USER], encoded_items, strict=True
    ):
        sequences.setdefault(int(raw_user), []).append(int(encoded_item))
    return sequences


def assert_train_only_sequences(
    sequences: dict[int, list[int]],
    train_events: np.ndarray,
    validation_events: np.ndarray,
    catalog: np.ndarray,
) -> None:
    """Assert that sequences exactly encode train and contain no validation rows."""
    expected = build_user_sequences(train_events, catalog)
    if sequences != expected:
        raise AssertionError("SASRec sequences do not exactly match the train partition")

    item_base = int(max(train_events[:, ITEM].max(), validation_events[:, ITEM].max())) + 1
    train_pairs = train_events[:, USER].astype(np.int64) * item_base + train_events[:, ITEM]
    validation_pairs = (
        validation_events[:, USER].astype(np.int64) * item_base
        + validation_events[:, ITEM]
    )
    if np.intersect1d(train_pairs, validation_pairs).size:
        raise AssertionError("train and validation user-item pairs overlap")

    if sum(map(len, sequences.values())) != len(train_events):
        raise AssertionError("sequence event count differs from train event count")
