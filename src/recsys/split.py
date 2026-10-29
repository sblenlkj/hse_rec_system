"""Leakage-safe chronological and diagnostic validation splitting."""

from __future__ import annotations

import numpy as np


USER, ITEM, RATING, TIMESTAMP = range(4)


def chronological_user_split(
    events: np.ndarray, train_fraction: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    """Split every user's ordered history into an early train and late holdout.

    The cut is ``floor(train_fraction * history_length)`` and is clamped so
    users with at least two events occur on both sides. Returned rows are
    sorted by ``(user_id, timestamp)``.
    """
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must have four columns: user, item, rating, timestamp")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if len(events) == 0:
        return events.copy(), events.copy()

    order = np.lexsort((events[:, TIMESTAMP], events[:, USER]))
    ordered = events[order]
    users = ordered[:, USER]
    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    ends = np.r_[starts[1:], len(ordered)]
    validation_mask = np.zeros(len(ordered), dtype=bool)
    for start, end in zip(starts, ends, strict=True):
        length = int(end - start)
        cut = int(np.floor(length * train_fraction))
        if length >= 2:
            cut = min(max(cut, 1), length - 1)
        validation_mask[start + cut : end] = True

    train = ordered[~validation_mask]
    validation = ordered[validation_mask]
    assert_temporal_split(train, validation)
    return train, validation


def temporal_ranker_split(
    events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each ordered user history into deterministic 80/10/10 layers.

    The direct cut points are ``floor(0.8 * n)`` and ``floor(0.9 * n)``.
    Histories in this project contain at least 14 events, so every user is
    represented in all three partitions. The explicit three-way split avoids
    rounding differences that arise from recursively splitting a 90% prefix.
    """
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must have four columns: user, item, rating, timestamp")
    if len(events) == 0:
        return events.copy(), events.copy(), events.copy()

    order = np.lexsort((events[:, TIMESTAMP], events[:, USER]))
    ordered = events[order]
    users = ordered[:, USER]
    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    ends = np.r_[starts[1:], len(ordered)]
    layer = np.zeros(len(ordered), dtype=np.int8)
    for start, end in zip(starts, ends, strict=True):
        length = int(end - start)
        if length < 3:
            raise ValueError("temporal 80/10/10 split requires at least 3 events per user")
        first_cut = min(max(int(np.floor(0.8 * length)), 1), length - 2)
        second_cut = min(
            max(int(np.floor(0.9 * length)), first_cut + 1), length - 1
        )
        layer[start + first_cut : start + second_cut] = 1
        layer[start + second_cut : end] = 2

    base_train = ordered[layer == 0]
    ranker_target = ordered[layer == 1]
    final_validation = ordered[layer == 2]
    assert_ranker_temporal_split(base_train, ranker_target, final_validation)
    return base_train, ranker_target, final_validation


def assert_ranker_temporal_split(
    base_train: np.ndarray,
    ranker_target: np.ndarray,
    final_validation: np.ndarray,
) -> None:
    """Assert ordering and pair disjointness for an 80/10/10 ranker split."""
    if not len(base_train) or not len(ranker_target) or not len(final_validation):
        raise AssertionError("all temporal ranker layers must be non-empty")
    assert_temporal_split(base_train, ranker_target)
    prefix = np.concatenate([base_train, ranker_target])
    assert_temporal_split(prefix, final_validation)

    max_item = int(
        max(
            base_train[:, ITEM].max(),
            ranker_target[:, ITEM].max(),
            final_validation[:, ITEM].max(),
        )
    ) + 1
    encoded = [
        part[:, USER].astype(np.int64) * max_item + part[:, ITEM]
        for part in (base_train, ranker_target, final_validation)
    ]
    if any(
        np.intersect1d(encoded[left], encoded[right]).size
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise AssertionError("a user-item pair occurs in multiple ranker layers")


def masked_user_split(
    events: np.ndarray, train_fraction: float = 0.9, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly mask roughly 10% of each user's distinct item interactions.

    Sampling is deterministic per user, so adding or reordering another user's
    events cannot change an existing user's split. Users need at least two
    distinct items to contribute validation data. Returned rows are sorted by
    ``(user_id, timestamp)`` and user-item pairs are kept wholly on one side.
    """
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must have four columns: user, item, rating, timestamp")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if len(events) == 0:
        return events.copy(), events.copy()

    order = np.lexsort((events[:, TIMESTAMP], events[:, USER]))
    ordered = events[order]
    users = ordered[:, USER]
    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    ends = np.r_[starts[1:], len(ordered)]
    validation_mask = np.zeros(len(ordered), dtype=bool)

    for start, end in zip(starts, ends, strict=True):
        user = int(ordered[start, USER])
        items = np.unique(ordered[start:end, ITEM])
        if len(items) < 2:
            continue
        validation_count = int(np.ceil((1.0 - train_fraction) * len(items)))
        validation_count = min(max(validation_count, 1), len(items) - 1)
        rng = np.random.default_rng(np.random.SeedSequence([seed, user]))
        held_out_items = rng.choice(items, size=validation_count, replace=False)
        validation_mask[start:end] = np.isin(
            ordered[start:end, ITEM], held_out_items, assume_unique=False
        )

    train = ordered[~validation_mask]
    validation = ordered[validation_mask]
    assert_masked_split(train, validation)
    return train, validation


def assert_masked_split(train: np.ndarray, validation: np.ndarray) -> None:
    """Raise when a user-item pair occurs on both sides of a masked split."""
    if len(train) == 0 or len(validation) == 0:
        return
    max_item = int(max(train[:, ITEM].max(), validation[:, ITEM].max())) + 1
    train_pairs = train[:, USER].astype(np.int64) * max_item + train[:, ITEM]
    validation_pairs = (
        validation[:, USER].astype(np.int64) * max_item + validation[:, ITEM]
    )
    if np.intersect1d(train_pairs, validation_pairs, assume_unique=False).size:
        raise AssertionError("a user-item pair occurs in both train and validation")


def assert_temporal_split(train: np.ndarray, validation: np.ndarray) -> None:
    """Raise when temporal ordering or user-item disjointness is violated."""
    if len(train) == 0 or len(validation) == 0:
        return

    n_users = int(max(train[:, USER].max(), validation[:, USER].max())) + 1
    train_max = np.full(n_users, np.iinfo(np.int64).min, dtype=np.int64)
    validation_min = np.full(n_users, np.iinfo(np.int64).max, dtype=np.int64)
    np.maximum.at(train_max, train[:, USER], train[:, TIMESTAMP])
    np.minimum.at(validation_min, validation[:, USER], validation[:, TIMESTAMP])
    common = (train_max != np.iinfo(np.int64).min) & (
        validation_min != np.iinfo(np.int64).max
    )
    leaking_users = np.flatnonzero(common & (train_max >= validation_min))
    if leaking_users.size:
        raise AssertionError(f"temporal leakage for user {int(leaking_users[0])}")

    max_item = int(max(train[:, ITEM].max(), validation[:, ITEM].max())) + 1
    train_pairs = train[:, USER].astype(np.int64) * max_item + train[:, ITEM]
    validation_pairs = validation[:, USER].astype(np.int64) * max_item + validation[:, ITEM]
    overlap = np.intersect1d(train_pairs, validation_pairs, assume_unique=False)
    if overlap.size:
        raise AssertionError("a user-item pair occurs in both train and validation")
