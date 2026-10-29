"""Leakage-safe chronological validation splitting."""

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
