"""Small train-only feature layer for classical recommendation experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from recsys.split import ITEM, RATING, USER


USER_FEATURE_NAMES = (
    "interaction_count",
    "mean_rating",
    "std_rating",
    "positive_count_rating_ge_4",
    "fraction_positive",
    "rating_1_fraction",
    "rating_2_fraction",
    "rating_3_fraction",
    "rating_4_fraction",
    "rating_5_fraction",
)
ITEM_AGGREGATE_NAMES = (
    "interaction_count",
    "mean_rating",
    "std_rating",
    "positive_count_rating_ge_4",
    "fraction_positive",
)


@dataclass(frozen=True)
class FeatureBundle:
    """Feature matrices and provenance for one exact fit event set."""

    user_features: np.ndarray
    item_features: np.ndarray
    item_genres: np.ndarray
    user_genre_rating_weighted: np.ndarray
    user_genre_positive: np.ndarray
    user_feature_names: tuple[str, ...]
    item_feature_names: tuple[str, ...]
    source_event_count: int
    source_fingerprint: str


def load_item_genres(path: Path) -> np.ndarray:
    """Load item IDs plus 18 binary genres into an item-indexed matrix."""
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != 19:
        raise ValueError("item_features must contain item_id and 18 genre columns")
    item_ids = raw[:, 0]
    if len(np.unique(item_ids)) != len(item_ids) or np.any(item_ids < 0):
        raise ValueError("item IDs must be unique and non-negative")
    genres = np.zeros((int(item_ids.max()) + 1, 18), dtype=np.float32)
    genres[item_ids] = raw[:, 1:]
    if not np.all((genres == 0) | (genres == 1)):
        raise ValueError("genre indicators must be binary")
    return genres


def build_train_features(
    train_events: np.ndarray,
    item_genres: np.ndarray,
    n_users: int | None = None,
    n_items: int | None = None,
) -> FeatureBundle:
    """Build all derived features from the supplied train events only."""
    if len(train_events) == 0:
        raise ValueError("train_events must not be empty")
    if n_users is None:
        n_users = int(train_events[:, USER].max()) + 1
    if n_items is None:
        n_items = max(int(train_events[:, ITEM].max()) + 1, len(item_genres))
    if len(item_genres) < n_items:
        raise ValueError("item_genres does not cover all requested item IDs")

    users = train_events[:, USER].astype(np.int64, copy=False)
    items = train_events[:, ITEM].astype(np.int64, copy=False)
    ratings = train_events[:, RATING].astype(np.float64, copy=False)
    positive = ratings >= 4

    user_counts, user_mean, user_std = _group_stats(users, ratings, n_users)
    user_positive = np.bincount(users[positive], minlength=n_users).astype(np.float64)
    user_rating_distribution = np.column_stack(
        [np.bincount(users[ratings == value], minlength=n_users) for value in range(1, 6)]
    ).astype(np.float64)
    user_features = np.column_stack(
        [
            user_counts,
            user_mean,
            user_std,
            user_positive,
            _safe_divide(user_positive, user_counts),
            _safe_divide(user_rating_distribution, user_counts[:, None]),
        ]
    ).astype(np.float32)

    item_counts, item_mean, item_std = _group_stats(items, ratings, n_items)
    item_positive = np.bincount(items[positive], minlength=n_items).astype(np.float64)
    item_aggregates = np.column_stack(
        [
            item_counts,
            item_mean,
            item_std,
            item_positive,
            _safe_divide(item_positive, item_counts),
        ]
    )
    item_features = np.column_stack([item_aggregates, item_genres[:n_items]]).astype(
        np.float32
    )

    event_genres = item_genres[items]
    weighted_profile = np.zeros((n_users, item_genres.shape[1]), dtype=np.float64)
    np.add.at(weighted_profile, users, event_genres * ratings[:, None])
    positive_profile = np.zeros_like(weighted_profile)
    np.add.at(positive_profile, users[positive], event_genres[positive])

    return FeatureBundle(
        user_features=user_features,
        item_features=item_features,
        item_genres=item_genres[:n_items].astype(np.float32, copy=True),
        user_genre_rating_weighted=_row_normalize(weighted_profile),
        user_genre_positive=_row_normalize(positive_profile),
        user_feature_names=USER_FEATURE_NAMES,
        item_feature_names=ITEM_AGGREGATE_NAMES
        + tuple(f"genre_{index}" for index in range(item_genres.shape[1])),
        source_event_count=len(train_events),
        source_fingerprint=events_fingerprint(train_events),
    )


def assert_features_fit_on_train(
    features: FeatureBundle, train_events: np.ndarray, validation_events: np.ndarray
) -> None:
    """Fail loudly unless feature provenance matches the exact train partition."""
    if features.source_event_count != len(train_events):
        raise AssertionError("feature event count does not match train")
    if features.source_fingerprint != events_fingerprint(train_events):
        raise AssertionError("features were not fitted on the exact train events")
    if len(validation_events):
        combined = np.concatenate([train_events, validation_events])
        if features.source_fingerprint == events_fingerprint(combined):
            raise AssertionError("features were fitted on train plus validation")


def events_fingerprint(events: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(events)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode())
    digest.update(str(contiguous.dtype).encode())
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _group_stats(
    groups: np.ndarray, values: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.bincount(groups, minlength=size).astype(np.float64)
    sums = np.bincount(groups, weights=values, minlength=size)
    square_sums = np.bincount(groups, weights=values * values, minlength=size)
    means = _safe_divide(sums, counts)
    variances = np.maximum(_safe_divide(square_sums, counts) - means * means, 0.0)
    return counts, means, np.sqrt(variances)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=output, where=denominator != 0)
    return output


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return _safe_divide(matrix, norms).astype(np.float32)
