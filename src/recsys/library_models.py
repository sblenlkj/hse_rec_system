"""Library-based collaborative filtering and simple genre recommenders."""

from __future__ import annotations

from collections import defaultdict

from implicit.nearest_neighbours import CosineRecommender
import numpy as np
from scipy import sparse

from recsys.features import FeatureBundle
from recsys.split import ITEM, USER


class ImplicitItemKNNRecommender:
    """Thin adapter around implicit's cosine nearest-neighbour recommender."""

    def __init__(self, n_neighbors: int = 200, num_threads: int = 1) -> None:
        self.n_neighbors = n_neighbors
        self.num_threads = num_threads
        self.model = CosineRecommender(K=n_neighbors, num_threads=num_threads)
        self.user_items: sparse.csr_matrix | None = None
        self.fallback: np.ndarray | None = None

    def fit(
        self, train_events: np.ndarray, n_users: int | None = None, n_items: int | None = None
    ) -> "ImplicitItemKNNRecommender":
        if n_users is None:
            n_users = int(train_events[:, USER].max()) + 1
        if n_items is None:
            n_items = int(train_events[:, ITEM].max()) + 1
        rows = train_events[:, USER].astype(np.int32, copy=False)
        columns = train_events[:, ITEM].astype(np.int32, copy=False)
        matrix = sparse.csr_matrix(
            (np.ones(len(train_events), dtype=np.float32), (rows, columns)),
            shape=(n_users, n_items),
        )
        matrix.data[:] = 1.0
        self.user_items = matrix
        counts = np.asarray(matrix.sum(axis=0)).ravel()
        self.fallback = np.lexsort((np.arange(n_items), -counts))
        self.model.fit(matrix, show_progress=False)
        return self

    def recommend_with_scores(
        self, users: np.ndarray, k: int = 10
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        if self.user_items is None or self.fallback is None:
            raise RuntimeError("fit must be called before recommend")
        raw_users = users.astype(np.int64, copy=False)
        item_ids, scores = self.model.recommend(
            raw_users,
            self.user_items[raw_users],
            N=k,
            filter_already_liked_items=True,
        )
        if item_ids.ndim == 1:
            item_ids, scores = item_ids[None, :], scores[None, :]
        output: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for row, raw_user in enumerate(raw_users):
            user = int(raw_user)
            valid = item_ids[row] >= 0
            chosen = item_ids[row][valid].astype(np.int64, copy=False).tolist()
            values = scores[row][valid].astype(np.float64, copy=False).tolist()
            if len(chosen) < k:
                seen = set(self.user_items[user].indices.tolist())
                chosen_set = set(chosen)
                for item in self.fallback:
                    value = int(item)
                    if value not in seen and value not in chosen_set:
                        chosen.append(value)
                        values.append(0.0)
                        chosen_set.add(value)
                    if len(chosen) == k:
                        break
            output[user] = (np.asarray(chosen[:k]), np.asarray(values[:k]))
        return output

    def recommend(self, users: np.ndarray, k: int = 10) -> dict[int, list[int]]:
        scored = self.recommend_with_scores(users, k=k)
        return {user: items.astype(int).tolist() for user, (items, _) in scored.items()}


class GenreProfileRecommender:
    """Cosine content ranking from a train-only user genre profile."""

    def __init__(self, profile: str = "positive", batch_size: int = 256) -> None:
        if profile not in {"positive", "rating_weighted"}:
            raise ValueError("profile must be positive or rating_weighted")
        self.profile = profile
        self.batch_size = batch_size
        self.user_profiles: np.ndarray | None = None
        self.item_profiles: np.ndarray | None = None
        self.seen: dict[int, set[int]] = {}
        self.fallback: np.ndarray | None = None

    def fit(
        self, train_events: np.ndarray, features: FeatureBundle
    ) -> "GenreProfileRecommender":
        self.user_profiles = (
            features.user_genre_positive
            if self.profile == "positive"
            else features.user_genre_rating_weighted
        )
        item_norms = np.linalg.norm(features.item_genres, axis=1, keepdims=True)
        self.item_profiles = np.divide(
            features.item_genres,
            item_norms,
            out=np.zeros_like(features.item_genres),
            where=item_norms != 0,
        )
        seen: defaultdict[int, set[int]] = defaultdict(set)
        for user, item in train_events[:, [USER, ITEM]]:
            seen[int(user)].add(int(item))
        self.seen = dict(seen)
        counts = np.bincount(train_events[:, ITEM], minlength=len(features.item_genres))
        self.fallback = np.lexsort((np.arange(len(counts)), -counts))
        return self

    def recommend(self, users: np.ndarray, k: int = 10) -> dict[int, list[int]]:
        if self.user_profiles is None or self.item_profiles is None or self.fallback is None:
            raise RuntimeError("fit must be called before recommend")
        output: dict[int, list[int]] = {}
        for offset in range(0, len(users), self.batch_size):
            batch = users[offset : offset + self.batch_size].astype(np.int64, copy=False)
            scores = self.user_profiles[batch] @ self.item_profiles.T
            for row, raw_user in enumerate(batch):
                user = int(raw_user)
                seen = self.seen.get(user, set())
                if seen:
                    scores[row, np.fromiter(seen, dtype=np.int64)] = -np.inf
                order = np.lexsort((np.arange(scores.shape[1]), -scores[row]))
                chosen = [int(item) for item in order[:k]]
                if len(chosen) < k:
                    chosen.extend(
                        int(item) for item in self.fallback if int(item) not in seen
                    )
                output[user] = chosen[:k]
        return output


def hybrid_recommend(
    knn_candidates: dict[int, tuple[np.ndarray, np.ndarray]],
    users: np.ndarray,
    features: FeatureBundle,
    alpha: float,
    k: int = 10,
) -> dict[int, list[int]]:
    """Rerank KNN candidates by normalized KNN and genre-affinity scores."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    item_norms = np.linalg.norm(features.item_genres, axis=1, keepdims=True)
    normalized_items = np.divide(
        features.item_genres,
        item_norms,
        out=np.zeros_like(features.item_genres),
        where=item_norms != 0,
    )
    output: dict[int, list[int]] = {}
    for raw_user in users:
        user = int(raw_user)
        items, knn_scores = knn_candidates[user]
        genre_scores = normalized_items[items] @ features.user_genre_positive[user]
        combined = _minmax(knn_scores) + alpha * _minmax(genre_scores)
        order = np.lexsort((items, -combined))
        output[user] = items[order[:k]].astype(int).tolist()
    return output


def _minmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(np.float64)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum == minimum:
        return np.zeros(len(values), dtype=np.float64)
    return (values - minimum) / (maximum - minimum)
