"""Thin adapters for matrix factorization models from ``implicit``."""

from __future__ import annotations

from implicit.als import AlternatingLeastSquares
import numpy as np
from scipy import sparse

from recsys.split import ITEM, RATING, USER


class ImplicitALSRecommender:
    """Deterministic CPU ALS with binary or rating confidence weights."""

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        weighting: str = "binary",
        num_threads: int = 1,
        random_state: int = 42,
    ) -> None:
        if weighting not in {"binary", "rating"}:
            raise ValueError("weighting must be binary or rating")
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.weighting = weighting
        self.num_threads = num_threads
        self.random_state = random_state
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            num_threads=num_threads,
            random_state=random_state,
        )
        self.user_items: sparse.csr_matrix | None = None
        self.fallback: np.ndarray | None = None

    def fit(
        self, train_events: np.ndarray, n_users: int | None = None, n_items: int | None = None
    ) -> "ImplicitALSRecommender":
        if n_users is None:
            n_users = int(train_events[:, USER].max()) + 1
        if n_items is None:
            n_items = int(train_events[:, ITEM].max()) + 1
        rows = train_events[:, USER].astype(np.int32, copy=False)
        columns = train_events[:, ITEM].astype(np.int32, copy=False)
        if self.weighting == "binary":
            values = np.ones(len(train_events), dtype=np.float32)
        else:
            values = train_events[:, RATING].astype(np.float32, copy=False)
        matrix = sparse.csr_matrix((values, (rows, columns)), shape=(n_users, n_items))
        matrix.eliminate_zeros()
        self.user_items = matrix
        counts = np.bincount(columns, minlength=n_items)
        self.fallback = np.lexsort((np.arange(n_items), -counts))
        self.model.fit(matrix, show_progress=False)
        return self

    def recommend(self, users: np.ndarray, k: int = 10) -> dict[int, list[int]]:
        scored = self.recommend_with_scores(users, k=k)
        return {user: items.astype(int).tolist() for user, (items, _) in scored.items()}

    def recommend_with_scores(
        self, users: np.ndarray, k: int = 10
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Return ranked unseen items and their native implicit ALS scores."""
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
            seen = set(self.user_items[user].indices.tolist())
            chosen_set = set(chosen)
            if len(chosen) < k:
                for raw_item in self.fallback:
                    item = int(raw_item)
                    if item not in seen and item not in chosen_set:
                        chosen.append(item)
                        values.append(0.0)
                        chosen_set.add(item)
                    if len(chosen) == k:
                        break
            output[user] = (np.asarray(chosen[:k]), np.asarray(values[:k]))
        return output
