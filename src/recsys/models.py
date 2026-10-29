"""Small, dependency-light personalized recommenders."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import sparse

from recsys.split import ITEM, USER


class ItemKNNRecommender:
    """Cosine ItemKNN over a binary user-item matrix."""

    def __init__(self, n_neighbors: int = 200, batch_size: int = 256) -> None:
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.user_items: sparse.csr_matrix | None = None
        self.similarity: sparse.csr_matrix | None = None
        self.fallback: np.ndarray | None = None

    def fit(
        self, train_events: np.ndarray, n_users: int | None = None, n_items: int | None = None
    ) -> "ItemKNNRecommender":
        if n_users is None:
            n_users = int(train_events[:, USER].max()) + 1
        if n_items is None:
            n_items = int(train_events[:, ITEM].max()) + 1
        rows = train_events[:, USER].astype(np.int32, copy=False)
        cols = train_events[:, ITEM].astype(np.int32, copy=False)
        data = np.ones(len(train_events), dtype=np.float32)
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        matrix.data[:] = 1.0
        matrix.eliminate_zeros()
        self.user_items = matrix

        counts = np.asarray(matrix.sum(axis=0)).ravel()
        self.fallback = np.lexsort((np.arange(n_items), -counts))
        inverse_norm = np.zeros(n_items, dtype=np.float32)
        nonzero = counts > 0
        inverse_norm[nonzero] = 1.0 / np.sqrt(counts[nonzero])

        similarity = (matrix.T @ matrix).tocoo()
        similarity.data *= inverse_norm[similarity.row] * inverse_norm[similarity.col]
        similarity = similarity.tocsr()
        similarity.setdiag(0.0)
        similarity.eliminate_zeros()
        self.similarity = _retain_top_k(similarity, self.n_neighbors)
        return self

    def recommend(self, users: np.ndarray, k: int = 10) -> dict[int, list[int]]:
        if self.user_items is None or self.similarity is None or self.fallback is None:
            raise RuntimeError("fit must be called before recommend")

        result: dict[int, list[int]] = {}
        for offset in range(0, len(users), self.batch_size):
            batch_users = users[offset : offset + self.batch_size].astype(np.int64, copy=False)
            scores = (self.user_items[batch_users] @ self.similarity).tocsr()
            for row_index, raw_user in enumerate(batch_users):
                user = int(raw_user)
                start, end = scores.indptr[row_index : row_index + 2]
                items = scores.indices[start:end]
                values = scores.data[start:end]
                seen = set(self.user_items[user].indices.tolist())
                if len(items):
                    candidate_order = np.lexsort((items, -values))
                    ranked = [int(items[i]) for i in candidate_order if int(items[i]) not in seen]
                else:
                    ranked = []
                chosen = ranked[:k]
                if len(chosen) < k:
                    chosen_set = set(chosen)
                    chosen.extend(
                        int(item)
                        for item in self.fallback
                        if int(item) not in seen and int(item) not in chosen_set
                    )
                result[user] = chosen[:k]
        return result


def _retain_top_k(matrix: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    """Keep the largest k values in every CSR row."""
    if k <= 0:
        raise ValueError("n_neighbors must be positive")
    row_indices: list[np.ndarray] = []
    row_values: list[np.ndarray] = []
    indptr = np.zeros(matrix.shape[0] + 1, dtype=np.int64)
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row : row + 2]
        indices = matrix.indices[start:end]
        values = matrix.data[start:end]
        if len(values) > k:
            keep = np.argpartition(values, -k)[-k:]
            indices = indices[keep]
            values = values[keep]
        order = np.argsort(indices)
        row_indices.append(indices[order])
        row_values.append(values[order])
        indptr[row + 1] = indptr[row] + len(indices)
    indices = np.concatenate(row_indices).astype(np.int32, copy=False)
    values = np.concatenate(row_values).astype(np.float32, copy=False)
    return sparse.csr_matrix((values, indices, indptr), shape=matrix.shape)
