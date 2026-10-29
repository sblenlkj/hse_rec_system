"""Train-only popularity recommenders."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from recsys.split import ITEM, RATING, USER


class PopularityRecommender:
    """Global popularity with per-user seen-item filtering."""

    def __init__(self, minimum_rating: int | None = None) -> None:
        self.minimum_rating = minimum_rating
        self.ranking: np.ndarray | None = None
        self.seen: dict[int, set[int]] = {}

    def fit(
        self,
        train_events: np.ndarray,
        n_items: int | None = None,
        n_users: int | None = None,
    ) -> "PopularityRecommender":
        del n_users  # Accepted for the shared recommender fit interface.
        if n_items is None:
            n_items = int(train_events[:, ITEM].max()) + 1
        selected = train_events
        if self.minimum_rating is not None:
            selected = train_events[train_events[:, RATING] >= self.minimum_rating]
        counts = np.bincount(selected[:, ITEM], minlength=n_items)
        # Stable tie-break: smaller item_id first.
        self.ranking = np.lexsort((np.arange(n_items), -counts))
        seen: defaultdict[int, set[int]] = defaultdict(set)
        for user, item in train_events[:, [USER, ITEM]]:
            seen[int(user)].add(int(item))
        self.seen = dict(seen)
        return self

    def recommend(self, users: np.ndarray, k: int = 10) -> dict[int, list[int]]:
        if self.ranking is None:
            raise RuntimeError("fit must be called before recommend")
        output: dict[int, list[int]] = {}
        for raw_user in users:
            user = int(raw_user)
            seen = self.seen.get(user, set())
            output[user] = [int(item) for item in self.ranking if int(item) not in seen][:k]
        return output
