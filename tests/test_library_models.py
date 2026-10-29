import unittest

import numpy as np

from recsys.features import build_train_features
from recsys.library_models import (
    GenreProfileRecommender,
    ImplicitItemKNNRecommender,
    hybrid_recommend,
)


class GenreModelsTest(unittest.TestCase):
    def test_implicit_adapter_returns_unique_unseen_items(self) -> None:
        events = np.array(
            [
                [0, 0, 5, 0],
                [0, 1, 4, 1],
                [1, 0, 5, 0],
                [1, 2, 4, 1],
                [2, 1, 5, 0],
                [2, 3, 4, 1],
            ],
            dtype=np.int64,
        )
        model = ImplicitItemKNNRecommender(n_neighbors=3, num_threads=1)
        model.fit(events, n_users=3, n_items=5)
        recommendations = model.recommend(np.array([0]), k=2)[0]
        self.assertEqual(len(recommendations), 2)
        self.assertEqual(len(set(recommendations)), 2)
        self.assertFalse({0, 1}.intersection(recommendations))

    def test_content_model_filters_seen_items(self) -> None:
        events = np.array(
            [[0, 0, 5, 0], [0, 1, 2, 1], [1, 2, 4, 0]], dtype=np.int64
        )
        genres = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
        features = build_train_features(events, genres, n_users=2, n_items=4)
        model = GenreProfileRecommender().fit(events, features)
        recommendations = model.recommend(np.array([0]), k=2)
        self.assertEqual(recommendations[0], [2, 3])

    def test_hybrid_can_promote_genre_match(self) -> None:
        events = np.array([[0, 0, 5, 0]], dtype=np.int64)
        genres = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
        features = build_train_features(events, genres, n_users=1, n_items=3)
        candidates = {0: (np.array([1, 2]), np.array([1.0, 0.9]))}
        recommendations = hybrid_recommend(
            candidates, np.array([0]), features, alpha=2.0, k=2
        )
        self.assertEqual(recommendations[0], [2, 1])


if __name__ == "__main__":
    unittest.main()
