import unittest

import numpy as np

from recsys.als_models import ImplicitALSRecommender


class ImplicitALSRecommenderTest(unittest.TestCase):
    def test_recommendations_are_unique_and_filter_seen_items(self) -> None:
        events = np.array(
            [
                [0, 0, 5, 0],
                [0, 1, 4, 1],
                [1, 0, 5, 0],
                [1, 2, 3, 1],
                [2, 1, 4, 0],
                [2, 3, 5, 1],
            ],
            dtype=np.int64,
        )
        model = ImplicitALSRecommender(
            factors=2, iterations=2, weighting="binary", num_threads=1, random_state=7
        ).fit(events, n_users=3, n_items=5)
        recommendations = model.recommend(np.array([0]), k=3)[0]

        self.assertEqual(len(recommendations), 3)
        self.assertEqual(len(set(recommendations)), 3)
        self.assertFalse({0, 1}.intersection(recommendations))

        items, scores = model.recommend_with_scores(np.array([0]), k=3)[0]
        self.assertEqual(items.tolist(), recommendations)
        self.assertEqual(len(scores), 3)


if __name__ == "__main__":
    unittest.main()
