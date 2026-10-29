import unittest

import numpy as np

from recsys.baselines import PopularityRecommender


class PopularityTest(unittest.TestCase):
    def test_seen_items_are_filtered(self) -> None:
        events = np.array(
            [[0, 0, 5, 0], [0, 1, 5, 1], [1, 0, 5, 0], [1, 2, 3, 1]],
            dtype=np.int64,
        )
        model = PopularityRecommender().fit(events, n_items=4)
        recommendations = model.recommend(np.array([0]), k=2)
        self.assertEqual(recommendations[0], [2, 3])


if __name__ == "__main__":
    unittest.main()
