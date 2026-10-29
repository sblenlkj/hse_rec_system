import unittest

import numpy as np

from recsys.features import assert_features_fit_on_train, build_train_features


class FeatureLayerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train = np.array(
            [[0, 0, 5, 0], [0, 1, 3, 1], [1, 1, 4, 0]], dtype=np.int64
        )
        self.validation = np.array([[0, 2, 1, 2], [1, 2, 5, 1]], dtype=np.int64)
        self.genres = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)

    def test_aggregates_and_profiles_are_train_only(self) -> None:
        features = build_train_features(self.train, self.genres, n_users=2, n_items=3)
        assert_features_fit_on_train(features, self.train, self.validation)
        self.assertEqual(features.user_features[0, 0], 2)
        self.assertEqual(features.user_features[0, 3], 1)
        self.assertAlmostEqual(features.item_features[1, 1], 3.5)
        np.testing.assert_allclose(features.user_genre_positive[0], [1, 0])

    def test_leakage_check_rejects_full_event_features(self) -> None:
        full = np.concatenate([self.train, self.validation])
        features = build_train_features(full, self.genres, n_users=2, n_items=3)
        with self.assertRaisesRegex(AssertionError, "train"):
            assert_features_fit_on_train(features, self.train, self.validation)


if __name__ == "__main__":
    unittest.main()
