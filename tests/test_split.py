import unittest

import numpy as np

from recsys.split import chronological_user_split, masked_user_split, temporal_ranker_split


class ChronologicalSplitTest(unittest.TestCase):
    def test_split_is_temporal_and_disjoint(self) -> None:
        events = np.array(
            [[1, item, 5, timestamp] for item, timestamp in enumerate(range(10))]
            + [[2, 20 + item, 4, timestamp] for item, timestamp in enumerate(range(20))],
            dtype=np.int64,
        )
        train, validation = chronological_user_split(events)
        self.assertEqual(len(train), 27)
        self.assertEqual(len(validation), 3)
        self.assertEqual(validation[:, 1].tolist(), [9, 38, 39])

    def test_masked_split_is_reproducible_and_disjoint(self) -> None:
        events = np.array(
            [
                [user, user * 20 + item, 1 + item % 5, item]
                for user in range(3)
                for item in range(10)
            ],
            dtype=np.int64,
        )
        first_train, first_validation = masked_user_split(events, seed=17)
        second_train, second_validation = masked_user_split(events[::-1], seed=17)

        np.testing.assert_array_equal(first_train, second_train)
        np.testing.assert_array_equal(first_validation, second_validation)
        self.assertEqual(len(first_validation), 3)
        train_pairs = set(map(tuple, first_train[:, :2]))
        validation_pairs = set(map(tuple, first_validation[:, :2]))
        self.assertTrue(train_pairs.isdisjoint(validation_pairs))

    def test_ranker_split_is_direct_80_10_10_and_temporal(self) -> None:
        events = np.array(
            [
                [user, user * 100 + item, 1 + item % 5, item]
                for user, length in ((0, 14), (1, 21))
                for item in range(length)
            ],
            dtype=np.int64,
        )
        base, target, final = temporal_ranker_split(events[::-1])

        self.assertEqual([len(base), len(target), len(final)], [27, 3, 5])
        self.assertEqual(target[:, 1].tolist(), [11, 116, 117])
        self.assertEqual(final[:, 1].tolist(), [12, 13, 118, 119, 120])
        self.assertTrue(set(map(tuple, base[:, :2])).isdisjoint(map(tuple, target[:, :2])))


if __name__ == "__main__":
    unittest.main()
