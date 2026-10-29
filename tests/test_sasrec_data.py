import unittest

import numpy as np

from recsys.sasrec_data import (
    assert_train_only_sequences,
    build_item_catalog,
    build_user_sequences,
    encode_items,
)


class SasRecDataTest(unittest.TestCase):
    def test_sequences_are_ordered_encoded_and_train_only(self) -> None:
        train = np.array(
            [
                [1, 20, 5, 2],
                [0, 10, 4, 0],
                [1, 10, 3, 0],
                [0, 30, 5, 1],
            ],
            dtype=np.int64,
        )
        validation = np.array(
            [[0, 40, 5, 2], [1, 30, 4, 3]], dtype=np.int64
        )
        catalog = build_item_catalog(train)
        sequences = build_user_sequences(train, catalog)

        self.assertEqual(catalog.tolist(), [10, 20, 30])
        self.assertEqual(sequences, {0: [0, 2], 1: [0, 1]})
        self.assertEqual(encode_items(np.array([30, 40]), catalog).tolist(), [2, -1])
        assert_train_only_sequences(sequences, train, validation, catalog)

    def test_leakage_assertion_rejects_validation_item_in_sequence(self) -> None:
        train = np.array([[0, 10, 5, 0], [0, 20, 4, 1]], dtype=np.int64)
        validation = np.array([[0, 30, 5, 2]], dtype=np.int64)
        catalog = build_item_catalog(train)
        with self.assertRaisesRegex(AssertionError, "exactly match"):
            assert_train_only_sequences(
                {0: [0, 1, 0]}, train, validation, catalog
            )


if __name__ == "__main__":
    unittest.main()
