import unittest

from recsys.metrics import recall_at_k


class RecallAtKTest(unittest.TestCase):
    def test_macro_recall_and_empty_ground_truth(self) -> None:
        ground_truth = {1: [10, 11], 2: [20], 3: []}
        recommendations = {1: [10, 99], 2: [98, 20], 3: [1]}
        self.assertAlmostEqual(recall_at_k(ground_truth, recommendations, k=2), 0.75)

    def test_duplicate_recommendations_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            recall_at_k({1: [10]}, {1: [10, 10]}, k=2)


if __name__ == "__main__":
    unittest.main()
