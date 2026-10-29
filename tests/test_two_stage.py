import unittest

import numpy as np

try:
    import pandas as pd
except ImportError:  # Optional ranking extra is not part of the minimal test environment.
    pd = None

from recsys.two_stage import (
    build_candidate_union,
    candidate_diagnostics,
    label_candidates,
    ranked_candidates_frame,
    scored_candidates_frame,
)


@unittest.skipIf(pd is None, "ranking optional dependencies are not installed")
class TwoStageCandidateTest(unittest.TestCase):
    def test_union_has_one_row_per_pair_and_stable_missing_values(self) -> None:
        scored = {
            0: (np.array([2, 3]), np.array([4.0, 2.0])),
            1: (np.array([3, 4]), np.array([1.0, 1.0])),
        }
        frames = {
            "sasrec": scored_candidates_frame(scored, "sasrec"),
            "als": scored_candidates_frame(
                {0: (np.array([3, 4]), np.array([0.8, 0.2])), 1: scored[1]}, "als"
            ),
            "knn": scored_candidates_frame(scored, "knn"),
            "popularity": ranked_candidates_frame({0: [4], 1: [2]}, "popularity"),
        }
        table = build_candidate_union(frames)

        self.assertFalse(table.duplicated(["user_id", "item_id"]).any())
        row = table[(table.user_id == 0) & (table.item_id == 2)].iloc[0]
        self.assertEqual(row.from_als, 0)
        self.assertEqual(row.als_rank, 201)
        self.assertEqual(row.als_score, 0.0)
        self.assertEqual(row.sasrec_score_norm, 1.0)

    def test_labels_and_candidate_recall_use_targets_only_as_ground_truth(self) -> None:
        scored = {0: (np.array([2, 3]), np.array([2.0, 1.0]))}
        frames = {
            source: scored_candidates_frame(scored, source)
            for source in ("sasrec", "als", "knn")
        }
        frames["popularity"] = ranked_candidates_frame({0: [4]}, "popularity")
        table = build_candidate_union(frames)
        target = np.array([[0, 3, 5, 2], [0, 5, 4, 3]], dtype=np.int64)

        self.assertEqual(label_candidates(table, target).sum(), 1)
        diagnostics = candidate_diagnostics(table, {0: [3, 5]})
        self.assertAlmostEqual(diagnostics["sasrec_recall_at_200"], 0.5)
        self.assertAlmostEqual(diagnostics["all_sources_union_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
