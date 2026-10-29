import unittest

import numpy as np

from recsys.split import chronological_user_split


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


if __name__ == "__main__":
    unittest.main()
