import unittest
import count_ion_SF
import numpy as np
import count_ion


class MyTestCase(unittest.TestCase):
    def test_match_head(self):
        print("TEST match_head")
        ions_state15 = {0: np.array([5, 1, 5, 1, 3]),
                        1: np.array([5, 4, 5, 1, 3]),
                        2: np.array([4, 5, 1, 3, 4])
                        }
        matched = count_ion_SF.match_head(ions_state15, seq=np.array([5, 1, 3]), forbidden=4)
        self.assertEqual(matched[0], (np.array([2]),))
        self.assertEqual(matched[1][0].size, 0)
        self.assertEqual(matched[2][0].size, 0)

    def test_match_tail(self):
        print("TEST match_tail")
        ions_state15 = {0: np.array([3, 1, 4, 5, 1]),
                        1: np.array([4, 5, 1, 5, 1]),
                        2: np.array([4, 5, 1, 3, 4])
                        }
        matched = count_ion_SF.match_tail(ions_state15, seq=np.array([4, 5, 1]), forbidden=3)
        self.assertEqual(matched[0], (np.array([2]),))
        self.assertEqual(matched[1], (np.array([0]),))
        self.assertEqual(matched[2][0].size, 0)


if __name__ == '__main__':
    unittest.main()
