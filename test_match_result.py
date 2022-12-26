import unittest
from match_result import *


class MyTestCase(unittest.TestCase):
    def test_equal(self):
        b1 = equal(PermEvent(1000, 100.0, True), PermEvent(1000, [50.0, 150.0], True))
        self.assertTrue(b1)
        b1 = equal(PermEvent(1000, 100.0, False), PermEvent(1000, [50.0, 150.0], True))
        self.assertFalse(b1)
        b1 = equal(PermEvent(1001, 100.0, True), PermEvent(1000, [50.0, 150.0], True))
        self.assertFalse(b1)
        b1 = equal(PermEvent(1000, 100.0, True), PermEvent(1000, [50.0, 99.0], True))
        self.assertFalse(b1)
        b1 = equal(PermEvent(1000, 100.0, True), PermEvent(1000, [101, 102], True))
        self.assertFalse(b1)

    def test_read_perm_up_down(self):
        perm_list = read_perm_up_down("test/04-result-match/analysis/02-xtck_hybrid_02/perm_up.dat",
                                      "test/04-result-match/analysis/02-xtck_hybrid_02/perm_down.dat",
                                      )
        for e1, e2 in zip(perm_list[:-1], perm_list[1:]):
            self.assertTrue(e1.time < e2.time)

        perm_list = read_perm_up_down("test/05-result-match/05-300mv/00/analysis/01-xtck_hybrid/perm_up.dat",
                                      "test/05-result-match/05-300mv/00/analysis/01-xtck_hybrid/perm_down.dat",
                                      )
        for e1, e2 in zip(perm_list[:-1], perm_list[1:]):
            # print(e2)
            self.assertTrue(e1.time < e2.time)

    def test_read_cylinder(self):
        perm_list = read_cylinder("test/04-result-match/analysis/03-Cylinkder/k_Cylinder.out")
        for e1, e2 in zip(perm_list[:-1], perm_list[1:]):
            self.assertTrue(e1.time[0] < e2.time[0])
            self.assertTrue(e1.time[1] < e2.time[1])
        index_list = []
        for e in perm_list:
            index_list.append(e.at_index)
        self.assertListEqual(index_list, [5961, 5995, 6061, 6064, 6057, 6052, 6034, 6068, 6048, 6051,
                                          5991, 6020, 6050, 6015, 5980, 6036, 6080, 6090, 6022, 5979,
                                          6004, 5981, 5988, 6013, 6053, 6014, 6049, 6087, 5963, 6036,
                                          6065, 6007, 5973, 6035, 6019, 6036, 5973])

    def test_lcs(self):
        print("TEST LCS 1...")
        seq_xtck = [PermEvent(1, 10, True),
                    PermEvent(2, 20, True),
                    PermEvent(2, 21, False),
                    PermEvent(2, 22, True),
                    PermEvent(3, 30, True),
                    ]
        seq_cylinder = [PermEvent(1, [5, 15], True),
                        PermEvent(2, [15, 25], True),
                        PermEvent(3, [25, 34], True)
                        ]
        e_list, l = longest_common_subsequence(seq_xtck, seq_cylinder)
        self.assertEqual(l, 3)
        self.assertEqual(len(e_list), 5)
        for e in e_list:
            print(e[0], ",", e[1])
        print("Done")

    def test_lcs2(self):
        print("TEST LCS 2...")
        seq_xtck = read_perm_up_down("test/05-result-match/05-300mv/00/analysis/01-xtck_hybrid/perm_up.dat",
                                     "test/05-result-match/05-300mv/00/analysis/01-xtck_hybrid/perm_down.dat"
                                     )
        seq_cylinder = read_cylinder("test/05-result-match/05-300mv/00/analysis/03-Cylinkder/k_Cylinder.out")
        e_list, l = longest_common_subsequence(seq_xtck, seq_cylinder)
        for e in e_list:
            for i in e:
                if not (i is None):
                    print(i.at_index, end=" ")
                    if i.up:
                        print("up   ", end="")
                    else:
                        print("down ", end="")
                else:
                    print("NONE      ", end="")
            if not e[1] is None:
                print(e[1].time)
            else:
                print()

        print("Done")


if __name__ == '__main__':
    unittest.main()
