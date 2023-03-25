import unittest
import count_ion_SF_cpp
import numpy as np
import mdtraj as md
import time


class MyTestCase(unittest.TestCase):
    def test_match_head(self):
        print("\nTEST match_head")
        ions_state15 = {0: np.array([5, 1, 5, 1, 3]),
                        1: np.array([5, 4, 5, 1, 3]),
                        2: np.array([4, 5, 1, 3, 4])
                        }
        matched = count_ion_SF_cpp.match_head(ions_state15, seq=np.array([5, 1, 3]), forbidden=4)
        self.assertEqual(matched[0], (np.array([2]),))
        self.assertEqual(matched[1][0].size, 0)
        self.assertEqual(matched[2][0].size, 0)
        # print(matched)

    def test_match_tail(self):
        print("\nTEST match_tail")
        ions_state15 = {0: np.array([1, 3, 1, 4, 5, 1]),
                        1: np.array([1, 4, 5, 1, 5, 1]),
                        2: np.array([1, 4, 5, 1, 3, 4])
                        }
        matched = count_ion_SF_cpp.match_tail(ions_state15, seq=np.array([4, 5, 1]), forbidden=3)
        self.assertEqual(matched[0], (np.array([3]),))
        self.assertEqual(matched[1], (np.array([1]),))
        self.assertEqual(matched[2][0].size, 0)
        # print(matched)

    def test_auto_find_SF_index_01(self):
        print("# TEST auto_find_SF_index NaK2K")
        traj_short = md.load("test/01-2POT/fix_c_10ns-pro-2POT.xtc",
                             top="test/01-2POT/02-pro-2POT.pdb")
        S00, S01, S12, S23, S34, S45 = count_ion_SF_cpp.auto_find_SF_index(traj_short)  # 0 based index
        self.assertEqual(S00, [int(i) - 1 for i in "118  254  390  526".split()])
        self.assertEqual(S01, [int(i) - 1 for i in "111  247  383  519".split()])
        self.assertEqual(S12, [int(i) - 1 for i in "90  226  362  498".split()])
        self.assertEqual(S23, [int(i) - 1 for i in "83  219  355  491".split()])
        self.assertEqual(S34, [int(i) - 1 for i in "67  203  339  475".split()])
        self.assertEqual(S45, [int(i) - 1 for i in "60  196  332  468".split()])

    def test_auto_find_SF_index_02(self):
        print("# TEST auto_find_SF_index TREK2 PDB 4BW5, chain A,B")
        traj = md.load("test/06-TREK-4BW5/4bw5.pdb")
        S00, S01, S12, S23, S34, S45 = count_ion_SF_cpp.auto_find_SF_index(
            traj,
            SF_seq="THR ILE GLY TYR GLY".split(),
            SF_seq2="THR VAL GLY PHE GLY".split())  # 0 based index
        self.assertEqual(S00, [int(i) - 1 for i in "753 1480 2564 3357".split()])
        self.assertEqual(S01, [int(i) - 1 for i in "741 1469 2552 3346".split()])
        self.assertEqual(S12, [int(i) - 1 for i in "737 1465 2548 3342".split()])
        self.assertEqual(S23, [int(i) - 1 for i in "729 1458 2540 3335".split()])
        self.assertEqual(S34, [int(i) - 1 for i in "722 1451 2533 3328".split()])
        self.assertEqual(S45, [int(i) - 1 for i in "724 1453 2535 3330".split()])

    def test_auto_find_SF_index_03(self):
        print("# TEST auto_find_SF_index TREK2 from TOM, chain A,A")
        traj = md.load("test/07-TREK2/TREK2_TOM_up.pdb")
        S00, S01, S12, S23, S34, S45 = count_ion_SF_cpp.auto_find_SF_index(
            traj,
            SF_seq="THR ILE GLY TYR GLY".split(),
            SF_seq2="THR VAL GLY PHE GLY".split())  # 0 based index
        self.assertEqual(S00, [int(i) - 1 for i in "1588 3327 5710 7449".split()])
        self.assertEqual(S01, [int(i) - 1 for i in "1581 3320 5703 7442".split()])
        self.assertEqual(S12, [int(i) - 1 for i in "1560 3300 5682 7422".split()])
        self.assertEqual(S23, [int(i) - 1 for i in "1553 3293 5675 7415".split()])
        self.assertEqual(S34, [int(i) - 1 for i in "1534 3277 5656 7399".split()])
        self.assertEqual(S45, [int(i) - 1 for i in "1527 3270 5649 7392".split()])

    def test_auto_find_SF_index_04(self):
        print("# TEST auto_find_SF_index TREK2 from TOM, chain A,B")
        traj = md.load("test/07-TREK2/TREK2_TOM_up_Chain_AB.pdb")
        S00, S01, S12, S23, S34, S45 = count_ion_SF_cpp.auto_find_SF_index(
            traj,
            SF_seq="THR ILE GLY TYR GLY".split(),
            SF_seq2="THR VAL GLY PHE GLY".split())  # 0 based index
        self.assertEqual(S00, [int(i) - 1 for i in "1588 3327 5710 7449".split()])
        self.assertEqual(S01, [int(i) - 1 for i in "1581 3320 5703 7442".split()])
        self.assertEqual(S12, [int(i) - 1 for i in "1560 3300 5682 7422".split()])
        self.assertEqual(S23, [int(i) - 1 for i in "1553 3293 5675 7415".split()])
        self.assertEqual(S34, [int(i) - 1 for i in "1534 3277 5656 7399".split()])
        self.assertEqual(S45, [int(i) - 1 for i in "1527 3270 5649 7392".split()])


    def test_PYSfilter_01(self):
        traj = md.load("test/03-find_SF_from_PDB/04-NaK2K-more/em.pdb")
        S00, S01, S12, S23, S34, S45 = count_ion_SF_cpp.auto_find_SF_index(traj)
        import os
        lib = os.path.join(os.path.dirname(__file__), "cpp/cmake-build-debug/")
        import sys
        sys.path.append(lib)
        from PYSfilter import Sfilter
        ion_index = count_ion_SF_cpp.find_K_index(traj, K_name="POT")
        wat_index = count_ion_SF_cpp.find_water_O_index(traj)
        SF = Sfilter("test/03-find_SF_from_PDB/04-NaK2K-more/fix_atom_c_kpro.xtc")
        SF.assign_state_double(S01, S23, S45,
                               S01 + S23 + S45,
                               ion_index,
                               wat_index, 0.25)
        ions_state_dict = count_ion_SF_cpp.ion_state_list_2_dict(SF.ion_state_list, ion_index)
        self.assertEqual(ions_state_dict[5960][0:5].tolist(), [5, 5, 5, 5, 5])
        self.assertEqual(ions_state_dict[5961][0:5].tolist(), [1, 1, 1, 1, 5])
        self.assertEqual(ions_state_dict[5962][0:5].tolist(), [1, 1, 1, 1, 1])
        self.assertEqual(ions_state_dict[5963][0:5].tolist(), [3, 3, 3, 3, 3])
        self.assertEqual(ions_state_dict[6068][31:36].tolist(), [4, 5, 5, 5, 5])

    def test_assign_state_12345_py(self):
        traj = md.load("test/03-find_SF_from_PDB/04-NaK2K-more/fix_atom_c_kpro.xtc",
                       top="test/03-find_SF_from_PDB/04-NaK2K-more/em.pdb")
        S00, S01, S12, S23, S34, S45 = count_ion_SF_cpp.auto_find_SF_index(traj)
        ion_index = count_ion_SF_cpp.find_K_index(traj, K_name="POT")
        wat_index = count_ion_SF_cpp.find_water_O_index(traj)
        ions_s_dict, wats_s_dict, traj_step = count_ion_SF_cpp.assign_state_12345_py(traj, ion_index, wat_index, 0.25,
                                                                                     S01, S23, S45)
        self.assertEqual(ions_s_dict[5960][0:5].tolist(), [5, 5, 5, 5, 5])
        self.assertEqual(ions_s_dict[5961][0:5].tolist(), [1, 1, 1, 1, 5])
        self.assertEqual(ions_s_dict[5962][0:5].tolist(), [1, 1, 1, 1, 1])
        self.assertEqual(ions_s_dict[5963][0:5].tolist(), [3, 3, 3, 3, 3])
        self.assertEqual(ions_s_dict[6068][31:36].tolist(), [4, 5, 5, 5, 5])
        self.assertAlmostEqual(traj_step, 20.0)

    def test_backend_time(self):
        xtc = "test/03-find_SF_from_PDB/04-NaK2K-more/fix_atom_c_kpro.xtc"
        pdb = "test/03-find_SF_from_PDB/04-NaK2K-more/em.pdb"
        S00, S01, S12, S23, S34, S45 = count_ion_SF_cpp.auto_find_SF_index(md.load(pdb))
        tick1 = time.time()
        traj_pdb = md.load(xtc, top=pdb)
        answer1 = count_ion_SF_cpp.assign_state_12345_py(
            traj_pdb,
            count_ion_SF_cpp.find_K_index(traj_pdb, K_name="POT"),
            count_ion_SF_cpp.find_water_O_index(traj_pdb),
            0.25, S01, S23, S45)
        tick2 = time.time()
        answer2 = count_ion_SF_cpp.assign_state_12345_cpp(
            xtc,
            count_ion_SF_cpp.find_K_index(traj_pdb, K_name="POT"),
            count_ion_SF_cpp.find_water_O_index(traj_pdb),
            0.25, S01, S23, S45)
        tick3 = time.time()
        for k in [5960, 5961]:
            self.assertListEqual(answer1[0][k].tolist(), answer2[0][k].tolist())
#        for k in answer1[1]:
#            self.assertListEqual(answer1[1][k].tolist(), answer2[1][k].tolist())
        self.assertAlmostEqual(answer1[2], answer2[2])  # traj time step (ps/frame)
        print("Time comparison ")
        print("PY: %.3f s, CPP %.3f s" % (tick2 - tick1, tick3 - tick2))
        print("C++ speed up %.3f times" % ((tick2 - tick1) / (tick3 - tick2)))


if __name__ == '__main__':
    unittest.main()
