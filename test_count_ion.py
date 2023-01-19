import unittest
import count_ion
import mdtraj as md
import numpy as np


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.traj_pdb = md.load("test/03-find_SF_from_PDB/02-NaK2K/em_amber.pdb")
        self.traj_short = md.load("test/01-2POT/fix_c_10ns-pro-2POT.xtc",
                                  top="test/01-2POT/02-pro-2POT.pdb")

    def test_potassium_state_assign_cylider(self):
        print("TEST: assign the state for each ion")
        top_ind = [111 - 1,
                   247 - 1,
                   383 - 1,
                   519 - 1]
        bottom_ind = [60 - 1,
                      196 - 1,
                      332 - 1,
                      468 - 1]
        center_ind = top_ind + bottom_ind
        traj = self.traj_short
        for ion_index in [[544, 545], [545, 544]]:
            ions_state_dict = count_ion.potassium_state_assign_cylider(top_ind, bottom_ind, center_ind,
                                                                       traj, ion_index)
            self.assertEqual(ions_state_dict[544][:11].tolist(), [1, 4, 3, 3, 4,
                                                                  4, 3, 3, 4, 3, 4])
            self.assertEqual(ions_state_dict[545][:11].tolist(), [1, 3, 3, 4, 4,
                                                                  3, 4, 4, 3, 4, 4])
            self.assertEqual(ions_state_dict[544][75], 2)
            self.assertEqual(ions_state_dict[545].shape, (101,))
            self.assertEqual(ions_state_dict[544].shape, (101,))

    def test_assign_ion_state_chunk(self):
        top_ind = [111 - 1,
                   247 - 1,
                   383 - 1,
                   519 - 1]
        bottom_ind = [60 - 1,
                      196 - 1,
                      332 - 1,
                      468 - 1]
        for ion_index in [[544, 545], [545, 544]]:
            ions_state_dict = count_ion.assign_ion_state_chunk(xtc_file="test/01-2POT/fix_c_10ns-pro-2POT.xtc",
                                                               stride=None,
                                                               top="test/01-2POT/02-pro-2POT.pdb",
                                                               chunk=10,
                                                               assign_fun=count_ion.potassium_state_assign_cylider,
                                                               top_ind=top_ind,
                                                               bottom_ind=bottom_ind,
                                                               center_ind=top_ind + bottom_ind,
                                                               ion_index=ion_index,
                                                               )
            i_non_iter = count_ion.potassium_state_assign_cylider(top_ind,
                                                                  bottom_ind,
                                                                  center_ind=top_ind + bottom_ind,
                                                                  traj=self.traj_short,
                                                                  ion_index=ion_index)
            self.assertListEqual(ions_state_dict[544].tolist(), i_non_iter[544].tolist())
            self.assertListEqual(ions_state_dict[545].tolist(), i_non_iter[545].tolist())


    def test_potassium_state_assign_cylider_double(self):
        S00, S01, S12, S23, S34, S45 = count_ion.auto_find_SF_index(self.traj_short)
        ions_state_dict = count_ion.potassium_state_assign_cylider_double(S01, S23, S45, S01 + S23 + S45,
                                                                          traj=self.traj_short, ion_index=[544, 545],
                                                                          rad=0.25
                                                                          )
        self.assertEqual(ions_state_dict[544][0:5].tolist(), [1, 4, 3, 3, 4])
        self.assertEqual(ions_state_dict[545][0:5].tolist(), [1, 3, 3, 4, 4])
        self.assertEqual(ions_state_dict[545][42:48].tolist(), [3, 5, 5, 1, 1, 4])



    def test_ion_state_short(self):
        print("TEST: short the sequence")
        chain = np.array("0 0 0 1 1 4 4 4 4 5 5 5".split())
        state_chain, resident_time_chain, end_time_chain = count_ion.ion_state_short(chain, 10)
        for i in range(4):
            self.assertEqual(state_chain[i],
                             [0, 1, 4, 5][i])
            self.assertAlmostEqual(resident_time_chain[i],
                                   [30., 20., 40., 30][i])
            self.assertAlmostEqual(end_time_chain[i],
                                   [20., 40., 80., 110][i])

    def test_match_sequence_numpy(self):
        print("TEST: sequence match")
        matched = count_ion.match_sequence_numpy(
            np.array([1, 2, 3, 4, 5, 6, 1, 2, 3]),
            np.array([1, 2]))
        self.assertEqual(matched[0].tolist(), [0, 6])

    def test_ion_state_short_map(self):
        print("TEST ion_state_short_map")
        ions_state_dict = {0: np.array("0 0 0 1 1 4 4 4 4 5 5 5".split()),
                           1: np.array("0 0 0 2 4 4 4 4 5 5 5".split())
                           }
        traj_timestep = 10
        d = count_ion.ion_state_short_map(ions_state_dict, traj_timestep)
        self.assertEqual(d[0][0].tolist(), np.array([0, 1, 4, 5]).tolist())
        self.assertEqual(d[1][0].tolist(), np.array([30, 20, 40, 30]).tolist())

    def test_match_seqs(self):
        print("TEST match sequence for all ions")
        ions_state = {0: np.array([1, 2, 3, 4, 5], dtype=int),
                      1: np.array([1, 2, 1, 2, 1], dtype=int)
                      }
        matched_dict = count_ion.match_seqs(ions_state, np.array([1, 2], dtype=int))
        self.assertEqual(matched_dict[0][0].tolist(), [0])
        self.assertEqual(matched_dict[1][0].tolist(), [0, 2])

        ions_state = {544: np.array([1, 2, 1, 4, 5], dtype=int),
                      545: np.array([1, 2, 1, 2, ], dtype=int)
                      }
        matched_dict = count_ion.match_seqs(ions_state, np.array([1, 2, 1], dtype=int))
        self.assertEqual(matched_dict[544][0].tolist(), [0])
        self.assertEqual(matched_dict[545][0].tolist(), [0])

    def test_find_P_index(self):
        print("TEST auto find index of P atom")
        up_leaf_index, low_leaf_index = count_ion.find_P_index(self.traj_pdb)
        self.assertEqual(up_leaf_index, (np.array(
            [6238, 6372, 6506, 6640, 6774, 6908, 7042, 7176, 7310, 7444,
             7578, 7712, 7846, 7980, 8114, 8248, 8382, 8516, 8650, 8784,
             8918, 9052, 9186, 9320, 9454, 9588, 9722, 9856, 9990, 10124,
             10258, 10392, 10526, 10660, 10794, 10928, 11062, 11196, 11330, 11464, 11598,
             11732, 11866, 12000, 12134, 12268, 12402, 12536, 12670, 12804, 12938,
             13072]) - 1).tolist())
        self.assertEqual(low_leaf_index, (np.array(
            [13206, 13340, 13474, 13608, 13742, 13876, 14010, 14144, 14278, 14412, 14546, 14680, 14814, 14948, 15082,
             15216, 15350, 15484, 15618, 15752, 15886, 16020, 16154, 16288, 16422, 16556, 16690, 16824, 16958, 17092,
             17226, 17360, 17494, 17628, 17762, 17896, 18030, 18164, 18298, 18432, 18566, 18700, 18834, 18968, 19102,
             19236, 19370, 19504, 19638, 19772, 19906, 20040, 20174, 20308, ]) - 1).tolist())


    def test_potassium_state_assign_membrane(self):
        print("TEST assign state for ion inside/outside membrane")
        traj = md.load("test/03-find_SF_from_PDB/02-NaK2K/fix_atom_c_1ns_10frame.xtc",
                       top = "test/03-find_SF_from_PDB/02-NaK2K/em_amber.pdb")
        up_leaf_index, low_leaf_index = count_ion.find_P_index(traj)
        ions_state_dict = count_ion.potassium_state_assign_membrane(traj,
                                                                    up_leaf_index,
                                                                    low_leaf_index,
                                                                    ion_index=[5960, 5961, 5962])
        self.assertEqual(ions_state_dict[5960].tolist(), [2, 1, 1, 1, 3])
        self.assertEqual(ions_state_dict[5961].tolist(), [2, 1, 1, 1, 3])
        self.assertEqual(ions_state_dict[5962].tolist(), [2, 1, 3, 3, 3])






if __name__ == '__main__':
    unittest.main()
