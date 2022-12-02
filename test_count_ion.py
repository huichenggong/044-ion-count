import unittest
import count_ion
import mdtraj as md
import numpy as np


class MyTestCase(unittest.TestCase):
    def setUp(self):
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
            ions_state_dict = count_ion.assign_ion_state_chunk("test/01-2POT/fix_c_10ns-pro-2POT.xtc",
                                                               "test/01-2POT/02-pro-2POT.pdb",
                                                               chunk=10,
                                                               assigh_fun=count_ion.potassium_state_assign_cylider,
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
        print(count_ion.match_sequence_numpy(
            np.array([1, 2, 3, 4, 5, 6, 1, 2, 3]),
            np.array([1, 2])))

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


if __name__ == '__main__':
    unittest.main()
