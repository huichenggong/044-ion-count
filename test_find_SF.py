import unittest
import count_ion
import mdtraj as md


class MyTestCase(unittest.TestCase):
    def test_auto_find_SF_index_1(self):
        pdb = "test/03-find_SF_from_PDB/01-small-SF/01-full.pdb"
        S00, S01, S12, S23, S34, S45 = count_ion.auto_find_SF_index(md.load(pdb))
        #print(S00, S01, S12, S23, S34, S45)
        self.assertEqual(S00, [int(at)-1 for at in "118  254  390  526".split()])
        self.assertEqual(S01, [int(at) - 1 for at in "111  247  383  519".split()])
        self.assertEqual(S12, [int(at) - 1 for at in "90  226  362  498".split()])
        self.assertEqual(S23, [int(at) - 1 for at in "83  219  355  491".split()])
        self.assertEqual(S34, [int(at) - 1 for at in "67  203  339  475".split()])
        self.assertEqual(S45, [int(at) - 1 for at in "60  196  332  468".split()])

    def test_auto_find_SF_index_2(self):
        pdb = "test/03-find_SF_from_PDB/02-NaK2K/em_amber.pdb"
        S00, S01, S12, S23, S34, S45 = count_ion.auto_find_SF_index(md.load(pdb))
        #print(S00, S01, S12, S23, S34, S45)
        self.assertEqual(S00, [int(at) - 1 for at in "754 2244 3734 5224".split()])
        self.assertEqual(S01, [int(at) - 1 for at in "747 2237 3727 5217".split()])
        self.assertEqual(S12, [int(at) - 1 for at in "726 2216 3706 5196".split()])
        self.assertEqual(S23, [int(at) - 1 for at in "719 2209 3699 5189".split()])
        self.assertEqual(S34, [int(at) - 1 for at in "703 2193 3683 5173".split()])
        self.assertEqual(S45, [int(at) - 1 for at in "700 2190 3680 5170".split()])

    def test_auto_find_SF_index_3(self):
        pdb = "test/03-find_SF_from_PDB/03-KV7.1/step6.0_minimization.pdb"
        SF_seq = ["THR", "ILE", "GLY", "TYR", "GLY"]
        S00, S01, S12, S23, S34, S45 = count_ion.auto_find_SF_index(md.load(pdb), SF_seq)
        #print(S00, S01, S12, S23, S34, S45)
        self.assertEqual(S00, [int(at) - 1 for at in "3458 9333 15208 21083".split()])
        self.assertEqual(S01, [int(at) - 1 for at in "3451 9326 15201 21076".split()])
        self.assertEqual(S12, [int(at) - 1 for at in "3430 9305 15180 21055".split()])
        self.assertEqual(S23, [int(at) - 1 for at in "3423 9298 15173 21048".split()])
        self.assertEqual(S34, [int(at) - 1 for at in "3404 9279 15154 21029".split()])
        self.assertEqual(S45, [int(at) - 1 for at in "3397 9272 15147 21022".split()])


if __name__ == '__main__':
    unittest.main()
