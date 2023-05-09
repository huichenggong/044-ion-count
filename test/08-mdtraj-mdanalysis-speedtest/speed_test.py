import time
import mdtraj as md
import MDAnalysis as mda
import numpy as np

pdb = "/home/chui/E29Project-2023-04-11/052-TRAAK-double-CYC/02-double-CYS-Charmm/01-C_1.00/01-EM/em.pdb"
xtc = "/home/chui/E29Project-2023-04-11/052-TRAAK-double-CYC/02-double-CYS-Charmm/HRE/02-1.00/25/fix_c_60ps.xtc"
# ILE130, VAL239 O

# mda iterload
tick = time.time()
u = mda.Universe(pdb, xtc)
tock1 = time.time()  # load xtc into mem
K_atoms = u.select_atoms("name POT")
O_atoms = u.select_atoms("((resid 130 and resname ILE) or (resid 239 and resname VAL)) and name O")
tock2 = time.time()  # translate selection
reslist_mdai = []
for ts in u.trajectory:
    reslist_mdai.append(K_atoms.positions[:, 2] - np.mean(O_atoms.positions[:, 2]))
tock = time.time()
print("\n# mdanalysis iter ##############")
print("Loading time       :", tock1 - tick)
print("translate selection:", tock2 - tock1)
print("Loop over array    :", tock - tock2)
print("Total time         :", tock - tick)


# mdtraj load and run
tick = time.time()
traj = md.load(xtc, top=pdb)
tock1 = time.time()  # load xtc into mem
K_index = traj.topology.select('name POT')
O_index = traj.topology.select('((resSeq 130 and resname ILE) or (resSeq 239 and resname VAL)) and name O')
tock2 = time.time()  # translate selection
k_traj = traj.atom_slice(K_index).xyz
o_traj = traj.atom_slice(O_index).xyz
tock3 = time.time()  # fetch np.array
reslist_mdtraj = []
for i in range(401):
    reslist_mdtraj.append(k_traj[i, :, 2] - np.mean(o_traj[i, :, 2]))
tock = time.time()
print("\n# mdtraj in memory #############")
print("Loading time       :", tock1 - tick)
print("translate selection:", tock2 - tock1)
print("fetch np.array     :", tock3 - tock2)
print("Loop over array    :", tock - tock3)
print("Total time         :", tock - tick)


# mda load and run
tick = time.time()
u = mda.Universe(pdb, xtc, in_memory=True)
tock1 = time.time()  # load xtc into mem
K_atoms = u.select_atoms("name POT")
O_atoms = u.select_atoms("((resid 130 and resname ILE) or (resid 239 and resname VAL)) and name O")
tock2 = time.time()  # translate selection
reslist_mda = []
for ts in u.trajectory:
    reslist_mda.append(K_atoms.positions[:, 2] - np.mean(O_atoms.positions[:, 2]))
tock = time.time()
print("\n# mdanalysis in memory #########")
print("Loading time       :", tock1 - tick)
print("translate selection:", tock2 - tock1)
print("Loop over array    :", tock - tock2)
print("Total time         :", tock - tick)




print(reslist_mdtraj[1][:5])
print(reslist_mda[1][:5])
print(reslist_mdai[1][:5])

# mdanalysis iter      : 11.10 s
# mdanalysis in memory : 11.81 s
# mdtraj in memory     : 21.38 s
