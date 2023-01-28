# 044-ion-count
tools for counting ion permeation

## 
mdtraj>=1.9.6 # 1.9.4 won't work
```bash
cd ./test/03-find_SF_from_PDB/04-NaK2K-more/

../../../count_ion.py \
  -pdb em.pdb \
  -xtc -xtc fix_atom_c_kpro.xtc \
  -K POT -volt 300 -chunk 1000 \
  -SF_seq THR VAL GLY TYR GLY > k_Cylinder_02.out
  

../../../count_ion_SF.py \
    -pdb em.pdb \
    -xtc fix_atom_c_kpro.xtc \
    -K POT -volt 300 -chunk 1000 \
    -SF_seq THR VAL GLY TYR GLY > k_Cylinder_03.out

# source GMXRC(2022.4) before using the c++ version
../../../count_ion_SF_cpp.py -pdb em.pdb -xtc fix_atom_c_kpro.xtc -SF_seq THR VAL GLY TYR GLY -K POT > count_ion_SF_cpp/count.out

match_result.py \
    -perm_up   02-xtck_hybrid_02/perm_up.dat \
    -perm_down 02-xtck_hybrid_02/perm_down.dat \
    -cylinder 03-Cylinkder/k_Cylinder_03.out > 03-Cylinkder/match02_03_03.dat
```
![permeation](044-ion-couting.png "permeation definition")