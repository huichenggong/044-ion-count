rsync -av --include="*/" \
  --include="0?-*/perm_up.dat" \
  --include="0?-*/perm_down.dat" \
  --include="03-Cylinkder/k_Cylinder.out" \
  --exclude="*" \
/home/chui/E29Project/033-NaK2K/3OUF/12-charge-Charmm/02-C-0.75/900mmol-96HID/05-300mv/ \
/home/chui/E29Project/044-ion-count/test/05-result-match/05-300mv/ -m 
