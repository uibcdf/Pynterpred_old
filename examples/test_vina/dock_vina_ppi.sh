obabel -i pdb Barnase.pdb  -o pdbqt -O Barnase.pdbqt -xr
obabel -i pdb Barstar.pdb  -o pdbqt -O Barstar.pdbqt -xnh
Linux-vina --ligand Barstar.pdbqt --receptor Barnase.pdbqt --center_x 0. --center_y 0. --center_z 0. --size_x 60. --size_y 60. --size_z 60 --exhaustiveness 10
obabel -m -i pdbqt ligand_out.pdbqt -o pdb out_.pdb -xnh
