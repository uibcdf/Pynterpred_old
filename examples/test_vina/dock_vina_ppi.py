from simtk.openmm import app

import sys
sys.path.append('/home/diego/Trabajo/Proyectos/Pynterpred_devel/')
import Pynterpred as pnt

from subprocess import call

comm1='obabel -i pdb Barnase.pdb  -o pdbqt -O Barnase.pdbqt -xr'

return_code = call(comm1,shell=True)
