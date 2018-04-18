import pynterpred as pnt
from simtk import unit

pnt.pH=7.0
pnt.forcefield='amber14-all.xml'

receptor = pnt.receptor('testsystems/E9034A_ETEC/longus_E9034A_ETEC.pdb')
ligand = pnt.ligand('testsystems/E9034A_ETEC/longus_E9034A_ETEC.pdb')

molcomplex = pnt.make_complex(receptor,ligand)


