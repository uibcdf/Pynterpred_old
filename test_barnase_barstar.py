import pynterpred as pnt
from simtk import unit

pnt.pH=7.0
pnt.forcefield='amber14-all.xml'

receptor = pnt.receptor('testsystems/Barnase-Barstar/Barnase.pdb',center=True)
ligand = pnt.ligand('testsystems/Barnase-Barstar/Barstar.pdb',center=True)
molcomplex = pnt.complex(receptor,ligand)

MMcontext = molcomplex.make_MMcontext()
MMcontext.translate_ligand([3.0,0.0,0.0]*unit.nanometers)
MMcontext.eval_potential_energy()







