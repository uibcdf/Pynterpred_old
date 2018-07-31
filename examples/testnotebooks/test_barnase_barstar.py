import pynterpred as pnt
from simtk import unit

pnt.set_forcefield('amber14-all.xml',pH=7.0)

receptor = pnt.receptor('testsystems/Barnase-Barstar/Barnase.pdb',center=True)
ligand = pnt.ligand('testsystems/Barnase-Barstar/Barstar.pdb',center=True)
molcomplex = pnt.complex(receptor,ligand)

MMcontext = molcomplex.make_MMcontext()
MMcontext.translate_ligand([3.0,0.0,0.0]*unit.nanometers)
MMcontext.eval_potential_energy()







