from simtk import openmm, unit
from simtk.openmm import app
import numpy as np

from copy import deepcopy
# from tqdm import tqdm

# import mdtraj
# import nglview


# def write_pdb(topology, positions, filename):
#     with open(filename, 'w') as outfile:
#         app.PDBFile.writeFile(topology, positions, outfile)

# def make_view(topology, positions):
#     mdtraj_aux_topology = mdtraj.Topology.from_openmm(topology)
#     traj_aux = mdtraj.Trajectory(positions/unit.nanometers, mdtraj_aux_topology)
#     view = nglview.show_mdtraj(traj_aux)
#     view.clear()
#     view.add_ball_and_stick('all')
#     view.center()
#     return view

global forcefield
global pH

class macromolecule():

    def __init__(self, pdb_file=None, forcefield=None, pH=7.0, addHs=True, center=False):


        self.pdb_file    = None
        self.forcefield  = None
        self.pH          = None
        self.modeller    = None
        self.topology    = None
        self.positions   = None
        self.n_atoms     = None
        self.__addHs_log = None

        if pdb_file:
            pdb_aux  = app.PDBFile(pdb_file)
            self.modeller = app.Modeller(pdb_aux.topology, pdb_aux.positions)
            self.forcefield = app.ForceField(forcefield)
            self.pH         = pH
            if addHs:
                self.__addHs_log = self.modeller.addHydrogens(self.forcefield, pH=self.pH)
            self.topology  = self.modeller.getTopology()
            self.positions = self.modeller.getPositions()
            self.n_atoms   = len(self.positions)

        pass


class receptor(macromolecule):

    pass


class ligand(macromolecule):

    pass

class complex(macromolecule):

    def __init__(self, receptor=None, ligand=None, offset=None):

        self.modeller   = app.Modeller(receptor.topology, receptor.positions)
        self.modeller.add(ligand.topology, ligand.positions)
        self.topology   = self.modeller.getTopology()
        self.positions  = self.modeller.getPositions()

        self.receptor   = receptor
        self.ligand     = ligand

    def make_MMcontext():

        return MMcontext(self.complex)

class MMcontext:

    def __init__(self, complex=None):

        self.complex=None
        self.integrator=None
        self.system=None
        self.context=None

        self.complex    = complex
        self.integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        self.system     = forcefield.createSystem(self.complex.topology, nonbondedMethod=app.NoCutoff,
                                                     constraints=app.HBonds, implicitSolvent=app.OBC2)
        self.context    = openmm.Context(self.system, self.integrator)

    def eval_potential_energy():
        return self.context.getState(getEnergy=True).getPotentialEnergy()

    def set_ligand_positions(positions=None):

        new_positions=deepcopy(self.complex.positions)
        for ind_atom in range(self.complex.ligand.n_atoms):
            new_positions[self.complex.receptor.n_atoms+ind_atom]=positions[ind_atom]

        self.context.setPositions(new_positions)
        pass

    def translate_ligand(translation=None):

        new_positions=deepcopy(self.complex.positions)
        for ind_atom in range(self.complex.ligand.n_atoms):
            new_positions[self.complex.receptor.n_atoms+ind_atom]=new_positions[self.complex.receptor.n_atoms+ind_atom]+translation

        self.context.setPositions(new_positions)
        pass

    def rotate_ligand(rotation=None):
        pass

