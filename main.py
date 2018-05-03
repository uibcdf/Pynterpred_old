from simtk import openmm, unit
from simtk.openmm import app
import numpy as np

import utils

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

class Macromolecule():

    def __init__(self, pdb_file=None, forcefield=None, pH=7.0, addHs=True, center=True):


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

            if center:
                geometrical_center=np.array(self.positions._value).mean(axis=0)
                positions_centered= self.positions - geometrical_center*unit.nanometers
                self.modeller.positions=positions_centered
                self.positions=self.modeller.getPositions()

        pass

    def setPositions(self,positions):

        self.modeller.positions  = positions
        self.positions          = self.modeller.getPositions()

        pass

class Receptor(Macromolecule):

    pass


class Ligand(Macromolecule):

    pass

class MolComplex(Macromolecule):

    def __init__(self, receptor=None, ligand=None):
        self.modeller   = app.Modeller(receptor.topology, receptor.positions)
        self.modeller.add(ligand.topology, ligand.positions)
        self.topology   = self.modeller.getTopology()
        self.receptor   = receptor
        self.ligand     = ligand

    def getMMContext(self):
        return MMContext(self.receptor, self.ligand)

class MMContext:

    def __init__(self, receptor,ligand):

        self.receptor   = deepcopy(receptor)
        self.ligand     = deepcopy(ligand)
        self.molcomplex = MolComplex(receptor,ligand)

        self._begins_receptor = 0
        self._begins_ligand   = self.receptor.n_atoms

        self.system     = None
        self.modeller   = None
        self.context    = None

        self.modeller   = self.molcomplex.modeller
        self.forcefield = receptor.forcefield
        self.system     = self.forcefield.createSystem(self.molcomplex.topology, nonbondedMethod=app.NoCutoff) # constraints=app.HBonds, implicit=)
        self.context    = openmm.Context(self.system, openmm.VerletIntegrator(1.0 * unit.femtoseconds))
        self.context.setPositions(np.vstack([self.receptor.positions,self.ligand.positions]))
        pass

    def get_ligand(self,conformation='context'): #'context'

        tmp_macromolecule = deepcopy(self.ligand)

        if conformation == 'original':
            pass
        elif conformation == 'context':
            tmp_macromolecule.setPositions(self.context.getPositions()[self._begins_ligand:])

        return tmp_macromolecule

    def get_receptor(self,conformation='context'):

        tmp_macromolecule = deepcopy(self.receptor)

        if conformation == 'original':
            pass
        elif conformation == 'context':
            tmp_macromolecule.setPositions(self.context.getPositions()[0:self._begins_ligand])

        return tmp_macromolecule

    def get_molcomplex(self,conformation='context'):

        tmp_macromolecule = deepcopy(self.molcomplex)

        if conformation == 'original':
            pass
        elif conformation == 'context':
            tmp_macromolecule.setPositions(self.context.getState(getPositions=True).getPositions())

        return tmp_macromolecule

    def get_potential_energy(self):
        return self.context.getState(getEnergy=True).getPotentialEnergy()

    def get_potential_energy_coupling(self):
        pass


    def center_ligand(self,center=None,conformation='original'):

        if conformation == 'original':
            self.context.setPositions(np.vstack([self.receptor.positions,self.ligand.positions+center]))
        elif conformation =='context':
            pass

        pass

    def rotate_ligand(self,qrotor=None):


        self.context.setPositions(np.vstack([self.receptor.positions,self.ligand.positions]))
        pass

    def translate_ligand(self,translation=None):

        aux_positions=self.context.getState(getPositions=True).getPositions()
        aux_positions[self._begins_ligand:]=aux_positions[self._begins_ligand:]+translation
        self.context.setPositions(aux_positions)

        pass

    def make_view(self):
        return utils.make_view(self.get_molcomplex())


def docking(receptor,ligand):

    #value_non_interacting = docker.non_interacting(receptor,ligand)

    rmax_receptor = utils.dist_furthest_atom_surface(receptor)
    rmax_ligand   = utils.dist_furthest_atom_surface(ligand)
    rmax_complex  = rmax_receptor+rmax_ligand+0.4

    centers     = docker.centers_in_region(region='layer', distribution='regular_cartesian', rmax=rmax_complex) #'regular_polar'
    #quaternions = docker.uniform_quaternions(size=None,random_state)

    #return lista_energias, lista_poses
    return centers
