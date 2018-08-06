import sys
import time
from .molecules import _Units as _Units, MolComplex as _MolComplex
from simtk import openmm, unit
from simtk.openmm import app
import numpy as np
import quaternion
from . import utils as utils
from copy import deepcopy
import dill as pickle

class MMContext:

    def __init__(self, molcomplex=None, forcefield=None,
                 integrator = "VerletIntegrator"):

        self._units      = _Units()
        self._integrator = "VerletIntegrator"

        self.molcomplex = None
        self.forcefield = None
        self.integrator = None
        self.system     = None
        self.context    = None

        if molcomplex is not None:
            self.molcomplex = utils.copy_molcomplex(molcomplex)
        elif (receptor is not None) and (ligand is not None):
            self.molcomplex = _MolComplex(receptor,ligand)

        if forcefield is None:
            self.forcefield = self.molcomplex.forcefield
        else:
            self.forcefield = forcefield

        self.system = self.forcefield.createSystem(self.molcomplex.topology, nonbondedMethod=app.NoCutoff) # constraints=app.HBonds, implicit=)

        self.integrator = integrator

        if self.integrator == "VerletIntegrator": #seguro que no hace falta...
            self.context = openmm.Context(self.system, openmm.VerletIntegrator(1.0 * unit.femtoseconds))

        self.context.setPositions(self.molcomplex.positions)

        pass

    def get_ligand(self):

        tmp_macromolecule = self.molcomplex.get_ligand()
        tmp_positions = self.get_ligand_positions()
        tmp_macromolecule.set_positions(tmp_positions)
        return tmp_macromolecule

    def get_receptor(self):

        tmp_macromolecule = self.molcomplex.get_receptor()
        tmp_positions = self.get_receptor_positions()
        tmp_macromolecule.set_positions(tmp_positions)
        return tmp_macromolecule

    def get_molcomplex(self):

        tmp_macromolecule = utils.copy_molcomplex(self.molcomplex)
        tmp_positions = self.get_molcomplex_positions()
        tmp_macromolecule.set_positions(tmp_positions)
        return tmp_macromolecule

    def get_ligand_positions(self):

        tmp_state = self.context.getState(getPositions=True)
        tmp_positions = tmp_state.getPositions(asNumpy=True)[self.molcomplex.ligand_atom_indices]
        del(tmp_state)
        return tmp_positions

    def get_receptor_positions(self):

        tmp_state = self.context.getState(getPositions=True)
        tmp_positions = tmp_state.getPositions(asNumpy=True)[self.molcomplex.receptor_atom_indices]
        del(tmp_state)
        return tmp_positions

    def get_molcomplex_positions(self):

        tmp_state = self.context.getState(getPositions=True)
        tmp_positions = tmp_state.getPositions(asNumpy=True)[self.molcomplex.complex_atom_indices]
        del(tmp_state)
        return tmp_positions

    def get_positions(self):
        tmp_state = self.context.getState(getPositions=True)
        tmp_positions = tmp_state.getPositions(asNumpy=True)
        return tmp_positions

    def set_ligand_positions(self,positions=None):

        tmp_positions = self.get_positions()
        tmp_positions[self.molcomplex.ligand_atom_indices]=positions
        self.molcomplex.set_ligand_positions(positions)
        self.context.setPositions(tmp_positions)

    def set_receptor_positions(self,positions=None):

        tmp_positions = self.get_positions()
        tmp_positions[self.molcomplex.receptor_atom_indices]=positions
        self.molcomplex.set_receptor_positions(positions)
        self.context.setPositions(tmp_positions)

    def set_molcomplex_positions(self,positions=None):

        tmp_positions = self.get_positions()
        tmp_positions[self.molcomplex.complex_atom_indices]=positions
        self.molcomplex.set_complex_positions(positions)
        self.context.setPositions(tmp_positions)

    def set_positions(self,positions=None):
        self.molcomplex.set_positions(positions)
        self.context.setPositions(positions)

    def get_potential_energy(self):
        return self.context.getState(getEnergy=True).getPotentialEnergy()

    def get_energy_units(self):
        tmp_energy = self.get_potential_energy()
        return tmp_energy.unit

    def get_potential_energy_uncoupled_complex(self):

        return self.get_potential_energy_ligand() + self.get_potential_energy_receptor()

    def get_potential_energy_ligand(self):

        tmp_ligand = self.get_ligand()
        tmp_system = tmp_ligand.forcefield.createSystem(tmp_ligand.topology, nonbondedMethod=app.NoCutoff) # constraints=app.HBonds, implicit=)
        tmp_context = openmm.Context(tmp_system, openmm.VerletIntegrator(1.0 * unit.femtoseconds))
        tmp_context.setPositions(tmp_ligand.positions)
        tmp_pe = tmp_context.getState(getEnergy=True).getPotentialEnergy()
        del(tmp_ligand, tmp_system, tmp_context)
        return tmp_pe

    def get_potential_energy_receptor(self):

        tmp_receptor = self.get_receptor()
        tmp_system = tmp_receptor.forcefield.createSystem(tmp_receptor.topology, nonbondedMethod=app.NoCutoff) # constraints=app.HBonds, implicit=)
        tmp_context = openmm.Context(tmp_system, openmm.VerletIntegrator(1.0 * unit.femtoseconds))
        tmp_context.setPositions(tmp_receptor.positions)
        tmp_pe = tmp_context.getState(getEnergy=True).getPotentialEnergy()
        del(tmp_receptor, tmp_system, tmp_context)
        return tmp_pe

    def get_potential_energy_molcomplex(self):

        tmp_molcomplex = self.get_molcomplex()
        tmp_system = tmp_molcomplex.forcefield.createSystem(tmp_molcomplex.topology, nonbondedMethod=app.NoCutoff) # constraints=app.HBonds, implicit=)
        tmp_context = openmm.Context(tmp_system, openmm.VerletIntegrator(1.0 * unit.femtoseconds))
        tmp_context.setPositions(tmp_molcomplex.positions)
        tmp_pe = tmp_context.getState(getEnergy=True).getPotentialEnergy()
        del(tmp_molcomplex, tmp_system, tmp_context)
        return tmp_pe

    def center_receptor(self,geometrical_center='heavy'):

        if geometrical_center == 'heavy':
            tmp_list = self.molcomplex.receptor._heavy_atoms_indices
        elif geometrical_center == 'CA':
            tmp_list = self.molcomplex.receptor._ca_atoms_indices
        elif geometrical_center == 'All':
            tmp_list = np.arange(self.molcomplex.receptor.n_atoms)

        tmp_positions = self.get_receptor_positions()
        geometrical_center_positions = utils.geometrical_center(tmp_positions,tmp_list)
        tmp_positions = tmp_positions - geometrical_center_positions
        self.set_receptor_positions(tmp_positions)

    def center_ligand(self,geometrical_center='heavy'):

        if geometrical_center == 'heavy':
            tmp_list = self.molcomplex.ligand._heavy_atoms_indices
        elif geometrical_center == 'CA':
            tmp_list = self.molcomplex.ligand._ca_atoms_indices
        elif geometrical_center == 'All':
            tmp_list = np.arange(self.molcomplex.ligand.n_atoms)

        tmp_positions = self.get_ligand_positions()
        geometrical_center_positions = utils.geometrical_center(tmp_positions,tmp_list)
        tmp_positions = tmp_positions - geometrical_center_positions
        self.set_ligand_positions(tmp_positions)

    def rotate_ligand(self,qrotor=None, geometrical_center='heavy', centered=False):

        if geometrical_center == 'heavy':
            tmp_list = self.molcomplex.ligand._heavy_atoms_indices
        elif geometrical_center == 'CA':
            tmp_list = self.molcomplex.ligand._ca_atoms_indices
        elif geometrical_center == 'All':
            tmp_list = np.arange(self.molcomplex.ligand.n_atoms)

        if type(qrotor) != quaternion.quaternion:
            qrotor = quaternion.as_quat_array(qrotor)


        tmp_positions = self.get_ligand_positions()
        tmp_unit = tmp_positions.unit

        if centered:
            tmp_positions=quaternion.rotate_vectors(qrotor,tmp_positions._value)*tmp_unit
        else:
            geometrical_center_positions = utils.geometrical_center(tmp_positions,tmp_list)
            tmp_positions = tmp_positions - geometrical_center_positions
            tmp_positions=quaternion.rotate_vectors(qrotor,tmp_positions._value)*tmp_unit
            tmp_positions = tmp_positions + geometrical_center_positions
            del(geometrical_center_positions)

        self.set_ligand_positions(tmp_positions)
        del(tmp_positions, tmp_unit, tmp_list)


    def translate_ligand(self,translation=None):

        tmp_positions=self.get_ligand_positions()
        tmp_positions+=translation
        self.set_ligand_positions(tmp_positions)
        del(tmp_positions)

    def make_conformation(self,center=None, qrotor=None, geometrical_center='heavy',centered=False):
        if centered == False:
            self.center_ligand(geometrical_center)
        tmp_positions = self.get_ligand_positions()
        tmp_unit = tmp_positions.unit
        tmp_positions=quaternion.rotate_vectors(qrotor,tmp_positions._value)*tmp_unit
        tmp_positions+=center
        self.set_ligand_positions(tmp_positions)

    def get_min_complex_distance(self):
        pass

    def get_rmsd_to_conformation(self,target='original',fit_receptor=True):
        pass


    def get_view(self):

        tmp_molecule = self.get_molcomplex()
        tmp_view = tmp_molecule.get_view()
        del(tmp_molecule)
        return tmp_view
