import sys
import time
from simtk import openmm, unit
from simtk.openmm import app
import numpy as np
import quaternion
from . import utils as utils
from copy import deepcopy
import dill as pickle

class _Units():

    def __init__(self):

        self.length=unit.nanometers
        self.time  =unit.picoseconds

class Macromolecule():

    def __init__(self, pdb_file=None, forcefield=None, pH=7.0, addHs=True, center=True,
                 mpi_comm=None):

        self._units          = _Units()
        self.pdb_file        = None
        self.forcefield      = None
        self.pH              = None
        self.modeller        = None
        self.topology        = None
        self.positions       = None
        self.n_atoms         = None
        self.__addHs_log     = None

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

            self._positions_centered=utils.geometrical_center_in_origin(self)
            if center:
                self.set_positions(self._positions_centered)

            if mpi_comm is not None:
                self.equal_positions_across_MPI_Universe(mpi_comm)

    def set_positions(self,positions):

        self.modeller.positions  = deepcopy(positions)
        self.positions           = self.modeller.getPositions()

    def get_positions(self,positions):

        return self.positions

    def equal_positions_across_MPI_Universe(self,mpi_comm=None):

        _my_rank = mpi_comm.Get_rank()

        if _my_rank == 0:
            universal_positions = self.positions._value
        else:
            _positions_shape = self.positions._value.shape
            _positions_dtype = self.positions._value.dtype
            universal_positions = np.empty(_positions_shape, dtype=_positions_dtype)

        mpi_comm.Bcast(universal_positions, root=0)

        if _my_rank != 0:
            _positions_unit = self.positions.unit
            self.set_positions(universal_positions*_positions_unit)
            del(_positions_shape, _positions_dtype, _positions_unit)

        del(universal_positions, _my_rank)
        mpi_comm.Barrier()

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

    def get_MMContext(self):
        return MMContext(self.receptor, self.ligand)

class MMContext:

    def __init__(self, receptor=None, ligand=None):

        self._units      = _Units()
        self._integrator = "VerletIntegrator"

        self.receptor   = None
        self.ligand     = None
        self.molcomplex = None
        self._begins_receptor = None
        self._begins_ligand   = None
        self.system     = None
        self.modeller   = None
        self.context    = None

        if (receptor is not None) and (ligand is not None):
            self.receptor   = receptor
            self.ligand     = ligand
            self.molcomplex = MolComplex(receptor,ligand)
            self._begins_receptor = 0
            self._begins_ligand   = self.receptor.n_atoms
            self.modeller   = self.molcomplex.modeller
            self.forcefield = receptor.forcefield
            self.system     = self.forcefield.createSystem(self.molcomplex.topology, nonbondedMethod=app.NoCutoff) # constraints=app.HBonds, implicit=)
            if self._integrator == "VerletIntegrator": #seguro que no hace falta...
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

        tmp_macromolecule = MolComplex(self.receptor,self.ligand)

        if conformation == 'original':
            pass
        elif conformation == 'context':
            tmp_macromolecule.set_positions(self.context.getState(getPositions=True).getPositions())

        return tmp_macromolecule

    def get_positions(self,molcomplex=True,receptor=False,ligand=False,conformation='context',centered=False):

        if conformation == 'original':

            if molcomplex:
                if centered:
                    print('not yet')
                else:
                    print('not yet')

            if receptor:
                if centered:
                    tmp_positions_receptor = self.receptor._positions_centered
                else:
                    tmp_positions_receptor = self.receptor.positions

            if ligand:
                if centered:
                    tmp_positions_ligand = self.ligand._positions_centered
                else:
                    tmp_positions_ligand = self.ligand._positions_centered

        elif conformation == 'context':

            tmp_positions_molcomplex=self.context.getState(getPositions=True).getPositions()

            if molcomplex:
                if centered:
                    tmp_positions_molcomplex=utils.center_positions_in_cartesian_origin(tmp_positions_molcomplex)

            if receptor:
                tmp_positions_receptor  =tmp_positions_molcomplex[:self._begins_ligand]
                if centered:
                    tmp_positions_receptor = utils.center_positions_in_cartesian_origin(tmp_positions_receptor)

            if ligand:
                tmp_positions_ligand    =tmp_positions_molcomplex[self._begins_ligand:]
                if centered:
                    tmp_positions_ligand = utils.center_positions_in_cartesian_origin(tmp_positions_ligand)

        tmp_out=[]
        if molcomplex:
            tmp_out.append(tmp_positions_molcomplex)
        if receptor:
            tmp_out.append(tmp_positions_receptor)
        if ligand:
            tmp_out.append(tmp_positions_ligand)

        if len(tmp_out)==1:
            return tmp_out[0]
        elif len(tmp_out):
            return tmp_out

    def set_positions(self,positions=None,molcomplex=None,receptor=None,ligand=None,conformation='context'):
        if positions is None:
            if conformation=='context':
                if (receptor is not None) or (ligand is not None):
                    tmp_positions=self.get_positions(molcomplex=True,receptor=False,ligand=False)
                    if receptor is not None:
                        tmp_positions[:self._begins_ligand]=receptor
                    if ligand is not None:
                        pass
                        tmp_positions[self._begins_ligand:]=ligand
                    self.context.setPositions(tmp_positions)
                elif molcomplex is not None:
                    self.context.setPositions(molcomplex)
            elif conformation=='original':
                print('Not yet')
        else:
                print('Not yet')


    def get_potential_energy(self):
        return self.context.getState(getEnergy=True).getPotentialEnergy()

    def get_energy_units(self):

        tmp_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
        return tmp_energy.unit

    def get_potential_energy_uncoupled_complex(self):

        return self.get_potential_energy_ligand("original") + self.get_potential_energy_receptor("original")

    def get_potential_energy_ligand(self,conformation="original"):

        tmp_ligand = self.get_ligand(conformation)
        tmp_system = tmp_ligand.forcefield.createSystem(tmp_ligand.topology, nonbondedMethod=app.NoCutoff) # constraints=app.HBonds, implicit=)
        tmp_context = openmm.Context(tmp_system, openmm.VerletIntegrator(1.0 * unit.femtoseconds))
        tmp_context.setPositions(tmp_ligand.positions)
        tmp_pe = tmp_context.getState(getEnergy=True).getPotentialEnergy()
        del(tmp_ligand, tmp_system, tmp_context)
        return tmp_pe

    def get_potential_energy_receptor(self,conformation="original"):

        tmp_receptor = self.get_receptor(conformation)
        tmp_system = tmp_receptor.forcefield.createSystem(tmp_receptor.topology, nonbondedMethod=app.NoCutoff) # constraints=app.HBonds, implicit=)
        tmp_context = openmm.Context(tmp_system, openmm.VerletIntegrator(1.0 * unit.femtoseconds))
        tmp_context.setPositions(tmp_receptor.positions)
        tmp_pe = tmp_context.getState(getEnergy=True).getPotentialEnergy()
        del(tmp_receptor, tmp_system, tmp_context)
        return tmp_pe



    def center_ligand(self,center=None,conformation='original'):

        tmp_positions = self.get_positions(ligand=True,conformation=conformation,centered=True)
        tmp_positions += center
        self.set_positions(ligand=tmp_positions)

    def rotate_ligand(self,qrotor=None,conformation='original'):

        tmp_ligand_positions = self.get_positions(ligand=True,conformation=conformation,centered=True)

        if conformation == 'original':
            tmp_ligand_positions=quaternion.rotate_vectors(qrotor,tmp_ligand_positions)*self._units.length
        elif conformation =='context':
            print("not working yet")

        self.set_positions(ligand=tmp_ligand_positions)

    def translate_ligand(self,translation=None):

        aux_positions=self.get_positions(ligand=True)
        aux_positions+=translation
        self.set_positions(ligand=aux_positions)

        pass

    def make_conformation(self,center=None,qrotor=None):
        tmp_positions_ligand=quaternion.rotate_vectors(qrotor,self.ligand._positions_centered)+center._value
        self.context.setPositions(np.vstack([self.receptor._positions_centered,tmp_positions_ligand]))

    def make_view(self,positions=None):

        return utils.make_view(self.get_molcomplex(conformation="context"),positions)

def predict (receptor_pdb_file=None, ligand_pdb_file=None, forcefield=None, pH=7.0,
               delta_x=0.5, nside=5, mpi_comm = None, verbose = False):

    from .region import Region as _Region
    from .docking import Docker as _Docker

    i_am_logger_out = False
    if verbose:
        if mpi_comm is None:
            i_am_logger_out = True
        elif (mpi_comm.Get_rank() == 0):
            i_am_logger_out = True

    if i_am_logger_out:
        print("Setting up the mechanical molecular context... ", end="", flush=True)
        time_start = time.time()
    tmp_receptor = Receptor(receptor_pdb_file,forcefield,pH, mpi_comm=mpi_comm)
    tmp_ligand   = Ligand(ligand_pdb_file,forcefield, pH, mpi_comm=mpi_comm)
    tmp_context  = MMContext(tmp_receptor,tmp_ligand)
    if i_am_logger_out:
        print(np.round(time.time()-time_start,2),'secs')
        time_start = time.time()
        print("Setting up the evaluation region... ", end="", flush=True)
    tmp_region   = _Region(tmp_receptor, tmp_ligand, delta_x=delta_x, nside=nside)
    tmp_docker  = _Docker(tmp_context, tmp_region, mpi_comm=mpi_comm)
    if i_am_logger_out:
        print(np.round(time.time()-time_start,2),'secs')
        print("Evaluation of", tmp_region.num_centers*tmp_region.num_rotations , "different relative orientations started...")
        time.sleep(0.5)
        time_start = time.time()
    tmp_docker.evaluation(verbose=verbose)
    if i_am_logger_out:
        time_finish = time.time()
        time.sleep(0.5)
        print("... done in", np.round(time_finish-time_start,2),'secs')
        print("--- FINISHED ---")
        print("Complex at infinite distance with Potential Energy:",
              tmp_docker.potential_energy_uncoupled)
        print("Best relative orientation with Potential Energy:",
              tmp_docker.potential_energies.min())
    del(tmp_receptor, tmp_ligand, tmp_region, tmp_context)
    del(_Region, _Docker)
    return tmp_docker

def load(filename=None):

        if filename.endswith('.dpkl'):

            from .docking import Docker as _Docker
            from .region import Region as _Region

            tmp_docker = _Docker()
            tmp_docker.mmcontext = MMContext()
            tmp_docker.region = _Region()

            with open(filename, 'rb') as pickle_file:
                tmp_docker.mmcontext.receptor = pickle.load(pickle_file)
                tmp_docker.mmcontext.ligand = pickle.load(pickle_file)
                tmp_docker.mmcontext.system = pickle.load(pickle_file)
                tmp_docker.mmcontext._units = pickle.load(pickle_file)
                tmp_docker.mmcontext._integrator = pickle.load(pickle_file)
                tmp_docker.region.centers = pickle.load(pickle_file)
                tmp_docker.region.ijk_centers = pickle.load(pickle_file)
                tmp_docker.region.nside = pickle.load(pickle_file)
                tmp_docker.region.rotations = quaternion.as_quat_array(pickle.load(pickle_file))
                tmp_docker.potential_energies = pickle.load(pickle_file)
                tmp_docker.potential_energy_uncoupled = pickle.load(pickle_file)
                tmp_docker._energy_units = pickle.load(pickle_file)

            tmp_docker.mmcontext.molcomplex = MolComplex(tmp_docker.mmcontext.receptor, tmp_docker.mmcontext.ligand)
            tmp_docker.mmcontext._begins_receptor = 0
            tmp_docker.mmcontext._begins_ligand   = tmp_docker.mmcontext.receptor.n_atoms
            tmp_docker.mmcontext.modeller   = tmp_docker.mmcontext.molcomplex.modeller
            tmp_docker.mmcontext.forcefield = tmp_docker.mmcontext.receptor.forcefield
            if tmp_docker.mmcontext._integrator == "VerletIntegrator":
                tmp_docker.mmcontext.context    = openmm.Context(tmp_docker.mmcontext.system, openmm.VerletIntegrator(1.0 * unit.femtoseconds))
            tmp_docker.mmcontext.context.setPositions(np.vstack([tmp_docker.mmcontext.receptor.positions,
                                                                 tmp_docker.mmcontext.ligand.positions]))

            tmp_docker.region.num_centers=tmp_docker.region.centers.shape[0]
            tmp_docker.region.num_rotations=tmp_docker.region.rotations.shape[0]

            del(_Docker,_Region,pickle_file)

            return tmp_docker

