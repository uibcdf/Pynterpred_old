import sys
import time
from simtk import openmm, unit
from simtk.openmm import app
import numpy as np
import quaternion
from . import utils as utils
from copy import deepcopy
import dill as pickle


# This block needs to be removed when
# pickle docking files are produced with this
# new classes schema
from .molecules import Receptor
from .molecules import Ligand
from .molecules import _Units

def predict (receptor_pdb_file=None, ligand_pdb_file=None, forcefield=None, pH=7.0,
               delta_x=0.5, nside=5, mpi_comm = None, verbose = False):

    from .molecules import Receptor as _Receptor
    from .molecules import Ligand as _Ligand
    from .mmcontext import MMContext as _MMContext
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
    tmp_receptor = _Receptor(receptor_pdb_file,forcefield,pH, mpi_comm=mpi_comm)
    tmp_ligand   = _Ligand(ligand_pdb_file,forcefield, pH, mpi_comm=mpi_comm)
    tmp_context  = _MMContext(tmp_receptor,tmp_ligand)
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

            from .molecules import MolComplex as _MolComplex
            from .docker import Docker as _Docker
            from .region import Region as _Region
            from .mmcontext import MMContext as _MMContext

            tmp_docker = _Docker()
            tmp_docker.mmcontext = _MMContext()
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

            tmp_docker.mmcontext.molcomplex = _MolComplex(tmp_docker.mmcontext.receptor, tmp_docker.mmcontext.ligand)
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

