import time
import numpy as np
import kinnetmt as knmt
from mdtraj import Trajectory as _mdtraj_trajectory, Topology as _mdtraj_topology
from nglview import show_mdtraj as _nv_show_mdtraj
from mpi4py import MPI
from simtk import unit
from copy import deepcopy
import quaternion
import dill as pickle
import tqdm

class Docker:

    def __init__ (self,mmcontext=None,region=None,mpi_comm=None):

        self.mmcontext = mmcontext
        self.region    = region
        self.potential_energies = None
        self.mpi_comm = None
        self.mpi_rank = None
        self.mpi_size = None
        self.potential_energy_uncoupled = None
        self._energy_units = None

        if mmcontext is not None:
            self.potential_energy_uncoupled = mmcontext.get_potential_energy_uncoupled_complex()
            self._energy_units = self.potential_energy_uncoupled.unit

        if mpi_comm is not None:
            self.mpi_comm = mpi_comm
            self.mpi_rank = mpi_comm.Get_rank()
            self.mpi_size = mpi_comm.Get_size()

        del(mmcontext,region)

        pass

    def get_conformations(self,centers_indices=None, rotations_indices=None, nodes_labels=None):

        if nodes_labels is not None:

            if type(nodes_labels) is str:
                tmp_center_rotation = node_label[1:-1].split(',')
                centers_indices = np.array(int(tmp_center_rotation[0]))
                rotations_indices = np.array(int(tmp_center_rotation[0]))

        tmp_centers = self.region.centers[centers_indices]
        tmp_rotations = self.region.rotations[rotations_indices]

        tmp_molcomplex = self.mmcontext.get_molcomplex()

        list_positions=[]
        for center,rotation in zip(tmp_centers,tmp_rotations):
            self.mmcontext.make_conformation(center*unit.nanometer,rotation)
            tmp_positions = self.mmcontext.get_positions(molcomplex=True, conformation='context', centered=False)
            list_positions.append(tmp_positions)

        tmp_molcomplex.set_positions(list_positions)

        return tmp_molcomplex

    def show_conformations(self,centers_indices=None, rotations_indices=None, nodes_labels=None):

        tmp_molcomplex = self.get_conformations(centers_indices, rotations_indices, nodes_labels)
        tmp_mdtraj_topol = _mdtraj_topology.from_openmm(tmp_molcomples.topology)
        tmp_mdtraj_traj = _mdtraj_trajectory(tmp_molcomplex,tmp_mdtraj_topol)
        tmp_view = _nv_show_mdtraj(tmp_mdtraj_traj)
        del(tmp_molcomplex, tmp_mdtraj_topol, tmp_mdtraj_traj)
        return tmp_view

    def evaluation(self,verbose=False):

        if self.mpi_comm is not None:

            list_centers_indices = np.array_split(np.arange(self.region.num_centers,dtype=int),
                                                  self.mpi_size)[self.mpi_rank]
            centers_iteration = self.region.centers[list_centers_indices]

            if verbose:
                center_start = list_centers_indices[0]
                center_end = list_centers_indices[-1]

            del(list_centers_indices)

        else:
            centers_iteration = self.region.centers

        if verbose:
            if self.mpi_comm is None:
                centers_iteration = tqdm.tqdm(centers_iteration)

        tmp_energies=[]
        time_start = time.time()
        for center in centers_iteration:
            for rotation in self.region.rotations:
                self.mmcontext.make_conformation(center*unit.nanometer,rotation)
                tmp_energies.append(self.mmcontext.get_potential_energy()._value)

        time_taken_iters = time.time()-time_start

        if self.mpi_comm is not None:

            self.mpi_comm.Barrier()
            time_start = time.time()
            self.potential_energies = self.mpi_comm.gather(tmp_energies, root = 0)
            del(tmp_energies)
            self.mpi_comm.Barrier()
            time_taken_collecting = time.time()-time_start
            if self.mpi_rank is 0:
                self.potential_energies = np.concatenate(self.potential_energies) * self._energy_units

            if verbose:
                print('In mpi rank', self.mpi_rank, 'centers from', center_start, 'to', center_end,
                      'in', np.round(time_taken_iters,2), 'seconds (',
                      np.round((center_end-center_start)/time_taken_iters,2),' its/sec)')

            if verbose and (self.mpi_rank == 0) :
                print('Time to collect potential energies:',np.round(time_taken_collecting,6))

        else:
            self.potential_energies=np.array(tmp_energies)*self._energy_units
            del(tmp_energies)

    def save(self,filename=None):

        with open(filename, 'wb') as pickle_file:
            pickle.dump(self.mmcontext.receptor, pickle_file)
            pickle.dump(self.mmcontext.ligand, pickle_file)
            pickle.dump(self.mmcontext.system, pickle_file)
            pickle.dump(self.mmcontext._units, pickle_file)
            pickle.dump(self.mmcontext._integrator, pickle_file)
            pickle.dump(self.region.centers, pickle_file)
            pickle.dump(self.region.ijk_centers, pickle_file)
            pickle.dump(self.region.nside, pickle_file)
            pickle.dump(quaternion.as_float_array(self.region.rotations), pickle_file)
            pickle.dump(self.potential_energies, pickle_file)
            pickle.dump(self.potential_energy_uncoupled, pickle_file)
            pickle.dump(self._energy_units, pickle_file)

        pickle_file.close()
        del(pickle_file)
        pass

    def get_PotentialEnergyNetwork(self,form='native.PotentialEnergyNetwork'):

        tmp_PotentialEnergyNetwork = self.region.make_ConformationalNetwork()

        for ii,jj in zip(list(tmp_PotentialEnergyNetwork.nodes), self.potential_energies):
            tmp_PotentialEnergyNetwork.nodes[ii]['Potential_Energy']=jj

        if form=='native.PotentialEnergyNetwork':
            tmp_net = knmt.load(tmp_PotentialEnergyNetwork,'native.PotentialEnergyNetwork')
        elif form=='networkx.Graph':
            tmp_net = tmp_PotentialEnergyNetwork

        return tmp_net

    #def get_potential_energy_1D_landscape(self):
    #    if self.PotentialEnergyNetwork is None:
    #        self.make_PotentialEnergyNetwork()
    #    tmp_xx = self.Native_PotentialEnergyNetwork.get_landscape_bottom_up()
    #    tmp_potential_energies = self.Native_PotentialEnergyNetwork.potential_energies * self._energy_units
    #    return tmp_xx, tmp_potential_energies


#    def _remove_forbidden_region(self,uncoupled_distance=100000.0*unit.nanometer):
#
#        self.mmcontext.center_ligand(center=[1.0,0.0,0.0]*uncoupled_distance)
#        uncoupled_energy= self.mmcontext.get_potential_energy()
#        nodes_to_remove = [node for node, attributes in self.region.net.nodes(data=True) if attributes['Potential_Energy']>uncoupled_energy]
#        print(len(nodes_to_remove))
#        self.region.net.remove_nodes_from(nodes_to_remove)


