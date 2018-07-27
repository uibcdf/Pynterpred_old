import time
import numpy as np
from mpi4py import MPI
from simtk import unit
from copy import deepcopy
import tqdm

class Docker:

    def __init__ (self,mmcontext=None,region=None,mpi_comm=None):

        self.mmcontext = mmcontext
        self.region    = region
        self.potential_energies = None
        self.PotentialEnergyNetwork = None
        self.mpi_comm = None
        self.mpi_rank = None
        self.mpi_size = None

        if mpi_comm is not None:
            self.mpi_comm = mpi_comm
            self.mpi_rank = mpi_comm.Get_rank()
            self.mpi_size = mpi_comm.Get_size()

        pass

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
                self.potential_energies = np.concatenate(self.potential_energies)

            if verbose:
                print('In mpi rank', self.mpi_rank, 'centers from', center_start, 'to', center_end,
                      'in', time_taken_iters, 'seconds (', (center_end-center_start)/time_taken_iters,' its/sec)')

            if verbose and (self.mpi_rank == 0) :
                print('Time to collect potential energies:',time_taken_collecting)

        else:
            self.potential_energies=tmp_energies
            del(tmp_energies)

    def make_PotentialEnergyNetwork(self):

        self.PotentialEnergyNetwork = self.region.make_ConformationalNetwork()

        for ii,jj in zip(list(self.PotentialEnergyNetwork.nodes), self.potential_energies):
            self.PotentialEnergyNetwork.nodes[ii]['Potential_Energy']=jj

    def get_potential_energy_1D_landscape(self):

        if self.PotentialEnergyNetwork is None:
            self.make_PotentialEnergyNetwork()

        tmp_net = knmt.load(self.net,'native.PotentialEnergyNetwork')
        tmp_xx = tmp_net.get_landscape_bottom_up()
        tmp_potential_energies = tmp_net.potential_energies
        del(tmp_net)
        return tmp_xx, tmp_potential_energies

#    def _remove_forbidden_region(self,uncoupled_distance=100000.0*unit.nanometer):
#
#        self.mmcontext.center_ligand(center=[1.0,0.0,0.0]*uncoupled_distance)
#        uncoupled_energy= self.mmcontext.get_potential_energy()
#        nodes_to_remove = [node for node, attributes in self.region.net.nodes(data=True) if attributes['Potential_Energy']>uncoupled_energy]
#        print(len(nodes_to_remove))
#        self.region.net.remove_nodes_from(nodes_to_remove)


