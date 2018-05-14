import numpy as np
from mpi4py import MPI

from copy import deepcopy

class Docker:

    def __init__ (self,mmcontext=None,region=None):
        self.mmcontext = mmcontext
        self.region    = region
        self.potential_energies =None
        pass

    def evaluation():
        tmp_energies=[]
        for qrotor in self.region.qrotors:
            for center in self.region.centers:
                self.mmcontext.make_conformation(center*unit.nanometer,qrotor)
                tmp_energies.append(self.mmcontext.get_potential_energy()._value)

        self.potential_energies=tmp_energies

def _run_single_docker(docker):
    tmp_energies=docker.evaluation()
    return tmp_energies


def docking (mmcontext=None, region=None, num_dockers=1, platform='CPUs'):

    if num_dockers==1:
        pass

    else:

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        print(rank,size)

    pass
