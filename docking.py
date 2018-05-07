from simtk import openmm, unit
from simtk.openmm import app
import numpy as np
import quaternion
import utils
from pathos.multiprocessing import ProcessPool
import time

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


def docking (mmcontext=None, region=None, num_dockers=1):

    tmp_docker=Docker()
    tmp_docker=mmcontext
    tmp_docker.region=region

    tmp_result=None

    if num_dockers>1:

        subregions=region.split_in_subregions(num_dockers)
        dockers   =[Docker(mmcontext,subregion) for subregion in subregions]

        pool = ProcessPool(nodes=num_dockers)
        results = pool.amap(_run_single_docker,dockers)
        while not results.ready():
            time.sleep(5); print(".", end=' ')
        print(results.get())

        # tmp_result=result.get(timeout=1)

    return tmp_result
