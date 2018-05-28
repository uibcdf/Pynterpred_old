import numpy as np
from mpi4py import MPI
from simtk import unit
from copy import deepcopy
import dill as pickle
import tqdm

class Docker:

    def __init__ (self,mmcontext=None,region=None,pickle_file=None):
        self.mmcontext = mmcontext
        self.region    = region
        self.potential_energies =None
        pass

    def evaluation(self):
        tmp_energies=[]
        for rotation in tqdm.tqdm(self.region.rotations):
            for center in self.region.centers:
                self.mmcontext.make_conformation(center*unit.nanometer,rotation)
                tmp_energies.append(self.mmcontext.get_potential_energy()._value)

        self.potential_energies=tmp_energies
        del(tmp_energies)
        self.region._set_nodes_attribute('Potential_Energy',self.potential_energies)
        self._remove_forbidden_region()

    def dump_energies(self,pickle_file=None):
        tmp_file=open(pickle_file,"wb")
        tmp_file.write(pickle.dumps(self.potential_energies))
        tmp_file.close()

    def _remove_forbidden_region(self,uncoupled_distance=100000.0*unit.nanometer):

        self.mmcontext.center_ligand(center=np.array([1.0,0.0,0.0])*uncoupled_distance)
        uncoupled_energy= self.mmcontext.get_potential_energy()
        nodes_to_remove = [node for node, attributes in self.region.net.nodes(data=True) if attributes['Potential_Energy']>uncoupled_energy._value]
        self.region.net.remove_nodes_from(nodes_to_remove)

    # def load(self,pickle_file=None):
        # tmp_file   = open(pickle_file, "rb")
        # tmp_docker = pickle.load(tmp_file)
        # self.mmcontext = tmp_docker.mmcontext
        # self.region    = tmp_docker.region
        # self.potential_energies = tmp_docker.potential_energies
        # del(tmp_docker)
        # tmp_file.close()

def _run_single_docker(docker):
    docker.evaluation()
    tmp_energies=docker.potential_energies
    return tmp_energies


def docking (mmcontext=None, region=None, num_dockers=1, platform='CPUs',ipp_client=None):

    if num_dockers==1:
        return _run_single_docker(Docker(mmcontext,region))
    else:
        if (ipp_client is not None) and (__IPYTHON__):
            print('not implemented yet')
            #    #print(len(ipp_client))
            #    #print('profile:',ipp_client.profile)
            #    #print('IDs:',ipp_client.ids)
            #    regions_list=region.split_in_subregions(num_dockers)
            #    dockers_list=[Docker(mmcontext,ii) for ii in regions_list]
            #    #resultado=ipp_client[:].map_sync(lambda x: _run_single_docker(x),dockers_list)
        else:
            print('not implemented yet')

