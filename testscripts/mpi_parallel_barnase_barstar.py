import os
import sys
import numpy as np
sys.path.append('/home/diego/Trabajo/Proyectos/Pynterpred_devel/')
import pynterpred as pnt
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

receptor = pnt.Receptor('testsystems/Barnase-Barstar/pdbs/Barnase.pdb','amber14-all.xml',pH=7.0)
ligand   = pnt.Ligand('testsystems/Barnase-Barstar/pdbs/Barstar.pdb','amber14-all.xml',pH=7.0)
context  = pnt.MMContext(receptor,ligand)
region   = pnt.Region(receptor, ligand, delta_x=0.25, nside=8)

regions_list=region.split_in_subregions(size)
my_region=regions_list[rank]
my_docker=pnt.Docker(context,my_region)

my_docker.evaluation()

total_energies = comm.gather(my_docker.potential_energies, root=0)

if rank==0:
    docker=pnt.Docker(context,region)
    docker.potential_energies=list(np.concatenate(total_energies))
    docker.dump_energies('pes_barnase_barstar.dill')
