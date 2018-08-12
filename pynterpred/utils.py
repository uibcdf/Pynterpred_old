import numpy as np
import mdtraj as md
import nglview
from os.path import dirname as _dirname, join as _join, abspath as _abspath
from copy import deepcopy
from simtk.openmm import app

test_systems_path=_abspath(_join(_dirname(__file__),'../examples/testsystems'))

def copy_molcomplex(molcomplex):

    tmp_complex = deepcopy(molcomplex)
    tmp_complex.modeller = app.Modeller(molcomplex.topology, molcomplex.positions)
    tmp_complex.topology = tmp_complex.modeller.getTopology()

    return tmp_complex

def make_view(topology,positions):

    if type(positions) == list:
        positions = np.array([tmp_pos._value for tmp_pos in positions])*positions[0].unit

    topology_mdtraj   = md.Topology.from_openmm(topology)
    positions_mdtraj  = positions._value
    aux_traj=md.Trajectory(positions_mdtraj, topology_mdtraj)

    return nglview.show_mdtraj(aux_traj)

def sasa_atoms(macromolecule,probe_radius=0.227,n_sphere_points=5000):
    #0.227 rdw NA
    topology_mdtraj   = md.Topology.from_openmm(macromolecule.topology)
    positions_mdtraj  = macromolecule.positions._value
    aux_traj=md.Trajectory(positions_mdtraj, topology_mdtraj)
    sasa = md.shrake_rupley(aux_traj,probe_radius=probe_radius, n_sphere_points=n_sphere_points, mode='atom')

    return sasa

def furthest_accessible_atom_to_center(macromolecule,probe_radius=0.227,n_sphere_points=5000,sasa_threshold=0.01):

    sasa = sasa_atoms(macromolecule,probe_radius=probe_radius, n_sphere_points=n_sphere_points)
    list_atoms_exposed=np.nonzero(sasa>=0.01)[1]

    positions   = np.array(macromolecule.positions._value)
    geom_center = positions.mean(0)
    vect_dists  = positions[list_atoms_exposed] - geom_center
    dists       = np.linalg.norm(vect_dists,axis=1)

    arg_exposed    = dists.argmax()

    return list_atoms_exposed[arg_exposed], dists[arg_exposed]

def closest_accessible_atom_to_center(macromolecule,probe_radius=0.227,n_sphere_points=5000,sasa_threshold=0.01):

    sasa = sasa_atoms(macromolecule,probe_radius=probe_radius, n_sphere_points=n_sphere_points)
    list_atoms_exposed=np.nonzero(sasa>=0.01)[1]

    positions   = np.array(macromolecule.positions._value)
    geom_center = positions.mean(0)
    vect_dists  = positions[list_atoms_exposed] - geom_center
    dists       = np.linalg.norm(vect_dists,axis=1)

    arg_exposed    = dists.argmin()

    return list_atoms_exposed[arg_exposed], dists[arg_exposed]

def heavy_atoms_indices(topology):

    tmp_list=[]
    ii=0
    for atom in topology.atoms():
        if atom.element.symbol != 'H':
            tmp_list.append(ii)
        ii+=1

    return np.array(tmp_list)

def ca_atoms_indices(topology):

    tmp_list=[]
    ii=0
    for atom in topology.atoms():
        if atom.name == 'CA':
            tmp_list.append(ii)
        ii+=1

    return np.array(tmp_list)


def geometrical_center(positions,atoms_list):

    tmp_unit = positions.unit
    tmp_positions = np.array(positions._value)[atoms_list]
    geometrical_center = tmp_positions.mean(axis=0)*tmp_unit
    del(tmp_positions,tmp_unit)
    return geometrical_center

def rgb2hex(rgb):

    r = int(rgb[0]) ; g = int(rgb[1]) ; b = int(rgb[2])
    hex = "0x{:02x}{:02x}{:02x}".format(r,g,b)
    return hex

def colorscale2hex(values,color_min=[255,0,0],color_max=[255,255,255],value_min=None,value_max=None,num_bins=254):

    if not value_min:
        value_min=values.min()
    if not value_max:
        value_max=values.max()

    color_bin=(np.array(color_max)-np.array(color_min))/float(num_bins)
    scale_bin=(value_max-value_min)/float(num_bins)

    colors_hex=[]
    for val in values:
        val_bin=(val-value_min)/scale_bin
        rgb_from_val=(color_bin*val_bin).astype(int)+np.array(color_min)
        colors_hex.append(rgb2hex(rgb_from_val))

    return colors_hex
