import numpy as np
import mdtraj as md
import nglview
from os.path import dirname as _dirname, join as _join

test_systems_path=_join(_dirname(__file__),'../examples/testsystems')

def make_view(macromolecule,positions=None):

    topology_mdtraj   = md.Topology.from_openmm(macromolecule.topology)
    if positions is not None:
        positions_mdtraj = positions
        if type(positions_mdtraj) in [list,tuple]:
            positions_mdtraj=np.array(positions_mdtraj)
    else:
        positions_mdtraj  = macromolecule.positions._value
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

def geometrical_center_in_origin(Macromolecule):

    # with heavy atoms only to avoid lack of reproducilibity when Hs added.

    tmp_unit = Macromolecule.positions.unit

    atoms_symbols = []
    for atom in Macromolecule.topology.atoms():
        atoms_symbols.append(atom.element.symbol)

    atoms_symbols = np.array(atoms_symbols)

    positions_value_heavy_atoms = np.array(Macromolecule.positions._value)[atoms_symbols!='H']
    geometrical_center=positions_value_heavy_atoms.mean(axis=0)*tmp_unit

    del(atoms_symbols,positions_value_heavy_atoms,tmp_unit)

    return (Macromolecule.positions - geometrical_center)

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
