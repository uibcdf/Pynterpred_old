import numpy as np
import mdtraj as md
import nglview

def make_view(macromolecule):

    topology_mdtraj   = md.Topology.from_openmm(macromolecule.topology)
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

