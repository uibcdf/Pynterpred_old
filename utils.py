import numpy as np


def dist_furthest_atom_surface(macromolecule): #sería mejor sacarlo con una grid de superficie como pymol o lo que sea

    positions   = np.array(macromolecule.positions._value)
    geom_center = positions.mean(0)
    vect_dists  = positions - geom_center
    dists       = np.linalg.norm(vect_dists,axis=1)
    max_dist    = dists.max()

    return max_dist

# Estrategia de movimiento outside-in para descartar

def centers_in_region(region="sphere",distribution="regular_cartesian", rmax=None, num_centers=None):
    '''region: sphere, cube
    distribution=regular_cartesian, regular_polar, uniform
    '''

    if region=="sphere":
        if distribution=="regular_cartesian":

            volume_explored = (4.0/3.0)*np.pi*(rmax**3)
            delta_x    = np.cbrt(volume_explored/num_centers)
            nx         = 2*rmax/delta_x
            x          = np.linspace(-rmax,rmax,delta_x)
            xv, yv, zv = np.meshgrid(x,x,x, indexing='ij')
            for ii in range(nx):
                for jj in range(nx):
                    for kk in range(nx):

