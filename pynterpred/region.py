import numpy as np
import networkx as nx
import kinnetmt as knmt
import quaternion
import healpy as hp
from . import utils
from sklearn.neighbors import NearestNeighbors

class Region():

    def __init__(self, receptor=None, ligand=None, centers='in_layer', centers_distribution='regular_cartesian',
                 delta_x=0.25, rotations='All', rotations_distribution='healpix', nside=8):

        self.centers=None
        self.ijk_centers=None
        self.num_centers=None

        self.rotations=None
        self.num_rotations=None
        self.nside=None

        if (receptor is not None) and (ligand is not None):

            if centers=='in_layer':
                self.centers_in_layer(centers_distribution, receptor, ligand, delta_x)
            if rotations_distribution=='healpix':
                self.rotations_in_quaternions_region(rotations, rotations_distribution, nside)

        pass

    ### def centers_in_sphere(self,centers_distribution="regular_cartesian", rmax=None, delta_x=None):
    ###     '''
    ###     centers_distribution=regular_cartesian, regular_polar, uniform
    ###     '''

    ###     if centers_distribution=="regular_cartesian":

    ###         volume_explored = (4.0/3.0)*np.pi*(rmax**3)
    ###         num_bins_x_2       = np.int(np.ceil(rmax/delta_x))
    ###         num_bins_x         =2*num_bins_x_2+1
    ###         prov_num_centers =num_bins_x*num_bins_x*num_bins_x
    ###         prov_centers    = np.empty((prov_num_centers,3),dtype=float)
    ###         prov_ijk_centers= np.empty((prov_num_centers,3),dtype=int)

    ###         h=0
    ###         for ii in range(-num_bins_x_2,num_bins_x_2+1):
    ###             for jj in range(-num_bins_x_2,num_bins_x_2+1):
    ###                 for kk in range(-num_bins_x_2,num_bins_x_2+1):
    ###                     prov_ijk_centers[h,:]=[ii,jj,kk]
    ###                     h+=1

    ###         prov_centers=delta_x*prov_ijk_centers
    ###         prov_dist_centers=np.linalg.norm(prov_centers,axis=1)
    ###         rank_dists = np.argsort(prov_dist_centers)
    ###         prov_dist_centers = prov_dist_centers[rank_dists]
    ###         prov_ijk_centers  = prov_ijk_centers[rank_dists]
    ###         prov_centers      = prov_centers[rank_dists]
    ###         mask_in_sphere    = (prov_dist_centers<=rmax)
    ###         prov_centers       = prov_centers[mask_in_sphere]
    ###         prov_ijk_centers   = prov_ijk_centers[mask_in_sphere]
    ###         prov_dist_centers = prov_dist_centers[mask_in_sphere]

    ###         self.centers = prov_centers[::-1]
    ###         self.ijk_centers = prov_ijk_centers[::-1]
    ###         self.num_centers = prov_centers.shape(0)
    ###         del(prov_centers,prov_ijk_centers,prov_dist_centers,mask_in_sphere)


    def centers_in_layer(self,centers_distribution="regular_cartesian", receptor=None, ligand=None,
                         delta_x=None, with_network=True):
        '''
        centers_distribution=regular_cartesian, regular_polar, uniform
        '''

        hbond_dist = 0.25
        receptor_positions = np.array(receptor.positions._value)

        if centers_distribution=="regular_cartesian":

            _, d_max_rec=utils.furthest_accessible_atom_to_center(receptor)
            _, d_max_lig=utils.furthest_accessible_atom_to_center(ligand)
            _, d_min_lig=utils.closest_accessible_atom_to_center(ligand)

            Lbox = 2*d_max_rec+2*d_max_lig+hbond_dist #añado hbond distance
            num_bins_x_2       = np.int(np.ceil((Lbox/2.0)/delta_x))
            num_bins_x         =2*num_bins_x_2+1
            prov_num_centers =num_bins_x*num_bins_x*num_bins_x

            prov_centers    = np.empty((prov_num_centers,3),dtype=float)
            prov_ijk_centers= np.empty((prov_num_centers,3),dtype=int)

            h=0
            for ii in range(-num_bins_x_2,num_bins_x_2+1):
                for jj in range(-num_bins_x_2,num_bins_x_2+1):
                    for kk in range(-num_bins_x_2,num_bins_x_2+1):
                        prov_ijk_centers[h,:]=[ii,jj,kk]
                        h+=1

            prov_centers=delta_x*prov_ijk_centers


            # removing out of sphere
            prov_dist_centers=np.linalg.norm(prov_centers,axis=1)
            mask_in_sphere    = (prov_dist_centers<=(Lbox/2.0+hbond_dist))
            prov_centers       = prov_centers[mask_in_sphere]
            prov_ijk_centers   = prov_ijk_centers[mask_in_sphere]


            # removing inside layer [dist<(hbond_dist+d_min_lig) of every receptor atom]

            raux=hbond_dist+d_min_lig
            for position_atom in receptor_positions:
                aux_vects = prov_centers-position_atom
                aux_dists = np.linalg.norm(aux_vects,axis=1)
                mask_in   = (aux_dists>=raux)
                prov_centers = prov_centers[mask_in]
                prov_ijk_centers = prov_ijk_centers[mask_in]

            # removing outside layer [dist>(hbond_dist+d_max_lig) of every receptor atom]

            raux=hbond_dist+d_max_lig
            mask_in=np.zeros(prov_centers.shape[0],dtype=bool)
            for ii in range(prov_centers.shape[0]):
                aux_vects = receptor_positions - prov_centers[ii]
                aux_dists = np.linalg.norm(aux_vects,axis=1)
                aux_min_dist = aux_dists.min()
                if aux_min_dist <= raux:
                    mask_in[ii]=True

            prov_centers = prov_centers[mask_in]
            prov_ijk_centers = prov_ijk_centers[mask_in]


            prov_dist_centers=np.linalg.norm(prov_centers,axis=1)
            rank_dists = np.argsort(prov_dist_centers)
            prov_dist_centers = prov_dist_centers[rank_dists]
            prov_ijk_centers  = prov_ijk_centers[rank_dists]
            prov_centers      = prov_centers[rank_dists]

            self.centers = prov_centers[::-1]
            self.ijk_centers = prov_ijk_centers[::-1]
            self.num_centers = prov_centers.shape[0]
            del(prov_centers,prov_ijk_centers,prov_dist_centers,mask_in)

    def rotations_in_quaternions_region(self, rotations='All', rotations_distribution='healpix',
                                        nside=8, with_network=True):

        if rotations=='All':
            if rotations_distribution=='healpix': #nside debe ser potencia de 2 para que sea regular y tengan 8 vecinos cada uno
                num_rotations=hp.nside2npix(nside)
                sphere_coors=hp.pix2ang(nside,np.arange(num_rotations),nest=False)
                rotations=quaternion.from_spherical_coords(sphere_coors[0],sphere_coors[1])
                self.rotations=rotations
                self.nside=nside
                self.num_rotations=num_rotations
                del(rotations)
        pass

    def make_ConformationalNetwork(self):

        neigh = NearestNeighbors(radius=1, metric='chebyshev')
        neigh.fit(self.ijk_centers)
        net_centers=nx.from_scipy_sparse_matrix(neigh.radius_neighbors_graph())
        del(neigh)

        net_rotations=nx.Graph()
        net_rotations.add_nodes_from(range(self.num_rotations))
        for ii in range(self.num_rotations):
            neighs=hp.get_all_neighbours(self.nside,ii,nest=False)
            neighs[neighs==-1]=0
            net_rotations.add_edges_from(zip(np.full(neighs.shape[0],ii), neighs))
        del(neighs)

        net=nx.cartesian_product(net_centers,net_rotations)

        del(net_rotations,net_centers)

        return net

    def extract_subregion(self,centers=None, rotations=None):

        pass

    def split_in_subregions(self,num_subregions=None):

        tmp_subregions=[]

        if self.num_rotations>=num_subregions:
            lists_rotation_indices=np.array_split(np.arange(self.num_rotations,dtype=int),num_subregions)
            tmp_subregions=[self.extract_subregion(rotations=list_rotations) for list_rotations in lists_rotation_indices]

        else:
            "not yet"

        return tmp_subregions

