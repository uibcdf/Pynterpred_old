import numpy as np
import quaternion
import healpy as hp
from . import utils

class Region():

    def __init__(self, receptor=None, ligand=None, centers='in_layer', centers_distribution='regular_cartesian', delta_x=0.25
                 ,rotations='All',rotations_distribution='healpix',nside=8):

        self.centers=None
        self.ijk_centers=None
        self.net_centers=None
        self.num_centers=None

        self.rotations=None
        self.net_rotations=None
        self.num_rotations=None
        self.nside=None

        if (receptor is not None) and (ligand is not None):
            if centers=='in_layer':
                self.centers_in_layer(centers_distribution, receptor, ligand, delta_x)
            if rotations_distribution=='healpix':
                self.rotations_in_quaternions_region(rotations,rotations_distribution,nside)

        pass

    def centers_in_sphere(self,centers_distribution="regular_cartesian", rmax=None, delta_x=None):
        '''
        centers_distribution=regular_cartesian, regular_polar, uniform
        '''

        if centers_distribution=="regular_cartesian":

            volume_explored = (4.0/3.0)*np.pi*(rmax**3)
            nx_2       = np.int(np.ceil(rmax/delta_x))
            nx         =2*nx_2+1
            prov_num_centers =nx*nx*nx
            prov_centers    = np.empty((prov_num_centers,3),dtype=float)
            prov_ijk_centers= np.empty((prov_num_centers,3),dtype=int)

            h=0
            for ii in range(-nx_2,nx_2+1):
                for jj in range(-nx_2,nx_2+1):
                    for kk in range(-nx_2,nx_2+1):
                        prov_ijk_centers[h,:]=[ii,jj,kk]
                        h+=1

            prov_centers=delta_x*prov_ijk_centers
            prov_dist_centers=np.linalg.norm(prov_centers,axis=1)
            rank_dists = np.argsort(prov_dist_centers)
            prov_dist_centers = prov_dist_centers[rank_dists]
            prov_ijk_centers  = prov_ijk_centers[rank_dists]
            prov_centers      = prov_centers[rank_dists]
            mask_in_sphere    = (prov_dist_centers<=rmax)
            prov_centers       = prov_centers[mask_in_sphere]
            prov_ijk_centers   = prov_ijk_centers[mask_in_sphere]
            prov_dist_centers = prov_dist_centers[mask_in_sphere]

            self.centers = prov_centers[::-1]
            self.ijk_centers = prov_ijk_centers[::-1]
            self.num_centers = prov_centers.shape(0)
            del(prov_centers,prov_ijk_centers,prov_dist_centers,mask_in_sphere)


    def centers_in_layer(self,centers_distribution="regular_cartesian", receptor=None, ligand=None, delta_x=None):
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
            nx_2       = np.int(np.ceil((Lbox/2.0)/delta_x))
            nx         =2*nx_2+1
            prov_num_centers =nx*nx*nx

            prov_centers    = np.empty((prov_num_centers,3),dtype=float)
            prov_ijk_centers= np.empty((prov_num_centers,3),dtype=int)

            h=0
            for ii in range(-nx_2,nx_2+1):
                for jj in range(-nx_2,nx_2+1):
                    for kk in range(-nx_2,nx_2+1):
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

        pass

    def rotations_in_quaternions_region(self,rotations='All',rotations_distribution='healpix',nside=8):

        if rotations=='All':
            if rotations_distribution=='healpix': #nside debe ser potencia de 2 para que sea regular y tengan 8 vecinos cada uno
                NSIDE=nside
                num_rotations=hp.nside2npix(NSIDE)
                sphere_coors=hp.pix2ang(NSIDE,np.arange(num_rotations),nest=False)
                rotations=quaternion.from_spherical_coords(sphere_coors[0],sphere_coors[1])

                self.rotations=rotations
                self.nside=nside
                self.num_rotations=num_rotations
                del(rotations)
        pass

    def extract_subregion(self,centers=None, rotations=None):

        tmp_subregion=Region()

        if centers is None:
            tmp_subregion.centers     = self.centers
            tmp_subregion.ijk_centers = self.ijk_centers
        else:
            tmp_subregion.centers     = self.centers[centers]
            tmp_subregion.ijk_centers = self.ijk_centers[centers]

        if rotations is None:
            tmp_subregion.rotations = self.rotations
        else:
            tmp_subregion.rotations = self.rotations[rotations]

        tmp_subregion.num_rotations = len(self.rotations)
        tmp_subregion.num_centers = len(self.centers)

        return tmp_subregion

    def split_in_subregions(self,num_subregions=None):

        tmp_subregions=[]

        if self.num_rotations>=num_subregions:
            lists_rotation_indices=np.array_split(np.arange(self.num_rotations,dtype=int),num_subregions)
            tmp_subregions=[self.extract_subregion(rotations=list_rotations) for list_rotations in lists_rotation_indices]

        else:
            "not yet"

        return tmp_subregions
