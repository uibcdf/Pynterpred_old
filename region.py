import numpy as np
import quaternion
import healpy as hp
from . import utils

class Region():

    def __init__(self, receptor=None, ligand=None, centers='in_layer', distribution='regular_cartesian', delta_x=0.25
                 ,qregion='All',qdistribution='healpix',nside=8):

        self.centers=None
        self.ijk_centers=None
        self.num_centers=None
        self.qrotors=None
        self.num_rotors=None
        self.nside=None

        if (receptor is not None) and (ligand is not None):
            if centers=='in_layer':
                self.centers_in_layer(distribution, receptor, ligand, delta_x)
            if qdistribution=='healpix':
                self.rotators_in_quaternions_region(qregion,qdistribution,nside)

        pass

    def centers_in_sphere(self,distribution="regular_cartesian", rmax=None, delta_x=None):
        '''
        distribution=regular_cartesian, regular_polar, uniform
        '''

        if distribution=="regular_cartesian":

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


    def centers_in_layer(self,distribution="regular_cartesian", receptor=None, ligand=None, delta_x=None):
        '''
        distribution=regular_cartesian, regular_polar, uniform
        '''

        hbond_dist = 0.25
        receptor_positions = np.array(receptor.positions._value)

        if distribution=="regular_cartesian":

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

    def rotators_in_quaternions_region(self,qregion='All',qdistribution='healpix',nside=8):

        if qregion=='All':
            if qdistribution=='healpix': #nside debe ser potencia de 2 para que sea regular y tengan 8 vecinos cada uno
                NSIDE=nside
                num_rotors=hp.nside2npix(NSIDE)
                sphere_coors=hp.pix2ang(NSIDE,np.arange(num_rotors),nest=False)
                qrotors=quaternion.from_spherical_coords(sphere_coors[0],sphere_coors[1])

                self.qrotors=qrotors
                self.nside=nside
                self.num_rotors=num_rotors
                del(qrotors)
        pass

    def extract_subregion(self,centers=None, qrotors=None):

        tmp_subregion=Region()

        if centers is None:
            tmp_subregion.centers     = self.centers
            tmp_subregion.ijk_centers = self.ijk_centers
        else:
            tmp_subregion.centers     = self.centers[centers]
            tmp_subregion.ijk_centers = self.ijk_centers[centers]

        if qrotors is None:
            tmp_subregion.qrotors = self.qrotors
        else:
            tmp_subregion.qrotors = self.qrotors[qrotors]

        tmp_subregion.num_rotors = len(self.qrotors)
        tmp_subregion.num_centers = len(self.centers)

        return tmp_subregion

    def split_in_subregions(self,num_subregions=None):

        tmp_subregions=[]

        if self.num_rotors>=num_subregions:
            lists_qrotor_indices=np.array_split(np.arange(self.num_rotors,dtype=int),num_subregions)
            tmp_subregions=[self.extract_subregion(qrotors=list_qrotors) for list_qrotors in lists_qrotor_indices]

        else:
            "not yet"

        return tmp_subregions
