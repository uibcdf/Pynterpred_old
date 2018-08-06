import sys
from simtk import unit
from simtk.openmm import app
from numpy import empty as _np_empty, arange as _arange, array as _array
from . import utils as utils
from copy import deepcopy

class _Units():

    def __init__(self):

        self.length=unit.nanometers
        self.time  =unit.picoseconds

class Macromolecule():

    def __init__(self, pdb_file=None, forcefield=None, pH=7.0, addHs=True, mpi_comm=None):

        self._units          = _Units()
        self.pdb_file        = None
        self.forcefield      = None
        self.pH              = None
        self.modeller        = None
        self.topology        = None
        self.positions       = None
        self.n_atoms         = None

        self._heavy_atoms_indices = None
        self._ca_atoms_indices = None
        self.__addHs_log     = None

        if pdb_file:

            pdb_aux  = app.PDBFile(pdb_file)
            self.modeller = app.Modeller(pdb_aux.topology, pdb_aux.positions)
            self.forcefield = app.ForceField(forcefield)
            self.pH         = pH
            if addHs:
                self.__addHs_log = self.modeller.addHydrogens(self.forcefield, pH=self.pH)
            self.topology  = self.modeller.getTopology()
            self.positions = self.modeller.getPositions()
            self.n_atoms   = len(self.positions)

            self._heavy_atoms_indices = utils.heavy_atoms_indices(self.topology)
            self._ca_atoms_indices = utils.ca_atoms_indices(self.topology)

            if mpi_comm is not None:
                self.equal_positions_across_MPI_Universe(mpi_comm)


    def set_positions(self,positions):

        self.modeller.positions  = deepcopy(positions)
        self.positions           = self.modeller.getPositions()

    def get_positions(self):

        return self.positions

    def center(self, geometrical_center = 'heavy'):

        if geometrical_center == 'heavy':
            tmp_list = self._heavy_atoms_indices
        elif geometrical_center == 'CA':
            tmp_list = self._ca_atoms_indices
        elif geometrical_center == 'All':
            tmp_list = _arange(self.n_atoms)

        tmp_positions = self.get_positions()
        geometrical_center_positions = utils.geometrical_center(tmp_positions,tmp_list)
        tmp_positions = tmp_positions - geometrical_center_positions
        self.set_positions(tmp_positions)

    def get_view(self):

        tmp_view = utils.make_view(self.topology,self.positions)
        return tmp_view

    def equal_positions_across_MPI_Universe(self,mpi_comm=None):

        _my_rank = mpi_comm.Get_rank()

        if _my_rank == 0:
            universal_positions = self.positions._value
        else:
            _positions_shape = self.positions._value.shape
            _positions_dtype = self.positions._value.dtype
            universal_positions = _np_empty(_positions_shape, dtype=_positions_dtype)

        mpi_comm.Bcast(universal_positions, root=0)

        if _my_rank != 0:
            _positions_unit = self.positions.unit
            self.set_positions(universal_positions*_positions_unit)
            del(_positions_shape, _positions_dtype, _positions_unit)

        del(universal_positions, _my_rank)
        mpi_comm.Barrier()

        pass

class Receptor(Macromolecule):

    pass


class Ligand(Macromolecule):

    pass

class MolComplex(Macromolecule):

    def __init__(self, receptor=None, ligand=None, forcefield=None, pH=7.0, addHs=True,
                 mpi_comm=None):

        if (receptor is not None) and (ligand is not None):
            if type(receptor) == str:
                receptor = Receptor(receptor,forcefield,pH,addHs,mpi_comm)
            if type(ligand) == str:
                ligand = Ligand(ligand,forcefield,pH,addHs,mpi_comm)

        self.modeller   = app.Modeller(receptor.topology, receptor.positions)
        self.modeller.add(ligand.topology, ligand.positions)
        self.topology   = self.modeller.getTopology()
        self.positions = self.modeller.getPositions() #np.vstack receptor.positions y ligand.positions
        self.receptor   = receptor
        self.receptor_atom_indices = _arange(0,receptor.n_atoms)
        self.ligand     = ligand
        self.ligand_atom_indices = _arange(receptor.n_atoms,receptor.n_atoms+ligand.n_atoms)
        self.complex_atom_indices = _arange(receptor.n_atoms+ligand.n_atoms)
        self.forcefield = self.receptor.forcefield


    def get_ligand(self):

        return deepcopy(self.ligand)

    def set_ligand_positions(self,positions):

        tmp_positions = self.modeller.getPositions()
        tmp_positions = _array(tmp_positions._value)*tmp_positions.unit
        tmp_positions[self.ligand_atom_indices]=positions
        self.modeller.positions  = tmp_positions
        self.positions           = self.modeller.getPositions()

    def get_ligand_positions(self):

        tmp_positions = self.modeller.getPositions()
        tmp_positions = _array(tmp_positions._value)*tmp_positions.unit
        tmp_positions = tmp_positions[self.ligand_atom_indices]
        return tmp_positions

    def get_receptor(self):

        return deepcopy(self.receptor)

    def set_receptor_positions(self,positions):

        tmp_positions = self.modeller.getPositions()
        tmp_positions = _array(tmp_positions._value)*tmp_positions.unit
        tmp_positions[self.receptor_atom_indices]=positions
        self.modeller.positions  = tmp_positions
        self.positions           = self.modeller.getPositions()

    def get_receptor_positions(self):
        tmp_positions = self.modeller.getPositions()
        tmp_positions = _array(tmp_positions._value)*tmp_positions.unit
        tmp_positions = tmp_positions[self.receptor_atom_indices]
        return tmp_positions

    def get_complex(self):

        return deepcopy(self)

    def set_complex_positions(self,positions):

        tmp_positions = self.modeller.getPositions()
        tmp_positions = _array(tmp_positions._value)*tmp_positions.unit
        tmp_positions[self.complex_atom_indices]=positions
        self.modeller.positions  = tmp_positions
        self.positions           = self.modeller.getPositions()

    def get_complex_positions(self):
        tmp_positions = self.modeller.getPositions()
        tmp_positions = _array(tmp_positions._value)*tmp_positions.unit
        tmp_positions = tmp_positions[self.complex_atom_indices]
        return tmp_positions


