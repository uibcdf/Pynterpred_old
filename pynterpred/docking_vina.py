import numpy as np
import tempfile
import shutil
import os
from os import path
from simtk.openmm import app
from subprocess import call, check_output
from glob import glob

def tempname(suffix='', create=False):
    if create:
        file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    else:
        file = tempfile.NamedTemporaryFile(delete=True, suffix=suffix)
    file.close()
    return file.name

def vina (receptor, ligand, center=None, extent=None, numposes=20, babelexe='obabel', vinaexe=None):

    """ Molecular docking, using Vina 
    >>> This is from HTMD, file dock.py.

    If centre and extent are not provided, docking will be performed over the whole protein

    Parameters
    ----------
    protein : :class:`Molecule <htmd.molecule.molecule.Molecule>` object
        Molecule object representing the receptor
    ligand : :class:`Molecule <htmd.molecule.molecule.Molecule>` object
        Molecule object representing the ligand to dock
    center : list
        3-vec centre of of the search bounding box (optional)
    extent : list
        3-vec linear extent of the search bounding box (optional)
    numposes : int
        Number of poses to return. Vina cannot return more than 20 poses.
    babelexe : str
        Path to babel executable.
    vinaexe : str
        Path to AutoDock Vina executable.

    Returns
    -------
    poses
        Molecule object representing the N<10 best poses
    scores
        3x num_poses matrix containing kcal, rmsd lb, rmsd ub

    Examples
    --------
    >>> poses, scoring = dock(protein, ligand)
    >>> poses, scoring = dock(protein, ligand, center=[ 10., 5., 12. ], extent=[ 15., 15., 15. ] )

    """

    buffer = 1.0 # 1 nanometer
    c_min = np.min(receptor.positions._value, 0).reshape((1, 3))[0]
    c_max = np.max(receptor.positions._value, 0).reshape((1, 3))[0]

    #if center is None:
    #    center = (buffer + (c_max))

    if center is None:
        center = (buffer + (c_max + c_min)) / 2
    if extent is None:
        extent = (c_max - c_min) + buffer

    # babel -i pdb protein.pdb  -o pdbqt protein.pdbqt -xr
    # babel -i pdb ligand.pdb   -o pdbqt ligand.pdbqt -xhn
    # vina --ligand ligand.pdbqt --receptor protein.pdbqt --center_x 0. --center_y 0. --center_z 0. --size_x 60. --size_y 60. --size_z 60 --exhaustiveness 10
    # babel -m -i pdbqt ligand_out.pdbqt -o pdb out_.pdb -xhn


    receptor_pdb = tempname(suffix=".pdb")
    ligand_pdb = tempname(suffix=".pdb")
    output_pdb = tempname(suffix="_.pdb")
    output_prefix = path.splitext(output_pdb)[0]

    receptor_pdbqt = tempname(suffix=".pdbqt")
    ligand_pdbqt = tempname(suffix=".pdbqt")
    output_pdbqt = tempname(suffix=".pdbqt")

    print('receptor pdb:',receptor_pdb)
    receptor_pdb_file = open(receptor_pdb, 'w')
    app.PDBFile.writeFile(receptor.topology, receptor.positions, receptor_pdb_file)
    receptor_pdb_file.close()

    print('ligand pdb:',ligand_pdb)
    ligand_pdb_file = open(ligand_pdb, 'w')
    app.PDBFile.writeFile(ligand.topology, ligand.positions, ligand_pdb_file)
    ligand_pdb_file.close()

    # Dirty hack to remove the 'END' line from the PDBs since babel hates it
    with open(receptor_pdb, 'r') as f:
        lines = f.readlines()
    with open(receptor_pdb, 'w') as f:
        f.writelines(lines[:-1])
    with open(ligand_pdb, 'r') as f:
        lines = f.readlines()
    with open(ligand_pdb, 'w') as f:
        f.writelines(lines[:-1])
    # End of dirty hack

    try:
        if vinaexe is None:
            import platform
            suffix = ''
            if platform.system() == "Windows":
                suffix = '.exe'
            vinaexe = '{}-vina{}'.format(platform.system(), suffix)

        vinaexe = shutil.which(vinaexe, mode=os.X_OK)
        if not vinaexe:
            raise NameError('Could not find vina, or no execute permissions are given')
    except:
        raise NameError('Could not find vina, or no execute permissions are given')
    try:
        babelexe = shutil.which(babelexe, mode=os.X_OK)
        if babelexe is None:
            raise NameError('Could not find babel, or no execute permissions are given')
    except:
        raise NameError('Could not find babel, or no execute permissions are given')


    print(center,extent)
    center=center*10.0
    extent=extent*10.0

    babel_receptor=babelexe+" -i pdb "+receptor_pdb+" -o pdbqt -O "+receptor_pdbqt+' -xr'
    babel_ligand  =babelexe+" -i pdb "+ligand_pdb+" -o pdbqt -O "+ligand_pdbqt+' -xnh'
    vina_command  =vinaexe+' --receptor '+receptor_pdbqt+' --ligand '+ligand_pdbqt+' --out '+output_pdbqt+ ' --center_x '+str(center[0])+' --center_y '+str(center[1])+' --center_z '+str(center[2])+ ' --size_x '+str(extent[0])+' --size_y '+str(extent[1])+' --size_z '+str(extent[2])+' --num_modes '+str(numposes)

    print(babel_receptor)
    print(babel_ligand)
    print(vina_command)

    call(babel_receptor,shell=True)

    with_charges=True
    if with_charges:
        #logger.info('Charges detected in ligand and will be used for docking.')
        print('Charges detected in ligand and will be used for docking.')
        call(babel_ligand,shell=True)
    else:
        #logger.info('Charges were not defined for all atoms. Will guess charges anew using gasteiger method.')
        call(babel_ligand+' --partialcharge gasteiger',shell=True)

    call(vina_command,shell=True)

    call([babelexe, '-m', '-i', 'pdbqt', output_pdbqt, '-o', 'pdb', '-O', output_pdb, '-xhn'],shell=True)
    pass

    from natsort import natsorted
    outfiles = natsorted(glob('{}*.pdb'.format(output_prefix)))

    scoring = []
    poses = []
    for i, ligf in enumerate(outfiles):
        scoring.append(_parseScoring(ligf))
        l = Molecule(ligf)
        l.viewname = 'Pose {}'.format(i)
        poses.append(l)

    os.remove(receptor_pdb)
    os.remove(ligand_pdb)
    os.remove(receptor_pdbqt)
    os.remove(ligand_pdbqt)
    os.remove(output_pdbqt)

    return poses, np.array(scoring)


