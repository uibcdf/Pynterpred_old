#!/bin/bash
##  Batch script for mpirun job on ixtlilton cluster.
#
#
#  Adjust your script as needed for your clusters!
#
# walltime : maximum wall clock time (hh:mm:ss)
#SBATCH -t 24:00:00
#
# join stdout
#SBATCH -o slurm-%j.out
# join stderr
#SBATCH -e slurm-%j.err
#
# specify partition
#SBATCH -p whole
#
# nodes: number of nodes
#   ppn: how many cores per node to use
###SBATCH --nodes=4 --ntasks-per-node=20
#SBATCH --nodes=5 --ntasks=90
#
# job name (default = name of script file)
#SBATCH -J docking-barnase_barstar

SLURM_LOGFILE=slurm-${SLURM_JOBID}.log

date > $SLURM_LOGFILE
echo "Running simulation via SLURM and MPI" >> $SLURM_LOGFILE
echo "Slurm job id:" $SLURM_JOBID >> $SLURM_LOGFILE
echo "Slurm job name:" $SLURM_JOB_NAME >> $SLURM_LOGFILE
echo "Slurm submit dir:" $SLURM_SUBMIT_DIR >> $SLURM_LOGFILE
echo "Slurm job node list:" $SLURM_JOB_NODELIST >> $SLURM_LOGFILE
echo "Slurm job num nodes:" $SLURM_JOB_NUM_NODES >> $SLURM_LOGFILE
echo "Slurm cpus on node:" $SLURM_CPUS_ON_NODE >> $SLURM_LOGFILE
echo "Slurm number of tasks:" $SLURM_NTASKS >> $SLURM_LOGFILE

mpirun -n $SLURM_NTASKS python barnase_barstar.py
