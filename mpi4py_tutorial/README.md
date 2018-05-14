which mpirun
mpirun -n 4 python script.py

https://scitas.epfl.ch/kb/Using+Jupyter

conda install ipyparallel
which ipcluster

#Para hacerlo con el slurm:
ipcluster start --init --profile=default --ip="*" -n=<ntasks> --engines=Slurm --SlurmEngineSetLauncher.timelimit=<timelimit> --SlurmEngineSetLauncher.queue=<partition> --SlurmEngineSetLauncher.account=<account>

https://ipython.org/ipython-doc/3/parallel/parallel_mpi.html
https://ipython.org/ipython-doc/3/parallel/parallel_process.html#parallel-process
