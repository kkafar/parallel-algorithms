#!/bin/bash -l

#SBATCH --account=plgar2023-cpu
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name="Sieve"

# As in quickstart
module load scipy-bundle/2021.10-intel-2021b
export SLURM_OVERLAP=1
mpiexec ./main.py 10 13 255 30

