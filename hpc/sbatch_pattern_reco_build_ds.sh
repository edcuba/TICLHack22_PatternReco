#!/bin/sh

#SBATCH --partition=genx
#SBATCH --time=1-0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=ticl_build_ds

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load dependencies
module load python

VEPATH=/mnt/ceph/users/ecuba/ve
source $VEPATH/bin/activate

python build_ds_pair.py
