#!/bin/sh

#SBATCH --partition=genx
#SBATCH --time=7-0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=reco_ds_pair

# Load dependencies
module load python

VEPATH=/mnt/ceph/users/ecuba/ve
source $VEPATH/bin/activate

python build_ds_pair.py
