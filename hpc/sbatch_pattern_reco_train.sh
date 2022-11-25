#!/bin/sh

#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=pattern_reco_train_lc_graphnet

#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --constraint=a100

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load dependencies
module load python

VEPATH=/mnt/ceph/users/ecuba/ve
source $VEPATH/bin/activate

python run_lc_graphnet.py
