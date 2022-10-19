module load python

VEPATH=/mnt/ceph/users/ecuba/ve
# python -m venv $VEPATH
source $VEPATH/bin/activate

module load jupyter-kernels
python -m make-custom-kernel ve