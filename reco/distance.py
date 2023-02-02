import pickle
from scipy.spatial.distance import cdist
from os.path import join


def get_z_map(data_root):
    z_map_path = join(data_root, "z_map.pt")
    return pickle.load(open(z_map_path, "rb"))


def apply_map(_target, _map, factor=1):
    if _map is None:
        return _target
    return [factor * _map[int(x)] for x in _target]


def euclidian_distance(X1, X2):
    # return minimum of pairwise distances
    # expensive for the full point cloud
    return cdist(X1, X2, metric="Euclidean").min()