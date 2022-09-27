from scipy.spatial.distance import cdist


def euclidian_distance(X1, X2):
    # return minimum of pairwise distances
    return cdist(X1, X2, metric="Euclidean").min()