import numpy as np
import awkward as ak


def get_bary(tracksters, _eid):
    return np.array([
        tracksters["barycenter_x"].array()[_eid],
        tracksters["barycenter_y"].array()[_eid],
        tracksters["barycenter_z"].array()[_eid]
    ]).T


def get_lc(tracksters, _eid):
    # this is not an entirely fair comparison:
    # the LC level methods should use sim LCs not only the CLUE3D ones
    # using sim data here is possible, but gets complicated
    x_lc = ak.flatten(tracksters["vertices_x"].array()[_eid])
    y_lc = ak.flatten(tracksters["vertices_y"].array()[_eid])
    z_lc = ak.flatten(tracksters["vertices_z"].array()[_eid])
    return np.array([x_lc, y_lc, z_lc]).T


def get_trackster_map(t_vertices, t_multiplicity, f_min=0):
    """
    Create mapping of vertex_id to an array of tupples:
        (trackster_id, energy_fraction)

    Input:
        t_vertices: array of vertex IDs in dimensions (tracksters, vertices)
        t_multiplicity: array of vertex multiplicities in dims (tracksters, vertices)

    Output:
        {vertex: [(trackster_id, energy_fraction)]}
    """
    i2te = {}
    for t_idx in range(len(t_vertices)):
        for i, m in zip(t_vertices[t_idx], t_multiplicity[t_idx]):
            f = 1. / m
            if f > f_min:
                if i not in i2te:
                    i2te[i] = []
                i2te[i].append((t_idx, f))
    return i2te


def remap_arrays_by_label(array, labels):
    h = max(labels) + 1
    rm = []

    for i in range(h):
        rm.append([])

    for l, i in zip(labels, array):
        if l >= 0:
            rm[l] += list(i)

    return ak.Array(rm)

def remap_items_by_label(array, labels):
    h = max(labels) + 1
    rm = []

    for i in range(h):
        rm.append([])

    for l, i in zip(labels, array):
        if l >= 0:
            rm[l].append(i)

    return ak.Array(rm)