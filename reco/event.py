import numpy as np
import awkward as ak
from .distance import euclidian_distance, apply_map


ARRAYS = [
    "vertices_x",
    "vertices_y",
    "vertices_z",
    "vertices_energy",
    "vertices_multiplicity",
    "vertices_indexes"
]


def get_bary(tracksters, _eid, z_map=None):
    return np.array([
        tracksters["barycenter_x"].array()[_eid],
        tracksters["barycenter_y"].array()[_eid],
        apply_map(tracksters["barycenter_z"].array()[_eid], z_map)
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


def remap_tracksters(tracksters, new_mapping, eid):
    """
        provide a mapping in format (source: target)
    """

    raw_e = tracksters["raw_energy"].array()[eid]
    new_idx_map = {}
    new_tracksters = []

    for tr_id in range(len(raw_e)):
        # only keep the tracksters that are not going to be merged
        if tr_id in new_mapping.keys():
            # small trackster, ignore
            continue

        # create the new entry
        new_tracksters.append([tr_id])
        new_idx_map[tr_id] = len(new_tracksters) - 1

    # now fill in the tracksters to be merged
    for little, big in new_mapping.items():
        new_big_idx = new_idx_map[big]
        new_tracksters[new_big_idx].append(little)

    datalist = list(tracksters[k].array()[eid] for k in ARRAYS)
    result = {
        k: ak.Array([ak.flatten(datalist[i][tlist]) for tlist in new_tracksters])
        for i, k in enumerate(ARRAYS)
    }

    # recompute barycentres
    tve = result["vertices_energy"]
    for coord in ("x", "y", "z"):
        _bary = [np.average(vx, weights=ve) for vx, ve in zip(result[f"vertices_{coord}"], tve)]
        result[f"barycenter_{coord}"] = ak.Array(_bary)

    return result


def get_candidate_pairs_little_big(
    clouds,
    inners,
    raw_energy,
    max_distance=10,
    energy_threshold=10
):
    dst_map = {}
    candidate_pairs = []
    for i, inners in enumerate(inners):
        for inner in inners:
            e_pair = (raw_energy[i], raw_energy[inner])
            if min(e_pair) < energy_threshold and max(e_pair) > energy_threshold:
                dst = euclidian_distance(clouds[i], clouds[inner])
                if dst <= max_distance:
                    pair = (i, inner) if e_pair[0] < e_pair[1] else (inner, i)
                    candidate_pairs.append(pair)
                    dst_map[pair] = dst
    return candidate_pairs, dst_map


def get_candidate_pairs_direct(coordinates, inners, max_distance=10):
    candidate_pairs = []
    dst_map = {}

    for i, inners in enumerate(inners):
        for inner in inners:
            dst = euclidian_distance(coordinates[i], coordinates[inner])
            if dst <= max_distance:
                candidate_pairs.append((i, inner))
                dst_map[(i, inner)] = dst

    return candidate_pairs, dst_map



def get_candidate_pairs(tracksters, graph, eid, max_distance=10, z_map=None):
    vx = tracksters["vertices_x"].array()[eid]
    vy = tracksters["vertices_y"].array()[eid]
    vz = tracksters["vertices_z"].array()[eid]
    clouds = [np.array([vx[tid], vy[tid], apply_map(vz[tid], z_map, factor=2)]).T for tid in range(len(vx))]
    inners = graph["linked_inners"].array()[eid]

    return get_candidate_pairs_direct(clouds, inners, max_distance=max_distance)

