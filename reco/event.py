import numpy as np
import awkward as ak
from .distance import euclidian_distance, apply_map
from .data import ARRAYS


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
        f_min: minimum energy fraction

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


def get_merge_map(pair_index, preds, threshold):
    """
        Performs the little-big mapping
        Respects the highest prediction score for each little
    """
    merge_map = {}
    score_map = {}

    # should always be (little, big)
    for (little, big), p in zip(pair_index, preds):
        if p > threshold:
            if not little in score_map or score_map[little] < p:
                merge_map[little] = [big]
                score_map[little] = p

    return merge_map


def get_merge_map_multi(pair_index, preds, threshold):
    """
        Performs the little-big mapping
        Respects the highest prediction score for each little
    """
    merge_map = {}

    # should always be (little, big)
    for (little, big), p in zip(pair_index, preds):
        if p > threshold:
            if not little in merge_map:
                merge_map[little] = []
            merge_map[little].append(big)
    return merge_map


def merge_tracksters(trackster_data, merged_tracksters, eid):
    datalist = list(trackster_data[k][eid] for k in ARRAYS)
    result = {
        k: ak.Array([ak.flatten(datalist[i][list(set(tlist))]) for tlist in merged_tracksters])
        for i, k in enumerate(ARRAYS)
    }
    # recompute barycentres
    tve = result["vertices_energy"]
    for coord in ("x", "y", "z"):
        _bary = [np.average(vx, weights=ve) for vx, ve in zip(result[f"vertices_{coord}"], tve)]
        result[f"barycenter_{coord}"] = ak.Array(_bary)
    return result


def remap_tracksters(trackster_data, pair_index, preds, eid, decision_th=0.5, pileup=False, allow_multiple=False):
    """
        provide a mapping in format (source: target)
    """
    if allow_multiple:
        new_mapping = get_merge_map_multi(pair_index, preds, decision_th)
    else:
        new_mapping = get_merge_map(pair_index, preds, decision_th)

    if pileup:
        # only include right-handed tracksters
        p_list = set(b for _, b in pair_index)
        new_tracksters = [[b] for b in p_list]
    else:
        # include all tracksters
        new_tracksters = [[i] for i in range(len(trackster_data["raw_energy"][eid]))]

    new_idx_map = {o[0]: i for i, o in enumerate(new_tracksters)}

    for l, bigs in new_mapping.items():
        for b in bigs:
            new_b_idx = new_idx_map[b]
            new_l_idx = new_idx_map.get(l, -1)

            if l == b or new_l_idx == new_b_idx:
                # sanity check: same trackster or already merged
                #   otherwise we delete the trackster
                continue

            if new_l_idx == -1:
                # assign to a trackster
                new_tracksters[new_b_idx].append(l)
                new_idx_map[l] = new_b_idx
            else:
                # merge tracksters
                new_tracksters[new_b_idx] += new_tracksters[new_l_idx]
                # forward dictionary references
                for k, v in new_idx_map.items():
                    if v == new_l_idx:
                        new_idx_map[k] = new_b_idx
                # remove the old record
                new_tracksters[new_l_idx] = []

    merged_tracksters = list(t for t in new_tracksters if t)
    return merge_tracksters(trackster_data, merged_tracksters, eid)


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


def get_candidate_pairs_little_big_planear(
    xy_cloud,
    layers_range,
    inners,
    raw_energy,
    max_distance=10,
    energy_threshold=10,
):
    candidate_pairs = []
    for i, inners in enumerate(inners):
        for inner in inners:
            e_pair = (raw_energy[i], raw_energy[inner])
            l_range1 = layers_range[i]
            l_range2 = layers_range[inner]
            l_distance = max((l_range1[0], l_range2[0])) - min((l_range1[1], l_range2[1]))
            if min(e_pair) < energy_threshold and max(e_pair) > energy_threshold and l_distance < 0:
                dst = euclidian_distance(xy_cloud[i], xy_cloud[inner])
                if dst <= max_distance:
                    pair = (i, inner) if e_pair[0] < e_pair[1] else (inner, i)
                    candidate_pairs.append(pair)
    return candidate_pairs


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

