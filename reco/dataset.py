import numpy as np
import awkward as ak

from .event import get_bary
from .matching import match_best_simtrackster
from .distance import euclidian_distance


def _bary_func(bary):
    return lambda tt_id, large_spt: list([euclidian_distance([bary[tt_id]], [bary[lsp]]) for lsp in large_spt])


def _pairwise_func(clouds):
    return lambda tt_id, large_spt: list([euclidian_distance(clouds[tt_id], clouds[lsp]) for lsp in large_spt])


def match_trackster_pairs(
    tracksters,
    simtracksters,
    associations,
    eid,
    energy_threshold=10,
    distance_type="pairwise",
    distance_threshold=10,
    confidence_threshold=0.5
):
    raw_e = tracksters["raw_energy"].array()[eid]
    raw_st = simtracksters["stsSC_raw_energy"].array()[eid]

    large_tracksters = np.where(raw_e > energy_threshold)[0]
    tiny_tracksters = np.where(raw_e <= energy_threshold)[0]

    if distance_type == "pairwise":
        vx = tracksters["vertices_x"].array()[eid]
        vy = tracksters["vertices_y"].array()[eid]
        vz = tracksters["vertices_z"].array()[eid]
        clouds = [np.array([vx[tid], vy[tid], vz[tid]]).T for tid in range(len(raw_e))]
        dst_func = _pairwise_func(clouds)
    elif distance_type == "bary":
        bary = get_bary(tracksters, eid)
        dst_func = _bary_func(bary)
    else:
        raise RuntimeError("Distance type '%s' not supported", distance_type)

    reco_fr, reco_st = match_best_simtrackster(tracksters, associations, eid)
    same_particle_tracksters = [np.where(np.array(reco_st) == st) for st in range(len(raw_st))]
    large_spts = [np.intersect1d(spt, large_tracksters) for spt in same_particle_tracksters]

    pairs = []
    for tt_id in tiny_tracksters:
        max_st = reco_st[tt_id]
        max_fr = reco_fr[tt_id]
        large_spt = large_spts[max_st]

        if len(large_spt) > 0 and max_fr >= confidence_threshold:
            # compute distances
            dists = dst_func(tt_id, large_spt)
            m_dist = min(dists)
            if m_dist < distance_threshold:
                pairs.append([tt_id, large_spt[np.argmin(dists)], m_dist])

    return pairs


def get_ground_truth(
        tracksters,
        simtracksters,
        associations,
        eid,
        energy_threshold=10,
        distance_type="pairwise",
        distance_threshold=10,
        confidence_threshold=0.5
    ):

    e_pairs = match_trackster_pairs(
        tracksters,
        simtracksters,
        associations,
        eid,
        energy_threshold=energy_threshold,
        distance_type=distance_type,
        distance_threshold=distance_threshold,
        confidence_threshold=confidence_threshold
    )

    raw_e = tracksters["raw_energy"].array()[eid]

    new_idx_map = {}
    merge_map = {little: big for little, big, _ in e_pairs}

    new_tracksters = []

    for tr_id in range(len(raw_e)):
        # only keep the tracksters that are not going to be merged
        if tr_id in merge_map.keys():
            # small trackster, ignore
            continue

        # create the new entry
        new_tracksters.append([tr_id])
        new_idx_map[tr_id] = len(new_tracksters) - 1

    # now fill in the tracksters to be merged
    for little, big in merge_map.items():
        new_big_idx = new_idx_map[big]
        new_tracksters[new_big_idx].append(little)

    ARRAYS = [
       "vertices_x",
       "vertices_y",
       "vertices_z",
       "vertices_energy",
       "vertices_multiplicity",
       "vertices_indexes"
    ]

    result = {
        k: ak.Array([ak.flatten(tracksters[k].array()[eid][tlist]) for tlist in new_tracksters])
        for k in ARRAYS
    }

    tve = result["vertices_energy"]

    result["barycenter_x"] = ak.Array([np.average(vx, weights=ve) for vx, ve in zip(result["vertices_x"], tve)])
    result["barycenter_y"] = ak.Array([np.average(vx, weights=ve) for vx, ve in zip(result["vertices_y"], tve)])
    result["barycenter_z"] = ak.Array([np.average(vx, weights=ve) for vx, ve in zip(result["vertices_z"], tve)])

    return result



