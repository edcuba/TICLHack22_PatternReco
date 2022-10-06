import torch
import numpy as np
import awkward as ak

from .energy import get_energy_map
from .dataset import get_ground_truth, get_pair_tensor_builder
from .event import get_trackster_map, remap_arrays_by_label, remap_tracksters, get_candidate_pairs



def f_score(precision, recall, beta=1):
    """
        F-Beta score
            when beta = 1, this is equal to F1 score
    """
    return ((1 + beta**2) * precision * recall) / ((precision * beta**2) + recall)


def B(i, j, mapping):
    """
    i and j are in the same trackster on one side
    are they also in a same trackster on the other side?
    """

    # no mapping, no match
    if not i in mapping or not j in mapping:
        return 0

    # get (sim-)tracksters of i and j on the other side
    it = np.array(mapping[i])
    jt = np.array(mapping[j])

    # find out how many are matching
    _, i_idx, j_idx, = np.intersect1d(it[:,0], jt[:,0], return_indices=True)

    # sum of the intersection
    i_match_sum = it[i_idx][:,1].sum()
    j_match_sum = jt[j_idx][:,1].sum()

    # sum of the union
    i_sum = it[:,1].sum()
    j_sum = jt[:,1].sum()

    return (i_match_sum + j_match_sum) / (i_sum + j_sum)


def get_pairwise_scores(i, V, i2t, te_map):
    """
    Compute the score per trackster/simtrackster
    normalized by the number of pairs or total pair energy
    Inputs:
        i       target LC
        V       L or C - all LCs of the trackster or simtrackster
        i2t     mapping from LC to its trackster or simtracksters on the oposite side
        te_map  (optional) LC -> energy in the given trackster
    Output:
        Normalized sum of the pair-wise scores
    """
    e_pairs = 0
    i_trackster_score = 0
    # for all vertices of the trackster
    for j in V:
        pair_score = B(i, j, i2t)
        if te_map:
            # multiply the score by the pair energy
            e_pair = (te_map[i] * te_map[j])**2
            pair_score *= e_pair
            e_pairs += e_pair
        i_trackster_score += pair_score

    if e_pairs:
        return i_trackster_score / e_pairs
    return i_trackster_score / len(V)


def bcubed(vertices, t_vertices, i2a, i2b, e_map=None):
    """
    Base algorithm for bcubed precision and recall
    Input:
        vertices: all layerclusters in the event
        t_vertices:
            precision: trackster to layercluster mapping
            recall: simtrackster to layercluster mapping
        i2a, i2b:
            precision:
                layercluster to tracksters map
                layercluster to simtracksters map
            recall: reverse order (i2b, i2a)
        e_map (optional):
            precision: LC to energy in a trackster
            recall: LC to energy in a simtrackster
    Returns: precision / recall for the given input
    """
    P = 0
    # for all reco/sim vertices
    for i in vertices:
        # get all tracksters/simtracksters i is in
        i_tracksters = i2a.get(i, [])

        # get score for each trackster/simtrackster i is in
        for i_trackster, _ in i_tracksters:
            # intersection required for recall (as the points are filtered)
            V = np.intersect1d(t_vertices[i_trackster], vertices)

            te_map = e_map[i_trackster] if e_map else None

            # normalize by the number of tracksters and add to the outer P sum
            P += get_pairwise_scores(i, V, i2b, te_map) / len(i_tracksters)

    # normalize the result
    return P / len(vertices)


def evaluate(t_indexes, st_indexes, t_energy, st_energy, v_multi, sv_multi, f_min=0, beta=2, noise=True):
    t_vertices = ak.flatten(t_indexes)
    st_vertices = ak.flatten(st_indexes) if noise else t_vertices

    # precompute LC -> Trackster mapping
    i2rt = get_trackster_map(t_indexes, v_multi)
    i2st = get_trackster_map(st_indexes, sv_multi, f_min=f_min)

    # precompute LC -> Energy mapping (same for all tracksters the LC is in)
    te_map = get_energy_map(t_indexes, t_energy, v_multi)
    ste_map = get_energy_map(st_indexes, st_energy, sv_multi)

    precision = bcubed(t_vertices, t_indexes, i2rt, i2st, e_map=te_map)
    recall = bcubed(st_vertices, st_indexes, i2st, i2rt, e_map=ste_map)

    return precision, recall, f_score(precision, recall, beta=beta)


def evaluate_remapped(t_indexes, st_indexes, t_energy, st_energy, v_multi, sv_multi, labels, f_min=0, noise=True):
    ri = remap_arrays_by_label(t_indexes, labels)
    re = remap_arrays_by_label(t_energy, labels)
    rm = remap_arrays_by_label(v_multi, labels)
    return evaluate(ri, st_indexes, re, st_energy, rm, sv_multi, f_min=f_min, noise=noise)


def run_evaluation(callable_fn, tracksters, simtracksters, associations, **kwargs):
    t_indexes = tracksters["vertices_indexes"].array()
    t_energy = tracksters["vertices_energy"].array()
    tv_multi = tracksters["vertices_multiplicity"].array()

    sv_i = simtracksters["stsSC_vertices_indexes"].array()

    mP = []
    mR = []
    mF = []

    for _eid in range(len(t_indexes)):
        labels = callable_fn(tracksters, _eid, **kwargs)

        gt = get_ground_truth(
            tracksters,
            simtracksters,
            associations,
            _eid,
        )

        gt_i = gt["vertices_indexes"]

        P, R, F = evaluate_remapped(
            t_indexes[_eid],
            gt_i,
            t_energy[_eid],
            gt["vertices_energy"],
            tv_multi[_eid],
            gt["vertices_multiplicity"],
            labels,
            noise=False
        )
        mP.append(P)
        mR.append(R)
        mF.append(F)
        NIn = len(t_indexes[_eid])
        NSim = len(sv_i[_eid])
        NGt = len(gt_i)
        NReco = max(labels) + 1
        print(f"E{_eid}: nTIn: {NIn}\tnTSim: {NSim}\tnTGt: {NGt}\tnTReco: {NReco}\tP: {P:.2f} R: {R:.2f} F:{F:.2f}")

    print(f"--- Mean results: p: {np.mean(mP):.2f} r: {np.mean(mR):.2f} f:{np.mean(mF):.2f} ---")


def pairwise_model_evaluation(
    tracksters,
    simtracksters,
    associations,
    graphs,
    model,
    scaler,
    decision_th,
    max_distance=10,
    energy_threshold=10,
):
    nt = tracksters["NTracksters"].array()
    model.eval()

    results = {
        "clue3d_to_sim": [],
        "target_to_sim": [],
        "reco_to_target": [],
        "reco_to_sim": []
    }

    for eid in range(len(nt)):
        # get candidate pairs
        candidate_pairs, dst_map = get_candidate_pairs(
            tracksters,
            graphs,
            eid,
            max_distance=max_distance,
        )
        builder = get_pair_tensor_builder(tracksters, eid, dst_map)

        # get target
        gt = get_ground_truth(
            tracksters,
            simtracksters,
            associations,
            eid,
            distance_threshold=max_distance,
            energy_threshold=energy_threshold,
        )

        # prepare features
        _scaled_samples = []
        for edge in candidate_pairs:
            sample = torch.tensor(builder(edge))
            scaled = scaler.transform(sample.reshape(1,-1))
            _scaled_samples.append(scaled)
        samples = torch.tensor(np.array(_scaled_samples)).type(torch.float)

        # predict edges
        preds = model(samples)
        out = (preds.reshape(1,-1)[0].detach().numpy() > decision_th)

        # map decisions
        # !must be little -> big
        # some tracksters happen to both on left and right here
        re = tracksters["raw_energy"].array()[eid]
        merge_map = {
            a: b for (a, b), o in zip(candidate_pairs, out)
            if o and re[a] < energy_threshold and re[b] > energy_threshold
        }

        # rebuild the event
        reco = remap_tracksters(tracksters, merge_map, eid)

        # reco
        re = reco["vertices_energy"]
        ri = reco["vertices_indexes"]
        rm = reco["vertices_multiplicity"]

        # target
        te = gt["vertices_energy"]
        ti = gt["vertices_indexes"]
        tm = gt["vertices_multiplicity"]

        # clue3D
        ce = tracksters["vertices_energy"].array()[eid]
        ci = tracksters["vertices_indexes"].array()[eid]
        cm = tracksters["vertices_multiplicity"].array()[eid]

        # simulation
        se = simtracksters["stsSC_vertices_energy"].array()[eid]
        si = simtracksters["stsSC_vertices_indexes"].array()[eid]
        sm = simtracksters["stsSC_vertices_multiplicity"].array()[eid]

        results["clue3d_to_sim"].append(evaluate(ci, si, ce, se, cm, sm, noise=False))
        results["target_to_sim"].append(evaluate(ti, si, te, se, tm, sm, noise=False))
        results["reco_to_target"].append(evaluate(ri, ti, re, te, rm, tm, noise=False))
        results["reco_to_sim"].append(evaluate(ri, si, re, se, rm, sm, noise=False))

        print(f"Event {eid}:")
        for key, values in results.items():
            vals = values[-1]
            print(f"\t{key}:\tP: {vals[0]:.2f} R: {vals[1]:.2f} F: {vals[2]:.2f}")

    print("----------")
    for key, values in results.items():
        avg_p = np.mean([x[0] for x in values])
        avg_r = np.mean([x[1] for x in values])
        avg_f = np.mean([x[2] for x in values])
        print(f"mean {key}:\tP: {avg_p:.2f} R: {avg_r:.2f} F: {avg_f:.2f}")

    return results