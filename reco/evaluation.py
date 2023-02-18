import torch
import numpy as np
import awkward as ak

from torch_geometric.data import Data
import torch_geometric.transforms as T


from .graphs import create_graph
from .energy import get_energy_map
from .data import FEATURE_KEYS
from .dataset import get_ground_truth
from .event import get_trackster_map, remap_arrays_by_label, remap_tracksters, get_candidate_pairs
from .features import get_graph_level_features

from .datasetPU import get_event_pairs


def f_score(precision, recall, beta=1):
    """
        F-Beta score
            when beta = 1, this is equal to F1 score
    """
    return ((1 + beta**2) * precision * recall) / ((precision * beta**2) + recall)


def B(it, jt):
    """
    Consider that it and jt are in the same objects in one clustering
    Are they in the same objects in the other clustering as well?
    """
    # compute the intersection
    fr_inter = 0.
    for it_x, it_f in it:
        for jt_x, jt_f in jt:
            if it_x == jt_x:
                fr_inter += it_f + jt_f

    # compute the intersection over union (sum of fractions must be 2)
    return fr_inter / 2


def get_pairwise_scores(i, V, i2t, te_map):
    """
    Compute the score per trackster/simtrackster
    normalized by the number of pairs or total pair energy
    Inputs:
        i       target LC
        V       L or C - all LCs of the trackster or simtrackster
        i2t     mapping from LC to its trackster or simtracksters on the oposite side
        te_map  LC -> energy in the given trackster
    Output:
        Normalized sum of the pair-wise scores
    """
    e_pairs = 0
    i_trackster_score = 0

    # for all vertices of the trackster
    for j in V:
        # get the pair energy
        e_pair = (te_map[i] * te_map[j])
        e_pairs += e_pair

        # get (sim-)tracksters of i and j on the other side
        if i in i2t and j in i2t:
            pair_score = B(i2t[i], i2t[j])
            # multiply the score by the pair energy
            i_trackster_score += pair_score * e_pair

    return i_trackster_score / e_pairs


def bcubed(vertices, t_vertices, i2a, i2b, e_map):
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
        e_map:
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
            V = t_vertices[int(i_trackster)]
            te_map = e_map[int(i_trackster)]

            # normalize by the number of tracksters and add to the outer P sum
            P += te_map[i] * get_pairwise_scores(i, V, i2b, te_map) / len(i_tracksters)

    total_e = sum(sum(te.values()) for te in e_map.values())

    # normalize the result
    return P / total_e


def evaluate(nhits, all_t_indexes, all_st_indexes, t_energy, st_energy, all_v_multi, all_sv_multi, f_min=0, beta=0.5, min_hits=1):

    # prepare RECO indexes
    lc_over_1_hit = ak.Array([nhits[t] > min_hits for t in all_t_indexes])
    t_indexes = all_t_indexes[lc_over_1_hit]
    v_multi = all_v_multi[lc_over_1_hit]
    t_vertices = ak.flatten(t_indexes)

    # prepare SIM indexes
    slc_over_1_hit = ak.Array([nhits[t] > min_hits for t in all_st_indexes])
    st_indexes = all_st_indexes[slc_over_1_hit]
    sv_multi = all_sv_multi[slc_over_1_hit]
    st_vertices = ak.Array(set(ak.flatten(st_indexes)))

    # precompute LC -> Trackster mapping
    i2rt = get_trackster_map(t_indexes, v_multi)
    i2st = get_trackster_map(st_indexes, sv_multi, f_min=f_min)

    # precompute LC -> Energy mapping (same for all tracksters the LC is in)
    te_map = get_energy_map(t_indexes, t_energy, v_multi)
    ste_map = get_energy_map(st_indexes, st_energy, sv_multi)

    precision = bcubed(t_vertices, t_indexes, i2rt, i2st, te_map)
    recall = bcubed(st_vertices, st_indexes, i2st, i2rt, ste_map)

    return precision, recall, f_score(precision, recall, beta=beta)


def evaluate_remapped(nhits, t_indexes, st_indexes, t_energy, st_energy, v_multi, sv_multi, labels, f_min=0):
    ri = remap_arrays_by_label(t_indexes, labels)
    re = remap_arrays_by_label(t_energy, labels)
    rm = remap_arrays_by_label(v_multi, labels)
    return evaluate(nhits, ri, st_indexes, re, st_energy, rm, sv_multi, f_min=f_min)


def baseline_evaluation(callable_fn, cluster_data, trackster_data, simtrackster_data, max_events=None, **kwargs):
    results = []
    n_events = len(trackster_data["vertices_indexes"])
    max_events = min(n_events, max_events) if max_events else n_events

    for eid in range(max_events):

        t_indexes = trackster_data["vertices_indexes"][eid]
        t_multiplicity = trackster_data["vertices_multiplicity"][eid]

        # simulation
        st_indexes = simtrackster_data["stsSC_vertices_indexes"][eid]
        st_multiplicity = simtrackster_data["stsSC_vertices_multiplicity"][eid]

        clusters_e = cluster_data["energy"][eid]
        nhits = cluster_data["cluster_number_of_hits"][eid]

        t_energy = ak.Array([clusters_e[indices] for indices in t_indexes])
        st_energy = ak.Array([clusters_e[indices] for indices in st_indexes])

        labels = callable_fn(trackster_data, eid, **kwargs)

        results.append((
            *evaluate_remapped(
                nhits,
                t_indexes,
                st_indexes,
                t_energy,
                st_energy,
                t_multiplicity,
                st_multiplicity,
                labels,
            ),
            max(labels) + 1,
        ))
        results
    return results


def pairwise_model_evaluation(
    cluster_data,
    trackster_data,
    simtrackster_data,
    assoc_data,
    model,
    decision_th=0.5,
    radius=10,
    max_events=100,
    reco_to_target=False,
    bigT_e_th=50,
    pileup=False,
):
    """
    Evaluation must be unbalanced
    """
    model.eval()

    results = {
        "clue3d_to_sim": [],
        "target_to_sim": [],
        "reco_to_sim": []
    }

    if reco_to_target:
        results["reco_to_target"] = []

    for eid in range(min([len(trackster_data["raw_energy"]), max_events])):
        print(f"Event {eid}:")

        dX, dY, pair_index = get_event_pairs(
            cluster_data,
            trackster_data,
            simtrackster_data,
            assoc_data,
            eid,
            radius,
            pileup=pileup,
            bigT_e_th=bigT_e_th,
        )

        if len(dX) == 0:
            print("\tNo data")
            continue

        # predict edges
        preds = model(torch.tensor(dX, dtype=torch.float)).detach().cpu().reshape(-1).tolist()
        truth = np.array(dY)

        clusters_e = cluster_data["energy"][eid]

        # clue3D
        ci = trackster_data["vertices_indexes"][eid]
        cm = trackster_data["vertices_multiplicity"][eid]
        ce = ak.Array([clusters_e[indices] for indices in ci])

        if pileup:
            # need to filter out all the unrelated stuff
            # keep only the big tracksters (right side)
            p_list = list(set(b for _, b in pair_index))
            ce = ce[p_list]
            ci = ci[p_list]
            cm = cm[p_list]

        # rebuild the event
        reco = remap_tracksters(trackster_data, pair_index, preds, eid, decision_th=decision_th, pileup=pileup)
        target = remap_tracksters(trackster_data, pair_index, truth, eid, decision_th=decision_th, pileup=pileup)

        # reco
        ri = reco["vertices_indexes"]
        rm = reco["vertices_multiplicity"]
        re = ak.Array([clusters_e[indices] for indices in ri])

        # target
        target_i = target["vertices_indexes"]
        target_m = target["vertices_multiplicity"]
        target_e = ak.Array([clusters_e[indices] for indices in target_i])

        # simulation
        si = simtrackster_data["stsSC_vertices_indexes"][eid]
        sm = simtrackster_data["stsSC_vertices_multiplicity"][eid]
        se = ak.Array([clusters_e[indices] for indices in si])

        nhits = cluster_data["cluster_number_of_hits"][eid]

        results["clue3d_to_sim"].append(evaluate(nhits, ci, si, ce, se, cm, sm))
        results["target_to_sim"].append(evaluate(nhits, target_i, si, target_e, se, target_m, sm))

        if reco_to_target:
            results["reco_to_target"].append(evaluate(nhits, ri, target_i, re, target_e, rm, target_m))

        results["reco_to_sim"].append(evaluate(nhits, ri, si, re, se, rm, sm))

        for key, values in results.items():
            vals = values[-1]
            print(f"\t{key}:\tP: {vals[0]:.3f} R: {vals[1]:.3f} F: {vals[2]:.3f}")

    print("-----")
    for key, values in results.items():
        avg_p = np.mean([x[0] for x in values])
        avg_r = np.mean([x[1] for x in values])
        avg_f = np.mean([x[2] for x in values])
        print(f"mean {key}:\tP: {avg_p:.3f} R: {avg_r:.3f} F: {avg_f:.3f}")

    return results


def graph_model_evaluation(
    tracksters,
    simtracksters,
    associations,
    graphs,
    model,
    decision_th,
    max_distance=10,
    energy_threshold=10,
    max_events=100,
    include_graph_features=False,
):
    nt = tracksters["NTracksters"].array()
    model.eval()

    results = {
        "clue3d_to_sim": [],
        "target_to_sim": [],
        "reco_to_target": [],
        "reco_to_sim": []
    }

    for eid in range(min([len(nt), max_events])):

        # clue3D
        cx = tracksters["vertices_x"].array()[eid]
        cy = tracksters["vertices_y"].array()[eid]
        cz = tracksters["vertices_z"].array()[eid]
        ce = tracksters["vertices_energy"].array()[eid]
        ci = tracksters["vertices_indexes"].array()[eid]
        cm = tracksters["vertices_multiplicity"].array()[eid]

        # get candidate pairs
        candidate_pairs, _ = get_candidate_pairs(
            tracksters,
            graphs,
            eid,
            max_distance=max_distance,
        )

        # get target
        gt = get_ground_truth(
            tracksters,
            simtracksters,
            associations,
            eid,
            distance_threshold=max_distance,
            energy_threshold=energy_threshold,
        )

        trackster_features = list([
            tracksters[k].array()[eid] for k in FEATURE_KEYS
        ])

        tx_list = []
        for tx in range(len(ce)):
            tx_features = [f[tx] for f in trackster_features]
            if include_graph_features:
                g = create_graph(cx[tx], cy[tx], cz[tx], ce[tx], ci[tx], N=2)
                tx_features += get_graph_level_features(g)
            tx_features += [len(ce[tx])]
            tx_list.append(tx_features)

        data = Data(
            x=torch.tensor(tx_list),
            edge_index=torch.tensor(candidate_pairs).T,
        )

        transform = T.Compose([T.NormalizeFeatures()])
        sample = transform(data)

        # predict edges
        preds = model(sample.x, sample.edge_index)
        out = (preds.reshape(-1) > decision_th)

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

    print("-----")
    for key, values in results.items():
        avg_p = np.mean([x[0] for x in values])
        avg_r = np.mean([x[1] for x in values])
        avg_f = np.mean([x[2] for x in values])
        print(f"mean {key}:\tP: {avg_p:.2f} R: {avg_r:.2f} F: {avg_f:.2f}")

    return results