import numpy as np


def f_score(A, B):
    return (2 * A * B) / (A + B)


def get_map(t_vertices, t_multiplicity):
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
            if i not in i2te:
                i2te[i] = []
            i2te[i].append((t_idx, 1. / m))
    return i2te


def get_E_map(t_vertices, t_energy):
    """
    Create mapping if vertex id to the total vertex energy
    """
    v2e = {}
    for t_idx in range(len(t_vertices)):
        for i, e in zip(t_vertices[t_idx], t_energy[t_idx]):
            v2e[i] = e
    return v2e


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
            e_pair = te_map[i] * te_map[j]
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
        i_tracksters = i2a[i]

        # get score for each trackster/simtrackster i is in
        for i_trackster, _ in i_tracksters:
            # intersection required for recall (as the points are filtered)
            V = np.intersect1d(t_vertices[i_trackster], vertices)

            # normalize by the number of tracksters and add to the outer P sum
            P += get_pairwise_scores(i, V, i2b, e_map) / len(i_tracksters)

    # normalize the result
    return P / len(vertices)