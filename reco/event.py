import awkward as ak


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


def remap_array_by_label(array, labels):
    h = max(labels) + 1
    rm = []

    for i in range(h):
        rm.append([])

    for l, i in zip(labels, array):
        rm[l] += list(i)

    return ak.Array(rm)
