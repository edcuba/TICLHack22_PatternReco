import numpy as np
import networkx as nx


def distance_matrix(trk_x, trk_y, trk_z):
    v_matrix = np.concatenate(([trk_x], [trk_y], [trk_z]))
    gram = v_matrix.T.dot(v_matrix)

    distance = np.zeros(np.shape(gram))
    for row in range(np.shape(distance)[0]):
        for col in range(np.shape(distance)[1]): #half of the matrix is sufficient, but then mask doesn't work properly
            distance[row][col] = (gram[row][row]-2*gram[row][col]+gram[col][col])**0.5 #= 0 if row==col else
    #print('\n'.join(['\t'.join([str("{:.2f}".format(cell)) for cell in row]) for row in distance]))
    return distance


def create_graph(trk_x, trk_y, trk_z, trk_energy, N=1):
    distance = distance_matrix(trk_x, trk_y, trk_z)
    G = nx.Graph()
    for i in range(len(trk_energy)):
        G.add_node(i, pos=(trk_x[i], trk_y[i], trk_z[i]), energy=trk_energy[i])

        # sort indices by distance
        idx_by_distance = np.argsort(distance[i])

        # filter nodes with lower energy
        idx_above_current = filter(lambda x: trk_energy[x] > trk_energy[i], idx_by_distance)

        # add first N nodes with a higher energy
        for c, idx in enumerate(idx_above_current):
            if c == N:
                break
            G.add_edge(i, idx)
    return G


def load_tree(tree, N=2):
    vx = tree['vertices_x'].array()
    vy = tree['vertices_y'].array()
    vz = tree['vertices_z'].array()
    energy = tree['vertices_energy'].array()
    labels = tree['trackster_label'].array()
    for tx, ty, tz, te, tl in zip(vx, vy, vz, energy, labels):
        yield create_graph(tx, ty, tz, te, N=N), tl, te


def load_pairs(tree, N=2):
    for tx, ty, tz, te, cx, cy, cz, ce, pl, pe, pf in zip(
        tree['trackster_x'].array(),
        tree['trackster_y'].array(),
        tree['trackster_z'].array(),
        tree['trackster_energy'].array(),
        tree['candidate_x'].array(),
        tree['candidate_y'].array(),
        tree['candidate_z'].array(),
        tree['candidate_energy'].array(),
        tree['pair_label'].array(),
        tree['pair_event'].array(),
        tree['pair_fileid'].array(),
    ):
        t = create_graph(tx, ty, tz, te, N=N)
        c = create_graph(cx, cy, cz, ce, N=N)
        yield t, c, pl, pe, pf


def get_graphs(tracksters, eid, N=2):
    vx = tracksters["vertices_x"].array()[eid]
    vy = tracksters["vertices_y"].array()[eid]
    vz = tracksters["vertices_z"].array()[eid]
    ve = tracksters["vertices_energy"].array()[eid]
    return [create_graph(x, y, z, e, N=N) for x, y, z, e in zip(vx, vy, vz, ve)]