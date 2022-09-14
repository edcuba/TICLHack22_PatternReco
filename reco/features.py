import numpy as np
import networkx as nx


def longest_path_from_highest_centrality(G):
    """
    Compute the longest path in graph from the point of highest (closeness) centrality
    """
    H = list(sorted(nx.closeness_centrality(G).items(), key=lambda x: x[1]))[-1][0]
    return max(nx.shortest_path_length(G, source=H).values())


def mean_edge_length(G):
    """
    Compute mean edge length in the graph
    """
    n_dists = 0
    for edge in G.edges():
        a, b = edge
        node_a = G.nodes[a]
        node_b = G.nodes[b]
        a_coord = np.array(node_a["pos"])
        b_coord = np.array(node_b["pos"])
        n_dists += np.linalg.norm(a_coord - b_coord)
    return np.mean(n_dists)


def mean_edge_energy_gap(G):
    """
    Compute mean edge energy difference in the graph
    """
    e_diffs = 0
    for edge in G.edges():
        a, b = edge
        node_a = G.nodes[a]
        node_b = G.nodes[b]
        e_diffs += abs(node_a["energy"] - node_b["energy"])

    return np.mean(e_diffs)