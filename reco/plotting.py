import numpy as np
import awkward as ak

import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


from .event import remap_arrays_by_label


def plot_trackster(ax, label, x, y, z, e):
    ax.scatter(x, y, z, label=label, s=np.array(e)*2)


def plot_tracksters(ax, vx, vy, vz, ve):
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    if ve is None:
        ve = np.ones(len(vx))*10
    for i, x, y, z, e in zip(range(len(vx)), vx, vy, vz, ve):
        plot_trackster(ax, i, x, y, z, e)


def get_event_window(ex, ey, ez, ee, zoom=1, e_threshold=2):

    ef = np.where(ak.flatten(ee) > e_threshold)

    x_max = max(ak.flatten(ex)[ef])
    x_min = min(ak.flatten(ex)[ef])
    y_max = max(ak.flatten(ey)[ef])
    y_min = min(ak.flatten(ey)[ef])
    z_max = max(ak.flatten(ez)[ef])
    z_min = min(ak.flatten(ez)[ef])

    x_zoom = (x_max - x_min) / zoom
    x_max -= x_zoom
    x_min += x_zoom

    y_zoom = (y_max - y_min) / zoom
    y_max -= y_zoom
    y_min += y_zoom

    z_zoom = (z_max - z_min) / zoom
    z_max -= z_zoom
    z_min += z_zoom

    return (x_min, x_max), (y_min, y_max), (z_min, z_max)



def plot_sim_reco(
    vx,
    vy,
    vz,
    ve,
    svx,
    svy,
    svz,
    sve,
    svi,
    svm,
    zoom=1,
    fig=None,
    align_dim=True,
    energy_threshold=1,
):
    # get approximate plottable area
    fig = fig or plt.figure(figsize=(12, 10))
    xlim, ylim, zlim = get_event_window(svx, svy, svz, sve, zoom=zoom, e_threshold=energy_threshold)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_zlim(zlim)
    ax2.set_xlabel("x (cm)")
    ax2.set_ylabel("y (cm)")
    ax2.set_zlabel("z (cm)")

    # keep only the lowest multiplicity per vertex
    plot_set = {}
    for vi, vm in zip(svi, svm):
        for i, m in zip(vi, vm):
            if i in plot_set and plot_set[i] < m:
                continue
            plot_set[i] = m

    for ti, tx, ty, tz, te, vi, vm in zip(range(len(svx)), svx, svy, svz, sve, svi, svm):
        _tx = []
        _ty = []
        _tz = []
        _te = []
        for x, y, z, e, i, m in zip(tx, ty, tz, te, vi, vm):
            # do not replot points with higher multiplicity
            if plot_set[i] != m:
                continue
            _tx.append(x)
            _ty.append(y)
            _tz.append(z)
            _te.append(e)
        plot_trackster(ax2, ti, _tx, _ty, _tz, _te)

    # plot reco
    if not align_dim:
        xlim, ylim, zlim = get_event_window(vx, vy, vz, ve, zoom=zoom, e_threshold=energy_threshold)

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_zlim(zlim)
    plot_tracksters(ax1, vx, vy, vz, ve)

    return ax1, ax2



def plot_event(tracksters, simtracksters, eid, legend=True):
    """
    Plot Reco and Sim tracksters in the event
    """
    # get the layerclusters for event eid
    vx = tracksters["vertices_x"].array()[eid]
    vy = tracksters["vertices_y"].array()[eid]
    vz = tracksters["vertices_z"].array()[eid]
    ve = tracksters["vertices_energy"].array()[eid]

    svx = simtracksters["stsSC_vertices_x"].array()[eid]
    svy = simtracksters["stsSC_vertices_y"].array()[eid]
    svz = simtracksters["stsSC_vertices_z"].array()[eid]
    sve = simtracksters["stsSC_vertices_energy"].array()[eid]
    svi = simtracksters["stsSC_vertices_indexes"].array()[eid]
    svm = simtracksters["stsSC_vertices_multiplicity"].array()[eid]

    plot_sim_reco(vx, vy, vz, ve, svx, svy, svz, sve, svi, svm, eid, legend=legend)


def plot_fractions_hist(all_fractions, complete_fractions, incomplete_fractions):
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(131)
    ax.set_title("Highest shared energy fraction")
    ax.hist(all_fractions, bins=20)
    ax = fig.add_subplot(132)
    ax.set_title("Energy fractions of complete tracksters")
    ax.hist(complete_fractions, bins=20)
    ax = fig.add_subplot(133)
    ax.set_title("Energy fractions of incomplete tracksters")
    ax.hist(incomplete_fractions, bins=20)


def plot_graph_3D(G, ax=None, edges=True, s=30):
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # 3D network plot
    with plt.style.context(('ggplot')):

        if ax == None:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        xi = []
        yi = []
        zi = []
        ei = []
        clr = []
        for k, value in pos.items():
            xi.append(value[0])
            yi.append(value[1])
            zi.append(value[2])
            ei.append(G.nodes[k].get("energy", 1))
            clr.append(G.nodes[k].get("color", "blue"))

        lc_size = np.array(ei) * s
        #ax.scatter(xi, yi, zi, s=lc_size, c=clr, cmap='rainbow')
        ax.scatter(xi, yi, zi, s=lc_size)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        if edges:
            for j in G.edges():
                x = np.array((pos[j[0]][0], pos[j[1]][0]))
                y = np.array((pos[j[0]][1], pos[j[1]][1]))
                z = np.array((pos[j[0]][2], pos[j[1]][2]))

                # Plot the connecting lines
                ax.plot(x, y, z, c=G.edges[j].get("color", "black"), alpha=0.4)
    if ax == None:
        plt.show()


def plot_remapped(tracksters, eid, labels):
    rx = remap_arrays_by_label(tracksters["vertices_x"].array()[eid], labels)
    ry = remap_arrays_by_label(tracksters["vertices_y"].array()[eid], labels)
    rz = remap_arrays_by_label(tracksters["vertices_z"].array()[eid], labels)
    re = remap_arrays_by_label(tracksters["vertices_energy"].array()[eid], labels)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_tracksters(ax, rx, ry, rz, re)
    plt.show()


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)