import torch
import uproot
import random
import numpy as np
from os import walk, path
from torch.utils.data import Dataset


from .graph_utils import create_graph
from .features import longest_path_from_highest_centrality, mean_edge_length, mean_edge_energy_gap
from .event import get_bary, remap_tracksters
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
                pairs.append((tt_id, large_spt[np.argmin(dists)], m_dist))

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

    merge_map = {little: big for little, big, _ in e_pairs}

    return remap_tracksters(tracksters, merge_map, eid)


class TracksterPairs(Dataset):

    def __init__(
            self,
            root_dir,
            transform=None,
            balanced=False,
            N_FILES=10,
            MAX_DISTANCE=10,
            ENERGY_THRESHOLD=10,
        ):
        self.N_FILES = N_FILES
        self.MAX_DISTANCE = MAX_DISTANCE
        self.ENERGY_THRESHOLD = ENERGY_THRESHOLD

        self.root_dir = root_dir
        self.transform = transform
        self.balanced = balanced

        fn = self.processed_paths[0]

        if not path.exists(fn):
            self.process()

        dx, dy = torch.load(fn)
        self.x = torch.tensor(dx).type(torch.float)
        self.y = torch.tensor(dy).type(torch.float)

    @property
    def raw_file_names(self):
        data_path = "/Users/ecuba/data/multiparticle_complet/"
        files = []
        for (_, _, filenames) in walk(data_path):
            files.extend(filenames)
            break
        full_paths = list([data_path + f for f in files])
        return full_paths[:self.N_FILES]

    @property
    def processed_file_names(self):
        bal = "b" if self.balanced else "nb"
        return list([f"pairs_10p_{self.N_FILES}f_d{self.MAX_DISTANCE}_e{self.ENERGY_THRESHOLD}_{bal}.pt"])

    @property
    def processed_paths(self):
        return [path.join(self.root_dir, "processed", fn) for fn in self.processed_file_names]

    def build_tensor(self, edge, *args):
        a, b = edge
        fa = [f[a] for f in args]
        fb = [f[b] for f in args]
        return fa + fb

    def process(self):
        """
            for now, the dataset fits into the memory
            store in one file and load at once

            if needed, store in multiple files (processed file names)
            and implement a get method
        """
        dataset_X = []
        dataset_Y = []
        for source in self.raw_file_names:
            print(f"Processing: {source}")

            tracksters = uproot.open({source: "ticlNtuplizer/tracksters"})
            simtracksters = uproot.open({source: "ticlNtuplizer/simtrackstersSC"})
            associations = uproot.open({source: "ticlNtuplizer/associations"})
            graph = uproot.open({source: "ticlNtuplizer/graph"})


            for eid in range(len(tracksters["vertices_x"].array())):

                vx = tracksters["vertices_x"].array()[eid]
                vy = tracksters["vertices_y"].array()[eid]
                vz = tracksters["vertices_z"].array()[eid]
                ve = tracksters["vertices_energy"].array()[eid]
                clouds = [np.array([vx[tid], vy[tid], vz[tid]]).T for tid in range(len(vx))]

                inners_list = graph["linked_inners"].array()[eid]

                candidate_pairs = []
                dst_map = {}

                for i, inners in enumerate(inners_list):
                    for inner in inners:
                        dst = euclidian_distance(clouds[i], clouds[inner])
                        if dst <= self.MAX_DISTANCE:
                            candidate_pairs.append((i, inner))
                            dst_map[(i, inner)] = dst

                gt_pairs = match_trackster_pairs(
                    tracksters,
                    simtracksters,
                    associations,
                    eid,
                    energy_threshold=self.ENERGY_THRESHOLD,
                    distance_threshold=self.MAX_DISTANCE,
                )

                ab_pairs = set([(a, b) for a, b, _ in gt_pairs])
                ba_pairs = set([(b, a) for a, b, _ in gt_pairs])
                c_pairs = set(candidate_pairs)

                matches = ab_pairs.union(ba_pairs).intersection(c_pairs)
                not_matches = c_pairs - matches

                if self.balanced:
                    # crucial step to get right!
                    take = min(len(matches), len(not_matches))
                    positive = random.sample(list(matches), k=take)
                    negative = random.sample(list(not_matches), k=take)
                else:
                    positive = matches
                    negative = not_matches

                bx = tracksters["barycenter_x"].array()[eid]
                by = tracksters["barycenter_y"].array()[eid]
                bz = tracksters["barycenter_z"].array()[eid]
                re = tracksters["raw_energy"].array()[eid]      # total energy
                reme = tracksters["raw_em_energy"].array()[eid] # electro-magnetic energy
                ev1 = tracksters["EV1"].array()[eid]            # eigenvalues of 1st principal component
                ev2 = tracksters["EV2"].array()[eid]            # eigenvalues of 2nd principal component
                ev3 = tracksters["EV3"].array()[eid]            # eigenvalues of 3rd principal component
                evx = tracksters["eVector0_x"].array()[eid]     # x of principal component
                evy = tracksters["eVector0_y"].array()[eid]     # y of principal component
                evz = tracksters["eVector0_z"].array()[eid]     # z of principal component
                sp1 = tracksters["sigmaPCA1"].array()[eid]      # error in the 1st principal axis
                sp2 = tracksters["sigmaPCA2"].array()[eid]      # error in the 2nd principal axis
                sp3 = tracksters["sigmaPCA3"].array()[eid]      # error in the 3rd principal axis

                # this is pricey (and so are the graph features)
                graphs = [create_graph(x, y, z, e, N=2) for x, y, z, e in zip(vx, vy, vz, ve)]

                labels = ((positive, 1), (negative, 0))
                for edges, label in labels:
                    for edge in edges:
                        a, b = edge

                        # individual metrics
                        sample = self.build_tensor(
                            # trackster metric
                            edge,
                            bx,
                            by,
                            bz,
                            re,
                            reme,
                            ev1,
                            ev2,
                            ev3,
                            evx,
                            evy,
                            evz,
                            sp1,
                            sp2,
                            sp3,
                        ) + [
                            # extra metrics
                            longest_path_from_highest_centrality(graphs[a]),
                            longest_path_from_highest_centrality(graphs[b]),
                            mean_edge_energy_gap(graphs[a]),
                            mean_edge_energy_gap(graphs[b]),
                            mean_edge_length(graphs[a]),
                            mean_edge_length(graphs[b]),
                            len(ve[a]),
                            len(ve[b]),
                            dst_map[(a, b)]
                        ]
                        dataset_X.append(sample)
                        dataset_Y.append(label)

        torch.save((dataset_X, dataset_Y), self.processed_paths[0])

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

    def __repr__(self):
        infos = [
            f"len={len(self)}",
            f"balanced={self.balanced}",
            f"max_distance={self.MAX_DISTANCE}",
            f"energy_threshold={self.ENERGY_THRESHOLD}"
        ]
        return f"<TracksterPairs {' '.join(infos)}>"