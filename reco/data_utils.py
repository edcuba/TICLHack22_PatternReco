import uproot
import random
import torch
import numpy as np
from os import walk

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from .graph_utils import load_tree, load_pairs
from .distance import euclidian_distance
from .dataset import match_trackster_pairs


class HGCALTracksters(InMemoryDataset):

    def __init__(self, root, kind="photon", transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        if kind == "photon":
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif kind == "pion":
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:
            raise RuntimeError("kind should be in ['pion', 'photon']")

    @property
    def raw_file_names(self):
        return ['trackster_tags_10ke_photon.root', 'trackster_tags_10ke_pion.root']

    @property
    def processed_file_names(self):
        return ['tags_photon.pt', 'tags_pion.pt']

    def process(self):
        for source, target in zip(self.raw_file_names, self.processed_paths):
            print(f"Processing: {source}")
            path = f"{self.root}/{source}"
            tracksters = uproot.open({path: "tracksters"})

            dataset = []
            for g, label, te in load_tree(tracksters, N=2):
                x = torch.tensor([pos for _, pos in g.nodes("pos")])
                edge_index = torch.tensor(list(g.edges())).T
                y = torch.tensor(label)
                dataset.append(Data(x, edge_index=edge_index, y=y, energy=torch.tensor(te)))

            data, slices = self.collate(dataset)
            torch.save((data, slices), target)


class TracksterPairs(InMemoryDataset):

    USE_FILES = 10
    MAX_DISTANCE = 10

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        data_path = "/Users/ecuba/data/multiparticle_complet/"
        files = []
        for (_, _, filenames) in walk(data_path):
            files.extend(filenames)
            break
        full_paths = list([data_path + f for f in files])
        return full_paths[:self.USE_FILES]

    @property
    def processed_file_names(self):
        return list([f'pairs_{i}.pt' for i in range(self.USE_FILES)])

    def build_tensor(self, edge, *args):
        a, b = edge
        fa = [f[a] for f in args]
        fb = [f[b] for f in args]
        return torch.tensor(fa + fb)

    def process(self):
        for source, target in zip(self.raw_file_names, self.processed_paths):
            print(f"Processing: {source}")

            tracksters = uproot.open({source: "ticlNtuplizer/tracksters"})
            simtracksters = uproot.open({source: "ticlNtuplizer/simtrackstersSC"})
            associations = uproot.open({source: "ticlNtuplizer/associations"})
            graph = uproot.open({source: "ticlNtuplizer/graph"})

            dataset = []

            for eid in range(len(tracksters["vertices_x"].array())):

                vx = tracksters["vertices_x"].array()[eid]
                vy = tracksters["vertices_y"].array()[eid]
                vz = tracksters["vertices_z"].array()[eid]
                clouds = [np.array([vx[tid], vy[tid], vz[tid]]).T for tid in range(len(vx))]

                inners_list = graph["linked_inners"].array()[eid]

                candidate_pairs = []

                for i, inners in enumerate(inners_list):
                    for inner in inners:
                        dst = euclidian_distance(clouds[i], clouds[inner])
                        if dst <= self.MAX_DISTANCE:
                            candidate_pairs.append((i, inner))

                gt_pairs = match_trackster_pairs(
                    tracksters,
                    simtracksters,
                    associations,
                    eid,
                    distance_threshold=self.MAX_DISTANCE,
                )

                ab_pairs = set([(a, b) for a, b, _ in gt_pairs])
                ba_pairs = set([(b, a) for a, b, _ in gt_pairs])
                c_pairs = set(candidate_pairs)

                matches = ab_pairs.union(ba_pairs).intersection(c_pairs)
                not_matches = c_pairs - matches

                take = min(len(matches), len(not_matches))

                positive = random.sample(list(not_matches), k=take)
                negative = random.sample(list(not_matches), k=take)

                bx = tracksters["barycenter_x"].array()[eid]
                by = tracksters["barycenter_y"].array()[eid]
                bz = tracksters["barycenter_z"].array()[eid]
                re = tracksters["raw_energy"].array()[eid]

                for edge in positive:
                    X = self.build_tensor(edge, bx, by, bz, re)
                    dataset.append(Data(
                        X.type(torch.float).reshape(1, -1),
                        y=torch.tensor(1),
                    ))

                for edge in negative:
                    X = self.build_tensor(edge, bx, by, bz, re)
                    dataset.append(Data(
                        X.type(torch.float).reshape(1, -1),
                        y=torch.tensor(0),
                    ))

            data, slices = self.collate(dataset)
            torch.save((data, slices), target)


class HGCALTracksterPairs(InMemoryDataset):

    def __init__(self, root, kind="photon", transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        if kind == "photon":
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif kind == "pion":
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:
            raise RuntimeError("kind should be in ['pion', 'photon']")

    @property
    def raw_file_names(self):
        return ['trackster_pairs_10ke_photon.root', 'trackster_pairs_10ke_pion.root']

    @property
    def processed_file_names(self):
        return ['pairs_photon.pt', 'pairs_pion.pt']

    def process(self):
        for source, target in zip(self.raw_file_names, self.processed_paths):
            print(f"Processing: {source}")
            path = f"{self.root}/{source}"
            pairs = uproot.open({path: "tracksters"})

            dataset = []
            for t, c, pl, pe, pf in load_pairs(pairs, N=2):
                # join both graphs into a single entry
                # and only add edges within the graphs
                x = torch.tensor(
                    list([pos for _, pos in t.nodes("pos")]) + list([pos for _, pos in c.nodes("pos")])
                )
                energy = torch.tensor(
                    list([e for _, e in t.nodes("energy")]) + list([e for _, e in c.nodes("energy")])
                )

                tlen = len(t.nodes())

                # offset the edges of the candidate graph
                edge_index = torch.tensor(
                    list(t.edges()) + list([(v0 + tlen, v1 + tlen) for v0, v1 in c.edges()])
                ).T

                y = torch.tensor(pl)
                dataset.append(Data(x, edge_index=edge_index, energy=energy, y=y, event=pe, fileid=pf))

            data, slices = self.collate(dataset)
            torch.save((data, slices), target)