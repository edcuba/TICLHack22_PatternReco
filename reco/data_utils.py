import uproot
import random
import torch
import numpy as np
from os import walk, path

from torch.utils.data import Dataset

from .distance import euclidian_distance
from .dataset import match_trackster_pairs

class TracksterPairs(Dataset):

    def __init__(
            self,
            root_dir,
            transform=None,
            N_FILES=10,
            MAX_DISTANCE=10,
            ENERGY_THRESHOLD=10,
        ):
        self.N_FILES = N_FILES
        self.MAX_DISTANCE = MAX_DISTANCE
        self.ENERGY_THRESHOLD = ENERGY_THRESHOLD

        self.root_dir = root_dir
        self.transform = transform

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
        return list([f"pairs_10p_{self.N_FILES}f_d{self.MAX_DISTANCE}_e{self.ENERGY_THRESHOLD}.pt"])

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
                    energy_threshold=self.ENERGY_THRESHOLD,
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
                    dataset_X.append(self.build_tensor(edge, bx, by, bz, re))
                    dataset_Y.append(1)

                for edge in negative:
                    dataset_X.append(self.build_tensor(edge, bx, by, bz, re))
                    dataset_Y.append(0)

        torch.save((dataset_X, dataset_Y), self.processed_paths[0])

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

    def __repr__(self):
        return f"<TracksterPairs len={len(self)}>"