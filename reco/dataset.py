import torch
import uproot
import random
import numpy as np
from os import walk, path
from torch.utils.data import Dataset

from .event import get_bary, get_candidate_pairs, remap_tracksters
from .matching import match_best_simtrackster, find_good_pairs
from .distance import euclidian_distance

from .graphs import get_graphs
from .features import get_graph_level_features


FEATURE_KEYS = [
    "barycenter_x",
    "barycenter_y",
    "barycenter_z",
    "raw_energy",       # total energy
    "raw_em_energy",    # electro-magnetic energy
    "EV1",              # eigenvalues of 1st principal component
    "EV2",
    "EV3",
    "eVector0_x",       # x of principal component
    "eVector0_y",
    "eVector0_z",
    "sigmaPCA1",        # error in the 1st principal axis
    "sigmaPCA2",
    "sigmaPCA3",
]



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
    confidence_threshold=0.5,
    best_only=True,
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

            if best_only:
                m_dist = min(dists)
                if m_dist < distance_threshold:
                    # this must be an array, not a tuple, awkward doesn't like tuples
                    pairs.append([tt_id, large_spt[np.argmin(dists)], m_dist])
            else:
                # add all possible edges, not just the best one
                for large_spt_id, dist in enumerate(dists):
                    if dist < distance_threshold:
                        pairs.append([tt_id, large_spt[large_spt_id], dist])

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


def build_pair_tensor(edge, features):
    a, b = edge
    fa = [f[a] for f in features]
    fb = [f[b] for f in features]
    return fa + fb


def get_pair_tensor_builder(tracksters, eid, dst_map):

    # extract all trackster features
    trackster_features = list([
        tracksters[k].array()[eid] for k in FEATURE_KEYS
    ])

    # this is pricey (and so are the graph features)
    graph_features = [
        get_graph_level_features(g) for g in get_graphs(tracksters, eid)
    ]

    ve = tracksters["vertices_energy"].array()[eid]

    return lambda edge: build_pair_tensor(edge, trackster_features) + [
        dst_map[(edge[0], edge[1])],    # pairwise distance
        len(ve[edge[0]]),               # num edges
        len(ve[edge[1]]),               # num edges
        *graph_features[edge[0]],
        *graph_features[edge[1]],
    ]


class TracksterPairs(Dataset):

    def __init__(
            self,
            name,
            root_dir,
            raw_data_path,
            transform=None,
            balanced=False,
            include_neutral=True,
            N_FILES=10,
            MAX_DISTANCE=10,
            ENERGY_THRESHOLD=10,
        ):
        self.name = name
        self.N_FILES = N_FILES
        self.MAX_DISTANCE = MAX_DISTANCE
        self.ENERGY_THRESHOLD = ENERGY_THRESHOLD
        self.raw_data_path = raw_data_path

        self.root_dir = root_dir
        self.transform = transform
        self.balanced = balanced
        self.include_neutral = include_neutral

        fn = self.processed_paths[0]

        if not path.exists(fn):
            self.process()

        dx, dy = torch.load(fn)
        self.x = torch.tensor(dx).type(torch.float)
        self.y = torch.tensor(dy).type(torch.float)

    @property
    def raw_file_names(self):
        files = []
        for (_, _, filenames) in walk(self.raw_data_path):
            files.extend(filenames)
            break
        full_paths = list([path.join(self.raw_data_path, f) for f in files])
        assert len(full_paths) >= self.N_FILES
        return full_paths[:self.N_FILES]

    @property
    def processed_file_names(self):
        infos = [
            self.name,
            f"{self.N_FILES}f",
            f"d{self.MAX_DISTANCE}",
            f"e{self.ENERGY_THRESHOLD}",
            "bal" if self.balanced else "nbal",
            "in" if self.include_neutral else "en",
        ]
        return list([f"pairs_{'_'.join(infos)}.pt"])

    @property
    def processed_paths(self):
        return [path.join(self.root_dir, fn) for fn in self.processed_file_names]

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

                candidate_pairs, dst_map = get_candidate_pairs(
                    tracksters,
                    graph,
                    eid,
                    max_distance=self.MAX_DISTANCE
                )

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
                neutral = find_good_pairs(tracksters, associations, not_matches, eid)

                if self.balanced:
                    # crucial step to get right!
                    take = min(len(matches), len(not_matches) - len(neutral))
                    positive = random.sample(list(matches), k=take)
                    negative = random.sample(list(not_matches - neutral), k=take)
                else:
                    positive = matches
                    negative = not_matches - neutral

                builder = get_pair_tensor_builder(tracksters, eid, dst_map)

                labels = [(positive, 1), (negative, 0)]
                if self.include_neutral:
                    labels.append((neutral, 0.5))

                for edges, label in labels:
                    for edge in edges:
                        sample = builder(edge)
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


class TracksterGraph(Dataset):

    def __init__(
            self,
            name,
            root_dir,
            raw_data_path,
            transform=None,
            balanced=False,
            include_neutral=True,
            N_FILES=10,
            MAX_DISTANCE=10,
            ENERGY_THRESHOLD=10,
        ):
        self.name = name
        self.N_FILES = N_FILES
        self.MAX_DISTANCE = MAX_DISTANCE
        self.ENERGY_THRESHOLD = ENERGY_THRESHOLD
        self.raw_data_path = raw_data_path

        self.root_dir = root_dir
        self.transform = transform
        self.balanced = balanced
        self.include_neutral = include_neutral

        fn = self.processed_paths[0]

        if not path.exists(fn):
            self.process()

        dx, dy = torch.load(fn)
        self.x = torch.tensor(dx).type(torch.float)
        self.y = torch.tensor(dy).type(torch.float)

    @property
    def raw_file_names(self):
        files = []
        for (_, _, filenames) in walk(self.raw_data_path):
            files.extend(filenames)
            break
        full_paths = list([path.join(self.raw_data_path, f) for f in files])
        assert len(full_paths) >= self.N_FILES
        return full_paths[:self.N_FILES]

    @property
    def processed_file_names(self):
        infos = [
            self.name,
            f"{self.N_FILES}f",
            f"d{self.MAX_DISTANCE}",
            f"e{self.ENERGY_THRESHOLD}",
            "bal" if self.balanced else "nbal",
            "in" if self.include_neutral else "en",
        ]
        return list([f"pairs_{'_'.join(infos)}.pt"])

    @property
    def processed_paths(self):
        return [path.join(self.root_dir, fn) for fn in self.processed_file_names]

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

                candidate_pairs, dst_map = get_candidate_pairs(
                    tracksters,
                    graph,
                    eid,
                    max_distance=self.MAX_DISTANCE
                )

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
                neutral = find_good_pairs(tracksters, associations, not_matches, eid)

                if self.balanced:
                    # crucial step to get right!
                    take = min(len(matches), len(not_matches) - len(neutral))
                    positive = random.sample(list(matches), k=take)
                    negative = random.sample(list(not_matches - neutral), k=take)
                else:
                    positive = matches
                    negative = not_matches - neutral

                builder = get_pair_tensor_builder(tracksters, eid, dst_map)

                labels = [(positive, 1), (negative, 0)]
                if self.include_neutral:
                    labels.append((neutral, 0.5))

                for edges, label in labels:
                    for edge in edges:
                        sample = builder(edge)
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



class PointCloudPairs(Dataset):

    def __init__(
            self,
            name,
            root_dir,
            raw_data_path,
            balanced=False,
            N_FILES=10,
            MAX_DISTANCE=10,
            ENERGY_THRESHOLD=10,
            padding=None,
    ):
        self.name = name
        self.N_FILES = N_FILES
        self.MAX_DISTANCE = MAX_DISTANCE
        self.ENERGY_THRESHOLD = ENERGY_THRESHOLD
        self.raw_data_path = raw_data_path
        self.root_dir = root_dir
        self.balanced = balanced

        fn = self.processed_paths[0]

        if not path.exists(fn):
            self.process()

        dx1, dx2, dy = torch.load(fn)
        self.x1 = dx1
        self.x2 = dx2
        self.y = torch.tensor(dy).type(torch.float)

        mx = max([len(x1[0]) + len(x2[0]) for x1, x2 in zip(self.x1, self.x2)])
        if not padding or padding < mx:
            print(f"Recommended padding: >{mx}")
        self.padding = padding


    @property
    def raw_file_names(self):
        files = []
        for (_, _, filenames) in walk(self.raw_data_path):
            files.extend(filenames)
            break
        full_paths = list([path.join(self.raw_data_path, f) for f in files])
        assert len(full_paths) >= self.N_FILES
        return full_paths[:self.N_FILES]

    @property
    def processed_file_names(self):
        infos = [
            self.name,
            f"{self.N_FILES}f",
            f"d{self.MAX_DISTANCE}",
            f"e{self.ENERGY_THRESHOLD}",
            "bal" if self.balanced else "nbal",
        ]
        return list([f"pc_pairs_{'_'.join(infos)}.pt"])

    @property
    def processed_paths(self):
        return [path.join(self.root_dir, fn) for fn in self.processed_file_names]

    def paddarray(self, arr):
        if self.padding:
            if len(arr) > self.padding:
                return arr[:self.padding]
            elif len(arr) < self.padding:
                return arr + [0] * (self.padding - len(arr))
        return arr

    def __getitem__(self, idx):
        x1 = self.x1[idx]
        x2 = self.x2[idx]
        y = self.y[idx]

        x = [self.paddarray(a + b) for a, b in zip(x1, x2)]
        y = self.paddarray([1] * len(x1[0]) + [y] * len(x2[0]))

        return torch.tensor(x), torch.tensor(y).type(torch.float)

    def __len__(self):
        return len(self.y)

    def __repr__(self):
        infos = [
            f"len={len(self)}",
            f"max_distance={self.MAX_DISTANCE}",
            f"balanced={self.balanced}",
            f"energy_threshold={self.ENERGY_THRESHOLD}"
        ]
        return f"<PointCloudPairs {' '.join(infos)}>"


    def process(self):
        dataset_X1 = []
        dataset_X2 = []
        dataset_Y = []

        for source in self.raw_file_names:
            print(f"Processing: {source}")

            tracksters = uproot.open({source: "ticlNtuplizer/tracksters"})
            simtracksters = uproot.open({source: "ticlNtuplizer/simtrackstersSC"})
            associations = uproot.open({source: "ticlNtuplizer/associations"})
            graph = uproot.open({source: "ticlNtuplizer/graph"})

            for eid in range(len(tracksters["vertices_x"].array())):

                candidate_pairs, _ = get_candidate_pairs(
                    tracksters,
                    graph,
                    eid,
                    max_distance=self.MAX_DISTANCE
                )

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
                neutral = find_good_pairs(tracksters, associations, not_matches, eid)

                if self.balanced:
                    # crucial step to get right!
                    take = min(len(matches), len(not_matches) - len(neutral))
                    positive = random.sample(list(matches), k=take)
                    negative = random.sample(list(not_matches - neutral), k=take)
                else:
                    positive = matches
                    negative = not_matches - neutral

                labels = [(positive, 1), (negative, 0)]

                vx = tracksters["vertices_x"].array()[eid].tolist()
                vy = tracksters["vertices_y"].array()[eid].tolist()
                vz = tracksters["vertices_z"].array()[eid].tolist()
                ve = tracksters["vertices_energy"].array()[eid].tolist()

                for edges, label in labels:
                    for (a, b) in edges:
                        x1 = [vx[a], vy[a], vz[a], ve[a]]
                        x2 = [vx[b], vy[b], vz[b], ve[b]]
                        dataset_X1.append(x1)
                        dataset_X2.append(x2)
                        dataset_Y.append(label)
        torch.save((dataset_X1, dataset_X2, dataset_Y), self.processed_paths[0])