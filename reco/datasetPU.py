import awkward as ak
from os import walk, path
import sys

import torch
import uproot

import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset


from .dataset import FEATURE_KEYS, build_pair_tensor

from .features import get_graph_level_features, get_min_max_z_points
from .graphs import create_graph


def get_trackster_representative_points(bx, by, bz, min_z, max_z):
    # take a line (0,0,0), (bx, by, bz) -> any point on the line is t*(bx, by, bz)
    # compute the intersection with the min and max layer
    # beginning of the line: (minx, miny, minz) = t*(bx, by, bz)
    # minx = t*bx
    # miny = t*by
    # minz = t*bz : t = minz / bz
    t_min = min_z / bz
    t_max = max_z / bz
    x1 = np.array((t_min * bx, t_min * by, min_z))
    x2 = np.array((t_max * bx, t_max * by, max_z))
    return x1, x2


def get_tracksters_in_cone(x1, x2, barycentres, radius=10):
    in_cone = []
    for i, x0 in enumerate(barycentres):
        # barycenter between the first and last layer
        if x0[2] > x1[2] - radius and x0[2] < x2[2] + radius:
            # distance from the particle axis less than X cm
            d = np.linalg.norm(np.cross(x0 - x1, x0 - x2)) / np.linalg.norm(x2 - x1)
            if d < radius:
                in_cone.append((i, d))
    return in_cone


def get_major_PU_tracksters(
    reco2sim,
    sim_raw_energy,
    score_threshold=0.2,
):
    # assuming only one simtrackster to keep things easy
    big = []

    for recoT_idx, (sim_indexes, shared_energies, scores) in enumerate(reco2sim):
        for simT_idx, shared_energy, score in zip(sim_indexes, shared_energies, scores):
            # 2 goals here:
            # - find the trackster with >50% shared energy
            # - find the tracksters with < 0.2 score
            if score > score_threshold: continue

            st_energy = sim_raw_energy[simT_idx]
            st_fraction = shared_energy / st_energy

            if st_fraction > 0.5:
                big.append(recoT_idx)

    return big


class TracksterPairsPU(Dataset):
    # output is about 250kb per file

    def __init__(
            self,
            name,
            root_dir,
            raw_data_path,
            transform=None,
            N_FILES=None,
            radius=10,
            score_threshold=0.2,
        ):
        self.name = name
        self.N_FILES = N_FILES
        self.RADIUS = radius
        self.SCORE_THRESHOLD = score_threshold
        self.raw_data_path = raw_data_path
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
        files = []
        for (_, _, filenames) in walk(self.raw_data_path):
            files.extend(filenames)
            break
        full_paths = list([path.join(self.raw_data_path, f) for f in files])

        if self.N_FILES is None:
            self.N_FILES = len(full_paths)

        return full_paths[:self.N_FILES]

    @property
    def processed_file_names(self):
        infos = [
            self.name,
            f"f{self.N_FILES or len(self.raw_file_names)}",
            f"r{self.RADIUS}",
            f"s{self.SCORE_THRESHOLD}"
        ]
        return list([f"TracksterPairsProdPU_{'_'.join(infos)}.pt"])

    @property
    def processed_paths(self):
        return [path.join(self.root_dir, fn) for fn in self.processed_file_names]

    def process(self):
        dataset_X = []
        dataset_Y = []

        assert len(self.raw_file_names) == self.N_FILES

        for source in self.raw_file_names:
            print(f"Processing: {source}", file=sys.stderr)

            tracksters = uproot.open({source: "ticlNtuplizer/tracksters"})
            simtracksters = uproot.open({source: "ticlNtuplizer/simtrackstersSC"})
            associations = uproot.open({source: "ticlNtuplizer/associations"})
            clusters = uproot.open({source: "ticlNtuplizer/clusters"})

            assoc_data = associations.arrays([
                "tsCLUE3D_recoToSim_SC",
                "tsCLUE3D_recoToSim_SC_sharedE",
                "tsCLUE3D_recoToSim_SC_score",
            ])

            trackster_data = tracksters.arrays([
                "barycenter_x",
                "barycenter_y",
                "barycenter_z",
                "vertices_indexes"
            ])

            cluster_data = clusters.arrays([
                "position_x",
                "position_y",
                "position_z",
            ])

            simtrackster_data = simtracksters.arrays([
                "stsSC_raw_energy"
            ])

            for eid in range(len(trackster_data["barycenter_x"])):

                # get LC info
                clusters_x = cluster_data["position_x"][eid]
                clusters_y = cluster_data["position_y"][eid]
                clusters_z = cluster_data["position_z"][eid]

                # get trackster info
                barycenter_x = trackster_data["barycenter_x"][eid]
                barycenter_y = trackster_data["barycenter_y"][eid]
                barycenter_z = trackster_data["barycenter_z"][eid]

                # reconstruct trackster LC info
                vertices_indices = trackster_data["vertices_indexes"][eid]
                vertices_x = ak.Array([clusters_x[indices] for indices in vertices_indices])
                vertices_y = ak.Array([clusters_y[indices] for indices in vertices_indices])
                vertices_z = ak.Array([clusters_z[indices] for indices in vertices_indices])

                # get associations data
                reco2sim_index = assoc_data["tsCLUE3D_recoToSim_SC"][eid]
                reco2sim_score = assoc_data["tsCLUE3D_recoToSim_SC_score"][eid]
                reco2sim_sharedE = assoc_data["tsCLUE3D_recoToSim_SC_sharedE"][eid]
                sim_raw_energy = simtrackster_data["stsSC_raw_energy"][eid]

                bigTs = get_major_PU_tracksters(
                    zip(reco2sim_index, reco2sim_sharedE, reco2sim_score),
                    sim_raw_energy,
                )

                if not bigTs:
                    continue

                assert len(bigTs) == 1  # 1 particle with PU
                bigT = bigTs[0]

                x1, x2 = get_trackster_representative_points(
                    barycenter_x[bigT],
                    barycenter_y[bigT],
                    barycenter_z[bigT],
                    min(vertices_z[bigT]),
                    max(vertices_z[bigT])
                )

                barycentres = np.array((barycenter_x, barycenter_y, barycenter_z)).T
                in_cone = get_tracksters_in_cone(x1, x2, barycentres, radius=self.RADIUS)

                trackster_features = list([
                    tracksters[k].array()[eid] for k in FEATURE_KEYS
                ])

                big_minP, big_maxP = get_min_max_z_points(
                    vertices_x[bigT],
                    vertices_y[bigT],
                    vertices_z[bigT],
                )

                for recoTxId, distance in in_cone:

                    if recoTxId == bigT:
                        continue    # do not connect to itself

                    features = build_pair_tensor((bigT, recoTxId), trackster_features)

                    minP, maxP = get_min_max_z_points(
                        vertices_x[recoTxId],
                        vertices_y[recoTxId],
                        vertices_z[recoTxId],
                    )

                    # add trackster axes
                    features += big_minP
                    features += big_maxP
                    features += minP
                    features += maxP

                    features.append(distance)
                    features.append(len(vertices_z[bigT]))
                    features.append(len(vertices_z[recoTxId]))

                    label = 1 - reco2sim_score[recoTxId][0]

                    dataset_X.append(features)
                    dataset_Y.append(label)

        torch.save((dataset_X, dataset_Y), self.processed_paths[0])

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

    def __repr__(self):
        infos = [
            f"len={len(self)}",
            f"radius={self.RADIUS}",
            f"score_threshold={self.SCORE_THRESHOLD}"
        ]
        return f"<TracksterPairsProdPU {' '.join(infos)}>"



class TracksterGraphPU(InMemoryDataset):

    def __init__(
            self,
            name,
            root_dir,
            raw_data_path,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            N_FILES=None,
            radius=10,
            score_threshold=0.2,
        ):
        self.name = name
        self.N_FILES = N_FILES
        self.raw_data_path = raw_data_path
        self.root_dir = root_dir
        self.RADIUS = radius
        self.SCORE_THRESHOLD = score_threshold
        super(TracksterGraphPU, self).__init__(root_dir, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        files = []
        for (_, _, filenames) in walk(self.raw_data_path):
            files.extend(filenames)
            break
        full_paths = list([path.join(self.raw_data_path, f) for f in files])
        if self.N_FILES:
            assert len(full_paths) >= self.N_FILES
        return full_paths[:self.N_FILES]

    @property
    def processed_file_names(self):
        infos = [
            self.name,
            f"f{self.N_FILES or len(self.raw_file_names)}",
            f"r{self.RADIUS}",
            f"s{self.SCORE_THRESHOLD}"
        ]
        return list([f"TracksterGraphPU_{'_'.join(infos)}.pt"])

    @property
    def processed_paths(self):
        return [path.join(self.root_dir, fn) for fn in self.processed_file_names]

    def process(self):
        data_list = []

        for source in self.raw_file_names:
            print(source, file=sys.stderr)

            tracksters = uproot.open({source: "ticlNtuplizer/tracksters"})
            associations = uproot.open({source: "ticlNtuplizer/associations"})
            simtracksters = uproot.open({source: "ticlNtuplizer/simtrackstersSC"})
            clusters = uproot.open({source: "ticlNtuplizer/clusters"})

            assoc_data = associations.arrays([
                "tsCLUE3D_recoToSim_SC",
                "tsCLUE3D_recoToSim_SC_sharedE",
                "tsCLUE3D_recoToSim_SC_score",
            ])

            trackster_data = tracksters.arrays([
                "barycenter_x",
                "barycenter_y",
                "barycenter_z",
                "vertices_indexes",
            ])

            cluster_data = clusters.arrays([
                "position_x",
                "position_y",
                "position_z",
                "energy",
            ])

            simtrackster_data = simtracksters.arrays([
                "stsSC_raw_energy"
            ])

            for eid in range(len(trackster_data["barycenter_x"])):

                # get LC info
                clusters_x = cluster_data["position_x"][eid]
                clusters_y = cluster_data["position_y"][eid]
                clusters_z = cluster_data["position_z"][eid]
                clusters_e = cluster_data["energy"][eid]

                # get trackster info
                barycenter_x = trackster_data["barycenter_x"][eid]
                barycenter_y = trackster_data["barycenter_y"][eid]
                barycenter_z = trackster_data["barycenter_z"][eid]

                # reconstruct trackster LC info
                vertices_indices = trackster_data["vertices_indexes"][eid]
                vertices_x = ak.Array([clusters_x[indices] for indices in vertices_indices])
                vertices_y = ak.Array([clusters_y[indices] for indices in vertices_indices])
                vertices_z = ak.Array([clusters_z[indices] for indices in vertices_indices])
                vertices_e = ak.Array([clusters_e[indices] for indices in vertices_indices])

                # get associations data
                reco2sim_index = assoc_data["tsCLUE3D_recoToSim_SC"][eid]
                reco2sim_score = assoc_data["tsCLUE3D_recoToSim_SC_score"][eid]
                reco2sim_sharedE = assoc_data["tsCLUE3D_recoToSim_SC_sharedE"][eid]
                sim_raw_energy = simtrackster_data["stsSC_raw_energy"][eid]

                bigTs = get_major_PU_tracksters(
                    zip(reco2sim_index, reco2sim_sharedE, reco2sim_score),
                    sim_raw_energy,
                )

                if not bigTs:
                    continue

                assert len(bigTs) == 1  # 1 particle with PU
                bigT = bigTs[0]

                x1, x2 = get_trackster_representative_points(
                    barycenter_x[bigT],
                    barycenter_y[bigT],
                    barycenter_z[bigT],
                    min(vertices_z[bigT]),
                    max(vertices_z[bigT])
                )

                barycentres = np.array((barycenter_x, barycenter_y, barycenter_z)).T
                in_cone = get_tracksters_in_cone(x1, x2, barycentres, radius=self.RADIUS)

                trackster_features = list([
                    tracksters[k].array()[eid] for k in FEATURE_KEYS
                ])

                node_features = []
                node_labels = []

                for recoTxId, distance in in_cone:

                    recoTx_graph = create_graph(
                        vertices_x[recoTxId],
                        vertices_y[recoTxId],
                        vertices_z[recoTxId],
                        vertices_e[recoTxId],
                    )

                    features = [int(recoTxId == bigT), distance, len(vertices_z[recoTxId])]
                    features += [f[recoTxId] for f in trackster_features]
                    features += get_graph_level_features(recoTx_graph)

                    node_features.append(features)
                    node_labels.append(1 - reco2sim_score[recoTxId][0])

                data_list.append(Data(
                    x=torch.tensor(node_features, dtype=torch.float),
                    y=torch.tensor(node_labels, dtype=torch.float),
                ))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        infos = [
            f"graphs={len(self)}",
            f"nodes={len(self.data.x)}",
            f"radius={self.RADIUS}",
            f"score_threshold={self.SCORE_THRESHOLD}",
        ]
        return f"TracksterGraphPU({', '.join(infos)})"
