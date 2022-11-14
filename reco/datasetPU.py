from os import walk, path

import torch
import uproot

import numpy as np
from torch.utils.data import Dataset

from .dataset import FEATURE_KEYS, build_pair_tensor

from .features import get_graph_level_features
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


def get_tracksters_in_cone(x1, x2, barycentres, radius=15):
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
    score_threshold=0.3,
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

    def __init__(
            self,
            name,
            root_dir,
            raw_data_path,
            transform=None,
            N_FILES=10,
            radius=15,
            score_threshold=0.3,
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
        return full_paths[:self.N_FILES]

    @property
    def processed_file_names(self):
        infos = [
            self.name,
            f"f{self.N_FILES}",
            f"r{self.RADIUS}",
            f"s{self.SCORE_THRESHOLD}"
        ]
        return list([f"TracksterPairsPU_{'_'.join(infos)}.pt"])

    @property
    def processed_paths(self):
        return [path.join(self.root_dir, fn) for fn in self.processed_file_names]

    def process(self):
        dataset_X = []
        dataset_Y = []

        assert len(self.raw_file_names) == self.N_FILES


        for source in self.raw_file_names:
            print(f"Processing: {source}")

            tracksters = uproot.open({source: "ticlNtuplizer/tracksters"})
            simtracksters = uproot.open({source: "ticlNtuplizer/simtrackstersSC"})
            associations = uproot.open({source: "ticlNtuplizer/associations"})

            reco2sim_index_ = associations["tsCLUE3D_recoToSim_SC"].array()
            reco2sim_shared_ = associations["tsCLUE3D_recoToSim_SC_sharedE"].array()
            reco2sim_score_ = associations["tsCLUE3D_recoToSim_SC_score"].array()

            sim_raw_energy_ = simtracksters["stsSC_raw_energy"].array()
            barycenter_x_ = tracksters["barycenter_x"].array()
            barycenter_y_ = tracksters["barycenter_y"].array()
            barycenter_z_ = tracksters["barycenter_z"].array()

            vertices_x_ = tracksters["vertices_x"].array()
            vertices_y_ = tracksters["vertices_y"].array()
            vertices_z_ = tracksters["vertices_z"].array()
            vertices_e_ = tracksters["vertices_energy"].array()

            for eid in range(len(vertices_z_)):
                vertices_x = vertices_x_[eid]
                vertices_y = vertices_y_[eid]
                vertices_z = vertices_z_[eid]
                vertices_e = vertices_e_[eid]
                barycenter_x = barycenter_x_[eid]
                barycenter_y = barycenter_y_[eid]
                barycenter_z = barycenter_z_[eid]

                reco2sim_score = reco2sim_score_[eid]

                bigTs = get_major_PU_tracksters(
                    zip(reco2sim_index_[eid], reco2sim_shared_[eid], reco2sim_score),
                    sim_raw_energy_[eid],
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
                in_cone = get_tracksters_in_cone(x1, x2, barycentres)

                trackster_features = list([
                    tracksters[k].array()[eid] for k in FEATURE_KEYS
                ])

                bigT_graph = create_graph(
                    vertices_x[bigT],
                    vertices_y[bigT],
                    vertices_z[bigT],
                    vertices_e[bigT],
                )

                bigT_graph_features = get_graph_level_features(bigT_graph)

                for recoTxId, distance in in_cone:
                    # get features for each reco trackster... pairwise?
                    # graph-wise?
                    # start pairwise (look at BiPartite Graphs in Pytorch Geometric)
                    if recoTxId == bigT:
                        continue    # do not connect to itself

                    recoTx_graph = create_graph(
                        vertices_x[recoTxId],
                        vertices_y[recoTxId],
                        vertices_z[recoTxId],
                        vertices_e[recoTxId],
                    )
                    recoTx_graph_features = get_graph_level_features(recoTx_graph)

                    features = build_pair_tensor((bigT, recoTxId), trackster_features)
                    features += bigT_graph_features
                    features += recoTx_graph_features

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
        return f"<TracksterPairsPU {' '.join(infos)}>"