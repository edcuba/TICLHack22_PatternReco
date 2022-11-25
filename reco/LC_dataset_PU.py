from os import walk, path
import sys
import torch
import uproot

import awkward as ak

import numpy as np
from torch_geometric.data import Data, InMemoryDataset

from .datasetPU import get_major_PU_tracksters, get_trackster_representative_points, get_tracksters_in_cone


class LCGraphPU(InMemoryDataset):
    # about 200kb per file

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
        super(LCGraphPU, self).__init__(root_dir, transform, pre_transform, pre_filter)
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
        return list([f"LCGraphPU_{'_'.join(infos)}.pt"])

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
                in_cone = get_tracksters_in_cone(x1, x2, barycentres)
                indexes = [idx for idx, _ in in_cone]

                tvx = ak.flatten(vertices_x[indexes])
                tvy = ak.flatten(vertices_y[indexes])
                tvz = ak.flatten(vertices_z[indexes])
                tve = ak.flatten(vertices_e[indexes])
                lc_labels = ak.flatten(list([1 - reco2sim_score[idx][0]] * len(vertices_z[idx]) for idx in indexes))

                data_list.append(Data(
                    x=torch.tensor((tvx, tvy, tvz, tve)).T,
                    y=torch.tensor(lc_labels)
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
        return f"LCGraphPU({', '.join(infos)})"
