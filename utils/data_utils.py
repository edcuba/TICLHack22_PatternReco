import uproot
import torch

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from .graph_utils import load_tree, load_pairs


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


class BaseTracksterPairs(InMemoryDataset):

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
        return ['base_pairs_photon.pt', 'base_pairs_pion.pt']

    def process(self):
        for source, target in zip(self.raw_file_names, self.processed_paths):
            print(f"Processing: {source}")
            path = f"{self.root}/{source}"
            pairs = uproot.open({path: "tracksters"})

            dataset = []
            te = pairs['trackster_energy'].array()

            for te, ce, pl in zip(
                pairs['trackster_energy'].array(),
                pairs['candidate_energy'].array(),
                pairs['pair_label'].array()
            ):
                dataset.append(
                    Data(torch.tensor([len(te), len(ce), sum(te), sum(ce)]).reshape(1, -1), y=torch.tensor(pl))
                )

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