import uproot
import torch

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from graph_utils import load_tree


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
        return ['tracksters_ds_100e.root', 'tracksters_ds_pion.root']

    @property
    def processed_file_names(self):
        return ['photon.pt', 'pion.pt']

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