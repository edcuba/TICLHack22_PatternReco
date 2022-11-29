# %%
import torch
import sys
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import random_split
from torch_geometric.nn import DynamicEdgeConv
from torch_geometric.loader import DataLoader

import sklearn.metrics as metrics

from reco.training import train_edge_pred, test_edge_pred, roc_auc
from reco.loss import FocalLoss
from reco.datasetLCPU import LCGraphPU

ds_name = "CloseByGamma200PUFull"

data_root = "/mnt/ceph/users/ecuba/processed"
raw_dir = f"/mnt/ceph/users/ecuba/{ds_name}"

# %%
# CUDA Setup
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
ds = LCGraphPU(
    ds_name + ".2",
    data_root,
    raw_dir,
    N_FILES=464,
    radius=10,
)

print(ds.processed_file_names)

# %%
ds_size = len(ds)
test_set_size = ds_size // 10
train_set_size = ds_size - test_set_size
train_set, test_set = random_split(ds, [train_set_size, test_set_size])
print(f"Train graphs: {len(train_set)}, Test graphs: {len(test_set)}")

# this is very nice - handles the dimensions automatically
train_dl = DataLoader(train_set, batch_size=32, shuffle=True)
test_dl = DataLoader(test_set, batch_size=32, shuffle=True)

# %%
print("Labels (one per layer-cluster):", len(ds.data.y))

# %%
balance = float(sum(ds.data.y > 0.8) / len(ds.data.y))
print(f"dataset balance: {balance * 100:.2f}%")

# %%
class EdgeConvBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, aggr="add", skip_link=False, k=8):
        super(EdgeConvBlock, self).__init__()

        convnetwork = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.dynamicgraphconv = DynamicEdgeConv(nn=convnetwork, aggr=aggr, k=k)
        self.skip_link = skip_link

    def forward(self, X, edge_index=None):
        H = self.dynamicgraphconv(X)

        if self.skip_link:
            return torch.hstack((H, X))

        return H


class LCGraphNet(nn.Module):
    def __init__(self, input_dim, output_dim=1, dropout=0.2, skip_link=False):
        super(LCGraphNet, self).__init__()
        # particlenet light

        hdim1 = 64
        in_dim2 = hdim1 + input_dim if skip_link else hdim1

        hdim2 = 128
        in_dim3 = hdim2 + in_dim2 if skip_link else hdim2

        hdim3 = 256

        # EdgeConv
        self.graphconv1 = EdgeConvBlock(input_dim, hdim1, skip_link=skip_link)
        self.graphconv2 = EdgeConvBlock(in_dim2, hdim2, skip_link=skip_link)

        self.edgenetwork = nn.Sequential(
            nn.Linear(in_dim3, hdim3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hdim3, output_dim),
            nn.Sigmoid()
        )

    def forward(self, X, _edge_index=None):
        H = self.graphconv1(X)
        H = self.graphconv2(H)
        return self.edgenetwork(H).squeeze(-1)

# %%
model = LCGraphNet(input_dim=ds.data.x.shape[1], skip_link=False)
epochs = 201
model_path = f"models/LCGraphNet.64.128.256.ns.{epochs}e-{ds_name}.{ds.RADIUS}.{ds.SCORE_THRESHOLD}.{ds.N_FILES}f.pt"

# %%
loss_func = FocalLoss(alpha=1-balance, gamma=2)

model = model.to(device)
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=1e-3)

for epoch in range(epochs):

    train_loss, train_true, train_pred = train_edge_pred(
        model,
        device,
        optimizer,
        loss_func,
        train_dl
    )

    train_auc = metrics.roc_auc_score((train_true > 0.8).astype(int), train_pred)
    scheduler.step()

    if epoch % 5 == 0:
        test_loss, test_true, test_pred = test_edge_pred(model, device, loss_func, test_dl)
        test_auc = metrics.roc_auc_score((test_true > 0.8).astype(int), test_pred)
        print(
            f"Epoch {epoch}:",
            f"\ttrain loss:{train_loss:.2f}\ttrain auc: {train_auc:.3f}",
            f"\t test loss:{test_loss:.2f} \t test auc: {test_auc:.3f}",
            file=sys.stderr
        )

torch.save(model.state_dict(), model_path)

# %%
print(roc_auc(model, device, test_dl))