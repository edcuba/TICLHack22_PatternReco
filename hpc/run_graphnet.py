# %%
import numpy as np
import torch
import sys

import torch.nn as nn
from torch.optim import SGD
from torch_cluster import knn_graph
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from reco.model import EdgeConvBlock

import sklearn.metrics as metrics

from reco.training import train_edge_pred, test_edge_pred, roc_auc
from reco.loss import FocalLoss
from reco.datasetPU import TracksterGraphPU


ds_name = "CloseByGamma200PUFull"

data_root = "data"
raw_dir = f"/Users/ecuba/data/{ds_name}"

# %%
# CUDA Setup
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
def create_mask(data):
    # extract the focus feature
    data.mask = (1 - data.x[:,0]).type(torch.bool)
    return data

def knn_transform(data):
    # pos coordinates are on position 3:6
    data.edge_index = knn_graph(data.x[:,3:6], k=8, loop=False)
    return data

transforms = T.Compose([knn_transform, create_mask])

# %%

ds = TracksterGraphPU(
    ds_name,
    data_root,
    raw_dir,
    transform=transforms,
    N_FILES=464,
    radius=10,
)

# %%
ds_size = len(ds)
test_set_size = ds_size // 10
train_set_size = ds_size - test_set_size
train_set, test_set = random_split(ds, [train_set_size, test_set_size])
print(f"Train graphs: {len(train_set)}, Test graphs: {len(test_set)}")

train_dl = DataLoader(train_set, batch_size=32, shuffle=True)
test_dl = DataLoader(test_set, batch_size=32, shuffle=True)

# %%
to_predict = []
for data in ds:
    to_predict += data.y[data.mask].tolist()
query_labels = np.array(to_predict)

print("Labels (one per trackster minus the main trackster):", len(query_labels))
print("Positive:", int((query_labels > 0.8).astype(int).sum()))
print("Negative:", int((query_labels < 0.8).astype(int).sum()))

# %%
balance = float(sum(query_labels > 0.8) / len(query_labels))
print(f"dataset balance: {balance * 100:.2f}%")

# %%
class TracksterGraphNet(nn.Module):
    def __init__(self, input_dim, output_dim=1, dropout=0.2):
        super(TracksterGraphNet, self).__init__()

        hdim1 = 64
        hdim2 = 128
        hdim_fc = 256

        # EdgeConv
        self.graphconv1 = EdgeConvBlock(input_dim, hdim1)
        self.graphconv2 = EdgeConvBlock(hdim1, hdim2)

        # Edge features from node embeddings for classification
        self.edgenetwork = nn.Sequential(
            nn.Linear(hdim2, hdim_fc),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hdim_fc, output_dim),
            nn.Sigmoid()
        )

    def forward(self, X, edge_index):
        H = self.graphconv1(X, edge_index)
        H = self.graphconv2(H, edge_index)
        return self.edgenetwork(H).squeeze(-1)

# %%
model = TracksterGraphNet(input_dim=ds.data.x.shape[1])
epochs = 101
model_path = f"models/TracksterGraphNet.KNN.mask.64.128.256.ns.{epochs}e-{ds_name}.{ds.RADIUS}.{ds.SCORE_THRESHOLD}.{ds.N_FILES}f.pt"

# alpha - percentage of negative edges
loss_func = FocalLoss(alpha=0.25, gamma=2)

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

    if epoch % 10 == 0:
        test_loss, test_true, test_pred = test_edge_pred(model, device, loss_func, test_dl)
        test_auc = metrics.roc_auc_score((test_true > 0.8).astype(int), test_pred)
        print(
            f"Epoch {epoch}",
            f"\ttrain loss:{train_loss:.3f}\ttrain auc: {train_auc:.3f}",
            f"\t test loss:{test_loss:.3f} \t test auc: {test_auc:.3f}",
            file=sys.stderr
        )

torch.save(model.state_dict(), model_path)
print(model_path)

# %%
print(roc_auc(model, device, test_dl))