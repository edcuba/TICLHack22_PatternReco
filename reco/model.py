import torch.nn as nn


class EdgeConvNet(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(EdgeConvNet, self).__init__()

        self.convnetwork = nn.Sequential(
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

    def forward(self, X):
        return self.convnetwork(X)