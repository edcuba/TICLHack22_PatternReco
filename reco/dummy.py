import torch
import numpy as np


class DummyModel:
    def eval(self):
        pass

class DummyPleaser(DummyModel):
    def __call__(self, X):
        return torch.tensor(np.ones(len(X)))

class DummyGuesser(DummyModel):
    def __call__(self, X):
        return torch.tensor(np.random.random(len(X)))

class DummyScaler:
    def transform(self, X):
        return X.tolist()

class GraphNaiveDummy(DummyModel):
    def __call__(self, _, edge_list):
        return torch.tensor(np.ones(edge_list.shape[1])).type(torch.float)

class GraphRandomDummy(DummyModel):
    def __call__(self, _, edge_list):
        return torch.tensor(np.random.random(edge_list.shape[1])).type(torch.float)
