from torch.utils.data import random_split
from torch_geometric.loader import DataLoader


def split_geo_train_test(ds, batch_size=64, test_set_fraction=0.1):

    ds_size = len(ds)
    test_set_size = ds_size // int(1. / test_set_fraction)
    train_set_size = ds_size - test_set_size

    print(f"Train set: {train_set_size}, Test set: {test_set_size}")

    train_set, test_set = random_split(ds, [train_set_size, test_set_size])

    # this is very nice - handles the dimensions automatically
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_dl, test_dl