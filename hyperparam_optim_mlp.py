import torch
import os
from ray import tune
from functools import partial

from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split, DataLoader
from reco.loss import FocalLoss

from reco.datasetPU import TracksterPairsPU
from reco.training import roc_auc, train_mlp


def get_model(ds, config):

    hdim1 = config["hdim1"]
    hdim2 = config["hdim2"]
    hdim3 = config["hdim3"]

    if config["activation"] == "relu":
        act = nn.ReLU
    elif config["activation"] == "sigmoid":
        act = nn.Sigmoid
    elif config["activation"] == "leakyrelu":
        act = nn.LeakyReLU
    else:
        raise RuntimeError("Activation function %s not recognized", config["activation"])

    model = nn.Sequential(
        nn.BatchNorm1d(ds.x.shape[1], affine=False),
        nn.Linear(ds.x.shape[1], hdim1),
        act(),
        nn.Linear(hdim1, hdim2),
        act(),
        nn.Linear(hdim2, hdim3),
        act(),
        nn.Linear(hdim3, 1),
        nn.Dropout(p=config["dropout"]),
        nn.Sigmoid()
    )

    return model





def train_model(config, ds_name=None, data_root=None, raw_dir=None, checkpoint_dir=None):

    print(ds_name, data_root, raw_dir)

    ds = TracksterPairsPU(
        ds_name,
        data_root,
        raw_dir,
        N_FILES=464,
        radius=10
    )
    ds_size = len(ds)

    epochs = 50
    model = get_model(ds, config)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    loss_obj = FocalLoss(alpha=config["focal_alpha"], gamma=2)

    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    else:
        raise RuntimeError("Optimizer %s not recognized", config["optimizer"])

    scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=1e-3)

    test_set_size = ds_size // 10
    train_set_size = ds_size - test_set_size
    train_set, test_set = random_split(ds, [train_set_size, test_set_size])
    train_dl = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_dl = DataLoader(test_set, batch_size=config["batch_size"], shuffle=True)

    for epoch in range(epochs):
        loss = train_mlp(model, device, optimizer, train_dl, loss_obj)
        val_auc = roc_auc(model, device, val_dl)
        scheduler.step()

        tune.report(loss=loss, auc=val_auc)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)


config = {
    "hdim1": tune.choice([16, 32, 64, 128, 256, 512]),
    "activation": tune.choice(["relu", "sigmoid", "leakyrelu"]),
    "hdim2": tune.choice([8, 16, 32, 64, 128, 256]),
    "hdim3": tune.choice([8, 16, 32, 64, 128, 256]),
    "lr": tune.loguniform(1e-4, 1e-2),
    "batch_size": tune.choice([8, 16, 32, 64]),
    "dropout": tune.loguniform(0.1, 0.5),
    "focal_alpha": tune.loguniform(0.1, 0.9),
    "optimizer": tune.choice(["sgd", "adam"]),
}

ray_scheduler = ASHAScheduler(
    metric="auc",
    mode="max",
    grace_period=1,
    reduction_factor=2
)

reporter = CLIReporter(
    metric_columns=["loss", "auc", "training_iteration"]
)


data_root = "/Users/ecuba/devel/pattern-reco/data"
ds_name = "CloseByGamma200PUFull"
raw_dir = f"/Users/ecuba/data/{ds_name}"

result = tune.run(
    partial(train_model, ds_name=ds_name, data_root=data_root, raw_dir=raw_dir),
    config=config,
    num_samples=10,
    scheduler=ray_scheduler,
    progress_reporter=reporter
)

best_trial = result.get_best_trial("auc", "max", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation auc: {}".format(best_trial.last_result["auc"]))