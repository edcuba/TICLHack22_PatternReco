import torch
import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, fbeta_score, balanced_accuracy_score



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


def precision_recall_curve(model, device, test_dl, beta=0.5):
    th_values = [i / 100. for i in range(1, 100)]
    precision = []
    recall = []
    fbeta = []
    b_acc = []
    cm = []
    acc = []

    for th in th_values:
        pred = []
        lab = []
        for b, l in test_dl:
            b = b.to(device)
            l = l.to(device)
            pred += (model(b) > th).type(torch.int).tolist()
            lab += (l > th).type(torch.int).tolist()
        precision.append(precision_score(lab, pred, zero_division=0))
        recall.append(recall_score(lab, pred))
        fbeta.append(fbeta_score(lab, pred, beta=beta))
        b_acc.append(balanced_accuracy_score(lab, pred))
        cm.append(confusion_matrix(lab, pred).ravel())
        acc.append(accuracy_score(lab, pred))

    plt.figure()
    plt.plot(th_values, precision, label="precision")
    plt.plot(th_values, b_acc, label="b_acc")
    plt.plot(th_values, recall, label="recall")
    plt.plot(th_values, fbeta, label=f"f{beta}")
    plt.xlabel("Threshold")
    plt.legend()
    plt.show()

    bi = np.argmax(fbeta)
    decision_th = th_values[bi]

    tn, fp, fn, tp = cm[bi]
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f'TH: {decision_th} Acc: {acc[bi]:.3f} BAcc: {b_acc[bi]:.3f} P: {precision[bi]:.3f} R: {recall[bi]:.3f} F{beta}: {fbeta[bi]:.3f}')