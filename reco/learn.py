import torch
import numpy as np


def train_edge_pred(model, device, optimizer, loss_func, train_dl, obj_cond=False):
    train_loss = 0.0
    model.train()

    train_true_seg = []
    train_pred_seg = []

    for data in train_dl:

        batch_size = len(data)
        data = data.to(device)

        optimizer.zero_grad()

        if obj_cond:
            seg_pred = model(data.x, data.edge_index, data.trackster_index)
        else:
            seg_pred = model(data.x, data.edge_index)

        loss = loss_func(seg_pred.view(-1, 1), data.y.view(-1, 1).type(torch.float))

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_size

        seg_np = data.y.cpu().numpy()
        pred_np = seg_pred.detach().cpu().numpy()

        train_true_seg.append(seg_np.reshape(-1))
        train_pred_seg.append(pred_np.reshape(-1))

    train_true_cls = np.concatenate(train_true_seg)
    train_pred_cls = np.concatenate(train_pred_seg)

    return train_loss, train_true_cls, train_pred_cls


@torch.no_grad()
def test_edge_pred(model, device, loss_func, test_dl, obj_cond=False):
    test_loss = 0.0
    model.eval()

    test_true_seg = []
    test_pred_seg = []
    for data in test_dl:

        batch_size = len(data)
        data = data.to(device)

        if obj_cond:
            seg_pred = model(data.x, data.edge_index, data.trackster_index)
        else:
            seg_pred = model(data.x, data.edge_index)

        loss = loss_func(seg_pred.view(-1, 1), data.y.view(-1, 1).type(torch.float))

        test_loss += loss.item() * batch_size

        seg_np = data.y.cpu().numpy()
        pred_np = seg_pred.detach().cpu().numpy()
        test_true_seg.append(seg_np.reshape(-1))
        test_pred_seg.append(pred_np.reshape(-1))

    test_true_cls = np.concatenate(test_true_seg)
    test_pred_cls = np.concatenate(test_pred_seg)

    return test_loss, test_true_cls, test_pred_cls
