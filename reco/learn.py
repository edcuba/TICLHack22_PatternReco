import torch
import numpy as np


def train_edge_pred(model, device, optimizer, loss_func, train_dl, batch_size):
    train_loss = 0.0
    model.train()

    train_true_seg = []
    train_pred_seg = []

    for data, edge_list, labels in train_dl:

        data, seg = data.to(device), labels.to(device)

        optimizer.zero_grad()

        seg_pred = model(data, edge_list.T.type(torch.long))
        loss = loss_func(seg_pred.view(-1, 1), seg.view(-1, 1).type(torch.float))

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_size

        seg_np = seg.cpu().numpy()
        pred_np = seg_pred.detach().cpu().numpy()

        train_true_seg.append(seg_np.reshape(-1))
        train_pred_seg.append(pred_np.reshape(-1))

    train_true_cls = np.concatenate(train_true_seg)
    train_pred_cls = np.concatenate(train_pred_seg)

    return train_loss, train_true_cls, train_pred_cls


@torch.no_grad()
def test_edge_pred(model, device, loss_func, test_dl, batch_size):
    test_loss = 0.0
    model.eval()

    test_true_seg = []
    test_pred_seg = []
    for data, edge_list, labels in test_dl:
        data, seg = data.to(device), labels.to(device)

        seg_pred = model(data, edge_list.T.type(torch.long))

        loss = loss_func(seg_pred.view(-1, 1), seg.view(-1, 1).type(torch.float))

        test_loss += loss.item() * batch_size

        seg_np = seg.cpu().numpy()
        pred_np = seg_pred.detach().cpu().numpy()
        test_true_seg.append(seg_np.reshape(-1))
        test_pred_seg.append(pred_np.reshape(-1))

    test_true_cls = np.concatenate(test_true_seg)
    test_pred_cls = np.concatenate(test_pred_seg)

    return test_loss, test_true_cls, test_pred_cls
