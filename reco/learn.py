import torch
import numpy as np


def train_edge_pred(model, device, optimizer, loss_func, train_dl):
    train_loss = 0.0
    count = 0.0
    model.train()

    train_true_seg = []
    train_pred_seg = []

    for data, edge_list, edge_length, labels in train_dl:

        if len(edge_list[0]) == 0:
            continue ## workaround

        data, seg = data[0].to(device), labels[0].to(device)
        batch_size = 1 #data.size()[0]

        optimizer.zero_grad()

        seg_pred = model(data, edge_list[0].T.type(torch.long))
        loss = loss_func(seg_pred.view(-1, 1), seg.view(-1, 1).type(torch.float))

        loss.backward()
        optimizer.step()

        count += batch_size
        train_loss += loss.item() * batch_size

        seg_np = seg.cpu().numpy()
        pred_np = seg_pred.detach().cpu().numpy()

        train_true_seg.append(seg_np.reshape(-1))
        train_pred_seg.append(pred_np.reshape(-1))

    train_true_cls = np.concatenate(train_true_seg).astype(int)
    train_pred_cls = (np.concatenate(train_pred_seg) > 0.5).astype(int)

    return train_loss, train_true_cls, train_pred_cls


@torch.no_grad()
def test_model(model, device, data):
    total = 0
    correct = 0
    for batch, labels in data:
        model.eval()
        batch = batch.to(device)
        labels = labels.to(device)
        z = model(batch).reshape(-1)
        prediction = (z > 0.5).type(torch.int)
        total += len(prediction)
        correct += sum(prediction == labels.type(torch.int))
    return (correct / total)