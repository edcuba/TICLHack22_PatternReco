import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, predictions, targets, gamma, alpha):
        """
        Binary focal loss, mean.

        https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/5
        with improvements for alpha.

        :param bce_loss: Binary Cross Entropy loss, a torch tensor.
        :param targets: a torch tensor containing the ground truth, 0s and 1s.
        :param gamma: focal loss power parameter, a float scalar.
        :param alpha: weight of the class indicated by 1, a float scalar.
        """
        bce_loss = F.binary_cross_entropy(predictions, targets)
        p_t = torch.exp(-bce_loss)
        alpha_tensor = (1 - alpha) + targets * (2 * alpha - 1)
        # alpha if target = 1 and 1 - alpha if target = 0
        f_loss = alpha_tensor * (1 - p_t) ** gamma * bce_loss
        return f_loss.mean()