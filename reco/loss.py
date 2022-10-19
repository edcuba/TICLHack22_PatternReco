import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predictions, targets):
        """
        Mean binary focal loss
        
        predictions: a torch tensor containing the predictions, 0s and 1s.
        targets: a torch tensor containing the ground truth, 0s and 1s.
        
        gamma: focal loss power parameter, a float scalar
            - how much importance is given to misclassified examples
            - 2 is a good start
        alpha: weight of the class indicated by 1, a float scalar.
            - foreground term
            - In practice may be set by inverse class frequency to begin with
        """
        bce_loss = F.binary_cross_entropy(predictions, targets)
        p_t = torch.exp(-bce_loss)
        alpha_tensor = (1 - self.alpha) + targets * (2 * self.alpha - 1)
        # alpha if target = 1 and 1 - alpha if target = 0
        f_loss = alpha_tensor * (1 - p_t) ** self.gamma * bce_loss
        return f_loss.mean()