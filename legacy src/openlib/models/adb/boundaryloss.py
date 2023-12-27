import torch
from torch import nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    """
    Deep Open Intent Classification with Adaptive Decision Boundary.
    https://arxiv.org/pdf/2012.10209.pdf
    """

    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, x, centroids, delta, labels):
        delta = F.softplus(delta)
        c = centroids[labels]
        d = delta[labels]

        euc_dis = torch.norm(x - c, 2, 1).view(-1)
        pos_mask = euc_dis > d
        neg_mask = euc_dis < d

        pos_loss = (euc_dis - d) * pos_mask
        neg_loss = (d - euc_dis) * neg_mask

        loss = pos_loss.mean() + neg_loss.mean()
        return loss
