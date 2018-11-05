import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target, size_average=True):
        distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((target).float() * torch.pow(distance, 2.0) +
                                      (1 - target).float() * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss_contrastive