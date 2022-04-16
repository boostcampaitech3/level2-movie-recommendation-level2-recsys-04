import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineLoss(nn.Module):
    def __init__(self, xent=.1, reduction="mean"):
        super(CosineLoss, self).__init__()
        self.xent = xent
        self.reduction = reduction

        self.y = torch.Tensor([1])

    def forward(self, input, target):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), self.y,
                                              reduction=self.reduction)
        cent_loss = F.cross_entropy(F.normalize(input), target, reduction=self.reduction)

        return cosine_loss + self.xent * cent_loss