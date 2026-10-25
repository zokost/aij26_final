"""Shared normalization, random seed and segmentation loss."""
import os
import random

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import cv2
import numpy as np
import torch
import torch.nn.functional as F

cv2.setNumThreads(0)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



class Loss(torch.nn.Module):
    def __init__(self, tversky_beta=.7, positive_balance=True):
        super().__init__()
        self.beta, self.positive_balance = tversky_beta, positive_balance

    def forward(self, logits, target):
        dims = (1, 2, 3)
        fraction = target.mean(dims).clamp_min(1e-6)
        positive_weight = (((1 - fraction) / fraction).clamp(max=200) ** .5).view(-1, 1, 1, 1)
        weights = 1 + target * (positive_weight - 1)
        bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = (bce_map * weights).sum() / weights.sum() if self.positive_balance else bce_map.mean()
        probability = logits.sigmoid()
        positive = target.sum(dims) > 0
        positive_loss = logits.sum() * 0.
        if bool(positive.any()):
            tp = (probability * target).sum(dims)[positive]
            fp = (probability * (1 - target)).sum(dims)[positive]
            fn = ((1 - probability) * target).sum(dims)[positive]
            if self.beta is None:
                denominator = (probability.sum(dims) + target.sum(dims))[positive]
                positive_loss = 1 - ((2 * tp + 1e-7) / (denominator + 1e-7)).mean()
            else:
                positive_loss = 1 - ((tp + 1e-7) /
                                     (tp + (1 - self.beta) * fp + self.beta * fn + 1e-7)).mean()
        negative_loss = probability.mean(dims)[~positive].mean() if bool((~positive).any()) else logits.sum() * 0.
        fraction = positive.float().mean()
        return .5 * bce + .5 * (fraction * positive_loss + (1 - fraction) * negative_loss)
