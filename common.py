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


def seed(value=42):
    """Seed data sampling, model initialization and CUDA without enabling autotuning."""
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed(worker_id):
    """Give each loader worker reproducible Python, NumPy and augmentation RNGs."""
    value = torch.initial_seed() % 2**32
    random.seed(value)
    np.random.seed(value)
    torch.utils.data.get_worker_info().dataset.transform.set_random_seed(value)



class Loss(torch.nn.Module):
    """Pixel-balanced BCE plus per-image overlap loss; empty targets penalize area.

    Tversky beta>0.5 increases the cost of missed changed pixels. Keeping empty
    images in a separate branch avoids treating their undefined overlap as Dice.
    Fractional targets are supported for the NB probability maps.
    """
    def __init__(self, tversky_beta=.7, positive_balance=True):
        super().__init__()
        self.beta, self.positive_balance = tversky_beta, positive_balance

    def forward(self, logits, target):
        dims = (1, 2, 3)
        fraction = target.mean(dims).clamp_min(1e-6)
        # Bound the inverse-area weight so tiny edits cannot dominate a batch.
        positive_weight = (((1 - fraction) / fraction).clamp(max=200) ** .5).view(-1, 1, 1, 1)
        weights = 1 + target * (positive_weight - 1)
        bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = (bce_map * weights).sum() / weights.sum() if self.positive_balance else bce_map.mean()
        probability = logits.sigmoid()
        positive = target.sum(dims) > 0
        # Keep a differentiable zero when a batch contains no edited images.
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
        # On clean frames, suppress predicted area directly instead of dividing
        # an overlap metric by an empty reference mask.
        negative_loss = probability.mean(dims)[~positive].mean() if bool((~positive).any()) else logits.sum() * 0.
        fraction = positive.float().mean()
        # Average overlap/area terms per image; large masks must not receive
        # extra weight merely because they contain more positive pixels.
        return .5 * bce + .5 * (fraction * positive_loss + (1 - fraction) * negative_loss)
