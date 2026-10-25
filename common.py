"""Fixed stage-one recipe: MiT-B2, decoder128, 704px, BCE/Tversky."""
import os
import random
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

cv2.setNumThreads(0)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model(pretrained=False):
    return smp.Segformer(encoder_name="mit_b2", encoder_weights="imagenet" if pretrained else None,
                         in_channels=3, classes=1, decoder_segmentation_channels=128)


def frames(root):
    """Preserve the original CSV ordering, clean-source sampling and split."""
    frame = pd.read_csv(Path(root) / "train.csv")
    for column in ("orgl_img_path", "chng_img_path", "gt_path"):
        frame[column] = frame[column].astype(str).str.replace("^stage1/", "", regex=True)
    sources = frame.orgl_img_path
    sources = sources[~sources.isin(["", "nan"])].drop_duplicates()
    count = int(len(frame) * .25 / .75)
    if count < len(sources):
        sources = sources.sample(n=count, random_state=42)
    negatives = pd.DataFrame({"orgl_img_path": sources.values,
                              "chng_img_path": sources.values, "gt_path": ""})
    frame = pd.concat([frame, negatives], ignore_index=True)
    groups = np.where(~frame.orgl_img_path.isin(["", "nan"]), frame.orgl_img_path,
                      np.where(~frame.gt_path.isin(["", "nan"]), frame.gt_path,
                               "solo_" + frame.index.astype(str)))
    train, val = next(GroupShuffleSplit(n_splits=1, test_size=.03, random_state=42).split(frame, groups=groups))
    return frame.iloc[train].copy(), frame.iloc[val].copy()


def read_pair(root, row):
    image = cv2.imread(str(root / row.chng_img_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(root / row.chng_img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if str(row.gt_path) in ("", "nan"):
        mask = np.zeros(image.shape[:2], np.uint8)
    else:
        mask = cv2.imread(str(root / row.gt_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(root / row.gt_path)
        mask = (mask > 127).astype(np.uint8)
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return image, mask


class Images(Dataset):
    def __init__(self, frame, root, training):
        self.frame, self.root, self.training = frame, Path(root), training
        operations = [A.Resize(704, 704, interpolation=cv2.INTER_AREA),
                      A.PadIfNeeded(704, 704, position="top_left", border_mode=cv2.BORDER_CONSTANT,
                                    fill=0, fill_mask=0), A.RandomCrop(704, 704)] if training else [
                                        A.Resize(704, 704, interpolation=cv2.INTER_AREA)]
        if training:
            operations += [A.HorizontalFlip(p=.5), A.VerticalFlip(p=.2), A.RandomRotate90(p=.5),
                           A.RandomBrightnessContrast(brightness_limit=.15, contrast_limit=.15, p=.3),
                           A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=8, p=.2)]
        self.transform = A.Compose(operations + [A.Normalize(mean=MEAN, std=STD), ToTensorV2()])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        image, mask = read_pair(self.root, self.frame.iloc[index])
        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], (transformed["mask"].float().unsqueeze(0) if self.training
                                      else torch.from_numpy(mask))


def validation_batch(batch):
    return torch.stack([x for x, _ in batch]), [mask for _, mask in batch]


class Loss(torch.nn.Module):
    def forward(self, logits, target):
        dims = (1, 2, 3)
        fraction = target.mean(dims).clamp_min(1e-6)
        positive_weight = (((1 - fraction) / fraction).clamp(max=200) ** .5).view(-1, 1, 1, 1)
        weights = 1 + target * (positive_weight - 1)
        bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).sum() / weights.sum()
        probability = logits.sigmoid()
        positive = target.sum(dims) > 0
        positive_loss = logits.sum() * 0.
        if bool(positive.any()):
            tp = (probability * target).sum(dims)[positive]
            fp = (probability * (1 - target)).sum(dims)[positive]
            fn = ((1 - probability) * target).sum(dims)[positive]
            positive_loss = 1 - ((tp + 1e-7) / (tp + .3 * fp + .7 * fn + 1e-7)).mean()
        negative_loss = probability.mean(dims)[~positive].mean() if bool((~positive).any()) else logits.sum() * 0.
        fraction = positive.float().mean()
        return .5 * bce + .5 * (fraction * positive_loss + (1 - fraction) * negative_loss)


@torch.inference_mode()
def evaluate(net, loader, device):
    """Monitoring only: original native-size Dice/FPR, fixed probability threshold .80."""
    net.eval()
    dice, false_alarms = [], []
    for images, masks in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(images.to(device, non_blocking=True))
        for i, gt in enumerate(masks):
            prediction = F.interpolate(logits[i:i + 1].float(), gt.shape, mode="bilinear",
                                       align_corners=False).sigmoid()[0, 0].cpu() >= .80
            gt = gt.bool()
            if bool(gt.any()):
                dice.append((2 * (prediction & gt).sum().item() + 1e-7) /
                            (prediction.sum().item() + gt.sum().item() + 1e-7))
            else:
                false_alarms.append(prediction.sum().item() > .01 * prediction.numel())
    d, f = float(np.mean(dice)), float(np.mean(false_alarms))
    return {"dice": d, "fpr": f, "aic": 2 * d * (1 - f) / (d + 1 - f + 1e-7)}
