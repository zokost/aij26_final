"""KDALL models, training inputs and deterministic FP32 prediction."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from common import MEAN, STD, seed, worker_seed

COLUMNS = ["orgl_img_path", "chng_img_path", "gt_path"]


def model(teacher=False, encoder="mit_b2", decoder_channels=320, pretrained=False):
    """Teachers are training-only; final inference uses the smaller student."""
    if teacher:
        return smp.UPerNet(encoder_name="tu-convnext_large.dinov3_lvd1689m",
                           encoder_weights=None, in_channels=3, classes=1)
    return smp.Segformer(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                         in_channels=3, classes=1, decoder_segmentation_channels=decoder_channels)


def read_rows(path, root):
    frame = pd.read_csv(path, keep_default_na=False)
    for column in COLUMNS:
        frame[column] = frame[column].map(lambda value: str(Path(root) / value) if value else "")
    return frame


def companions(frame):
    """Pair edited NB examples with genuine clean negatives and empty masks."""
    return pd.DataFrame(dict(orgl_img_path=frame.orgl_img_path,
                             chng_img_path=frame.orgl_img_path, gt_path=""))


class Images(Dataset):
    """Load native RGB/targets and apply synchronized segmentation augmentation.

    soft=True preserves fractional NB probabilities; official binary PNGs
    remain 0/1. A missing mask path denotes a genuine clean image.
    """
    def __init__(self, frame, size, soft, augmentation="minimal", seed_value=42):
        self.frame, self.soft = frame.reset_index(drop=True), soft
        if augmentation not in ("minimal", "forensic"):
            raise ValueError(augmentation)
        # The final recipe uses squish resize, mild flips and JPEG. Resizing
        # happens before the crop, so this crop does not implement positive zoom.
        operations = [
            A.Resize(size, size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(size, size, position="top_left", border_mode=cv2.BORDER_CONSTANT,
                          fill=0, fill_mask=0), A.RandomCrop(size, size),
            A.HorizontalFlip(p=.5), A.VerticalFlip(p=.2),
            A.ImageCompression(quality_range=(90, 100), p=.2)]
        if augmentation == "forensic":
            # Optional research profile; it is NOT enabled for submission121.
            # Acquisition/redistribution perturbations: resize, blur and recompression.
            # Photometric operations change the image only; geometric transforms
            # apply the same mapping to image and fractional segmentation target.
            operations += [A.OneOf([
                A.Downscale(scale_range=(.5, .9), interpolation_pair={
                    "downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR}, p=1),
                A.GaussianBlur(blur_limit=(3, 5), p=1)], p=.2),
                A.ImageCompression(quality_range=(60, 95), p=.2)]
        self.transform = A.Compose(operations + [A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
        self.transform.set_random_seed(seed_value)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = cv2.imread(row.chng_img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(row.chng_img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = np.zeros(image.shape[:2], np.uint8)
        if row.gt_path:
            target = cv2.imread(row.gt_path, cv2.IMREAD_GRAYSCALE)
            if target is None:
                raise FileNotFoundError(row.gt_path)
            target = target.astype(np.float32) / 255. if self.soft else (target > 127).astype(np.uint8)
            if target.shape != image.shape[:2]:
                target = cv2.resize(target, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        result = self.transform(image=image, mask=target)
        return result["image"], result["mask"].float().unsqueeze(0)


class Predictor:
    """Rebuild checkpoint architecture and use deterministic FP32 evaluation."""
    def __init__(self, checkpoint):
        seed()
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.size = int(saved["cfg"]["img_size"])
        cfg = saved["cfg"]
        self.net = model(cfg["arch"] == "upernet", encoder=cfg.get("encoder", "mit_b2"),
                         decoder_channels=cfg.get("decoder_channels", 320))
        self.net.load_state_dict(saved["model"], strict=True)
        del saved
        self.net.cuda().eval()

    @torch.inference_mode()
    def __call__(self, paths):
        return probabilities(self.net, paths, self.size)


@torch.inference_mode()
def probabilities(net, paths, size):
    """Shared FP32 preprocessing and native-resolution prediction for evaluation/inference."""
    images, sizes = [], []
    mean, std = np.asarray(MEAN, np.float32), np.asarray(STD, np.float32)
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sizes.append(image.shape[:2])
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        image = (image.astype(np.float32) / 255. - mean) / std
        images.append(torch.from_numpy(image.transpose(2, 0, 1)))
    logits = net(torch.stack(images).to(next(net.parameters()).device, non_blocking=True))
    # Restore logits before sigmoid/thresholding. Interpolating a binary mask
    # instead would lose small structures and change native-resolution scores.
    return [F.interpolate(logits[i:i + 1].float(), shape, mode="bilinear", align_corners=False)
            .sigmoid()[0, 0].cpu().numpy() for i, shape in enumerate(sizes)]
