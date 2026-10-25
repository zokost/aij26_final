"""Frozen #116 calibration: one segmentation forward, lower-only mask expansion."""
from hashlib import sha256
from pathlib import Path
import pickle

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import MEAN, STD
from .runtime import Predictor
from .calibration_features import handcrafted, WithEncoderROIs

THRESHOLDS = np.asarray([.50, .60, .70, .75, .80, .85, .90, .95])
ARTIFACTS = {
    'policy.pkl': 'a37021a48a8c5d0421af703563d123554ecd57264d4b23ac0ea5a09d49837ceb',
    'encoder_transform.pkl': 'd46444cf30aa7ae0a941e200b619b2caabf8b03c73d8173257c31acbed2ab403',
}


def map_features(probability):
    """22 native-size probability statistics, in the original training order."""
    hist = np.histogram(probability, bins=[0, .02, .1, .25, .5, .75, .9, .98, 1.])[0] / probability.size
    areas = [(probability >= t).mean() for t in THRESHOLDS]
    small = cv2.resize(probability, (32, 32), interpolation=cv2.INTER_AREA)
    spatial = [np.abs(np.diff(small, axis=0)).mean(), np.abs(np.diff(small, axis=1)).mean()]
    return np.asarray([*hist, *areas, probability.max(), probability.mean(),
                       *spatial, np.log(probability.shape[0]), np.log(probability.shape[1])], np.float64)


def choose_thresholds(model, features, base_areas, pixel_counts):
    """Preserve every image-level flag under the official >=1% definition."""
    gains = model.predict(features)
    choice = gains.argmax(1)
    choice[gains[np.arange(len(choice)), choice] <= 1e-8] = 3
    choice[base_areas < .01 * pixel_counts] = 3
    return THRESHOLDS[choice]


class CalibratedPredictor:
    def __init__(self, checkpoint, calibration):
        calibration = Path(calibration)
        # Only load known trusted pickle artifacts; pickle is executable input.
        loaded = {}
        for name, digest in ARTIFACTS.items():
            raw = (calibration / name).read_bytes()
            if sha256(raw).hexdigest() != digest:
                raise ValueError(f'Unexpected calibration artifact: {name}')
            loaded[name] = pickle.loads(raw)
        self.policy = loaded['policy.pkl']
        self.transform = loaded['encoder_transform.pkl']
        if self.policy['conservative'] or list(self.policy['indices']) != list(range(276)):
            raise ValueError('Require the frozen all276_positive_gain policy')
        predictor = Predictor(checkpoint)
        self.size = predictor.size
        self.net = WithEncoderROIs(predictor.net).eval()
        cv2.setNumThreads(1)

    @torch.inference_mode()
    def __call__(self, paths):
        images, tensors = [], []
        mean, std = np.asarray(MEAN, np.float32), np.asarray(STD, np.float32)
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            small = cv2.resize(image, (640,640), interpolation=cv2.INTER_AREA)
            tensors.append(torch.from_numpy(((small.astype(np.float32)/255.-mean)/std).transpose(2,0,1)))
        logits, encoder = self.net(torch.stack(tensors).cuda())
        probabilities = [F.interpolate(logits[j:j+1].float(), im.shape[:2], mode='bilinear', align_corners=False).sigmoid()[0,0].cpu().numpy() for j,im in enumerate(images)]
        hand = [handcrafted(im,p) for im,p in zip(images,probabilities)]
        x = np.asarray([np.concatenate([map_features(p), *values]) for p,values in zip(probabilities,hand)])
        embedding = self.transform['pca'].transform(self.transform['scaler'].transform(encoder.cpu().numpy()))
        x = np.column_stack([x,embedding])
        areas = np.asarray([(p>=.75).sum() for p in probabilities])
        pixels = np.asarray([p.size for p in probabilities])
        thresholds = choose_thresholds(self.policy['model'], x, areas, pixels)
        masks = [p>=t for p,t in zip(probabilities,thresholds)]
        for p,mask in zip(probabilities,masks):
            baseline = p>=.75
            if np.any(baseline & ~mask) or bool(baseline.sum()>=.01*p.size) != bool(mask.sum()>=.01*p.size):
                raise RuntimeError('Calibration invariant failed')
        return masks
