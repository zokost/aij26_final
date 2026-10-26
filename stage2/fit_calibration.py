"""Fit the fixed lower-only NC17 policy on prepared NB/stage2 partitions."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from common import MEAN, STD
from .calibration import BASE, CANDIDATES, choose_thresholds
from .calibration_features import WithEncoderROIs, handcrafted, map_features
from .runtime import Predictor


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def extract(checkpoint, entries, root):
    """One FP32 forward per batch supplies both logits and encoder regions."""
    predictor = Predictor(checkpoint)
    if predictor.size != 640:
        raise ValueError('Calibration requires the 640px student')
    net = WithEncoderROIs(predictor.net).eval()
    mean, std = np.asarray(MEAN, np.float32), np.asarray(STD, np.float32)
    values = dict(handcrafted=[], encoder=[], dice=[], area=[], pixels=[])
    with torch.inference_mode():
        for start in range(0, len(entries), 4):
            chunk = entries.iloc[start:start + 4]
            images, inputs = [], []
            for path in chunk.path:
                image = cv2.imread(str(root / path))
                if image is None:
                    raise FileNotFoundError(root / path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                small = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
                inputs.append(torch.from_numpy(((small.astype(np.float32) / 255. - mean) / std).transpose(2, 0, 1)))
            logits, encoder = net(torch.stack(inputs).cuda())
            encoder = encoder.cpu().numpy()
            for i, (row, image) in enumerate(zip(chunk.itertuples(), images)):
                p = F.interpolate(logits[i:i+1].float(), image.shape[:2], mode='bilinear',
                                  align_corners=False).sigmoid()[0, 0].cpu().numpy()
                gt = np.zeros(p.shape, bool)
                valid = np.ones(p.shape, bool)
                for column in ('mask', 'valid'):
                    path = getattr(row, column)
                    if not path:
                        continue
                    mask = cv2.imread(str(root / path), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise FileNotFoundError(root / path)
                    if mask.shape != p.shape:
                        if column == 'mask':
                            mask = cv2.resize(mask, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_NEAREST)
                        else:
                            raise ValueError(f'Expected native-size {column}: {path}')
                    if column == 'mask':
                        gt = mask > 127
                    else:
                        valid = mask > 127
                gt &= valid
                if bool(gt.any()) != row.positive:
                    raise ValueError(f'Mask and positive flag disagree: {row.path}')
                areas, dice = [], []
                for threshold in CANDIDATES:
                    predicted = p >= threshold
                    areas.append(int(predicted.sum()))
                    predicted &= valid
                    denominator = int(predicted.sum() + gt.sum())
                    dice.append(2 * int((predicted & gt).sum()) / denominator if denominator else 1.)
                values['handcrafted'].append(np.concatenate([map_features(p), *handcrafted(image, p)]))
                values['encoder'].append(encoder[i])
                values['dice'].append(dice)
                values['area'].append(areas)
                values['pixels'].append(p.size)
            print(f'Calibration features: {min(start + 4, len(entries))}/{len(entries)}', flush=True)
    net.handle.remove()
    return {key: np.asarray(value) for key, value in values.items()}


def fit_policy(data, entries):
    """Controls never contribute to the scaler, PCA or tree targets."""
    fit = entries.partition.eq('policy_fit').to_numpy()
    scaler = StandardScaler().fit(data['encoder'][fit])
    pca = PCA(n_components=32, svd_solver='randomized', random_state=45)
    pca.fit(scaler.transform(data['encoder'][fit]))
    transform = dict(scaler=scaler, pca=pca)
    x = np.column_stack([data['handcrafted'], pca.transform(scaler.transform(data['encoder']))])
    base_index = len(CANDIDATES) - 1
    # Regress improvement over the baseline, not the absolute Dice. The base
    # candidate has exactly zero gain and serves as the fallback action.
    target = data['dice'] - data['dice'][:, base_index:base_index+1]
    xs, ys, weights, counts = [], [], [], {}
    for source in ('NB', 'stage2'):
        # Trees only act on positive images already passing the area guard.
        # Fit each source with total weight0.5 to prevent the larger pool from
        # dominating; clean controls are retained for reporting FPR separately.
        eligible = (fit & entries.source.eq(source).to_numpy() & entries.positive.to_numpy()
                    & (data['area'][:, base_index] >= .01 * data['pixels']))
        count = int(eligible.sum())
        if not count or not np.isfinite(target[eligible]).all():
            raise ValueError(f'No finite eligible calibration targets for {source}')
        counts[source] = count
        xs.append(x[eligible]); ys.append(target[eligible])
        weights.append(np.full(count, .5 / count))
    model = ExtraTreesRegressor(n_estimators=32, max_depth=5, min_samples_leaf=64,
                               random_state=45, bootstrap=False, n_jobs=4)
    model.fit(np.concatenate(xs), np.concatenate(ys), sample_weight=np.concatenate(weights))
    # Apply the frozen decision rule to control rows after fitting is complete.
    # These metrics do not feed back into the scaler, PCA or tree parameters.
    choice = choose_thresholds(model, x, data['area'][:, base_index], data['pixels'])
    indices = np.searchsorted(CANDIDATES, choice)
    controls = {}
    for source in ('NB', 'stage2'):
        control = entries.source.eq(source).to_numpy() & ~fit
        positive = control & entries.positive.to_numpy()
        negative = control & ~entries.positive.to_numpy()
        if not positive.any() or not negative.any():
            raise ValueError(f'Both positive and negative controls required for {source}')
        dice = float(data['dice'][positive, indices[positive]].mean())
        flags = data['area'][np.arange(len(entries)), indices] >= .01 * data['pixels']
        fpr = float(flags[negative].mean())
        controls[source] = dict(dice_positive=dice, fpr_negative=fpr,
                                aic=2 * dice * (1 - fpr) / (dice + 1 - fpr) if dice + 1 - fpr else 0.)
    return dict(model=model, base=float(BASE), candidates=CANDIDATES.tolist(), eligible=counts), transform, controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--entries', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    entries = pd.read_csv(args.entries, keep_default_na=False)
    if (not {'path', 'mask', 'valid', 'source', 'partition', 'positive'} <= set(entries)
            or not set(entries.source) == {'NB', 'stage2'}
            or not set(entries.partition) == {'policy_fit', 'control'}
            or not entries.path.is_unique or entries.positive.dtype != bool):
        raise ValueError('Expected unique NB/stage2 entries with fixed fit/control partitions')
    args.out.mkdir(parents=True, exist_ok=True)
    identity = dict(checkpoint_sha256=digest(args.checkpoint), entries_sha256=digest(args.entries))
    # Feature caches are valid only for the exact checkpoint and ordered CSV.
    cache, meta = args.out / 'features.npz', args.out / 'features.json'
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == identity:
        with np.load(cache, allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
    else:
        data = extract(args.checkpoint, entries, args.root)
        temporary = cache.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **data)
        temporary.replace(cache)
        meta.write_text(json.dumps(identity, indent=2) + '\n')
    policy, transform, controls = fit_policy(data, entries)
    for name, artifact in (('policy.pkl', policy), ('encoder_transform.pkl', transform)):
        with (args.out / name).open('wb') as stream:
            pickle.dump(artifact, stream)
    metadata = dict(**identity, base=float(BASE), candidates=CANDIDATES.tolist(),
                    artifacts={name: digest(args.out / name) for name in ('policy.pkl', 'encoder_transform.pkl')},
                    eligible=policy['eligible'], controls=controls)
    (args.out / 'calibration.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
