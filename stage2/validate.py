"""NB pair validation with approximate masks and paired clean originals."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .runtime import COLUMNS, Predictor, probabilities, read_rows

DEFAULT_CSV = Path(__file__).with_name("nb_validation.csv")


def validation_rows(csv, root):
    frame = read_rows(csv, root)
    if frame.empty or not frame[COLUMNS].astype(bool).all().all():
        raise ValueError("NB validation requires nonempty image, original and mask paths")
    for column in ("orgl_img_path", "chng_img_path"):
        if not frame[column].is_unique:
            raise ValueError(f"Repeated NB validation path: {column}")
    if set(frame.orgl_img_path) & set(frame.chng_img_path):
        raise ValueError("NB original and edited paths overlap")
    for column in COLUMNS:
        for path in frame[column]:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    return frame


def check_disjoint(validation, training):
    """Check both source and edited paths, including aliases resolved by the filesystem."""
    def paths(frame):
        unique = {p for c in ("orgl_img_path", "chng_img_path") for p in frame[c] if p}
        return {str(Path(p).resolve()) for p in unique}
    overlap = paths(validation) & paths(training)
    if overlap:
        raise ValueError(f"NB validation overlaps training: {len(overlap)} paths; {sorted(overlap)[:3]}")


def evaluate_nb(net, frame, size, threshold=.75, batch_size=4):
    """Mean per-image positive Dice and image-level FPR at >=1% predicted area."""
    if frame.empty or not 0 < threshold < 1 or batch_size < 1:
        raise ValueError("Expected nonempty NB frame, threshold in (0, 1) and positive batch size")
    dice, alarms = [], []
    training = net.training
    net.eval()
    try:
        for start in range(0, len(frame), batch_size):
            chunk = frame.iloc[start:start + batch_size]
            edits = probabilities(net, chunk.chng_img_path.tolist(), size)
            originals = probabilities(net, chunk.orgl_img_path.tolist(), size)
            for row, edit, original in zip(chunk.itertuples(), edits, originals):
                if not np.isfinite(edit).all() or not np.isfinite(original).all():
                    raise FloatingPointError(f"Non-finite NB prediction: {row.chng_img_path}")
                mask = cv2.imread(row.gt_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(row.gt_path)
                if mask.shape != edit.shape or not np.isin(mask, [0, 255]).all() or not mask.any():
                    raise ValueError(f"Expected nonempty native-size binary NB mask: {row.gt_path}")
                target, predicted = mask > 127, edit >= threshold
                dice.append(float(2 * (predicted & target).sum() / (predicted.sum() + target.sum())))
                alarms.append(bool(int((original >= threshold).sum()) >= .01 * original.size))
    finally:
        net.train(training)
    positive, fpr = float(np.mean(dice)), float(np.mean(alarms))
    denominator = positive + 1 - fpr
    return dict(n_pairs=len(frame), threshold=threshold, dice_positive=positive, fpr_negative=fpr,
                aic=2 * positive * (1 - fpr) / denominator if denominator else 0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--threshold", type=float, default=.75)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = validation_rows(args.csv, args.root)
    predictor = Predictor(args.checkpoint)
    metrics = evaluate_nb(predictor.net, frame, predictor.size, args.threshold, args.batch_size)
    metrics.update(checkpoint=str(args.checkpoint), img_size=predictor.size, dtype="FP32",
                   manifest_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                   mask_type="approximate pair-difference masks")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
