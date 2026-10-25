"""Generate soft NB targets using each pair's held-out teacher."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .runtime import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--teachers", type=Path, nargs=3, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pairs = pd.read_csv(args.pairs)
    if len(pairs) != 9693 or not pairs.chng_img_path.is_unique:
        raise ValueError("Expected the fixed 9693-pair NB pool")
    args.out.mkdir(parents=True, exist_ok=False)
    for fold, checkpoint in enumerate(args.teachers):
        predictor = Predictor(checkpoint)
        part = pairs[pairs.fold == fold]
        for offset in range(0, len(part), 8):
            chunk = part.iloc[offset:offset + 8]
            probabilities = predictor([args.root / path for path in chunk.chng_img_path])
            for row, probability in zip(chunk.itertuples(), probabilities):
                probability = probability.astype(np.float32)
                probability[probability < .02] = 0
                path = args.out / f"{int(row.pair_id)}.png"
                if not cv2.imwrite(str(path), np.round(probability * 255).astype(np.uint8)):
                    raise OSError(path)
        del predictor
        torch.cuda.empty_cache()
        print(f"fold {fold}: {len(part)} targets", flush=True)


if __name__ == "__main__":
    main()
