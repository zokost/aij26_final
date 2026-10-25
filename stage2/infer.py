"""KDALL FP32 inference: squish640, threshold0.75, native-size binary PNG."""
import argparse
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from .runtime import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    paths = frame.img_path.astype(str).tolist()
    if not frame.img_path.is_unique or not len(paths):
        raise ValueError("Expected a nonempty CSV with unique img_path values")
    predictor = Predictor(args.checkpoint)
    if predictor.size != 640:
        raise ValueError("KDALL requires img_size640")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "predictions").mkdir()
    rows = []
    for offset in range(0, len(paths), 4):
        chunk = paths[offset:offset + 4]
        for path, probability in zip(chunk, predictor([args.root / path for path in chunk])):
            relative = f"predictions/{Path(path).stem}_pred.png"
            Image.fromarray((probability >= .75).astype(np.uint8) * 255).save(args.out / relative, optimize=True)
            with Image.open(args.root / path) as original, Image.open(args.out / relative) as mask:
                if mask.size != original.size or mask.mode != "L" or not set(np.unique(np.asarray(mask))) <= {0, 255}:
                    raise ValueError(relative)
            rows.append((path, relative))
        print(f"{min(offset + 4, len(paths))}/{len(paths)}", flush=True)
    submission = pd.DataFrame(rows, columns=["img_path", "prediction_path"])
    if not submission.prediction_path.is_unique:
        raise ValueError("Repeated output mask path")
    submission.to_csv(args.out / "submission.csv", index=False)
    with zipfile.ZipFile(args.out / "submission.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.out / "submission.csv", "submission.csv")
        for _, relative in rows:
            archive.write(args.out / relative, relative)
    with zipfile.ZipFile(args.out / "submission.zip") as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(paths) + 1:
            raise ValueError("Invalid ZIP")


if __name__ == "__main__":
    main()
