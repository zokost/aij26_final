"""Stage-one final submission: deterministic FP32, three aspect canvases, threshold.70."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from common import MEAN, STD, seed
from .common import model

CANVASES = ((832, 608), (704, 704), (608, 832))
LOG_ASPECTS = np.log([w / h for h, w in CANVASES])


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("work/final.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/test_stage1"))
    parser.add_argument("--out", type=Path, default=Path("work/submission"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    seed()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    net = model()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net.load_state_dict(checkpoint["model"], strict=True)
    net.cuda().eval()
    frame = pd.read_csv(args.data / "test.csv")
    paths = frame.img_path.astype(str).tolist()
    if not frame.img_path.is_unique:
        raise ValueError("Duplicate test paths")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "predictions").mkdir()
    groups, sizes = {i: [] for i in range(3)}, []
    for i, path in enumerate(paths):
        with Image.open(args.data / path) as im:
            w, h = im.size
        sizes.append((h, w))
        groups[int(np.argmin(np.abs(np.log(w / h) - LOG_ASPECTS)))].append(i)
    mean, std = np.asarray(MEAN, np.float32), np.asarray(STD, np.float32)
    rows = []
    for canvas, indices in groups.items():
        ch, cw = CANVASES[canvas]
        for start in range(0, len(indices), args.batch_size):
            chunk = indices[start:start + args.batch_size]
            batch = []
            for i in chunk:
                image = cv2.imread(str(args.data / paths[i]), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(args.data / paths[i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (cw, ch), interpolation=cv2.INTER_AREA)
                image = (image.astype(np.float32) / 255.0 - mean) / std
                batch.append(torch.from_numpy(image.transpose(2, 0, 1)))
            logits = net(torch.stack(batch).cuda(non_blocking=True))
            for j, i in enumerate(chunk):
                probability = F.interpolate(logits[j:j + 1].float(), sizes[i], mode="bilinear",
                                            align_corners=False).sigmoid()[0, 0]
                mask = (probability >= .70).to(torch.uint8).cpu().numpy() * 255
                relative = f"predictions/{Path(paths[i]).stem}_pred.png"
                if not cv2.imwrite(str(args.out / relative), mask):
                    raise OSError(relative)
                rows.append((paths[i], relative))
            print(f"canvas{canvas}: {min(start + len(chunk), len(indices))}/{len(indices)}", flush=True)
    order = {path: i for i, path in enumerate(paths)}
    rows.sort(key=lambda row: order[row[0]])
    submission = pd.DataFrame(rows, columns=["img_path", "prediction_path"])
    if not submission.prediction_path.is_unique or submission.img_path.tolist() != paths:
        raise ValueError("Submission coverage/order differs")
    submission.to_csv(args.out / "submission.csv", index=False)
    for i, (_, relative) in enumerate(rows):
        with Image.open(args.out / relative) as mask:
            assert mask.mode == "L" and mask.size == (sizes[i][1], sizes[i][0])
            assert set(np.unique(np.asarray(mask))) <= {0, 255}
    with zipfile.ZipFile(args.out / "submission.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.out / "submission.csv", "submission.csv")
        for _, relative in rows:
            archive.write(args.out / relative, relative)
    with zipfile.ZipFile(args.out / "submission.zip") as archive:
        assert archive.testzip() is None and len(archive.namelist()) == len(paths) + 1
    print(args.out / "submission.zip")


if __name__ == "__main__":
    main()
