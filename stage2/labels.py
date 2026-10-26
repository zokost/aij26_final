"""Export soft NB masks or native stage2 logits from held-out teachers."""
import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .runtime import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--teachers", type=Path, nargs=3, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument('--kind',choices=['nb','stage2'],default='nb')
    args = parser.parse_args()
    pairs = pd.read_csv(args.pairs)
    expected=9693 if args.kind=='nb' else 19837
    if len(pairs)!=expected or not pairs.chng_img_path.is_unique or not set(pairs.fold)<=set(range(3)):
        raise ValueError(f'Expected {expected} unique images with fold0/1/2')
    if args.kind=='stage2' and pairs.groupby('orgl_img_path').fold.nunique().max()!=1:
        raise ValueError('Each original source must belong to one OOF fold')
    args.out.mkdir(parents=True, exist_ok=False)
    records=[]
    for fold, checkpoint in enumerate(args.teachers):
        predictor = Predictor(checkpoint)
        part = pairs[pairs.fold == fold]
        batch=8 if args.kind=='nb' else 4
        for offset in range(0, len(part), batch):
            chunk = part.iloc[offset:offset + batch]
            if args.kind=='stage2':
                images=[];shapes=[]
                from common import MEAN, STD
                for path in chunk.chng_img_path:
                    image=cv2.imread(str(args.root/path))
                    if image is None:raise FileNotFoundError(path)
                    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);shapes.append(image.shape[:2])
                    small=cv2.resize(image,(predictor.size,predictor.size),interpolation=cv2.INTER_AREA)
                    images.append(torch.from_numpy(((small.astype(np.float32)/255.-np.asarray(MEAN,np.float32))/np.asarray(STD,np.float32)).transpose(2,0,1)))
                with torch.inference_mode():
                    logits=predictor.net(torch.stack(images).cuda())
                    for i,row in enumerate(chunk.itertuples()):
                        native=F.interpolate(logits[i:i+1].float(),shapes[i],mode='bilinear',align_corners=False)[0,0].cpu().numpy().astype(np.float16)
                        if not np.isfinite(native).all():raise FloatingPointError(row.chng_img_path)
                        path=args.out/(hashlib.sha256(row.chng_img_path.encode()).hexdigest()+'.npz')
                        np.savez_compressed(path,logits=native)
                        records.append(dict(chng_img_path=row.chng_img_path,teacher_fold=fold,logits_path=str(path.resolve())))
                continue
            probabilities=predictor([args.root/path for path in chunk.chng_img_path])
            for row, probability in zip(chunk.itertuples(), probabilities):
                probability = probability.astype(np.float32)
                probability[probability < .02] = 0
                path = args.out / f"{int(row.pair_id)}.png"
                if not cv2.imwrite(str(path), np.round(probability * 255).astype(np.uint8)):
                    raise OSError(path)
        del predictor
        torch.cuda.empty_cache()
        print(f"fold {fold}: {len(part)} targets", flush=True)
    if args.kind=='stage2':pd.DataFrame(records).to_csv(args.out/'index.csv',index=False)


if __name__ == "__main__":
    main()
