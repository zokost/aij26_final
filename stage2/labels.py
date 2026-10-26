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
    parser.add_argument('--resume',action='store_true',help='Reuse complete target files after an interrupted export')
    args = parser.parse_args()
    pairs = pd.read_csv(args.pairs)
    expected=9693 if args.kind=='nb' else 19837
    if len(pairs)!=expected or not pairs.chng_img_path.is_unique or not set(pairs.fold)<=set(range(3)):
        raise ValueError(f'Expected {expected} unique images with fold0/1/2')
    if args.kind=='stage2' and pairs.groupby('orgl_img_path').fold.nunique().max()!=1:
        raise ValueError('Each original source must belong to one OOF fold')
    args.out.mkdir(parents=True, exist_ok=args.resume)
    records=[]
    for fold, checkpoint in enumerate(args.teachers):
        # A row receives exactly one prediction: from the teacher trained
        # without its fold. This is cross-fitting, not a three-model ensemble.
        predictor = Predictor(checkpoint)
        part = pairs[pairs.fold == fold]
        batch=8 if args.kind=='nb' else 4
        for offset in range(0, len(part), batch):
            chunk = part.iloc[offset:offset + batch]
            if args.resume:
                pending=[]
                for row in chunk.itertuples():
                    name=(hashlib.sha256(row.chng_img_path.encode()).hexdigest()+'.npz') if args.kind=='stage2' else f'{int(row.pair_id)}.png'
                    path=args.out/name
                    complete=False
                    if path.is_file():
                        try:
                            if args.kind=='stage2':
                                with np.load(path,allow_pickle=False) as saved:
                                    value=saved['logits']
                                    complete=value.ndim==2 and value.dtype==np.float16 and np.isfinite(value).all()
                            else:
                                value=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
                                complete=value is not None
                        except (OSError,ValueError,KeyError,EOFError):
                            complete=False
                    if complete:
                        if args.kind=='stage2':records.append(dict(chng_img_path=row.chng_img_path,teacher_fold=fold,logits_path=str(path.resolve())))
                    else:pending.append(row.Index)
                chunk=chunk.loc[pending]
                if chunk.empty:continue
            if args.kind=='stage2':
                # Stage2 keeps native float16 logits. The student converts them
                # to probabilities in float32 alongside the unchanged hard GT.
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
                        temporary=path.with_suffix('.tmp')
                        with temporary.open('wb') as stream:np.savez_compressed(stream,logits=native)
                        temporary.replace(path)
                        records.append(dict(chng_img_path=row.chng_img_path,teacher_fold=fold,logits_path=str(path.resolve())))
                continue
            probabilities=predictor([args.root/path for path in chunk.chng_img_path])
            for row, probability in zip(chunk.itertuples(), probabilities):
                # NB follows the original compact target format: quantized
                # probabilities, not a binary thresholded teacher mask.
                probability = probability.astype(np.float32)
                probability[probability < .02] = 0
                path = args.out / f"{int(row.pair_id)}.png"
                temporary=path.with_name(path.stem+'.tmp.png')
                if not cv2.imwrite(str(temporary), np.round(probability * 255).astype(np.uint8)):
                    raise OSError(temporary)
                temporary.replace(path)
        del predictor
        torch.cuda.empty_cache()
        print(f"fold {fold}: {len(part)} targets", flush=True)
    if args.kind=='stage2':pd.DataFrame(records).to_csv(args.out/'index.csv',index=False)


if __name__ == "__main__":
    main()
