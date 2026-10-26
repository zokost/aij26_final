"""Submission121: original batch4 inference or accelerated FP32 batch1."""
import argparse
import json
import time
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import torch

from .calibration import CalibratedPredictor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--batch-size',type=int,choices=[1,4],default=4)
    parser.add_argument('--compile',action='store_true',help='Experimental Inductor compilation; validate output differences separately')
    parser.add_argument('--benchmark-only',action='store_true')
    args=parser.parse_args()
    if args.compile and args.batch_size!=1:
        parser.error('--compile requires --batch-size1')
    frame=pd.read_csv(args.csv)
    if not len(frame) or not frame.img_path.is_unique:
        raise ValueError('Expected a nonempty CSV with unique img_path values')
    args.out.mkdir(parents=True,exist_ok=False)
    if not args.benchmark_only:(args.out/'predictions').mkdir()
    predictor=CalibratedPredictor(args.checkpoint,args.calibration,graph=args.batch_size==1,
        compile_net=args.compile,opencv_threads=args.threads)
    warm=frame.img_path.astype(str).tolist()[:args.batch_size]
    for _ in range(5):
        if args.batch_size==1:predictor.predict(args.root/warm[0])
        else:predictor.predict_many([args.root/p for p in warm])
    rows=[];timings=[]
    try:
        paths=frame.img_path.astype(str).tolist()
        for offset in range(0,len(paths),args.batch_size):
            chunk=paths[offset:offset+args.batch_size]
            if args.batch_size==1:
                mask,threshold,timing,_=predictor.predict(args.root/chunk[0])
                masks=[mask];measurements=[dict(img_path=chunk[0],threshold=float(threshold),**timing)]
            else:
                torch.cuda.synchronize();start=time.perf_counter()
                masks=predictor.predict_many([args.root/p for p in chunk]);torch.cuda.synchronize()
                per_image=(time.perf_counter()-start)*1000/len(chunk)
                measurements=[dict(img_path=p,total=per_image) for p in chunk]
            timings.extend(measurements)
            for i,(path,mask) in enumerate(zip(chunk,masks),offset):
                if not args.benchmark_only:
                    relative=f'predictions/{i:06d}.png'
                    Image.fromarray(mask.astype(np.uint8)*255).save(args.out/relative)
                    rows.append((path,relative))
            if offset%100==0:print(f'{min(offset+len(chunk),len(paths))}/{len(frame)}',flush=True)
    finally:predictor.close()
    measurements=pd.DataFrame(timings);measurements.to_csv(args.out/'timings.csv',index=False)
    latency=measurements.total.to_numpy()
    summary=dict(n=len(frame),batch=args.batch_size,precision='FP32',device=torch.cuda.get_device_name(),opencv_threads=args.threads,
        compiled=args.compile,mean_ms=float(latency.mean()),median_ms=float(np.median(latency)),p95_ms=float(np.quantile(latency,.95)),
        p99_ms=float(np.quantile(latency,.99)),max_ms=float(latency.max()),over50=int((latency>50).sum()),
        includes='JPEG read/decode, resize, normalization, FP32 network, native probability map, calibration, binary mask',
        excludes='One-time initialization/compilation, PNG encoding and ZIP writing')
    summary['timing_mode']='Single-image wall time' if args.batch_size==1 else 'Batch wall time divided by image count'
    (args.out/'latency.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not args.benchmark_only:
        submission=pd.DataFrame(rows,columns=['img_path','prediction_path'])
        submission.to_csv(args.out/'submission.csv',index=False)
        with zipfile.ZipFile(args.out/'submission.zip','w',zipfile.ZIP_DEFLATED) as archive:
            archive.write(args.out/'submission.csv','submission.csv')
            for _,relative in rows:archive.write(args.out/relative,relative)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
