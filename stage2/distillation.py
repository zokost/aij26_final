"""Additional stage2 supervision from source-held-out teacher logits."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import Loss
from .runtime import Images


class DistillationLoss(torch.nn.Module):
    def __init__(self, weight=.6, **kwargs):
        super().__init__()
        self.supervised=Loss(**kwargs)
        self.weight=weight

    def forward(self, logits, targets):
        gt,teacher,valid=targets.split(1,dim=1)
        pixel_loss=F.binary_cross_entropy_with_logits(logits.float(),teacher,reduction='none')
        auxiliary=(pixel_loss*valid).sum()/valid.sum().clamp_min(1)
        return self.supervised(logits,gt)+self.weight*auxiliary


def target_index(path, root, base):
    frame=pd.read_csv(path,keep_default_na=False)
    if not {'chng_img_path','logits_path'}<=set(frame):
        raise ValueError('OOF index needs chng_img_path and logits_path')
    index={str((root/row.chng_img_path).resolve()):root/row.logits_path for row in frame.itertuples()}
    expected={str(Path(p).resolve()) for p in base.chng_img_path}
    if len(frame)!=19837 or len(index)!=19837 or set(index)!=expected:
        raise ValueError('Require exact once-only OOF coverage of19837 stage2 rows')
    for p in index.values():
        if not p.is_file():raise FileNotFoundError(p)
    return index


class DistillationImages(Images):
    def __init__(self,*args,teacher_index,**kwargs):
        super().__init__(*args,**kwargs)
        self.teacher_index=teacher_index

    def __getitem__(self,index):
        row=self.frame.iloc[index]
        image=cv2.imread(row.chng_img_path)
        if image is None:raise FileNotFoundError(row.chng_img_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        target=np.zeros(image.shape[:2],np.float32)
        if row.gt_path:
            mask=cv2.imread(row.gt_path,0)
            if mask is None:raise FileNotFoundError(row.gt_path)
            target=mask.astype(np.float32)/255.
            if target.shape!=image.shape[:2]:
                target=cv2.resize(target,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
        teacher,valid=np.zeros_like(target),np.zeros_like(target)
        path=self.teacher_index.get(str(Path(row.chng_img_path).resolve()))
        if path is not None:
            with np.load(path,allow_pickle=False) as saved:logits=saved['logits'].astype(np.float32)
            if logits.shape!=target.shape or not np.isfinite(logits).all():
                raise ValueError(f'Invalid native teacher logits: {path}')
            teacher=np.exp(-np.logaddexp(0.,-logits));valid.fill(1.)
        result=self.transform(image=image,mask=np.stack([target,teacher,valid],axis=-1))
        return result['image'],result['mask'].float().permute(2,0,1)
