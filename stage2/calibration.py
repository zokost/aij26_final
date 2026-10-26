"""Frozen submission121 FP32 inference and optional CUDA Graph acceleration.

No retraining or threshold changes. Fast map histogram, single-thread trees,
optional eager CUDA Graph and early exit under the frozen 1% guard.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import pickle,time
from threading import Lock
from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
from common import MEAN,STD
from .runtime import Predictor
from .calibration_features import (WithEncoderROIs, FEATURE_THRESHOLDS, map_features,
                                   map_from_counts, prepare_rgb, selected_handcrafted)
BASE=np.float64(.675)
CANDIDATES=np.asarray([.5,.6,.65,.675])


def choose_thresholds(model,features,base_areas,pixel_counts):
    """Expand only masks already at or above the official1% alarm boundary."""
    gains=model.predict(features)
    choice=gains.argmax(1)
    thresholds=CANDIDATES[choice].copy()
    thresholds[gains[np.arange(len(choice)),choice]<=1e-8]=BASE
    thresholds[base_areas<.01*pixel_counts]=BASE
    return thresholds


class CalibratedPredictor:
    def __init__(self,checkpoint,calibration,graph=True,compile_net=False,opencv_threads=4):
        cv2.setNumThreads(1);threadpool_limits(limits=1)
        predictor=Predictor(checkpoint)
        if predictor.size!=640:
            raise ValueError('Expected a 640px checkpoint')
        self.net=WithEncoderROIs(predictor.net).eval()
        if compile_net:
            self.net=torch.compile(self.net,mode='max-autotune-no-cudagraphs')
        torch.set_num_threads(3)
        folder=Path(calibration)
        policy_name='policy.pkl' if (folder/'policy.pkl').exists() else 'refit_submit.pkl'
        policy_bytes=(folder/policy_name).read_bytes()
        transform_bytes=(folder/'encoder_transform.pkl').read_bytes()
        if sha256(policy_bytes).hexdigest()!='0bf2ba074b3609aefea2eba383ab512f5d635311315c76a79e3b08472023de25':
            raise ValueError('Expected frozen submission121 policy')
        if sha256(transform_bytes).hexdigest()!='fc865212594a3e98eb3e9b1465d75225782a422bebc636f4a7f99ebc949b9758':
            raise ValueError('Expected matching submission121 scaler/PCA')
        self.policy=pickle.loads(policy_bytes)
        self.transform=pickle.loads(transform_bytes)
        self.policy['model'].n_jobs=1
        self.used=set(int(i) for tree in self.policy['model'].estimators_ for i in tree.tree_.feature if i>=0)
        self.cuts=torch.tensor([.02,.1,.25,.5,.75,.9,.98,*FEATURE_THRESHOLDS.tolist()],dtype=torch.float64,device='cuda')
        self.mean=np.asarray(MEAN,np.float32);self.std=np.asarray(STD,np.float32)
        # RGB is uint8: a 256x3 lookup gives bit-exact original NumPy
        # normalization, without CPU image-wide float operations or GPU division.
        table=(np.arange(256,dtype=np.float32)[:,None]/255.-self.mean)/self.std
        self.lookup=torch.from_numpy(table).cuda();self.channels=torch.arange(3,device='cuda')
        self.pool=ThreadPoolExecutor(max_workers=1)
        self.lock=Lock()
        self.graph=None
        if graph:
            self.static=torch.zeros((640,640,3),dtype=torch.uint8,device='cuda')
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.inference_mode(),torch.cuda.stream(stream):
                for _ in range(5):self.forward_rgb(self.static)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph=torch.cuda.CUDAGraph()
            with torch.inference_mode(),torch.cuda.graph(self.graph):
                self.static_outputs=self.forward_rgb(self.static)
            torch.cuda.synchronize()
        cv2.setNumThreads(opencv_threads)

    def close(self):
        self.pool.shutdown(wait=True)

    def __call__(self,paths):
        return [self.predict(path)[0] for path in paths]

    @torch.inference_mode()
    def predict_many(self,paths):
        """Original batch4 FP32 forward for exact submitted-mask reproduction."""
        images=[];inputs=[]
        for path in paths:
            decoded=cv2.imread(str(path))
            if decoded is None:raise FileNotFoundError(path)
            image=cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB);images.append(image)
            small=cv2.resize(image,(640,640),interpolation=cv2.INTER_AREA)
            inputs.append(torch.from_numpy(((small.astype(np.float32)/255.-self.mean)/self.std).transpose(2,0,1)))
        with self.lock:
            logits,encoder=self.net(torch.stack(inputs).cuda())
            probabilities=[F.interpolate(logits[i:i+1].float(),image.shape[:2],mode='bilinear',align_corners=False).sigmoid()[0,0].cpu().numpy() for i,image in enumerate(images)]
            embedding=self.transform['pca'].transform(self.transform['scaler'].transform(encoder.cpu().numpy()))
            masks=[]
            for i,(image,p) in enumerate(zip(images,probabilities)):
                base=p>=BASE
                if base.sum()<.01*p.size:
                    masks.append(base);continue
                x=np.concatenate([map_features(p),*selected_handcrafted(image,p,self.used),embedding[i]])[None]
                threshold=choose_thresholds(self.policy['model'],x,np.asarray([base.sum()]),np.asarray([p.size]))[0]
                masks.append(p>=threshold)
        return masks

    def forward_rgb(self,rgb):
        normalized=self.lookup[rgb.long(),self.channels]
        return self.net(normalized.permute(2,0,1)[None])

    def predict(self,path,force_full=False):
        # CUDA Graph static buffers are shared by calls on this instance.
        with self.lock:
            return self._predict(path,force_full)

    @torch.inference_mode()
    def _predict(self,path,force_full=False):
        timings={};torch.cuda.synchronize();started=last=time.perf_counter()
        def tick(name):
            nonlocal last
            now=time.perf_counter();timings[name]=(now-last)*1000;last=now
        decoded=cv2.imread(str(path))
        if decoded is None:
            raise FileNotFoundError(path)
        image=cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB)
        small=cv2.resize(image,(640,640),interpolation=cv2.INTER_AREA)
        rgb_future=self.pool.submit(prepare_rgb,image)
        inputs=torch.from_numpy(small);tick('read_preprocess')
        if self.graph is None:logits,encoder=self.forward_rgb(inputs.cuda())
        else:
            self.static.copy_(inputs);self.graph.replay();logits,encoder=self.static_outputs
        native=F.interpolate(logits.float(),image.shape[:2],mode='bilinear',align_corners=False).sigmoid()[0,0]
        counts=(native.double()[None]<self.cuts[:,None,None]).sum((1,2)).cpu().numpy()
        counts[7:]=native.numel()-counts[7:]
        p=native.cpu().numpy()
        tick('forward_native_copy')
        base=p>=np.float64(.675);eligible=bool(base.sum()>=.01*p.size)
        if not eligible and not force_full:
            timings['total']=(time.perf_counter()-started)*1000
            return base,np.float64(.675),timings,None
        encoder=encoder.cpu().numpy()
        maps=map_from_counts(p,counts);tick('map_features')
        hand=selected_handcrafted(image,p,self.used,rgb_future.result());tick('handcrafted')
        emb=self.transform['pca'].transform(self.transform['scaler'].transform(encoder))
        x=np.concatenate([maps,*hand,emb[0]])[None]
        threshold=choose_thresholds(self.policy['model'],x,np.asarray([base.sum()]),np.asarray([p.size]))[0];tick('pca_trees')
        mask=p>=threshold;tick('mask');timings['total']=(time.perf_counter()-started)*1000
        return mask,threshold,timings,x
