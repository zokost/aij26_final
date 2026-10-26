"""Probability, image-region and same-forward encoder features for calibration."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

FEATURE_THRESHOLDS=np.asarray([.5,.6,.7,.75,.8,.85,.9,.95])

def mean(values):
    return float(values.mean()) if values.size else 0.

def component_features(mask):
    """Describe mask topology, dominant component, compactness and border contact."""
    n, labels, sizes, centroids=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    areas=sizes[1:,cv2.CC_STAT_AREA]
    count=len(areas);total=int(mask.sum());frame=mask.size
    largest=np.zeros_like(mask);center=np.zeros(2)
    if count:
        order=np.argsort(areas)[::-1]
        largest=labels==order[0]+1
        center=centroids[order[0]+1]/np.asarray([mask.shape[1],mask.shape[0]])
        shares=areas[order]/max(total,1)
        ys,xs=np.nonzero(mask)
        bbox=(xs.max()-xs.min()+1)*(ys.max()-ys.min()+1)
    else:
        shares=np.zeros(2);bbox=1
    contours,hierarchy=cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    perimeter=sum(cv2.arcLength(c,True) for c in contours)
    holes=int((hierarchy[0,:,3]>=0).sum()) if hierarchy is not None else 0
    border=np.zeros_like(mask);border[[0,-1],:]=True;border[:,[0,-1]]=True
    row=[np.log1p(count),np.log1p((areas>=max(2,.001*frame)).sum()),
         float(areas.max()/frame) if count else 0.,float(shares[0]),
         float(shares[1]) if count>1 else 0.,float(areas.mean()/frame) if count else 0.,
         float(np.log1p(perimeter**2/(4*np.pi*max(total,1)))),np.log1p(holes),
         float((mask&border).sum()/max(total,1)),float(total/bbox)]
    return row,largest,center

def map_features(probability):
    """22 native-size probability statistics, in the original training order."""
    hist = np.histogram(probability, bins=[0, .02, .1, .25, .5, .75, .9, .98, 1.])[0] / probability.size
    areas = [(probability >= t).mean() for t in FEATURE_THRESHOLDS]
    small = cv2.resize(probability, (32, 32), interpolation=cv2.INTER_AREA)
    spatial = [np.abs(np.diff(small, axis=0)).mean(), np.abs(np.diff(small, axis=1)).mean()]
    return np.asarray([*hist, *areas, probability.max(), probability.mean(),
                       *spatial, np.log(probability.shape[0]), np.log(probability.shape[1])], np.float64)

def map_from_counts(p,counts):
    cuts=counts[:7];hist=np.diff(np.r_[0,cuts,p.size])/p.size
    areas=counts[7:]/p.size
    small=cv2.resize(p,(32,32),interpolation=cv2.INTER_AREA)
    spatial=[np.abs(np.diff(small,axis=0)).mean(),np.abs(np.diff(small,axis=1)).mean()]
    return np.asarray([*hist,*areas,p.max(),p.mean(),*spatial,np.log(p.shape[0]),np.log(p.shape[1])],np.float64)

def prepare_rgb(image):
    """Compute cheap appearance channels on a 256px aspect-preserving view."""
    h,w=image.shape[:2];scale=256/max(h,w)
    shape=(max(1,int(round(w*scale))),max(1,int(round(h*scale))))
    im=cv2.resize(image,shape,interpolation=cv2.INTER_AREA).astype(np.float32)/255.
    gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY);saturation=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)[:,:,1]
    dx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3);dy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3);rgb_gradient=np.sqrt(dx*dx+dy*dy)
    blurred=cv2.GaussianBlur(gray,(0,0),1);local_std=np.sqrt(np.maximum(cv2.GaussianBlur(gray*gray,(0,0),1)-blurred*blurred,0))
    return [im[:,:,0],im[:,:,1],im[:,:,2],gray,saturation,rgb_gradient,np.abs(cv2.Laplacian(gray,cv2.CV_32F)),np.abs(gray-blurred),local_std],rgb_gradient

def selected_handcrafted(image,probability,used,prepared=None):
    """Evaluate only the feature columns actually referenced by frozen trees.

    Unused columns are zero; used columns preserve the original arithmetic.
    """
    h,w=probability.shape;scale=256/max(h,w)
    shape=(max(1,int(round(w*scale))),max(1,int(round(h*scale))))
    q=cv2.resize(probability,shape,interpolation=cv2.INTER_AREA)
    # Fixed feature layout: probability22 | geometry49 | boundary54 | RGB119 |
    # encoder PCA32. Offsets below must match the fitted tree feature indices.
    geometry=[];largest=[];centers=[]
    for t in [.5,.75,.9,.95]:
        row,m,c=component_features(q>=t);geometry.extend(row);largest.append(m);centers.append(c)
    for i in range(3):
        union=(largest[i]|largest[i+1]).sum()
        geometry.extend([float((largest[i]&largest[i+1]).sum()/max(union,1)),float(np.linalg.norm(centers[i]-centers[i+1])),float((largest[i]&largest[i+1]).sum()/max(largest[i].sum(),1))])
    kernel=np.ones((7,7),np.uint8)
    gx=cv2.Sobel(q,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(q,cv2.CV_32F,0,1,ksize=3);gradient=np.sqrt(gx*gx+gy*gy)
    boundary=np.zeros(54,np.float32);base_regions=None;base_band=None
    # Compare confident interiors, inner/outer boundary bands and background.
    # Appearance differences near a boundary help distinguish plausible edits
    # from over-segmentation without running another image model.
    for index,t in enumerate([.75,.9]):
        mask=(q>=t).astype(np.uint8);dilated=cv2.dilate(mask,kernel)>0;eroded=cv2.erode(mask,kernel)>0
        regions=[eroded,(mask>0)&~eroded,dilated&~(mask>0),~dilated];band=dilated&~eroded
        for r,region in enumerate(regions):
            offset=index*27+r*6;need=[k for k in range(6) if offset+k+71 in used]
            if not need:continue
            values=q[region]
            if not values.size:continue
            if 0 in need:boundary[offset]=values.mean()
            if 1 in need:boundary[offset+1]=values.std()
            quant=[k for k in [2,3,4] if k in need]
            if quant:
                result=np.quantile(values,[[.1,.5,.9][k-2] for k in quant])
                for k,value in zip(quant,result):boundary[offset+k]=value
            if 5 in need:
                p=np.clip(values,1e-6,1-1e-6);boundary[offset+5]=mean(-p*np.log(p)-(1-p)*np.log(1-p))
        boundary[index*27+24:index*27+27]=[mean(gradient[band]),mean(((q>.5)&(q<.95))[band]),float(band.sum()/max(mask.sum(),1))]
        if index==0:base_regions,base_band=regions,band
    channels,rgb_gradient=prepare_rgb(image) if prepared is None else prepared
    rgb=np.zeros(119,np.float32)
    for r,region in enumerate(base_regions):
        for c,channel in enumerate(channels):
            offset=r*27+c*3;need=[k for k in range(3) if offset+k+125 in used]
            if not need:continue
            values=channel[region]
            if not values.size:continue
            if 0 in need:rgb[offset]=values.mean()
            if 1 in need:rgb[offset+1]=values.std()
            if 2 in need:rgb[offset+2]=np.quantile(values,.9)
    for c,channel in enumerate(channels):
        if 233+c in used:
            inner,outer=mean(channel[base_regions[1]]),mean(channel[base_regions[2]])
            rgb[108+c]=(inner-outer)/(abs(inner)+abs(outer)+1e-6)
    if 242 in used:
        a,b=gradient[base_band],rgb_gradient[base_band]
        rgb[117]=float(np.corrcoef(a,b)[0,1]) if len(a)>2 and a.std()>1e-8 and b.std()>1e-8 else 0.
    if 243 in used:rgb[118]=mean(rgb_gradient[base_band])/(mean(rgb_gradient[~base_band])+1e-6)
    return [np.asarray(geometry,np.float32),boundary,rgb]

def handcrafted(image,probability):
    """Full frozen feature vector, using the same implementation as inference."""
    return selected_handcrafted(image,probability,set(range(276)))

class WithEncoderROIs(torch.nn.Module):
    """Reuse four encoder scales in mask/boundary/background regions.

    Pooling yields 3072 values per image; the fitted scaler/PCA reduces them
    to32. The .75 feature-region cutoff is fixed independently of the policy's
    .675 base threshold. No second encoder or teacher runs at inference.
    """
    def __init__(self,net):
        super().__init__()
        self.net=net
        self.maps=None
        self.handle=net.encoder.register_forward_hook(self.capture)

    def capture(self,module,inputs,outputs):
        # SMP exposes the four MiT stages as the last four encoder outputs.
        self.maps=outputs[-4:]

    def forward(self,x):
        logits=self.net(x)
        mask=(logits.sigmoid()>=.75).float()
        dilated=F.max_pool2d(mask,11,1,5)
        eroded=1-F.max_pool2d(1-mask,11,1,5)
        regions=torch.cat([mask,dilated-eroded,1-dilated],dim=1)
        pooled=[]
        for feature in self.maps:
            weight=F.interpolate(regions,size=feature.shape[-2:],mode='area')
            # Area-resampled region weights retain narrow structures on coarse
            # feature maps. Empty regions divide by a clamped denominator.
            denominator=weight.sum((2,3)).clamp_min(1e-6)
            value=torch.einsum('bchw,brhw->brc',feature,weight)/denominator[:,:,None]
            pooled.append(value.flatten(1))
        return logits,torch.cat(pooled,dim=1)
