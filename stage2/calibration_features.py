"""Deployable map geometry, local boundary/RGB features and same-forward ROIs."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def mean(values):
    return float(values.mean()) if values.size else 0.


def stats(values, quantiles=True):
    if not values.size:
        return [0.]* (6 if quantiles else 3)
    if quantiles:
        p=np.clip(values,1e-6,1-1e-6)
        return [mean(values),float(values.std()),*np.quantile(values,[.1,.5,.9]).tolist(),
                mean(-p*np.log(p)-(1-p)*np.log(1-p))]
    return [mean(values),float(values.std()),float(np.quantile(values,.9))]


def component_features(mask):
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


def handcrafted(image,probability):
    """Aspect-preserving analysis raster: longest side256; no GT input."""
    h,w=probability.shape
    scale=256/max(h,w)
    shape=(max(1,int(round(w*scale))),max(1,int(round(h*scale))))
    q=cv2.resize(probability,shape,interpolation=cv2.INTER_AREA)
    im=cv2.resize(image,shape,interpolation=cv2.INTER_AREA).astype(np.float32)/255.
    geometry=[];largest=[];centers=[]
    for t in [.5,.75,.9,.95]:
        row,m,c=component_features(q>=t)
        geometry.extend(row);largest.append(m);centers.append(c)
    for i in range(3):
        union=(largest[i]|largest[i+1]).sum()
        geometry.extend([float((largest[i]&largest[i+1]).sum()/max(union,1)),
                         float(np.linalg.norm(centers[i]-centers[i+1])),
                         float((largest[i]&largest[i+1]).sum()/max(largest[i].sum(),1))])
    kernel=np.ones((7,7),np.uint8)
    gx=cv2.Sobel(q,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(q,cv2.CV_32F,0,1,ksize=3)
    gradient=np.sqrt(gx*gx+gy*gy)
    boundary=[];base_regions=None;base_band=None
    for t in [.75,.9]:
        mask=(q>=t).astype(np.uint8)
        dilated=cv2.dilate(mask,kernel)>0
        eroded=cv2.erode(mask,kernel)>0
        regions=[eroded,(mask>0)&~eroded,dilated&~(mask>0),~dilated]
        band=dilated&~eroded
        for region in regions:
            boundary.extend(stats(q[region]))
        boundary.extend([mean(gradient[band]),mean(((q>.5)&(q<.95))[band]),
                         float(band.sum()/max(mask.sum(),1))])
        if t==.75:
            base_regions,base_band=regions,band
    gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    saturation=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)[:,:,1]
    dx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    dy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    rgb_gradient=np.sqrt(dx*dx+dy*dy)
    blurred=cv2.GaussianBlur(gray,(0,0),1)
    local_std=np.sqrt(np.maximum(cv2.GaussianBlur(gray*gray,(0,0),1)-blurred*blurred,0))
    channels=[im[:,:,0],im[:,:,1],im[:,:,2],gray,saturation,rgb_gradient,
              np.abs(cv2.Laplacian(gray,cv2.CV_32F)),np.abs(gray-blurred),local_std]
    rgb=[]
    for region in base_regions:
        for channel in channels:
            rgb.extend(stats(channel[region],False))
    for channel in channels:
        inner,outer=mean(channel[base_regions[1]]),mean(channel[base_regions[2]])
        rgb.append((inner-outer)/(abs(inner)+abs(outer)+1e-6))
    a,b=gradient[base_band],rgb_gradient[base_band]
    correlation=float(np.corrcoef(a,b)[0,1]) if len(a)>2 and a.std()>1e-8 and b.std()>1e-8 else 0.
    rgb.extend([correlation,mean(rgb_gradient[base_band])/(mean(rgb_gradient[~base_band])+1e-6)])
    result=[np.asarray(a,np.float32) for a in [geometry,boundary,rgb]]
    assert [len(a) for a in result]==[49,54,119]
    assert all(np.isfinite(a).all() for a in result)
    return result


class WithEncoderROIs(torch.nn.Module):
    """Reuse the four MiT feature maps; no second encoder forward."""
    def __init__(self,net):
        super().__init__()
        self.net=net
        self.maps=None
        self.handle=net.encoder.register_forward_hook(self.capture)

    def capture(self,module,inputs,outputs):
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
            denominator=weight.sum((2,3)).clamp_min(1e-6)
            value=torch.einsum('bchw,brhw->brc',feature,weight)/denominator[:,:,None]
            pooled.append(value.flatten(1))
        return logits,torch.cat(pooled,dim=1)
