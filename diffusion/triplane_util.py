import torch
import torch.nn.functional as F
import numpy as np
from diffusion.nn import decompose_featmaps, compose_featmaps

def augment(triplane, p, tri_size=(128,128,32)):
    H, W, D = tri_size
    triplane = torch.from_numpy(triplane).float()
    feat_xy, feat_xz, feat_zy = decompose_featmaps(triplane,tri_size, False)
    if p == 0:
        feat_xy = torch.flip(feat_xy, [2])
        feat_zy = torch.flip(feat_zy, [2])
    elif p == 1:
        feat_xy = torch.flip(feat_xy, [1])
        feat_xz = torch.flip(feat_xz, [1])
    elif p == 2:
        feat_xy = torch.flip(feat_xy, [2])
        feat_zy = torch.flip(feat_zy, [2])
        feat_xy = torch.flip(feat_xy, [1])
        feat_xz = torch.flip(feat_xz, [1])
    elif p == 3: 
        feat_xy += torch.randn_like(feat_xy) * 0.05
        feat_xz += torch.randn_like(feat_xz) * 0.05
        feat_zy += torch.randn_like(feat_zy) * 0.05
    elif p == 4 :
        size = torch.randint(0, 3, (1,)).item()
        s = 80 + size*16
        region = 128-s
        x, y = torch.randint(0, region, (2,)).tolist()
        feat_xy = feat_xy[:, y:y+s, x:x+s]
        feat_xz = feat_xz[:, y:y+s, :]
        feat_zy = feat_zy[:, :, x:x+s]
        feat_xy = F.interpolate(feat_xy.unsqueeze(0).float(), size=(H, W), mode='bilinear').squeeze(0)
        feat_xz = F.interpolate(feat_xz.unsqueeze(0).float(), size=(H, D), mode='bilinear').squeeze(0)
        feat_zy = F.interpolate(feat_zy.unsqueeze(0).float(), size=(D, W), mode='bilinear').squeeze(0)
        
    triplane, _ = compose_featmaps(feat_xy, feat_xz, feat_zy, tri_size, False)
    return np.array(triplane)
