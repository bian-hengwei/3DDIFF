import torch
import numpy as np

from pathlib import Path

from diffusion.triplane_util import augment


class TriplaneDataset(torch.utils.data.Dataset):
    def __init__(self, args, imageset):
        self.args = args
        self.imageset = imageset 
        
        self.grid_size = (128, 128, 128)
        self.tri_size = (128, 128, 128)
        
        tri_path = Path('data/triplane')
        files = list(tri_path.glob('*.npy'))
        files = sorted(files)
        self.im_idx = files
   
    def __len__(self):
        return len(self.im_idx)  
    
    def __getitem__(self, index):
        triplane = np.load(self.im_idx[index]).squeeze()
        condition = np.zeros_like(triplane)
        path = str(self.im_idx[index])
        
        if (not self.args.diff_net_type == 'unet_voxel') and (self.imageset == 'train') :
            # rotation
            q = torch.randint(0, 3, (1,)).item()    
            if q == 0:
                triplane = torch.from_numpy(triplane).permute(0, 2, 1).numpy()
                condition = torch.from_numpy(condition).permute(0, 2, 1).numpy()
                        
            # other augmentations (flip, crop, noise.)
            p = torch.randint(0, 6, (1,)).item()
            triplane = augment(triplane, p, self.tri_size)
            condition = augment(condition, p, self.tri_size)
                    
        return triplane, {'y':condition, 'H':self.tri_size[0], 'W':self.tri_size[1], 'D':self.tri_size[2], 'path':(path)}
