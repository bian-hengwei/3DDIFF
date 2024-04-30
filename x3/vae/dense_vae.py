import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import x3.utils.nn as x3nn

from x3.vae.vae import V3DEncoder, V3DDecoder, V3DVAE


class DenseEncoder(V3DEncoder):
    def __init__(self, input_dim, base_channels, channels_multiple, latent_dim):
        modules = dict(
            linear1=x3nn.Linear3D,
            gcr1=x3nn.gcr,
            maxpool=nn.MaxPool3d,
            to_dense=nn.Sequential,
            gcr2=x3nn.gcr,
            linear2=x3nn.Linear3D,
        )
        super(DenseEncoder, self).__init__(input_dim, base_channels, channels_multiple, latent_dim, modules)


class DenseDecoder(V3DDecoder):
    def __init__(self, upsample_count, latent_dim):
        modules = dict(
            block=x3nn.block,
            dconv=x3nn.dconv,
        )
        super(DenseDecoder, self).__init__(upsample_count, latent_dim, modules)

    def upsample(self, x, upsample_mask, delete_mask):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        upsample_mask = upsample_mask.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        delete_mask = delete_mask.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        return x, upsample_mask, delete_mask

    def filter(self, x, block, upsample_mask, delete_mask, structure=None, alpha=0.):
        out = block(x)
        probabilities = torch.softmax(out, dim=1)

        if structure is not None and random.random() < alpha:
            structure_one_hot = F.one_hot(structure.to(torch.int64), num_classes=3).permute(0, 4, 1, 2, 3)  # shape: (N, 3, D, H, W)
            structure_one_hot = structure_one_hot.to(dtype=probabilities.dtype)
            probabilities = structure_one_hot

        classification = torch.argmax(probabilities, dim=1)  # 0 means keep, 1 means upsample, 2 means delete

        # mask all points we should not delete
        mask = ((upsample_mask == 0) | ((classification == 0) | (classification == 1))).float()
        mask = mask.unsqueeze(1).repeat(1, x.shape[1], 1, 1, 1)  # shape: (N, C, D, H, W)
        x = x * mask

        # upsample mask stores those voxels that haven't been marked as either keep or delete
        upsample_mask_mask = ((classification == 0) | (classification == 2)).to(torch.int8)
        upsample_mask = upsample_mask * (1 - upsample_mask_mask)
        
        # delete mask stores those voxels that have been marked as delete
        delete_mask_mask = (classification == 2).to(torch.int8)
        delete_mask = 1 - ((1 - delete_mask) * (1 - delete_mask_mask))

        return x, out, upsample_mask, delete_mask

    def forward(self, x, structures=None, alpha=None):
        structure_preds = list()  # store block outputs for loss
        # upsample_mask (N, D, H, W) == 1 means upsampling allowed
        upsample_mask = torch.ones((x.shape[0], *x.shape[2:]), dtype=torch.int8).to(x.device)
        # delete_mask (N, D, H, W) == 1 means already deleted
        delete_mask = torch.zeros((x.shape[0], *x.shape[2:]), dtype=torch.int8).to(x.device)
        
        for i, block in enumerate(self.blocks):
            structure = structures[i] if structures is not None else None
            x, classification, upsample_mask, delete_mask = self.filter(x, block, upsample_mask, delete_mask, structure, alpha)
            structure_preds.append(classification)
            
            if i < len(self.blocks) - 1:
                x, upsample_mask, delete_mask = self.upsample(x, upsample_mask, delete_mask)
                x = self.conv_layers[i](x)
                mask = delete_mask.to(torch.int8).unsqueeze(1).repeat(1, x.shape[1], 1, 1, 1)
                x = x * (1 - mask)
                
        return x, structure_preds


class DenseVAE(V3DVAE):
    def __init__(self, input_dim, base_channels, channels_multiple, latent_dim):
        modules = dict(encoder=DenseEncoder, decoder=DenseDecoder)
        super(DenseVAE, self).__init__(input_dim, base_channels, channels_multiple, latent_dim, modules)
