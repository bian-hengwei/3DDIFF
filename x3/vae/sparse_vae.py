import random
import spconv.pytorch as spconv
import torch
import torch.nn.functional as F

import x3.utils.nn as x3nn

from x3.vae.vae import V3DEncoder, V3DDecoder, V3DVAE


class SparseEncoder(V3DEncoder):
    def __init__(self, input_dim, base_channels, channels_multiple, latent_dim):
        modules = dict(
            linear1=x3nn.SparseLinear,
            gcr1=x3nn.spgcr,
            maxpool=spconv.SparseMaxPool3d,
            to_dense=spconv.Identity,
            gcr2=x3nn.spgcr,
            linear2=x3nn.SparseLinear,
        )
        super(SparseEncoder, self).__init__(input_dim, base_channels, channels_multiple, latent_dim, modules)


class SparseDecoder(V3DDecoder):
    def __init__(self, upsample_count, latent_dim):
        modules = dict(
            block=x3nn.spblock,
            dconv=x3nn.spdconv,
        )
        super(SparseDecoder, self).__init__(upsample_count, latent_dim, modules)

    def upsample(self, x, upsample_mask):
        N = x.indices.shape[0]
        up_indices = torch.zeros((N, 8, 4)).to(x.indices)
        for i in range(8):
            scale = torch.tensor([1, 2, 2, 2]).to(x.indices)
            offset = torch.tensor([0, i // 4, (i // 2) % 2, i % 2]).to(x.indices)
            up_indices[:, i, :] = scale * x.indices + offset
        up_indices = up_indices.view(-1, 4)
        up_features = x.features.repeat_interleave(8, dim=0)
        up_upsample_mask = upsample_mask.repeat_interleave(8)
        x = spconv.SparseConvTensor(up_features, up_indices, [s * 2 for s in x.spatial_shape], x.batch_size)
        return x, up_upsample_mask

    def filter(self, x, block, upsample_mask, structure=None, alpha=0.):
        out = block(x)

        if structure is not None and random.random() < alpha:
            classification = structure[x.indices[:, 0], x.indices[:, 1], x.indices[:, 2], x.indices[:, 3]]
        else:
            probabilities = torch.softmax(out.features, dim=1)
            classification = torch.argmax(probabilities, dim=1)  # 0 means keep, 1 means upsample, 2 means delete

        # mask all points we should not delete
        keep_mask = ((upsample_mask == 0) | ((classification == 0) | (classification == 1)))
        indices = x.indices[keep_mask]
        features = x.features[keep_mask]
        x = spconv.SparseConvTensor(features, indices, x.spatial_shape, x.batch_size)

        # upsample mask stores those voxels that haven't been marked as either keep or delete
        upsample_mask_mask = ((classification == 0) | (classification == 2))
        upsample_mask[upsample_mask_mask] = 0
        upsample_mask = upsample_mask[keep_mask]

        return x, out, upsample_mask

    def forward(self, x, structures=None, alpha=0.):
        structure_preds = list()  # store block outputs for loss
        # upsample_mask (N) == 1 means upsampling allowed
        upsample_mask = torch.ones((x.indices.shape[0]), dtype=torch.int8).to(x.indices.device)
        
        for i, block in enumerate(self.blocks):
            structure = structures[i] if structures is not None else None
            x, classification, upsample_mask = self.filter(x, block, upsample_mask, structure, alpha)
            structure_preds.append(classification)
            
            if i < len(self.blocks) - 1:
                x, upsample_mask = self.upsample(x, upsample_mask)
                x = self.conv_layers[i](x)
                
        return x, structure_preds


class SparseVAE(V3DVAE):
    def __init__(self, input_dim, base_channels, channels_multiple, latent_dim):
        modules = dict(encoder=SparseEncoder, decoder=SparseDecoder)
        super(SparseVAE, self).__init__(input_dim, base_channels, channels_multiple, latent_dim, modules)

    def latent_sample(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar.features)
            eps = torch.randn_like(std)
            return spconv.SparseConvTensor(mu.features + eps * std, mu.indices, mu.spatial_shape, mu.batch_size)
        return mu

    def kl_loss(self, mu, logvar):
        return (-0.5 * torch.sum(1 + logvar.features - mu.features.pow(2) - logvar.features.exp())) / mu.indices.shape[0]

    def structure_loss(self, structures, structures_pred):
        loss = 0.
        for gt, pr in zip(structures, structures_pred):
            indices = pr.indices
            gt = gt[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]
            loss += F.cross_entropy(pr.features, gt.long())
        return loss / structures[0].shape[0]
        return loss
    
    def loss(self, x_gt, x_pred, mu, logvar, structures, structures_pred, beta=0.0015):
        return super().loss(x_gt, x_pred.dense(), mu, logvar, structures, structures_pred, beta)
