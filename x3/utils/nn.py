import torch
import torch.nn as nn
import spconv.pytorch as spconv


class Linear3D(nn.Module):  # linear does not work well for 3D data
    def __init__(self, in_features, out_features):
        super(Linear3D, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        N, C, D, H, W = x.size()
        x = x.permute(0, 2, 3, 4, 1).reshape(-1, C)
        x = self.linear(x)
        x = x.reshape(N, D, H, W, -1).permute(0, 4, 1, 2, 3)
        return x
    

class SparseLinear(nn.Linear):
    def forward(self, x):
        features = super().forward(x.features)
        return spconv.SparseConvTensor(features, x.indices, x.spatial_shape, x.batch_size)
    

class SparseGroupNorm(nn.GroupNorm):  # from torchsparse
    def forward(self, x):
        features, indices, spatial_shape, batch_size = x.features, x.indices, x.spatial_shape, x.batch_size
        num_channels = features.shape[1]
        
        gn_features = torch.zeros_like(features).to(features)
        for b in range(batch_size):
            batch_indices = indices[:, 0] == b
            batch_features = features[batch_indices]
            batch_features = batch_features.transpose(0, 1).reshape(1, num_channels, -1)
            batch_features = super().forward(batch_features)
            batch_features = batch_features.reshape(num_channels, -1).transpose(0, 1)
            gn_features[batch_indices] = batch_features.to(gn_features.dtype)

        return spconv.SparseConvTensor(gn_features, indices, spatial_shape, batch_size)
    

class SparseLeakyReLU(nn.LeakyReLU):
    def forward(self, x):
        features = super().forward(x.features)
        return spconv.SparseConvTensor(features, x.indices, x.spatial_shape, x.batch_size)
    

def gcr(a, b, num_groups=4, kernel_size=3):
    return nn.Sequential(
        nn.GroupNorm(num_groups, a),
        nn.Conv3d(a, b, kernel_size, padding=1),
        nn.LeakyReLU(),
    )
    

def spgcr(a, b, num_groups=4, kernel_size=3):
    return nn.Sequential(
        SparseGroupNorm(num_groups, a),
        spconv.SubMConv3d(a, b, kernel_size, padding=1),
        SparseLeakyReLU(),
    )
    

def block(a, b, num_groups=4, kernel_size=3):
    return nn.Sequential(
        nn.GroupNorm(num_groups, a),
        nn.Conv3d(a, a, kernel_size, padding=1),
        nn.LeakyReLU(),
        nn.Conv3d(a, b, 1),
    )
    

def spblock(a, b, num_groups=4, kernel_size=3):
    return nn.Sequential(
        SparseGroupNorm(num_groups, a),
        spconv.SubMConv3d(a, a, kernel_size, padding=1),
        SparseLeakyReLU(),
        spconv.SubMConv3d(a, b, 1),
    )


def dconv(a, b, c, num_groups=4, kernel_size=3):
    return nn.Sequential(
        nn.GroupNorm(num_groups, a),
        nn.Conv3d(a, b, kernel_size, padding=1),
        nn.LeakyReLU(),
        nn.GroupNorm(num_groups, b),
        nn.Conv3d(b, c, kernel_size, padding=1),
        nn.LeakyReLU(),
    )


def spdconv(a, b, c, num_groups=4, kernel_size=3):
    return nn.Sequential(
        SparseGroupNorm(num_groups, a),
        spconv.SubMConv3d(a, b, kernel_size, padding=1),
        SparseLeakyReLU(),
        SparseGroupNorm(num_groups, b),
        spconv.SubMConv3d(b, c, kernel_size, padding=1),
        SparseLeakyReLU(),
    )
