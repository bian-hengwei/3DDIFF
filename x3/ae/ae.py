import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def compose_triplane(feat_maps):
    h_xy, h_xz, h_yz = feat_maps  # (H, W), (H, D), (W, D)
    B, C, H, W, D = *h_xy.shape, h_xz.shape[-1]
    h_new, w_new = max(H, W), max(W, D)
    h_xy = F.pad(h_xy, (0, w_new - W, 0, h_new - H))
    h_xz = F.pad(h_xz, (0, w_new - D, 0, h_new - H))
    h_yz = F.pad(h_yz, (0, w_new - D, 0, h_new - W))
    h = torch.cat([h_xy, h_xz, h_yz], dim=1)  # (B, 3C, h, w)
    return h, (B, C, H, W, D)


def decompose_triplane(composed_map, sizes):
    B, C, H, W, D = sizes
    C = composed_map.shape[1] // 3
    h_xy = composed_map[:, : C, : H, : W]
    h_xz = composed_map[:, C: 2 * C, : H, : D]
    h_yz = composed_map[:, 2 * C:, : W, : D]
    return h_xy, h_xz, h_yz
    

class TriplaneEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv0 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate')
        self.convblock1 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm3d(out_channels)
        )
        self.convblock2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm3d(out_channels)
        )
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):  # [b, channels, X, Y, Z]
        x = self.conv0(x)  # [b, channels, X, Y, Z]

        residual_feat = x
        x = self.convblock1(x)  # [b, channels, X, Y, Z]
        x = x + residual_feat   # [b, channels, X, Y, Z]

        residual_feat = x
        x = self.convblock2(x)
        x = x + residual_feat  # [b, channels, X, Y, Z]
        
        triplane = [(self.norm(x.mean(dim=dims)) * 0.5).tanh() for dims in [4, 3, 2]]  # [b, C, X, Y]
        return triplane


class TriplaneResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_layers = nn.Conv2d(in_channels * 3, out_channels * 3, groups=3, kernel_size=3, stride=1, padding=1)
        self.norms = nn.ModuleList([nn.InstanceNorm2d(out_channels, eps=1e-6, affine=True) for _ in range(3)])
        self.out_layers = nn.Sequential(nn.SiLU(), 
            zero_module(nn.Conv2d(out_channels * 3, out_channels * 3, groups=3, kernel_size=3, stride=1, padding=1)))
        self.shortcut = nn.Conv2d(in_channels * 3, out_channels * 3, groups=3, kernel_size=1, stride=1, padding=0)

    def forward(self, triplane):
        h_original, (B, C, H, W, D) = compose_triplane(triplane)
        h = self.in_layers(h_original)
        triplane = decompose_triplane(h, (B, C, H, W, D))
        triplane = [self.norms[i](triplane[i]) for i in range(3)]
        h, _ = compose_triplane(triplane)
        h = self.out_layers(h)
        h = h + self.shortcut(h_original)
        return decompose_triplane(h, (B, C, H, W, D))


class TriplaneMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_hidden_layers):
        super().__init__()
        first_layer_list = [nn.Linear(in_channels, hidden_channels), nn.ReLU()]
        for _ in range(num_hidden_layers // 2):
            first_layer_list.append(nn.Linear(hidden_channels, hidden_channels))
            first_layer_list.append(nn.ReLU())
        self.first_layers = nn.Sequential(*first_layer_list)

        second_layer_list = [nn.Linear(in_channels + hidden_channels, hidden_channels), nn.ReLU()]
        for _ in range(num_hidden_layers // 2 - 1):
            second_layer_list.append(nn.Linear(hidden_channels, hidden_channels))
            second_layer_list.append(nn.ReLU())
        second_layer_list.append(nn.Linear(hidden_channels, 1))
        self.second_layers = nn.Sequential(*second_layer_list)

    def forward(self, x):
        h = self.first_layers(x)
        h = torch.cat([x, h], dim=-1)
        h = self.second_layers(h)
        return h
    

class TriplaneDecoder(nn.Module):
    def __init__(self, in_channels, block_hidden_channels, mlp_hidden_channels, mlp_hidden_layers, grid_size):
        super().__init__()
        self.block = TriplaneResidualBlock(in_channels, block_hidden_channels)
        self.mlp = TriplaneMLP(block_hidden_channels + 3 * 2 * 6, mlp_hidden_channels, mlp_hidden_layers)
        self.grid_size = grid_size
        
    def forward(self, triplane, query):
        triplane = self.block(triplane)
        batch_size = triplane[0].shape[0]
        
        dims = [[0, 1], [0, 2], [1, 2]]
        feats = sum([self.sample_feature_plane2D(triplane[i], query[..., dims[i]], batch_size) for i in range(3)])

        pos_encoding = list()
        for freq in range(6):
            pos_encoding.append(torch.sin((2. ** freq) * query))
            pos_encoding.append(torch.cos((2. ** freq) * query))
        pos_encoding = torch.cat(pos_encoding, dim=-1)  # [bs, N, 6 * self.pos_num_freq]

        preds = self.mlp(torch.cat([feats, pos_encoding], dim=-1)) # h: [bs, N, 1]
        return preds.squeeze(-1)  # B, N

    def sample_feature_plane2D(self, feat_map, x, batch_size):
        sample_coords = x.view(batch_size, 1, -1, 2) # sample_coords: [bs, 1, N, 2]
        feat = F.grid_sample(feat_map, sample_coords.flip(-1), align_corners=False, padding_mode='border') # feat : [bs, C, 1, N]
        feat = feat[:, :, 0, :] # feat : [bs, C, N]
        feat = feat.transpose(1, 2) # feat : [bs, N, C]
        return feat


class TriplaneAE(nn.Module):
    def __init__(self, in_channels, encoder_channels, block_hidden_channels, mlp_hidden_channels, mlp_hidden_layers, grid_size):
        super().__init__()
        self.encoder = TriplaneEncoder(in_channels, encoder_channels)
        self.decoder = TriplaneDecoder(encoder_channels, block_hidden_channels, mlp_hidden_channels, mlp_hidden_layers, grid_size)

    def forward(self, x, query):
        triplane = self.encoder(x)
        return self.decoder(triplane, query)

    def reconstruction_loss(self, gt, pred):
        return F.binary_cross_entropy_with_logits(pred, gt.float())
    
    def metrics(self, gt, pred):
        pred_prob = F.sigmoid(pred)
        pred_binary = pred_prob > 0.5
        true_positives = ((pred_binary == 1) & (gt == 1)).sum()
        predicted_positives = (pred_binary == 1).sum()
        actual_positives = (gt == 1).sum()
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        return precision, recall

    def loss(self, x_gt, x_pred):
        reconstruction_loss = self.reconstruction_loss(x_gt, x_pred)
        return reconstruction_loss
