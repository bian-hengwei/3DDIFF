import torch
import torch.nn as nn
import torch.nn.functional as F


class V3DEncoder(nn.Module):
    def __init__(self, input_dim, base_channels, channels_multiple, latent_dim, modules):
        super(V3DEncoder, self).__init__()
        
        # lift to d-dimentional feature
        self.linear = modules['linear1'](input_dim, base_channels)

        # coarsening voxels to bottleneck dimension
        self.conv_layers = list()
        for i in range(len(channels_multiple) - 1):
            self.conv_layers.append(modules['gcr1'](base_channels * channels_multiple[i], base_channels * channels_multiple[i + 1]))
            self.conv_layers.append(modules['maxpool'](2, 2))
        self.conv_layers = nn.Sequential(*self.conv_layers)
        
        # Optional layer to convert bottleneck to dense
        self.to_dense_layer = modules['to_dense']()

        # convert to latent tensor X of the same shape and sparsity pattern
        bottleneck_channels = base_channels * channels_multiple[-1]
        self.b2l_conv_layers = nn.Sequential(
            modules['gcr2'](bottleneck_channels, bottleneck_channels),
            modules['gcr2'](bottleneck_channels, bottleneck_channels),
            modules['gcr2'](bottleneck_channels, bottleneck_channels),
            modules['gcr2'](bottleneck_channels, latent_dim),
        )

        self.linear_mu = modules['linear2'](latent_dim, latent_dim)
        self.linear_logvar = modules['linear2'](latent_dim, latent_dim)

    def forward(self, x):  # input: (N, D, H, W, C)
        x = self.linear(x)
        x = self.conv_layers(x)
        x = self.to_dense_layer(x)
        x = self.b2l_conv_layers(x)
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)
        return x, mu, logvar
    

class V3DDecoder(nn.Module):
    def __init__(self, upsample_count, latent_dim, modules):
        super(V3DDecoder, self).__init__()

        self.blocks = nn.ModuleList()
        for _ in range(upsample_count):
            self.blocks.append(modules['block'](latent_dim, 3))

        self.conv_layers = nn.ModuleList()
        for _ in range(upsample_count - 1):
            self.conv_layers.append(modules['dconv'](latent_dim, latent_dim, latent_dim))
            

class V3DVAE(nn.Module):
    def __init__(self, input_dim, base_channels, channels_multiple, latent_dim, modules):
        super(V3DVAE, self).__init__()
        self.encoder = modules['encoder'](input_dim, base_channels, channels_multiple, latent_dim)
        self.decoder = modules['decoder'](len(channels_multiple), latent_dim)
        
    def forward(self, x, structures=None, alpha=0.):
        x, mu, logvar = self.encoder(x)
        x = self.latent_sample(mu, logvar)
        x, structure_preds = self.decoder(x, structures, alpha)
        return x, mu, logvar, structure_preds

    def latent_sample(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def reconstruction_loss(self, gt, pred):
        pred = torch.sign(pred.abs().max(dim=1).values)
        return (gt - pred).abs().sum() * 100 / (gt.shape[0] * gt.shape[1] * gt.shape[2] * gt.shape[3])

    def kl_loss(self, mu, logvar):
        return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / (mu.shape[0] * mu.shape[2] * mu.shape[3] * mu.shape[4])

    def structure_loss(self, structures, structures_pred):
        loss = 0.
        for gt, pr in zip(structures, structures_pred):
            loss += F.cross_entropy(pr, gt.long())
        return loss

    def loss(self, x_gt, x_pred, mu, logvar, structures, structures_pred, beta=0.0015):
        reconstruction_loss = self.reconstruction_loss(x_gt, x_pred)
        kl_loss = beta * self.kl_loss(mu, logvar)
        structure_loss = self.structure_loss(structures, structures_pred)
        return {
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
            'structure_loss': structure_loss,
        }
    
    def metrics(self, gt, pred):
        pred = torch.sign(pred.abs().max(dim=1).values)
        true_positives = ((pred == 1) & (gt == 1)).sum()
        predicted_positives = (pred == 1).sum()
        actual_positives = (gt == 1).sum()
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        return precision, recall
