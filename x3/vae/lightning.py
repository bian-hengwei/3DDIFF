import lightning as L
import spconv.pytorch as spconv
import torch

from x3.vae.dense_vae import DenseVAE
from x3.vae.mix_vae import MixVAE
from x3.vae.sparse_vae import SparseVAE


class VAELightningModule(L.LightningModule):
    def __init__(self, input_dim, base_channels, channels_multiple, 
            latent_dim, alpha, alpha_decay_rate, beta, spatial_shape, mode):
        super().__init__()

        ModelClass = dict(dense=DenseVAE, mix=MixVAE, sparse=SparseVAE)[mode]
        self.model = ModelClass(input_dim, base_channels, channels_multiple, latent_dim)
            
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.alpha = alpha  # probability to choose ground truth structure
        self.alpha_decay_rate = alpha_decay_rate
        self.beta = beta
        self.spatial_shape = spatial_shape
        self.mode = mode

    def configure_optimizers(self):
        return self.optim

    def forward(self, x, structures=None, alpha=None):
        return self.model(x, structures, alpha)
    
    def load_batch(self, batch):
        volume = batch['volume']
        structures = batch['structure']
        x = batch['x']
        if self.mode != 'dense':
            coords, feats = x['coords'], x['feats']
            x = spconv.SparseConvTensor(feats, coords, self.spatial_shape, structures[0].shape[0])
        return volume, x, structures

    def training_step(self, batch, batch_idx):
        volume, x, structures = self.load_batch(batch)

        x, mu, logvar, structure_preds = self(x, structures, self.alpha)
        
        losses = self.model.loss(volume, x, mu, logvar, structures, structure_preds, self.beta)
        reconstruction_loss = losses['reconstruction_loss']
        kl_loss = losses['kl_loss']
        structure_loss = losses['structure_loss']
        loss = 0.0 * (1 - self.alpha) * reconstruction_loss + 1.0 * kl_loss + 1.0 * structure_loss
        
        self.log('reconstruction_loss', reconstruction_loss, prog_bar=True)
        self.log('kl_loss', kl_loss)
        self.log('structure_loss', structure_loss, prog_bar=True)
        self.log('loss', loss)
        
        precision, recall = self.model.metrics(volume, x)
        self.log('precision', precision, prog_bar=True)
        self.log('recall', recall, prog_bar=True)
        self.log('f1', 2 * precision * recall / (precision + recall), prog_bar=True)
        
        return loss

    def on_train_epoch_end(self):
        self.alpha = max(self.alpha - self.alpha_decay_rate, 0.)
