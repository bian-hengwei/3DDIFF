import lightning as L
import torch

from x3.ae.ae import TriplaneAE

class AELightningModule(L.LightningModule):
    def __init__(self, in_channels, encoder_channels, block_hidden_channels, mlp_hidden_channels, mlp_hidden_layers, 
            lr, lr_scheduler_steps, lr_scheduler_decay, grid_size):
        super().__init__()        
        self.model = TriplaneAE(in_channels, encoder_channels, 
            block_hidden_channels, mlp_hidden_channels, mlp_hidden_layers, grid_size)
        self.lr = lr
        self.lr_scheduler_steps = lr_scheduler_steps
        self.lr_scheduler_decay = lr_scheduler_decay
        self.grid_size = grid_size

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = dict(
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.lr_scheduler_steps, self.lr_scheduler_decay),
            interval = 'epoch',
        )
        return [optimizer], [scheduler]
        
    def forward(self, x, query):
        return self.model(x, query)

    def load_batch(self, batch):
        volume = batch['volume']
        x = batch['x']
        query = batch['query']
        coords = batch['coords']
        return volume, x, query, coords

    def training_step(self, batch, batch_idx):
        volume, x, query, coords = self.load_batch(batch)
        pred = self(x, query)
        batches = torch.arange(x.shape[0])[:, None]
        volume_query = volume[batches, coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]]
        loss = self.model.loss(volume_query, pred)
        self.log('loss', loss, prog_bar=True, on_epoch=True)
        precision, recall = self.model.metrics(volume_query, pred)
        self.log('precision', precision, prog_bar=True, on_epoch=True)
        self.log('recall', recall, prog_bar=True, on_epoch=True)
        return loss
