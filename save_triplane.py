import hydra
import numpy as np
import spconv.pytorch as spconv
import torch
import tqdm

from omegaconf import DictConfig
from pathlib import Path

from diffusion.nn import compose_featmaps
from x3.dataset.shapenet import ShapeNetDataModule
from x3.ae.lightning import AELightningModule


@torch.inference_mode()
def save_triplane(config):
    triplane_path = Path(config.triplane_path)
    triplane_path.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = str(next(Path('ckpts').glob(f'{config.exp}.ckpt')))
    
    datamodule = ShapeNetDataModule(**config.dataset, triplane=True)
    datamodule.setup()
    
    model = AELightningModule.load_from_checkpoint(ckpt_path, 
        in_channels=datamodule.train_dataset.input_dim, **config.ae.params)
    model.cuda()
    model.eval()
    
    dataloader = iter(datamodule.train_dataloader())
    
    for batch in tqdm.tqdm(dataloader):
        x = batch['x'].cuda()
        names = batch['name'][0]
    
        triplanes = model.model.encoder(x)
        triplanes, _ = compose_featmaps(triplanes[0].squeeze(), triplanes[1].squeeze(), triplanes[2].squeeze(), (128, 128, 128))
        
        np.save(str(triplane_path / f'{names}.npy'), triplanes.cpu().numpy())


@hydra.main(config_path='conf', config_name='reconstruct_ae', version_base=None)
def main(config: DictConfig):
    print(config)  # batch_size should be 1
    save_triplane(config)


if __name__ == '__main__':
    main()
