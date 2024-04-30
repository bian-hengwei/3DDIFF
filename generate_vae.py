import hydra
import numpy as np
import spconv.pytorch as spconv
import torch
import tqdm

from omegaconf import DictConfig
from pathlib import Path

from x3.dataset.shapenet import ShapeNetDataModule
from x3.utils.visualization_utils import coords_to_cube_mesh
from x3.vae.lightning import VAELightningModule


def volume_to_mesh(volume, filename):
    if len(volume.shape) == 4:
        volume = volume.abs().sum(dim=0) > 0
    coords = np.argwhere(volume.cpu().numpy())
    coords_to_cube_mesh(coords, filename)


@torch.inference_mode()
def generate(config):
    output_path = Path(config.output_path.replace('visualization', 'generation'))
    output_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(next(Path('ckpts').glob(f'{config.exp}_*.ckpt')))
    
    datamodule = ShapeNetDataModule(**config.dataset, dense=config.mode == 'dense')
    datamodule.setup()
    
    model = VAELightningModule.load_from_checkpoint(
        ckpt_path,
        input_dim=datamodule.train_dataset.input_dim,
        **config.vae.params,
        mode=config.mode
    )
    model.cuda()
    model.eval()
    
    vis_count = 0
    z_spatial_shape = np.array(config.vae.spatial_shape) // (2 ** (len(config.vae.channels_multiple) - 1))
    z = torch.randn(config.vis_count, config.vae.latent_dim, *z_spatial_shape).cuda()
    x, _ = model.model.decoder(z)
    [volume_to_mesh(x[i], str(output_path / f'{i}_gt.obj')) for i in range(len(x))]


@hydra.main(config_path='conf', config_name='reconstruct_mix', version_base=None)
def main(config: DictConfig):
    print(config)
    generate(config)


if __name__ == '__main__':
    main()
