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
def reconstruct(config):
    output_path = Path(config.output_path)
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
    
    dataloader = iter(datamodule.train_dataloader())
    
    vis_count = 0
    precision_all, recall_all = 0, 0
    
    for batch in tqdm.tqdm(dataloader):
        volume = batch['volume'].cuda()
        structures = [s.cuda() for s in batch['structure']]
        x = batch['x']
        if config.mode != 'dense':
            coords, feats = x['coords'].cuda(), x['feats'].cuda()
            x = spconv.SparseConvTensor(feats, coords, model.spatial_shape, structures[0].shape[0])
        else:
            x = x.cuda()

        x, mu, logvar, structure_preds = model(x)
        
        # losses = model.model.loss(volume, x, mu, logvar, structures, structure_preds)
        
        if config.mode == 'sparse':
            x = x.dense()
            
        precision, recall = model.model.metrics(volume, x)
        precision_all += precision
        recall_all += recall
        
        i = 0
        while vis_count < config.vis_count and i < x.shape[0]:
            # volume_to_mesh(volume[i], str(output_path / f'{vis_count}_gt.obj'))
            # volume_to_mesh(x[i], str(output_path / f'{vis_count}_re.obj'))
            structure = structures[i][1]
            structure[structure == 0] = 1
            structure[structure == 2] = 0
            volume_to_mesh(structure, str(output_path / f'{vis_count}_structure_1.obj'))
            vis_count += 1
            i += 1
            print(f'{vis_count} / {config.vis_count}')
            exit(0)
    print(f'Precision: {precision_all / len(dataloader)}, Recall: {recall_all / len(dataloader)}')


@hydra.main(config_path='conf', config_name='reconstruct_sparse', version_base=None)
def main(config: DictConfig):
    print(config)
    reconstruct(config)


if __name__ == '__main__':
    main()
