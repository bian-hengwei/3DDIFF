import hydra
import numpy as np
import spconv.pytorch as spconv
import torch
import tqdm

from omegaconf import DictConfig
from pathlib import Path

from x3.dataset.shapenet import ShapeNetDataModule
from x3.utils.visualization_utils import coords_to_cube_mesh
from x3.ae.lightning import AELightningModule


def volume_to_mesh(volume, filename):
    if len(volume.shape) == 4:
        volume = volume.abs().sum(dim=0) > 0
    coords = np.argwhere(volume.cpu().numpy())
    coords_to_cube_mesh(coords, filename)
    

def pred_to_voxels(preds, coords, grid_size):
    output = torch.zeros((preds.shape[0], *grid_size), device=preds.device)
    for i in range(preds.shape[0]):
        output[i, coords[i, :, 0], coords[i, :, 1], coords[i, :, 2]] = preds[i]
    return output


@torch.inference_mode()
def reconstruct(config):
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(next(Path('ckpts').glob(f'{config.exp}.ckpt')))
    
    datamodule = ShapeNetDataModule(**config.dataset, triplane=True)
    datamodule.setup()
    
    model = AELightningModule.load_from_checkpoint(ckpt_path, 
        in_channels=datamodule.train_dataset.input_dim, **config.ae.params)
    model.cuda()
    model.eval()
    
    dataloader = iter(datamodule.train_dataloader())
    
    vis_count = 0
    precision_all, recall_all = 0, 0
    
    for batch in tqdm.tqdm(dataloader):
        volume = batch['volume'].cuda()
        x = batch['x'].cuda()
        query = batch['query'].cuda()
        coords = batch['coords'].cuda()
        
        pred = model(x, query)
        
        batches = torch.arange(x.shape[0])[:, None]
        volume_query = volume[batches, coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]]
            
        precision, recall = model.model.metrics(volume_query, pred)
        precision_all += precision
        recall_all += recall
        
        voxels = None
        if vis_count < config.vis_count:
            voxels = pred_to_voxels(pred, coords, config.ae.grid_size)
        
        i = 0
        while vis_count < config.vis_count and i < x.shape[0]:
            voxel = voxels[i]
            voxel = torch.nn.functional.sigmoid(voxel)
            voxel = (voxel > 0.5).int()
            volume_to_mesh(volume[i], str(output_path / f'{vis_count}_gt.obj'))
            volume_to_mesh(voxel, str(output_path / f'{vis_count}_re.obj'))
            vis_count += 1
            i += 1
            print(f'{vis_count} / {config.vis_count}')
    print(f'Precision: {precision_all / len(dataloader)}, Recall: {recall_all / len(dataloader)}')


@hydra.main(config_path='conf', config_name='reconstruct_ae', version_base=None)
def main(config: DictConfig):
    print(config)
    reconstruct(config)


if __name__ == '__main__':
    main()
