"""OLD VERSION"""

import argparse
import torch
import tqdm

from pathlib import Path

from x3.dataset.shapenet import ShapeNetDataModule
from x3.utils.visualization_utils import coords_to_cube_mesh
from x3.vae.vae import VAELightningModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='ckpts/last.ckpt')
    parser.add_argument('--root', type=str, default='data/ShapeNetCore/03001627')  # data/ShapeNetCore/03001627
    parser.add_argument('--out_root', type=str)  # data/ShapeNetX3/03001627
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--glob_pattern', type=str, default='*/models/model_normalized.surface.binvox')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=None)
    args = parser.parse_args()

    if args.out_root is None:
        args.out_root = args.root.replace('ShapeNetCore', 'ShapeNetX3')
    root = Path(args.root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    names = sorted(list(root.glob(args.glob_pattern)))
    names = [name.parent.parent.name for name in names]

    datamodule = ShapeNetDataModule(
        root.parent, out_root.parent, root.name, 128 // args.resolution, load_cache=False, debug_length=1, batch_size=1
    )
    datamodule.setup()
    model = VAELightningModule.load_from_checkpoint(
        args.ckpt,
        spatial_shape=[128, 128, 128],
        input_dim=64,
        base_channels=64,
        channels_multiple=[1, 2, 4, 8],
        latent_dim=16,
        batch_size=16,
        dense=False
    )
    model.to('cuda')
    model.eval()

    N = len(datamodule.train_dataset)
    start_index = args.start_index
    end_index = args.end_index if args.end_index is not None else N
    end_index = min(N, end_index)
    print(f'Processing total {N} objects from {root.absolute()}. Output directory is {out_root.absolute()}.')

    dataloader = iter(datamodule.train_dataloader())

    for index in tqdm.tqdm(range(start_index, end_index)):
        data = next(dataloader)
        item = data[0].cuda()
        structures = [d.cuda() for d in data[1]]
        coords_to_cube_mesh(item.C, out_root / names[index] / 'gt.obj')
        with torch.no_grad():
            # output = model(item)[0]
            x, mu, logvar = model.model.encoder(item)
            x.F = model.model.latent_sample(mu, logvar)
            x, structures_pred = model.model.decoder(x, structures, 0.1)
            output = x
        coords_to_cube_mesh(output.C, out_root / names[index] / 're.obj')

        # Path('temp').mkdir(exist_ok=True)
        # model.train()
        # data = next(dataloader)
        # data[0].cuda()
        # [d.cuda() for d in data[1]]
        # model.training_step(data, 0)
        break
