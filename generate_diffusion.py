import argparse
import numpy as np
import torch
import tqdm

from functools import lru_cache
from pathlib import Path

from diffusion.script_util import create_model_and_diffusion_from_args
from diffusion.triplane_util import decompose_featmaps
from x3.ae.lightning import AELightningModule
from x3.utils.visualization_utils import coords_to_cube_mesh


@lru_cache(4)
def voxel_coord(voxel_shape):
    x = np.arange(voxel_shape[0])
    y = np.arange(voxel_shape[1])
    z = np.arange(voxel_shape[2])
    Y, X, Z = np.meshgrid(x, y, z)
    voxel_coord = np.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1)
    return voxel_coord


def make_query(grid_size):
    gs = grid_size[1:]
    coords = torch.from_numpy(voxel_coord(gs))
    coords = coords.reshape(-1, 3)
    query = torch.zeros(coords.shape, dtype=torch.float32)
    query[:, 0] = 2 * coords[:, 0] / float(gs[0] - 1) - 1
    query[:, 1] = 2 * coords[:, 1] / float(gs[1] - 1) - 1
    query[:, 2] = 2 * coords[:, 2] / float(gs[2] - 1) - 1
    
    query = query.reshape(-1, 3)
    return coords.unsqueeze(0), query.unsqueeze(0)


def build_sampling_model(args):
    grid_size = (1, 128, 128, 128)
    H, W, D = 128, 128, 128

    model, diffusion = create_model_and_diffusion_from_args(args)
    model.load_state_dict(torch.load(DIFF_PATH, map_location="cpu"))
    model = model.cuda().eval()

    ae = AELightningModule.load_from_checkpoint(
        CKPT_PATH, 
        in_channels=39, 
        encoder_channels=16,
        block_hidden_channels=64,
        mlp_hidden_channels=256,
        mlp_hidden_layers=4,
        lr=1e-4,
        lr_scheduler_steps=[30, 40],
        lr_scheduler_decay=0.5,
        grid_size=[128, 128, 128]
    )
    ae.cuda()
    ae.eval()

    sample_fn = (diffusion.p_sample_loop if not args.repaint else diffusion.p_sample_loop_scene_repaint)
    C = args.geo_feat_channels
    coords, query = make_query(grid_size)
    coords, query = coords.cuda(), query.cuda()    
    out_shape = [args.batch_size, C, H + D, W + D]

    return model, ae, sample_fn, coords, query, out_shape, H, W, D, grid_size, args
    
    
def volume_to_mesh(volume, filename):
    if len(volume.shape) == 4:
        volume = volume.abs().sum(dim=0) > 0
    coords = np.argwhere(volume.cpu().numpy())
    coords_to_cube_mesh(coords, filename)
    

def pred_to_voxels(preds, coords, grid_size):
    output = torch.zeros(grid_size, device=preds.device)
    output[coords[..., 0], coords[..., 1], coords[..., 2]] = preds
    return output


def sample(args):
    model, ae, sample_fn, coords, query, out_shape, H, W, D, grid_size, args = build_sampling_model(args)
    args.grid_size = grid_size
    with torch.no_grad():
        condition = np.zeros(out_shape)
        cond = {'y': condition, 'H': H, 'W': W, 'D': D, 'path': args.save_path}

        for r in tqdm.trange(args.num_samples):
            samples = sample_fn(model, out_shape, progress=False, model_kwargs=cond)         
            xy_feat, xz_feat, yz_feat = decompose_featmaps(samples, (H, W, D))
            pred = ae.model.decoder([xy_feat, xz_feat, yz_feat], query)
            
            voxel = pred_to_voxels(pred, coords, grid_size[1:])
            voxel = torch.nn.functional.sigmoid(voxel)
            voxel = (voxel > 0.5).int()
            volume_to_mesh(voxel, str(Path(args.save_path) / f'{r}.obj'))
            
            
def sample_parser():
    parser = argparse.ArgumentParser()
    
    group = parser.add_argument_group("sampling")
    group.add_argument("--triplane", default=True)
    group.add_argument("--pos", default=True, type=bool)
    group.add_argument("--voxel_fea", default=False)
    group.add_argument('--ssc_refine', default=False, type=bool)
    group.add_argument("--refine_dataset", default='monoscene', choices=['monoscene', 'occdepth', 'scpnet', 'ssasc', 'lmsc', 'motionsc', 'sscfull'])
    group.add_argument("--triplane_loss_type", type=str, default='l2', choices=['l1',  'l2',])
    group.add_argument("--batch_size", type=int, default=1)
    group.add_argument("--diff_net_type", type=str, default='unet_tri')
    group.add_argument("--repaint", default=False, type=bool)
    
    group = parser.add_argument_group("diffusion")
    group.add_argument("--steps", type=int, default=100, help="diffusion step")
    group.add_argument("--is_rollout", type=bool, default=True)
    group.add_argument('--mult_channels', default=(1, 2, 4))
    group.add_argument("--diff_lr", type=float, default=5e-4, help="initial learning rate for diffusion training")
    group.add_argument("--schedule_sampler", type=str, default="uniform", help="schedule sampler")
    group.add_argument("--ema_rate", type=float, default=0.9999, help="ema rate")
    group.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    group.add_argument("--log_interval", type=int, default=500, help="log interval")
    group.add_argument("--save_interval", type=int, default=1000, help="save interval")
    group.add_argument("--use_fp16", type=bool, default=False)
    group.add_argument("--predict_xstart", type=bool, default=True)
    group.add_argument("--learn_sigma", type=bool, default=False)
    group.add_argument("--timestep_respacing", default='')
    group.add_argument("--conv_down", default=True)
    group.add_argument("--diff_n_iters", type=int, default=50000, help="lr ann eal steps for diffusion training")
    group.add_argument("--tri_z_down", default=False)
    group.add_argument('--tri_unet_updown', type=bool, default=True)
    group.add_argument("--model_channels", default=64, help="model channels")
    
    group = parser.add_argument_group("encoding")
    group.add_argument("--feat_channel_up", type=int, default=64, help="conv feature dimension")
    group.add_argument("--mlp_hidden_channels", type=int, default=256, help="mlp hidden dimension")
    group.add_argument("--mlp_hidden_layers", type=int, default=4, help="mlp hidden layers")
    group.add_argument("--invalid_class", type=bool, default=False)
    group.add_argument("--padding_mode", default='replicate')
    group.add_argument("--lovasz", default=True)
    group.add_argument("--geo_feat_channels", type=int, default=16, help="geometry feature dimension")
    group.add_argument("--z_down", default=False)
    
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--save_path", type=str, default='generation/triplane')

    parser.add_argument("--dataset",  default='kitti', choices=['kitti', 'carla'])
    parser.add_argument("--num_samples", type=int, default=10)
    
    parser.add_argument("--ckpt_path", type=str, default='ckpts/ae-v1.ckpt')
    parser.add_argument("--diff_path", type=str, default="out/triplane/ema_0.9999_030000.pt")
    parser.add_argument("--diff_step")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = sample_parser()
    if args.diff_step:
        args.save_path = args.save_path + args.diff_step
        args.diff_path = args.diff_path.replace(args.diff_path[-9: -3], args.diff_step)
    
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    CKPT_PATH = args.ckpt_path
    DIFF_PATH = args.diff_path
    sample(args)
