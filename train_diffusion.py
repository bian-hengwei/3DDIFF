import argparse

from torch.utils.data import DataLoader

from diffusion import logger
from diffusion.dataset import TriplaneDataset
from diffusion.resample import create_named_schedule_sampler
from diffusion.script_util import create_model_and_diffusion_from_args
from diffusion.train_util import TrainLoop


def cycle(dl):
    while True:
        for data in dl:
            yield data


def train_diffusion(args) :
    log_dir = args.save_path
    logger.configure(dir=log_dir)
    
    ds = TriplaneDataset(args, 'train')
    collate_fn = None
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    dl = cycle(dl)
    val_dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)
    val_dl = cycle(val_dl)  # cheating

    model, diffusion = create_model_and_diffusion_from_args(args)
    model.to('cuda')
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    TrainLoop(
        diffusion_net = args.diff_net_type, 
        triplane_loss_type = args.triplane_loss_type,
        timestep_respacing = args.timestep_respacing,
        training_step = args.steps, 
        model=model,
        diffusion=diffusion,
        data=dl,
        val_data=val_dl,
        ssc_refine = args.ssc_refine,
        batch_size=args.batch_size,
        microbatch=-1,
        lr=args.diff_lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.diff_n_iters,
    ).run_loop()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--save_path", type=str, default='')
    parser.add_argument('--ssc_refine', action='store_true')
    parser.add_argument("--ssc_refine_dataset", default='monoscene', choices=['monoscene', 'occdepth', 'scpnet', 'ssasc'])
    
    parser.add_argument("--dataset", default='kitti', choices=['kitti', 'carla'])
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for diffusion training")
    parser.add_argument("--resume_checkpoint", type=str, default = None)
    parser.add_argument("--triplane_loss_type", type=str, default='l2', choices=['l1', 'l2'])
    
    parser.add_argument("--triplane", default=True)
    parser.add_argument("--pos", default=True, type=bool)
    parser.add_argument("--voxel_fea", default=False, type=bool)
    
    group = parser.add_argument_group("encoding")
    group.add_argument("--feat_channel_up", type=int, default=64, help="conv feature dimension")
    group.add_argument("--mlp_hidden_channels", type=int, default=256, help="mlp hidden dimension")
    group.add_argument("--mlp_hidden_layers", type=int, default=4, help="mlp hidden layers")
    group.add_argument("--invalid_class", type=bool, default=False)
    group.add_argument("--padding_mode", default='replicate')
    group.add_argument("--lovasz", default=True)
    group.add_argument("--geo_feat_channels", type=int, default=16, help="geometry feature dimension")
    group.add_argument("--z_down", default=False)
    
    
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
    
    args = parser.parse_args()
        
    args.data_path = 'data/triplane'
    
    if args.voxel_fea:
        args.diff_net_type = "unet_voxel"
    else :
        args.diff_net_type = "unet_tri" if args.triplane else "unet_bev"

    train_diffusion(args)
