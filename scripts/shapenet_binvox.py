import argparse
import subprocess
import tqdm

from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glob_pattern', type=str, default='*/models/model_normalized.obj')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--root', type=str, default='data/ShapeNetCore/03001627')  # data/ShapeNetCore/03001627
    args = parser.parse_args()

    root = Path(args.root)
    mesh_paths = sorted(list(root.glob(args.glob_pattern)))
    print(f'Processing total {len(mesh_paths)} meshes from {root.absolute()}.')

    for mesh_path in tqdm.tqdm(mesh_paths):
        command = f'cuda_voxelizer -f {mesh_path} -s {args.resolution} -o binvox'
        subprocess.run(command, shell=True)
