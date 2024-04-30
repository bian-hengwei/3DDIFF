import argparse
import concurrent.futures
import sys
from pathlib import Path

import mesh2sdf
import numpy as np
import tqdm
import trimesh


def mesh_to_sdf(mesh_path, sdf_path, resolution, mesh_scale):
    if sdf_path.exists():
        return

    level = 2 / resolution
    mesh = trimesh.load(mesh_path, force='mesh')

    # unit cube bounded from (0, 0, 0) to (1, 1, 1) 
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 1.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale + 0.5

    sdf = mesh2sdf.compute(mesh.vertices, mesh.faces, resolution, fix=True, level=level)
    np.save(sdf_path, sdf)


def process_mesh(mesh_path, out_root, resolution, mesh_scale):
    sdf_path = out_root / mesh_path.parent.parent.name / 'sdf.npy'
    sdf_path.parent.mkdir(exist_ok=True)
    mesh_to_sdf(mesh_path, sdf_path, resolution, mesh_scale)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glob_pattern', type=str, default='*/models/model_normalized.obj')
    parser.add_argument('--mesh_scale', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--out_root', type=str)  # data/ShapeNetX3/03001627
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--root', type=str, default='data/ShapeNetCore/03001627')  # data/ShapeNetCore/03001627
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=None)
    args = parser.parse_args()
    if args.out_root is None:
        args.out_root = args.root.replace('ShapeNetCore', 'ShapeNetX3')

    root = Path(args.root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    mesh_paths = sorted(list(root.glob(args.glob_pattern)))
    start_index = args.start_index
    end_index = args.end_index if args.end_index is not None else len(mesh_paths)
    end_index = min(len(mesh_paths), end_index)
    mesh_paths = mesh_paths[start_index: end_index]
    print(
        f'Processing total {len(mesh_paths)} meshes from {root.absolute()}. Output directory is {out_root.absolute()}.'
    )

    if args.num_workers == 1:
        for mesh_path in tqdm.tqdm(mesh_paths):
            sdf_path = out_root / mesh_path.parent.parent.name / 'sdf.npy'
            sdf_path.parent.mkdir(exist_ok=True)
            mesh_to_sdf(mesh_path, sdf_path, args.resolution, args.mesh_scale)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_mesh, mesh_path, out_root, args.resolution, args.mesh_scale) for
                mesh_path in mesh_paths]
            for _ in tqdm.tqdm(
                    concurrent.futures.as_completed(futures), total=len(mesh_paths), mininterval=10, ascii=True,
                    file=sys.stdout, dynamic_ncols=False
            ):
                pass
