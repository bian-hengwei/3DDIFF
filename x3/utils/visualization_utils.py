import numpy as np
import torch

from io import TextIOWrapper


VERTEX_OFFSETS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ]
)

FACES = [
    [1, 2, 3, 4],
    [5, 8, 7, 6],
    [1, 5, 6, 2],
    [2, 6, 7, 3],
    [3, 7, 8, 4],
    [4, 8, 5, 1]
]


def write_cube(obj_file: TextIOWrapper, coord: np.ndarray, vertex_count: int, multiplier: int) -> int:
    """Write a cube to the .obj file at the given coordinates.
    """
    x, y, z = coord

    for offset in VERTEX_OFFSETS:
        vertex = (np.array([x, y, z]) + offset) * multiplier
        obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

    for face in FACES:
        obj_file.write("f" + "".join([f" {vertex_index + vertex_count}" for vertex_index in face]) + "\n")

    return vertex_count + 8


def coords_to_cube_mesh(coords: np.ndarray | torch.Tensor, file_path: str, multiplier: int = 1):
    """Write a .obj file for a 3D mesh of cubes at the given coordinates.
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()

    channels = coords.shape[1]
    assert channels in [3, 4], f'Invalid number of channels: {channels}'
    if channels == 4:
        coords = coords[:, 1:]  # remove batch index

    obj_file = open(file_path, 'w')
    vertex_count = 0
    for coord in coords:
        vertex_count = write_cube(obj_file, coord, vertex_count, multiplier)
    obj_file.close()
