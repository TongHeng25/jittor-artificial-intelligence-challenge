from itertools import permutations, product
from numpy import ndarray
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Union

import numpy as np

def axis_angle_to_matrix(axis_angle: ndarray) -> ndarray:
    res = np.pad(R.from_rotvec(axis_angle).as_matrix(), ((0, 0), (0, 1), (0, 1)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
    assert res.ndim == 3
    res[:, -1, -1] = 1
    return res

def rotation_distance(R1: ndarray, R2: ndarray) -> float:
    trace = np.clip((np.trace(R1.T @ R2) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(trace)
    return angle

def guess_orientation(m: ndarray) -> ndarray:
    if m.shape == (4,4):
        m = m[:3,:3]
    elif m.shape != (3,3):
        raise ValueError("m must be 3x3 or 4x4 matrix")
    
    axes = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    std_orientations = []
    for perm in permutations(axes):
        for signs in product(*[[-1,1]]*3):
            R = np.column_stack([perm[i] * signs[i] for i in range(3)])
            std_orientations.append(R)
    distances = [rotation_distance(m, R) for R in std_orientations]
    idx = np.argmin(distances)
    return std_orientations[idx]

def orientation_str_to_matrix(orientation: str) -> ndarray:
    if len(orientation) != 6:
        raise ValueError("`orientation` must have length 6, e.g. '+x+y+z'")    
    axis_map = {'x': np.array([1,0,0]),
                'y': np.array([0,1,0]),
                'z': np.array([0,0,1])}
    parts = [orientation[i:i+2] for i in range(0, 6, 2)]
    axes = [p[1] for p in parts]
    if sorted(axes) != ['x', 'y', 'z']:
        raise ValueError("`orientation` must include x, y, z exactly once.")
    matrix = np.column_stack([
        (1 if p[0] == '+' else -1) * axis_map[p[1]] 
        for p in parts
    ])
    return matrix