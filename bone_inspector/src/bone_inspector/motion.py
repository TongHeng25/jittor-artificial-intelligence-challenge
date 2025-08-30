from collections import defaultdict
from dataclasses import dataclass
from numpy import ndarray
from typing import Any, Optional, Union, Tuple, List

import bpy
import numpy as np

def linear_blend_skinning(
    vertices: ndarray,
    matrix_local: ndarray,
    matrix: ndarray,
    skin: ndarray,
    pad: int=0,
    value: float=0.,
) -> ndarray:
    """
    Args:
        vertex: (N, 4-pad)
        matrix_local: (J, 4, 4)
        matrix: targert pose, (J, 4, 4)
        skin: (N, J)
        value: 0 for interpolation(normals) and 1 for motion
    Returns:
        (N, 3)
    """
    assert vertices.shape[-1] + pad == 4
    assert isinstance(vertices, ndarray)
    assert isinstance(matrix_local, ndarray)
    assert isinstance(matrix, ndarray)
    assert isinstance(skin, ndarray)
    J = matrix_local.shape[0]
    N = vertices.shape[0]
    # (4, N)
    padded = np.pad(vertices, ((0, 0), (0, pad)), 'constant', constant_values=(0, value)).T
    # (J, 4, 4)
    trans = matrix @ np.linalg.inv(matrix_local)
    weighted_per_bone_matrix = []
    # (J, N)
    mask = (skin > 0).T
    for i in range(J):
        offset = np.zeros((4, N), dtype=np.float32)
        offset[:, mask[i]] = (trans[i] @ padded[:, mask[i]]) * skin.T[i, mask[i]]
        weighted_per_bone_matrix.append(offset)
    weighted_per_bone_matrix = np.stack(weighted_per_bone_matrix)
    
    g = np.sum(weighted_per_bone_matrix, axis=0)
    sum_skin = np.sum(skin, axis=1)
    is_zero = np.abs(sum_skin) < 1e-8
    final = np.zeros_like(g[:3, :])
    non_zero = ~is_zero
    if np.any(non_zero):
        final[:, non_zero] = g[:3, non_zero] / sum_skin[np.newaxis, non_zero]
    return final.T

def get_matrix(
    matrix_world: ndarray,
    matrix_local: ndarray,
    matrix_basis: ndarray,
    parents: List[Union[None, int]],
    roll: Optional[ndarray]=None,
) -> ndarray:
    """
    Get matrix using forward kinetics.
    """
    J = matrix_local.shape[0]
    assert matrix_local.shape == matrix_basis.shape
    assert matrix_local.shape == (J, 4, 4)
    assert len(parents) == J
    matrix = np.zeros((J, 4, 4))
    for i in range(J):
        if i==0:
            matrix[i] = matrix_local[i] @ matrix_basis[i]
        else:
            pid = parents[i]
            matrix_parent = matrix[pid]
            matrix_local_parent = matrix_local[pid]
            
            matrix[i] = (
                matrix_parent @
                (np.linalg.inv(matrix_local_parent) @ matrix_local[i]) @
                matrix_basis[i]
            )
    if roll is not None:
        matrix[:, :3, :3] = matrix[:, :3, :3] @ roll
    return matrix_world @ matrix

def get_matrix_basis(
    matrix: ndarray,
    matrix_world: ndarray,
    matrix_local: ndarray,
    parents: List[Union[None, int]],
) -> ndarray:
    """
    Solve matrix_basis given matrix, matrix_world and matrix_local.
    """
    J = matrix.shape[0]
    assert matrix_local.shape == matrix.shape
    assert matrix_local.shape == (J, 4, 4)
    assert len(parents) == J
    matrix_mid = np.linalg.inv(matrix_world) @ matrix
    matrix_basis = np.zeros((J, 4, 4))
    for i in range(J):
        if i==0:
            matrix_basis[i] = np.linalg.inv(matrix_local[i]) @ matrix_mid[i]
        else:
            pid = parents[i]
            matrix_parent = matrix_mid[pid]
            matrix_local_parent = matrix_local[pid]
            
            matrix_basis[i] = np.linalg.inv(
                matrix_parent @
                (np.linalg.inv(matrix_local_parent) @ matrix_local[i])
            ) @ matrix_mid[i]
    return matrix_basis