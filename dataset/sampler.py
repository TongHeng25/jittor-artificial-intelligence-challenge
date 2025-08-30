import numpy as np
from numpy import ndarray
from typing import Tuple
from abc import ABC, abstractmethod

class Sampler(ABC):
    def __init__(self):
        pass
    
    def _sample_barycentric(
        self,
        vertex_groups: ndarray,
        faces: ndarray,
        face_index: ndarray,
        random_lengths: ndarray,
    ):
        v_origins = vertex_groups[faces[face_index, 0]]
        v_vectors = vertex_groups[faces[face_index, 1:]]
        v_vectors -= v_origins[:, np.newaxis, :]
        
        sample_vector = (v_vectors * random_lengths).sum(axis=1)
        v_samples = sample_vector + v_origins
        return v_samples
    
    @abstractmethod
    def sample(
        self,
        vertices: ndarray,
        vertex_normals: ndarray,
        face_normals: ndarray,
        vertex_groups: dict[str, ndarray],
        faces: ndarray,
    ) -> Tuple[ndarray, ndarray, dict[str, ndarray]]:
        '''
        Args:
            vertices: (N, 3)
            vertex_normals: (N, 3)
            face_normals: (F, 3)
            vertex_groups: dict{name: shape (N, x)}
            face: (F, 3)
        Returns:
            vertices, vertex_normals, vertex_groups
        '''
        return vertices, vertex_normals, vertex_groups

class SamplerMix(Sampler):
    '''
    Pick `vertex_samples` samples with SamplerOrigin, then pick `num_vertices`-`vertex_samples` samples
    with SamplerRandom.
    '''
    def __init__(self, num_samples: int, vertex_samples: int):
        super().__init__()
        self.num_samples = num_samples
        self.vertex_samples = vertex_samples
    
    def sample(
        self,
        vertices: ndarray,
        vertex_normals: ndarray,
        face_normals: ndarray,
        vertex_groups: dict[str, ndarray],
        faces: ndarray,
    ) -> Tuple[ndarray, ndarray, dict[str, ndarray]]:
        '''
        Args:
            vertices: (N, 3)
            vertex_normals: (N, 3)
            face_normals: (F, 3)
            vertex_groups: dict{name: shape (N, x)}
            face: (F, 3)
        Returns:
            vertices, vertex_normals, vertex_groups
        '''
        if self.num_samples==-1:
            return vertices, vertex_normals, vertex_groups
        
        # 1. sample vertices
        num_samples = self.num_samples
        perm = np.random.permutation(vertices.shape[0])
        vertex_samples = min(self.vertex_samples, vertices.shape[0])
        num_samples -= vertex_samples
        perm = perm[:vertex_samples]
        n_vertices = vertices[perm]
        n_normal = vertex_normals[perm]
        n_v = {name: v[perm] for name, v in vertex_groups.items()}
        
        # 2. sample surface
        perm = np.random.permutation(num_samples)
        vertex_samples, face_index, random_lengths = sample_surface(
            num_samples=num_samples,
            vertices=vertices,
            faces=faces,
            return_weight=True,
        )
        vertex_samples = np.concatenate([n_vertices, vertex_samples], axis=0)
        normal_samples = np.concatenate([n_normal, face_normals[face_index]], axis=0)
        vertex_groups_samples = {}
        for n, v in vertex_groups.items():
            g = self._sample_barycentric(
                vertex_groups=v,
                faces=faces,
                face_index=face_index,
                random_lengths=random_lengths,
            )
            vertex_groups_samples[n] = np.concatenate([n_v[n], g], axis=0)
        return vertex_samples, normal_samples, vertex_groups_samples

def sample_surface(
    num_samples: int,
    vertices: ndarray,
    faces: ndarray,
    return_weight: bool=False,
):
    '''
    Randomly pick samples according to face area.
    
    See sample_surface: https://github.com/mikedh/trimesh/blob/main/trimesh/sample.py
    '''
    # get face area
    offset_0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    offset_1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_weight = np.cross(offset_0, offset_1, axis=-1)
    face_weight = (face_weight * face_weight).sum(axis=1)
    
    weight_cum = np.cumsum(face_weight, axis=0)
    face_pick = np.random.rand(num_samples) * weight_cum[-1]
    face_index = np.searchsorted(weight_cum, face_pick)
    
    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vertices[faces[:, 0]]
    tri_vectors = vertices[faces[:, 1:]]
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]
    
    # randomly generate two 0-1 scalar components to multiply edge vectors b
    random_lengths = np.random.rand(len(tri_vectors), 2, 1)
    
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)
    
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    vertex_samples = sample_vector + tri_origins
    if not return_weight:
        return vertex_samples
    return vertex_samples, face_index, random_lengths