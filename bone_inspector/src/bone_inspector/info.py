from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from numpy import ndarray
from scipy.spatial import KDTree
from typing import Any, Optional, Union, Tuple, List, Literal, Dict

import bpy
import fast_simplification
import gc
import logging
import math
import numpy as np
import os
import trimesh

from .export import Exporter
from .motion import linear_blend_skinning, get_matrix, get_matrix_basis
from .utils import axis_angle_to_matrix, guess_orientation, orientation_str_to_matrix

@dataclass
class VoxelInfo():
    origin: ndarray
    voxel_size: float
    coords: ndarray
    
    _voxel: Optional[ndarray]=None
    
    def _make_voxel(self):
        if self._voxel is None:
            max_coords = np.max(self.coords, axis=0)
            shape = tuple(max_coords + 1)
            voxel = np.zeros(shape, dtype=bool)
            voxel[tuple(self.coords.T)] = True
            self._voxel = voxel
    
    @property
    def voxel(self):
        self._make_voxel()
        return self._voxel
    
    def export_pc(self, path: str):
        pc = self.origin + (self.coords + 0.5) * self.voxel_size
        Exporter.export_pc(vertices=pc, path=path)
    
    def projection_fill(self):
        """
        Fill in holes.
        """
        self._make_voxel()
        grids = np.indices(self._voxel.shape)
        x_coord = grids[0, ...]
        y_coord = grids[1, ...]
        z_coord = grids[2, ...]
        
        INF = 2147483647
        x_tmp = x_coord.copy()
        x_tmp[~self._voxel] = INF
        x_min = x_tmp.min(axis=0)
        
        x_tmp[~self._voxel] = -1
        x_max = x_tmp.max(axis=0)
        
        y_tmp = y_coord.copy()
        y_tmp[~self._voxel] = INF
        y_min = y_tmp.min(axis=1)
        
        y_tmp[~self._voxel] = -1
        y_max = y_tmp.max(axis=1)
        
        z_tmp = z_coord.copy()
        z_tmp[~self._voxel] = INF
        z_min = z_tmp.min(axis=2)
        z_tmp[~self._voxel] = -1
        z_max = z_tmp.max(axis=2)
        
        in_x = (x_coord >= x_min[None, :, :]) & (x_coord <= x_max[None, :, :])
        in_y = (y_coord >= y_min[:, None, :]) & (y_coord <= y_max[:, None, :])
        in_z = (z_coord >= z_min[:, :, None]) & (z_coord <= z_max[:, :, None])
        
        count = in_x.astype(int) + in_y.astype(int) + in_z.astype(int)
        fill_mask = count >= 2
        self._voxel = self._voxel | fill_mask
        x, y, z = np.where(self._voxel)
        self.coords = np.stack([x, y, z], axis=1)
    
    def inside(self, point: Union[ndarray, Tuple]) -> bool:
        self._make_voxel()
        point = np.asarray(point)
        idx = np.floor((point - self.origin) / self.voxel_size).astype(int)
        if np.any(idx < 0) or np.any(idx >= self._voxel.shape):
            return False
        return self._voxel[tuple(idx)]

@dataclass
class MeshInfo():
    name: Optional[str]=None
    # (N, 3)
    vertices: Optional[ndarray]=None
    
    # (N, 3)
    vertex_normals: Optional[ndarray]=None # handled by trimesh
    
    # (F, 3)
    face_normals: Optional[ndarray]=None # handled by trimesh
    
    # (F, 3)
    faces: Optional[ndarray]=None
    
    # (N, J)
    skin: Optional[ndarray]=None
    
    @property
    def N(self):
        return None if self.vertices is None else self.vertices.shape[0]
    
    @property
    def F(self):
        return None if self.faces is None else self.faces.shape[0]
    
    def data(self, float_dtype=np.float32, int_dtype=np.int32):
        data = {}
        if self.name is not None:
            data["name"] = self.name
        if self.vertices is not None:
            data["vertices"] = np.array(self.vertices, dtype=float_dtype)
        if self.vertex_normals is not None:
            data["vertex_normals"] = np.array(self.vertex_normals, dtype=float_dtype)
        if self.face_normals is not None:
            data["face_normals"] = np.array(self.face_normals, dtype=float_dtype)
        if self.faces is not None:
            data["faces"] = np.array(self.faces, dtype=int_dtype)
        if self.skin is not None:
            data["skin"] = np.array(self.skin, dtype=float_dtype)
        return data
    
    @classmethod
    def from_data(cls, data: Dict, float_dtype=np.float32, int_dtype=np.int32) -> 'MeshInfo':
        name = data.get("name", None)
        if name is not None and isinstance(name, ndarray):
            name = name.item()
        name = str(name)
        vertices = data.get("vertices", None)
        if vertices is not None:
            vertices = np.array(vertices, dtype=float_dtype)
        vertex_normals = data.get("vertex_normals", None)
        if vertex_normals is not None:
            vertex_normals = np.array(vertex_normals, dtype=float_dtype)
        face_normals = data.get("face_normals", None)
        if face_normals is not None:
            face_normals = np.array(face_normals, dtype=float_dtype)
        faces = data.get("faces", None)
        if faces is not None:
            faces = np.array(faces, dtype=int_dtype)
        skin = data.get("skin", None)
        if skin is not None:
            skin = np.array(skin, dtype=float_dtype)
        return cls(
            name=name,
            vertices=vertices,
            vertex_normals=vertex_normals,
            face_normals=face_normals,
            faces=faces,
            skin=skin,
        )
        
    def transform(self, trans: ndarray):
        """
        Affine transformation via 4x4 matrix.
        """
        assert trans.shape == (4, 4)
        def _apply(v: ndarray, trans: ndarray) -> ndarray:
            return np.matmul(v, trans[:3, :3].transpose()) + trans[:3, 3]
        self.vertices = _apply(self.vertices, trans)
        mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
        self.vertex_normals = np.array(mesh.vertex_normals, dtype=np.float32)
        self.face_normals = np.array(mesh.face_normals, dtype=np.float32)
    
    def voxelization(self, voxel_size: float=0.1) -> VoxelInfo:
        try:
            import open3d as o3d
        except Exception as e:
            raise RuntimeError(f"cannot import open3d: {str(e)}")
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(self.vertices.copy())
        mesh_o3d.triangles = o3d.utility.Vector3iVector(self.faces)
        voxel = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh_o3d, voxel_size=voxel_size)
        coords = np.array([pt.grid_index for pt in voxel.get_voxels()])
        return VoxelInfo(
            origin=voxel.origin,
            voxel_size=voxel_size,
            coords=coords,
        )
    
    def keep_indices(self, indices: Union[List[int], ndarray]):
        indices = np.asarray(indices)
        old_to_new = -np.ones(self.N, dtype=int)
        old_to_new[indices] = np.arange(len(indices))

        # update vertex arrays
        self.vertices = self.vertices[indices]
        if self.vertex_normals is not None:
            self.vertex_normals = self.vertex_normals[indices]
        if self.skin is not None:
            self.skin = self.skin[indices]

        # update faces and face_normals
        if self.faces is not None:
            mask = np.all(np.isin(self.faces, indices), axis=1)
            new_faces = self.faces[mask]
            new_faces = old_to_new[new_faces]

            self.faces = new_faces
            if self.face_normals is not None:
                self.face_normals = self.face_normals[mask]
    
    def normalize_into(self, continuous_range: Tuple[float, float]):
        """
        Normalize vertices into a cube.
        """
        if self.vertices is None:
            raise ValueError("Cannot normalize mesh with no vertices.")
        
        v_min = self.vertices.min(axis=0)
        v_max = self.vertices.max(axis=0)
        scale_range = v_max - v_min
        
        # normalize to [0, 1]
        vertices_normalized = (self.vertices - v_min) / scale_range
        
        # then scale to the desired range
        target_min, target_max = continuous_range
        scale = target_max - target_min
        self.vertices = vertices_normalized * scale + target_min
    
    def construct(self, max_f: Optional[int]=None):
        """
        Build normals from trimesh.
        """
        vertices = self.vertices.copy()
        mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        if max_f and mesh.faces.shape[0] > max_f:
            processed_vertices = np.array(mesh.vertices, dtype=np.float32)
            processed_faces = np.array(mesh.faces, dtype=np.int32)
            processed_vertices, processed_faces = fast_simplification.simplify(processed_vertices, processed_faces, target_count=max_f)
            mesh = trimesh.Trimesh(vertices=processed_vertices, faces=processed_faces)
        self.vertices = np.array(mesh.vertices, dtype=np.float32)
        self.vertex_normals = np.array(mesh.vertex_normals, dtype=np.float32)
        self.faces = np.array(mesh.faces, dtype=np.int32)
        self.face_normals = np.array(mesh.face_normals, dtype=np.float32)
        if self.skin is not None:
            new_skin = np.array(self.skin, dtype=np.float32)
            # sample nearest
            tree = KDTree(vertices)
            _, indices = tree.query(self.vertices)
            new_skin = new_skin[indices]
        else:
            new_skin = None
        self.skin = new_skin
    
    def permute(self, perm):
        if skin is not None:
            self.skin = self.skin[:, perm]
    
    def normalize_skin(self):
        self.skin = self.skin / np.maximum(self.skin.sum(axis=1, keepdims=True), 1e-6)
    
    def apply_pose(self, matrix: ndarray, matrix_local: ndarray, inplace: bool=True) -> 'MeshInfo':
        vertices = linear_blend_skinning(
            vertices=self.vertices,
            matrix_local=matrix_local,
            matrix=matrix,
            skin=self.skin,
            pad=1,
            value=1.,
        )
        ret = deepcopy(self)
        # change normals
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces, process=False)
        ret.vertices = vertices
        ret.vertex_normals = np.array(mesh.vertex_normals, dtype=np.float32)
        ret.face_normals = np.array(mesh.face_normals, dtype=np.float32)
        if inplace:
            self.vertices = vertices
            self.vertex_normals = np.array(mesh.vertex_normals, dtype=np.float32)
            self.face_normals = np.array(mesh.face_normals, dtype=np.float32)
            return ret
        return ret
    
    def export_mesh(self, path: str):
        Exporter.export_mesh(vertices=self.vertices, faces=self.faces, path=path)
    
    def export_pc(self, path: str, with_norm: bool=True, norm_size: float=0.01):
        Exporter.export_pc(
            vertices=self.vertices,
            path=path,
            vertex_normals=self.vertex_normals if with_norm else None,
            normal_size=norm_size,
        )

@dataclass
class ArmatureInfo():
    """
    Rest pose in true world, which is matrix_world @ matrix_local.
    """
    # (4, 4)
    matrix_world: ndarray
    
    # (J, 4, 4)
    matrix_local: ndarray
    
    # (J)
    parents: List[Union[None, int]]
    
    # (J)
    lengths: Optional[ndarray]=None
    
    # (J)
    bone_names: Optional[List[str]]=None
    
    # (frames, J, 4, 4)
    matrix_basis: Optional[ndarray]=None
    name: Optional[str]=None
    
    @property
    def joints(self) -> ndarray:
        return (self.matrix_world @ self.matrix_local)[:, :3, 3]
    
    @property
    def tails(self):
        if self.lengths is None:
            return None
        x = np.array([0.0, 1.0, 0.0])
        x = self.lengths * x[:, np.newaxis]
        y = np.zeros((self.J, 3))
        for i in range(self.J):
            y[i] = self.matrix_world[:3, :3] @ self.matrix_local[i, :3, :3] @ x[:, i]
        return self.joints + y
    
    @property
    def J(self) -> int:
        return self.joints.shape[0]
    
    @property
    def frames(self):
        if self.matrix_basis is None:
            return None
        return self.matrix_basis.shape[0]
    
    def get_tails_offset(self, matrix: ndarray):
        if self.lengths is None:
            return None
        x = np.array([0.0, 1.0, 0.0])
        x = self.lengths * x[:, np.newaxis]
        y = np.zeros((self.J, 3))
        for i in range(self.J):
            y[i] = matrix[i, :3, :3] @ x[:, i]
        return y
    
    def data(self, float_dtype=np.float32, int_dtype=np.int32) -> Dict:
        data = {
            "matrix_world": np.array(self.matrix_world, dtype=float_dtype),
            "matrix_local": np.array(self.matrix_local, dtype=float_dtype),
            "parents": np.array([-1 if p is None else p for p in self.parents], dtype=int_dtype),
        }
        
        if self.lengths is not None:
            data["lengths"] = np.array(self.lengths, dtype=np.float32)
        if self.bone_names is not None:
            data["bone_names"] = self.bone_names
        if self.matrix_basis is not None:
            data["matrix_basis"] = np.array(self.matrix_basis, dtype=np.float32)
        if self.name is not None:
            data["name"] = self.name
        return data
    
    @classmethod
    def from_data(cls, data: dict, float_dtype=np.float32, int_dtype=np.int32) -> 'ArmatureInfo':
        """
        从 numpy-friendly dict 恢复 ArmatureInfo
        """
        matrix_world = np.array(data["matrix_world"], dtype=float_dtype)
        matrix_local = np.array(data["matrix_local"], dtype=float_dtype)

        parents_arr = np.array(data["parents"], dtype=int_dtype)
        parents = [None if p == -1 else int(p) for p in parents_arr]
        
        lengths = None
        if "lengths" in data:
            lengths = np.array(data["lengths"], dtype=float_dtype)
        bone_names = None
        if "bone_names" in data:
            bone_names = list(data["bone_names"])
        matrix_basis = None
        if "matrix_basis" in data:
            matrix_basis = np.array(data["matrix_basis"], dtype=float_dtype)
        name = data.get('name', None)
        if name is not None and isinstance(name, np.ndarray):
            name = name.item()
        name = str(name)
        return cls(
            matrix_world=matrix_world,
            matrix_local=matrix_local,
            parents=parents,
            lengths=lengths,
            bone_names=bone_names,
            matrix_basis=matrix_basis,
            name=name,
        )
    
    def permute(self, perm, new_parents):
        self.matrix_local = self.matrix_local[perm]
        self.new_parents = new_parents
        if self.matrix_basis is not None:
            self.matrix_basis = self.matrix_basis[:, perm]
    
    def get_frame(self, frame: int):
        if self.matrix_basis is None:
            return None
        return self.matrix_basis[frame]
    
    def transform(self, trans: ndarray):
        """
        Affine transformation via 4x4 matrix.
        """
        assert trans.shape == (4, 4)
        self.matrix_world = trans @ self.matrix_world
    
    def random_pose(self, degree: float=15.0) -> ndarray:
        matrix_basis = axis_angle_to_matrix(
            (np.random.rand(self.J, 3) - 0.5) * degree / 180 * np.pi * 2
        )
        return get_matrix(
            matrix_world=np.eye(4),
            matrix_local=self.matrix_local,
            matrix_basis=matrix_basis,
            parents=self.parents,
        )
    
    def get_pose(self, matrix_basis: Optional[ndarray]=None, frame: Optional[int]=None) -> ndarray:
        assert not(matrix_basis is None and frame is None), "`matrix_basis` or `frame` must be assigned"
        if matrix_basis is not None:
            pass
        else:
            assert self.frames is not None, "do not have any frames"
            frame = min(frame, self.frames-1)
            matrix_basis = self.matrix_basis[frame]
        matrix = get_matrix(
            matrix_world=self.matrix_world,
            matrix_local=self.matrix_local,
            matrix_basis=matrix_basis,
            parents=self.parents
        )
        return matrix
    
    def apply_pose(self, matrix_basis: ndarray, inplace: bool=True) -> ndarray:
        matrix = get_matrix(
            matrix_world=np.eye(4),
            matrix_local=self.matrix_local,
            matrix_basis=matrix_basis,
            parents=self.parents
        )
        if inplace:
            self.matrix_local = matrix
            return matrix.copy()
        return matrix
    
    def retarget(
        self,
        source: 'ArmatureInfo',
        exact: bool=True,
        do_not_align: bool=False,
        ignore_missing_bone: bool=False,
        start: Optional[int]=None,
        end: Optional[int]=None,
        mapping: Optional[Dict[str, str]]=None,
    ) -> 'ArmatureInfo':
        """
        Transfer animation from source.
        """
        if source.frames is None:
            raise ValueError("There is no animation in source.")
        if start is None:
            start = 0
        if end is None:
            end = source.frames
        assert 0 <= start < end <= source.frames
        tgt_bone_names = self.bone_names
        if tgt_bone_names is None:
            logging.warning("`bone_names` is None for target")
            tgt_bone_names = [f"bone_{i}" for i in range(self.J)]
        src_bone_names = source.bone_names
        if src_bone_names is None:
            logging.warning("`bone_names` is None for source")
            src_bone_names = [f"bone_{i}" for i in range(source.J)]
        if mapping is None:
            mapping = {}
        else:
            src_bone_names = [n if n not in mapping else mapping[n] for n in src_bone_names]
        src_name_to_id = {name: i for i, name in enumerate(src_bone_names)}
        tgt_name_to_id = {name: i for i, name in enumerate(tgt_bone_names)}
        if exact:
            if set(src_name_to_id.keys()) != set(tgt_name_to_id.keys()):
                raise ValueError("Mismatch between bone names.")
        elif not ignore_missing_bone:
            for k in tgt_name_to_id.keys():
                if k not in src_name_to_id:
                    raise ValueError(f"Missing bone in source: {k}")
        src_align = deepcopy(source)
        src_align.change_matrix_local(matrix_world=self.matrix_world)
        matrix_basis = np.zeros((end-start, self.J, 4, 4), dtype=np.float32)
        matrix_basis[...] = np.eye(4)
        matrix_local = self.matrix_local.copy()
        for (k, v) in tgt_name_to_id.items():
            if k in src_name_to_id:
                matrix_local[v] = src_align.matrix_local[src_name_to_id[k]]
        if do_not_align:
            matrix_align = np.zeros((self.J, 4, 4), dtype=np.float32)
            matrix_align[...] = np.eye(4)
        else:
            matrix_align = get_matrix_basis(
                matrix=matrix_local,
                matrix_world=np.eye(4),
                matrix_local=self.matrix_local,
                parents=self.parents,
            )
        for (k, v) in tgt_name_to_id.items():
            if k not in src_name_to_id:
                matrix_align[v] = np.eye(4)
        for (k, v) in tgt_name_to_id.items():
            if k not in src_name_to_id:
                continue
            for i in range(start, end):
                matrix_basis[i-start, v] = matrix_align[v] @ src_align.matrix_basis[i, src_name_to_id[k]]
        return ArmatureInfo(
            matrix_world=self.matrix_world.copy(),
            matrix_local=self.matrix_local.copy(),
            matrix_basis=matrix_basis,
            parents=self.parents.copy(),
            lengths=None if self.lengths is None else self.lengths.copy(),
            bone_names=None if self.bone_names is None else self.bone_names.copy(),
        )
    
    def export_skeleton(self, path: str, simple: bool=True, ignore_tail: bool=True, frame: Optional[int]=None):
        _j = self.joints
        if frame is not None:
            assert self.frames is not None, "do not have any frames"
            frame = min(frame, self.frames-1)
            matrix = self.get_pose(frame=frame)
            _j = matrix[:, :3, 3]
        if ignore_tail:
            Exporter.export_skeleton(joints=_j, parents=self.parents, path=path, simple=simple)
        else:
            Exporter.export_skeleton(joints=_j, parents=self.parents, path=path, tails=_j+self.get_tails_offset(matrix=matrix), simple=simple)
    
    def export_animation(
        self,
        path: str,
        start: Optional[int]=None,
        end: Optional[int]=None,
    ):
        if self.frames is None:
            raise ValueError("Do not have animation.")
        start = 0 if start is None else start
        end = self.frames if end is None else end
        assert 0 <= start < end <= self.frames
        Exporter.export_asset(
            path=path,
            bone_names=self.bone_names,
            parents=self.parents,
            tails=self.tails,
            matrix_local=self.matrix_local,
            matrix_basis=self.matrix_basis[start:end],
            matrix_world=self.matrix_world,
        )
    
    def export_fbx(self, path: str):
        # TODO
        Exporter.export_asset(
            path=path,
            joints=self.joints,
            tails=self.tails,
            parents=self.parents,
            bone_names=self.bone_names,
        )
        
    def change_matrix_local(
        self,
        matrix_local: Optional[ndarray]=None,
        matrix_world: Optional[ndarray]=None,
        src_orientation: Optional[str]=None,
        tgt_orientation: Optional[str]=None,
        roll: Optional[Dict[str, float]]=None,
    ):
        """
        Change the matrix_local(another armature's matrix_local) so that two armatures can align.
        
        By default convert into world space(matrix_local'=matrix_world @ matrix_local, matrix=identity). By default target has the same bone orientation with source's.
        
        `orientation` must be a str, like '+x+z-y', '+x+y+z', '+y-z+x'.
        """
        src_axis = np.eye(4)
        tgt_axis = np.eye(4)
        # blender does not have any function to know the actual orientation
        # have to guess or expect it from user
        if src_orientation is None:
            src_axis[:3, :3] = guess_orientation(self.matrix_local[0])
        else:
            src_axis[:3, :3] = orientation_str_to_matrix(src_orientation)
        
        if tgt_orientation is None:
            if matrix_local is None and matrix_world is None:
                # change to world, so set to blender's defualt axis
                tgt_axis[:3, :3] = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0,-1.0],
                    [0.0, 1.0, 0.0],
                ])
            elif matrix_local is None:
                # change matrix_world, so copy from self's axis
                tgt_axis = src_axis.copy()
            else:
                # copy from target's axis
                tgt_axis[:3, :3] = guess_orientation(matrix_local[0])
        else:
            tgt_axis[:3, :3] = orientation_str_to_matrix(tgt_orientation)
        
        _f = False
        if matrix_world is None:
            matrix_world = np.eye(4)
            _f = True
        # why need this matrix? because `matrix` is just a derived term from `matrix_local` and `matrix_basis`, need this to correctly compute `matrix_basis`
        view_in_another_space = matrix_world @ tgt_axis @ np.linalg.inv(self.matrix_world @ src_axis)
        view_in_another_space = view_in_another_space[:3, :3]
        roll_matrix = np.zeros((self.J, 3, 3))
        roll_matrix[...] = np.eye(3)
        name_to_id = {k: i for (i, k) in enumerate(self.bone_names)}
        if roll is not None:
            if isinstance(roll, float):
                roll_matrix[:, :3, :3] = 0.
                roll_matrix[:, 1, 1] = 1.
                roll_matrix[:, 0, 0] = math.cos(roll)
                roll_matrix[:, 2, 0] = -math.sin(roll)
                roll_matrix[:, 0, 2] = math.sin(roll)
                roll_matrix[:, 2, 2] = math.cos(roll)
            else:
                for k, v in roll.items():
                    id = name_to_id[k]
                    roll_matrix[id, :3, :3] = 0.
                    roll_matrix[id, 1, 1] = 1.
                    roll_matrix[id, 0, 0] = math.cos(v)
                    roll_matrix[id, 2, 0] = -math.sin(v)
                    roll_matrix[id, 0, 2] = math.sin(v)
                    roll_matrix[id, 2, 2] = math.cos(v)
        if matrix_local is None and _f:
            # convert into world space
            # note the coordinates are now in world space, but rotation is not
            matrix_local = self.matrix_world @ self.matrix_local
            # only apply rotation transition
            matrix_local[:, :3, :3] = view_in_another_space @ matrix_local[:, :3, :3]
        elif matrix_local is None:
            # just change world_matrix
            matrix_local = self.matrix_local
        
        if self.frames is not None:
            for frame in range(self.frames):
                matrix = get_matrix(
                    matrix_world=self.matrix_world,
                    matrix_local=self.matrix_local,
                    matrix_basis=self.matrix_basis[frame],
                    parents=self.parents,
                    roll=roll_matrix,
                )
                # only apply rotation transition
                matrix[:, :3, :3] = view_in_another_space @ matrix[:, :3, :3]
                self.matrix_basis[frame] = get_matrix_basis(
                    matrix=matrix,
                    matrix_world=matrix_world,
                    matrix_local=matrix_local,
                    parents=self.parents,
                )
        
        self.matrix_local = matrix_local.copy()
        self.matrix_world = matrix_world.copy()