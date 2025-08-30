from copy import deepcopy
from dataclasses import dataclass
from numpy import ndarray
from typing import Optional, List, Union, Dict

import numpy as np

from .error import ExtractError
from .export import Exporter
from .info import ArmatureInfo, MeshInfo, VoxelInfo
from .warning import ExtractWarning

@dataclass
class Asset():
    warnings: Optional[List[ExtractWarning]]=None
    error: Optional[ExtractError]=None
    meshes: Optional[List[MeshInfo]]=None
    armature: Optional[ArmatureInfo]=None
    
    def data(self, float_dtype=np.float32, int_dtype=np.int32) -> Dict:
        d = {
            'meshes': None,
            'armature': None,
        }
        if self.meshes is not None:
            d['meshes'] = []
            for mesh in self.meshes:
                d['meshes'].append(mesh.data())
        if self.armature is not None:
            d['armature'] = self.armature.data()
        return d
    
    @classmethod
    def from_data(cls, data: Dict, float_dtype=np.float32, int_dtype=np.int32) -> 'Asset':
        meshes = None
        if data.get('meshes') is not None:
            meshes = [
                MeshInfo.from_data(m, float_dtype, int_dtype)
                for m in data['meshes']
            ]
        armature = None
        if data.get('armature') is not None:
            armature = ArmatureInfo.from_data(data['armature'])
        return Asset(
            meshes=meshes,
            armature=armature,
        )
    
    def add_warning(self, warning: Union[ExtractWarning, List[ExtractWarning], 'Asset']):
        if self.warnings is None:
            self.warnings = []
        if isinstance(warning, ExtractWarning):
            self.warnings.append(warning)
        elif isinstance(warning, list):
            self.warnings.extend(warning)
        elif isinstance(warning, Asset):
            if warning.warnings:
                self.warnings.extend(warning.warnings)
    
    def voxelization(self, voxel_size: float=0.1) -> VoxelInfo:
        mesh = self.merge_meshes(inplace=False)
        return mesh.voxelization(voxel_size=voxel_size)
    
    def transform(self, trans: ndarray):
        if self.meshes is not None:
            for mesh in self.meshes:
                mesh.transform(trans=trans)
        if self.armature is not None:
            self.armature.transform(trans=trans)
    
    def keep(self, names: Union[List[str], Dict]):
        if self.armature is None:
            return
        assert self.armature.bone_names[0] in names, f"root {self.armature.bone_names[0]} must be kept"
        arranged_names = [k for k in self.armature.bone_names if k in names]
        name_to_id = {k: i for (i, k) in enumerate(self.armature.bone_names)}
        ids = [i for (i, k) in enumerate(self.armature.bone_names) if k in arranged_names]
        parents = []
        n_id = {}
        def find_parent(name):
            while name not in n_id:
                pid = self.armature.parents[name_to_id[name]]
                if pid is None:
                    break
                name = self.armature.bone_names[pid]
            if name not in n_id:
                return None
            return n_id[name]
        for (i, n) in enumerate(arranged_names):
            if i == 0:
                parents.append(None)
            else:
                parents.append(find_parent(n))
            n_id[n] = i
        self.armature.matrix_local = self.armature.matrix_local[ids]
        if self.armature.matrix_basis is not None:
            self.armature.matrix_basis = self.armature.matrix_basis[:, ids]
        self.armature.parents = parents
        self.armature.bone_names = [self.armature.bone_names[x] for x in ids]
        for i in range(self.armature.J):
            if i == 0:
                continue
        self.armature.lengths = self.armature.lengths[ids]
        if self.meshes is not None:
            for mesh in self.meshes:
                if mesh.skin is not None:
                    mesh.skin = mesh.skin[:, ids]
    
    def set_order_by_names(self, new_names: List[str]):
        assert len(new_names) == len(self.armature.bone_names)
        name_to_id = {name: id for (id, name) in enumerate(self.armature.bone_names)}
        new_name_to_id = {name: id for (id, name) in enumerate(new_names)}
        perm = []
        new_parents = []
        for (new_id, name) in enumerate(new_names):
            perm.append(name_to_id[name])
            pid = self.armature.parents[name_to_id[name]]
            if new_id == 0:
                assert pid is None, 'first bone is not root bone'
            else:
                pname = self.armature.bone_names[pid]
                pid = new_name_to_id[pname]
                assert pid < new_id, 'new order does not form a tree'
            new_parents.append(pid)
        self.armature.permute(perm, new_parents)
        for i in range(len(self.meshes)):
            self.mesh[i].permute(perm)
        self.armature.bone_names = new_names
    
    def mesh_with_pose(
        self,
        matrix_basis: Optional[ndarray]=None,
        degree: float=15.0
    ) -> MeshInfo:
        if self.meshes is None:
            raise ValueError("Meshes are missing.")
        if self.armature is None:
            raise ValueError("Armature is missing.")
        if matrix_basis is None:
            matrix = self.armature.random_pose(degree=degree)
        else:
            matrix = self.armature.apply_pose(matrix_basis=matrix_basis, inplace=False)
        if len(self.meshes) == 1:
            mesh = deepcopy(self.meshes[0])
        else:
            mesh = self.merge_meshes(inplace=False)
        return mesh.apply_pose(matrix=matrix, matrix_local=self.armature.matrix_local, inplace=False)
    
    def merge_meshes(self, inplace: bool=True) -> MeshInfo:
        """
        Merge multiple meshes into one.
        """
        if not self.meshes:
            raise ValueError("Meshes are missing.")
        if len(self.meshes) == 1:
            return deepcopy(self.meshes[0])
        
        vertices_list = []
        vertex_normals_list = []
        faces_list = []
        face_normals_list = []
        skin_list = []
        
        vertex_offset = 0
        for mesh in self.meshes:
            vertices_list.append(mesh.vertices)
            vertex_normals_list.append(mesh.vertex_normals if mesh.vertex_normals is not None else np.zeros_like(mesh.vertices))
            faces_list.append(mesh.faces + vertex_offset)
            face_normals_list.append(mesh.face_normals if mesh.face_normals is not None else np.zeros((mesh.faces.shape[0], 3)))
            if skin_list is not None and mesh.skin is None:
                skin_list = None
            else:
                skin_list.append(mesh.skin)
            vertex_offset += mesh.N
        
        if skin_list is not None:
            skin_list = np.vstack(skin_list)
        merged_mesh = MeshInfo(
            name="merged",
            vertices=np.vstack(vertices_list),
            vertex_normals=np.vstack(vertex_normals_list),
            faces=np.vstack(faces_list),
            face_normals=np.vstack(face_normals_list),
            skin=skin_list,
        )
        if inplace:
            self.meshes = [merged_mesh]
            return deepcopy(merged_mesh)
        return merged_mesh
    
    def normalize_skin(self):
        if self.meshes is not None:
            for mesh in self.meshes:
                mesh.normalize_skin()
    
    def _get_mesh_list(self):
        _vertices = []
        _skin = []
        _faces = []
        _mesh_names = []
        for mesh in self.meshes:
            _vertices.append(mesh.vertices)
            _skin.append(mesh.skin)
            _faces.append(mesh.faces)
            _mesh_names.append(mesh.name+"GG")
        if all(x is None for x in _vertices):
            _vertices = None
        if all(x is None for x in _skin):
            _skin = None
        if all(x is None for x in _faces):
            _faces = None
        if all(x is None for x in _mesh_names):
            _mesh_names = None
        return {
            'vertices': _vertices,
            'skin': _skin,
            'faces': _faces,
            'mesh_names': _mesh_names,
        }
    
    def export_armature(
        self,
        path: str,
        armature_only: bool=False,
        **kwargs,
    ):
        if not armature_only and self.meshes is None:
            raise ValueError("'armature_only' is False, but meshes are missing.")
        d = self._get_mesh_list()
        Exporter.export_asset(
            path=path,
            vertices=d['vertices'],
            skin=d['skin'],
            faces=d['faces'],
            mesh_names=d['mesh_names'],
            joints=self.armature.joints,
            tails=self.armature.tails,
            bone_names=self.armature.bone_names,
            parents=self.armature.parents,
            matrix_world=self.armature.matrix_world,
            **kwargs,
        )
    
    def export_animation(
        self,
        path: str,
        armature_only: bool=False,
        start: Optional[int]=None,
        end: Optional[int]=None,
        **kwargs,
    ):
        if self.armature is None or self.armature.matrix_basis is None:
            raise ValueError("Cannot export animation because animation is missing.")
        
        if armature_only:
            self.armature.export_animation(path=path, start=start, end=end)
            return
        if self.meshes is None:
            raise ValueError("`armature_only` is False, but meshes are missing.")
        
        start = 0 if start is None else start
        end = self.armature.frames if end is None else end
        assert 0 <= start < end <= self.armature.frames
        d = self._get_mesh_list()
        Exporter.export_asset(
            path=path,
            vertices=d['vertices'],
            skin=d['skin'],
            faces=d['faces'],
            mesh_names=d['mesh_names'],
            bone_names=self.armature.bone_names,
            parents=self.armature.parents,
            matrix_local=self.armature.matrix_local,
            matrix_basis=self.armature.matrix_basis[start:end],
            matrix_world=self.armature.matrix_world,
            **kwargs,
        )
