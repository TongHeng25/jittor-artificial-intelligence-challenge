from collections import defaultdict
from numpy import ndarray
from typing import List, Union, Tuple, Optional

import logging
import numpy as np
import os
import sys

try:
    import open3d as o3d
    OPEN3D_EQUIPPED = True
except:
    logging.warning("do not have open3d")
    OPEN3D_EQUIPPED = False

class Exporter():
    
    @classmethod
    def _safe_make_dir(cls, path):
        if os.path.dirname(path) == '':
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    @classmethod
    def _check_format(cls, ext, oks):
        if ext not in oks:
            raise ValueError(f"Unsupported format: {ext}, expect: {','.join(oks)}")
    
    @classmethod
    def export_skeleton(
        cls,
        joints: ndarray,
        parents: List[Union[int, None]],
        path: str,
        tails: Optional[ndarray]=None,
        simple: bool=True,
    ):
        name, ext = os.path.splitext(path)
        ext = ext.lower()[1:]
        cls._check_format(ext, ['obj'])
        path = name + f".{ext}"
        cls._safe_make_dir(path)
        J = joints.shape[0]
        lines = []
        file = open(path, 'w')
        if simple:
            lines.append("o skeleton\n")
            _joints = []
            for id in range(J):
                pid = parents[id]
                if tails is None and (pid is None or pid == -1):
                    continue
                bx, by, bz = joints[id]
                if tails is None:
                    ex, ey, ez = joints[pid]
                else:
                    ex, ey, ez = tails[id]
                _joints.extend([
                    f"v {bx:.8f} {bz:.8f} {-by:.8f}\n",
                    f"v {ex:.8f} {ez:.8f} {-ey:.8f}\n",
                    f"v {ex:.8f} {ez:.8f} {-ey + 0.00001:.8f}\n"
                ])
            lines.extend(_joints) 
            # faces
            lines.extend([f"f {id*3+1} {id*3+2} {id*3+3}\n" for id in range(J)])
        else:
            raise NotImplementedError()
        file.writelines(lines)
    
    @classmethod
    def export_mesh(cls, vertices: ndarray, faces: ndarray, path: str):
        name, ext = os.path.splitext(path)
        ext = ext.lower()[1:]
        cls._check_format(ext, ['obj', 'ply'])
        path = name + f".{ext}"
        cls._safe_make_dir(path)
        if ext == 'ply':
            if not OPEN3D_EQUIPPED:
                raise RuntimeError("open3d is not available")
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            o3d.io.write_triangle_mesh(path, mesh)
            return
        file = open(path, 'w')
        lines = ["o mesh\n"]
        _vertices = []
        for co in vertices:
            _vertices.append(f"v {co[0]} {co[2]} {-co[1]}\n")
        lines.extend(_vertices)
        _faces = []
        for face in faces:
            _faces.append(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        lines.extend(_faces)
        file.writelines(lines)
    
    @classmethod
    def export_pc(
        cls,
        vertices: ndarray,
        path: str,
        vertex_normals: Union[ndarray, None]=None,
        normal_size: float=0.01,
    ):
        name, ext = os.path.splitext(path)
        ext = ext.lower()[1:]
        cls._check_format(ext, ['obj', 'ply'])
        path = name + f".{ext}"
        cls._safe_make_dir(path)
        if path.endswith('.ply'):
            if vertex_normals is not None:
                logging.warning("normal result will not be displayed in .ply format")
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(vertices)
            # segment fault when numpy >= 2.0 !!
            o3d.io.write_point_cloud(path, pc)
            return
        with open(path, 'w') as file:
            file.write("o pc\n")
            _vertex = []
            for co in vertices:
                _vertex.append(f"v {co[0]} {co[2]} {-co[1]}\n")
            file.writelines(_vertex)
            if vertex_normals is not None:
                new_path = path.replace('.obj', '_normal.obj')
                nfile = open(new_path, 'w')
                nfile.write("o normal\n")
                _normal = []
                for i in range(vertices.shape[0]):
                    co = vertices[i]
                    x = vertex_normals[i, 0]
                    y = vertex_normals[i, 1]
                    z = vertex_normals[i, 2]
                    _normal.extend([
                        f"v {co[0]} {co[2]} {-co[1]}\n",
                        f"v {co[0]+0.0001} {co[2]} {-co[1]}\n",
                        f"v {co[0]+x*normal_size} {co[2]+z*normal_size} {-(co[1]+y*normal_size)}\n",
                        f"f {i*3+1} {i*3+2} {i*3+3}\n",
                    ])
                nfile.writelines(_normal)
    
    @classmethod
    def _make_armature(
        cls,
        vertices: Optional[Union[ndarray, List[ndarray]]]=None,
        skin: Optional[Union[ndarray, List[ndarray]]]=None,
        faces: Optional[Union[ndarray, List[ndarray]]]=None,
        mesh_names: Optional[Union[str, List[str]]]=None,
        joints: Optional[ndarray]=None,
        parents: Optional[List[Union[int, None]]]=None,
        bone_names: Optional[List[str]]=None,
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        use_extrude_bone: bool=True,
        use_connect_unique_child: bool=True,
        extrude_from_parent: bool=True,
        tails: Optional[ndarray]=None,
        matrix_local: Optional[ndarray]=None,
        matrix_basis: Optional[ndarray]=None,
        matrix_world: Optional[ndarray]=None,
    ):
        np.savez("d1.npz", matrix_local=matrix_local, matrix_basis=matrix_basis, matrix_world=matrix_world)
        # check types
        _map = {
            'vertices': vertices,
            'skin': skin,
            'faces': faces,
            'mesh_names': mesh_names,
        }
        _values = [isinstance(v, list) for v in _map.values() if v is not None]
        _keys = [k for k, v in _map.items() if v is not None]
        if len(set(_values)) > 1:
            raise ValueError(f"The following arguments must be a list or a single value: {','.join(_keys)}.")
        if len(_values) > 0 and _values[0]:
            _len = [len(v) for _, v in _map.items() if v is not None]
            if len(set(_len)) > 1:
                raise ValueError(f"The following arguments must have the same length: {','.join(_keys)}.")
        if len(_values) > 0 and not _values[0]:
            # turn into array
            if vertices is not None:
                vertices = [vertices]
            if skin is not None:
                skin = [skin]
            if faces is not None:
                faces = [faces]
            if mesh_names is not None:
                mesh_names = [mesh_names]
        
        import bpy # type: ignore
        from mathutils import Vector, Matrix # type: ignore
        
        # make collection
        collection = bpy.data.collections.new('new_collection')
        bpy.context.scene.collection.children.link(collection)
        
        # make mesh
        if vertices is not None:
            if mesh_names is None:
                mesh_names = [f"mesh_{i}" for i in range(len(vertices))]
            for i in range(len(vertices)):
                mesh = bpy.data.meshes.new(f"data_{mesh_names[i]}")
                if faces is None:
                    faces = []
                v = np.linalg.inv(matrix_world[:3, :3]) @ vertices[i].T
                mesh.from_pydata(v.T, [], faces[i])
                mesh.update()
        
                # make object from mesh
                object = bpy.data.objects.new(mesh_names[i], mesh)
            
                # add object to scene collection
                collection.objects.link(object)
        
        # deselect mesh
        bpy.ops.object.armature_add(enter_editmode=True)
        armature = bpy.data.armatures.get('Armature')
        edit_bones = armature.edit_bones
        
        # if there is matrix_local, inherit joints from it
        if joints is None and matrix_local is not None:
            joints = matrix_local[:, :3, 3]
        
        if joints is not None and parents is not None and bone_names is not None:
            J = joints.shape[0]
            connects = [False for _ in range(J)]
            children = defaultdict(list)
            for i in range(1, J):
                children[parents[i]].append(i)
            # make tails
            if tails is None:
                tails = joints.copy()
                tails[:, 2] += extrude_size
                if use_extrude_bone:
                    for i in range(J):
                        if len(children[i]) != 1 and extrude_from_parent and i != 0:
                            pjoint = joints[parents[i]]
                            joint = joints[i]
                            d = joint - pjoint
                            if np.linalg.norm(d) < 0.000001:
                                d = np.array([0., 0., 1.]) # in case son.head == parent.head
                            else:
                                d = d / np.linalg.norm(d)
                            tails[i] = joint + d * extrude_size
                if use_connect_unique_child:
                    for i in range(J):
                        if len(children[i]) == 1:
                            child = children[i][0]
                            tails[i] = joints[child]
                        if parents[i] is not None and len(children[parents[i]]) == 1:
                            connects[i] = True
            if add_root:
                bone_root = edit_bones.get('Bone')
                bone_root.name = 'Root'
                bone_root.tail = Vector((joints[0, 0], joints[0, 1], joints[0, 2]))
            else:
                bone_root = edit_bones.get('Bone')
                bone_root.name = bone_names[0]
                bone_root.head = Vector((joints[0, 0], joints[0, 1], joints[0, 2]))
                bone_root.tail = Vector((joints[0, 0], joints[0, 1], joints[0, 2] + extrude_size))
            
            def extrude_bone(
                edit_bones,
                name: str,
                parent_name: str,
                head: Tuple[float, float, float],
                tail: Tuple[float, float, float],
                connect: bool
            ):
                bone = edit_bones.new(name)
                bone.head = Vector((head[0], head[1], head[2]))
                bone.tail = Vector((tail[0], tail[1], tail[2]))
                bone.name = name
                parent_bone = edit_bones.get(parent_name)
                bone.parent = parent_bone
                bone.use_connect = connect
                assert not np.isnan(head).any(), f"nan found in head of bone {name}"
                assert not np.isnan(tail).any(), f"nan found in tail of bone {name}"
            
            for i in range(J):
                if add_root is False and i==0:
                    continue
                edit_bones = armature.edit_bones
                pname = 'Root' if parents[i] is None else bone_names[parents[i]]
                extrude_bone(edit_bones, bone_names[i], pname, joints[i], tails[i], connects[i])
            for i in range(J):
                bone = edit_bones.get(bone_names[i])
                bone.head = Vector((joints[i, 0], joints[i, 1], joints[i, 2]))
                bone.tail = Vector((tails[i, 0], tails[i, 1], tails[i, 2]))
            
            # set vertex groups
            if vertices is not None and skin is not None:
                # must set to object mode to enable parent_set
                bpy.ops.object.mode_set(mode='OBJECT')
                objects = bpy.data.objects
                for o in bpy.context.selected_objects:
                    o.select_set(False)
                for i in range(len(vertices)):
                    ob = objects[mesh_names[i]]
                    armature = bpy.data.objects['Armature']
                    ob.select_set(True)
                    armature.select_set(True)
                    bpy.ops.object.parent_set(type='ARMATURE_NAME')
                    vis = []
                    for x in ob.vertex_groups:
                        vis.append(x.name)
                    #sparsify
                    argsorted = np.argsort(-skin[i], axis=1)
                    vertex_group_reweight = skin[i][np.arange(skin[i].shape[0])[..., None], argsorted]
                    if group_per_vertex == -1:
                        group_per_vertex = vertex_group_reweight.shape[-1]
                    if not do_not_normalize:
                        vertex_group_reweight = vertex_group_reweight / vertex_group_reweight[..., :group_per_vertex].sum(axis=1)[...,None]

                    for v, w in enumerate(skin[i]):
                        for ii in range(group_per_vertex):
                            i = argsorted[v, ii]
                            if i >= J:
                                continue
                            n = bone_names[i]
                            if n not in vis:
                                continue
                            ob.vertex_groups[n].add([v], vertex_group_reweight[v, ii], 'REPLACE')
            # set animation
            if matrix_local is not None and matrix_basis is not None:
                def to_matrix(x):
                    return Matrix((x[0, :], x[1, :], x[2, :], x[3, :])) # THE VERY IDIOT INITIALIZATION METHOD OF THE MATHUTILS
                if matrix_world is None:
                    logging.warning("No matrix_world found, automatically set to identity and pose may be wrong.")
                    matrix_world = to_matrix(np.eye(4))
                bpy.ops.object.mode_set(mode='OBJECT')
                objects = bpy.data.objects
                for o in bpy.context.selected_objects:
                    o.select_set(False)
                matrix_world = to_matrix(matrix_world) # must convert to Matrix first to avoid transpose
                armature = bpy.data.objects['Armature']
                armature.select_set(True)
                armature.matrix_world = matrix_world
                frames = matrix_basis.shape[0]
                
                # change matrix_local
                bpy.context.view_layer.objects.active = armature
                bpy.ops.object.mode_set(mode='EDIT')
                for (id, name) in enumerate(bone_names):
                    # matrix_local of pose bone
                    bpy.context.active_object.data.edit_bones[id].matrix = to_matrix(matrix_local[id])
                bpy.ops.object.mode_set(mode='OBJECT')
                for (id, name) in enumerate(bone_names):
                    pbone = armature.pose.bones.get(name)
                    for frame in range(frames):
                        bpy.context.scene.frame_set(frame + 1)
                        q = to_matrix(matrix_basis[frame, id])
                        if pbone.rotation_mode == "QUATERNION":
                            pbone.rotation_quaternion = q.to_quaternion()
                            pbone.keyframe_insert(data_path = 'rotation_quaternion')
                        else:
                            pbone.rotation_euler = q.to_euler()
                            pbone.keyframe_insert(data_path = 'rotation_euler')
                        pbone.location = q.to_translation()
                        pbone.keyframe_insert(data_path = 'location')
                bpy.ops.object.mode_set(mode='OBJECT')
    
    @classmethod
    def _clean_bpy(cls):
        import bpy # type: ignore
        for c in bpy.data.actions:
            bpy.data.actions.remove(c)
        for c in bpy.data.armatures:
            bpy.data.armatures.remove(c)
        for c in bpy.data.cameras:
            bpy.data.cameras.remove(c)
        for c in bpy.data.collections:
            bpy.data.collections.remove(c)
        for c in bpy.data.images:
            bpy.data.images.remove(c)
        for c in bpy.data.materials:
            bpy.data.materials.remove(c)
        for c in bpy.data.meshes:
            bpy.data.meshes.remove(c)
        for c in bpy.data.objects:
            bpy.data.objects.remove(c)
        for c in bpy.data.textures:
            bpy.data.textures.remove(c)
    
    @classmethod
    def export_asset(
        cls,
        path: str,
        vertices: Optional[Union[ndarray, List[ndarray]]]=None,
        skin: Optional[Union[ndarray, List[ndarray]]]=None,
        faces: Optional[Union[ndarray, List[ndarray]]]=None,
        mesh_names: Optional[Union[str, List[str]]]=None,
        joints: Optional[ndarray]=None,
        parents: Optional[List[Union[int, None]]]=None,
        bone_names: Optional[List[str]]=None,
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        use_extrude_bone: bool=True,
        use_connect_unique_child: bool=True,
        extrude_from_parent: bool=True,
        tails: Optional[ndarray]=None,
        matrix_local: Optional[ndarray]=None,
        matrix_basis: Optional[ndarray]=None,
        matrix_world: Optional[ndarray]=None,
    ):
        import bpy # type: ignore
        cls._safe_make_dir(path=path)
        cls._clean_bpy()
        cls._make_armature(
            vertices=vertices,
            skin=skin,
            faces=faces,
            mesh_names=mesh_names,
            joints=joints,
            parents=parents,
            bone_names=bone_names,
            extrude_size=extrude_size,
            group_per_vertex=group_per_vertex,
            add_root=add_root,
            do_not_normalize=do_not_normalize,
            use_extrude_bone=use_extrude_bone,
            use_connect_unique_child=use_connect_unique_child,
            extrude_from_parent=extrude_from_parent,
            tails=tails,
            matrix_local=matrix_local,
            matrix_basis=matrix_basis,
            matrix_world=matrix_world,
        )
        
        _, ext = os.path.splitext(path)
        ext = ext.lower()[1:]
        if ext == 'fbx':
            if vertices is None and matrix_basis is not None:
                logging.warning("Exporting animation, but fbx format is deprecated because the rest pose will not be exported in bpy4.2. Use glb/gltf format instead. See: https://blender.stackexchange.com/questions/273398/blender-export-fbx-lose-the-origin-rest-pose.")
            bpy.ops.export_scene.fbx(filepath=path, check_existing=False, add_leaf_bones=False)
        elif ext == 'glb' or ext == 'gltf':
            bpy.ops.export_scene.gltf(filepath=path)
        else:
            raise ValueError(f"Unsupported format: {ext}")

def _trans_to_m(v: ndarray):
    m = np.eye(4)
    m[0:3, 3] = v
    return m

def _scale_to_m(r: ndarray):
    m = np.zeros((4, 4))
    m[0, 0] = r
    m[1, 1] = r
    m[2, 2] = r
    m[3, 3] = 1.
    return m