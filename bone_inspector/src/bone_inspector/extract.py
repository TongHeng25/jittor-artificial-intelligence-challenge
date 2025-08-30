from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from numpy import ndarray
from scipy.spatial import KDTree
from typing import Any, Optional, Union, Tuple, List

import bpy
import fast_simplification
import gc
import logging
import numpy as np
import os
import trimesh

from .asset import Asset
from .error import *
from .export import Exporter
from .info import MeshInfo, ArmatureInfo
from .motion import linear_blend_skinning
from .warning import *

@dataclass
class ExtractOption():
    extract_mesh: bool=True
    extract_armature: bool=True
    extract_skin: bool=True
    extract_track: bool=True
    merge_meshes: bool=True
    
    # change roll of the bone to zero
    zero_roll: bool=False
    
    # remove roots with no skin
    remove_root: bool=True
    
    # remove bones with no skin
    remove_dummy: bool=False
    
    # trim to (approximately) assigned number of faces
    max_f: Optional[int]=None

def clean_bpy():
    """
    Clean all the data in bpy.
    """
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    data_types = [
        bpy.data.actions,
        bpy.data.armatures,
        bpy.data.cameras,
        bpy.data.collections,
        bpy.data.curves,
        bpy.data.images,
        bpy.data.lights,
        bpy.data.materials,
        bpy.data.meshes,
        bpy.data.objects,
        bpy.data.textures,
        bpy.data.worlds,
        bpy.data.node_groups
    ]
    for data_collection in data_types:
        for item in data_collection:
            data_collection.remove(item)
    gc.collect()

def load(filepath: str, extract_option: ExtractOption):
    """
    Load mesh and armature from a file.
    """
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()[1:]
    old_objs = set(bpy.context.scene.objects)
    
    if not os.path.exists(filepath):
        raise error_file_does_not_exist(filepath)
    
    if ext == "vrm":
        # enable vrm addon and load vrm model
        try:
            bpy.ops.preferences.addon_enable(module='vrm')
            bpy.ops.import_scene.vrm(
                filepath=filepath,
                use_addon_preferences=True,
                extract_textures_into_folder=False,
                make_new_texture_folder=False,
                set_shading_type_to_material_on_import=False,
                set_view_transform_to_standard_on_import=True,
                set_armature_display_to_wire=True,
                set_armature_display_to_show_in_front=True,
                set_armature_bone_shape_to_default=True,
                disable_bake=True, # customized option for better performance
            )
        except:
            raise error_incorrect_vrm_addon()
    elif ext == "obj":
        bpy.ops.wm.obj_import(filepath=filepath)
    elif ext == "fbx":
        # do not ignore leaf bones !!!
        bpy.ops.import_scene.fbx(filepath=filepath, ignore_leaf_bones=False, use_image_search=False)
    elif ext == "glb" or ext == "gltf":
        bpy.ops.import_scene.gltf(filepath=filepath, import_pack_images=False)
    elif ext == "dae":
        bpy.ops.wm.collada_import(filepath=filepath)
    elif ext == "blend":
        with bpy.data.libraries.load(filepath) as (data_from, data_to):
            data_to.objects = data_from.objects
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
    elif ext == "bvh":
        bpy.ops.import_anim.bvh(filepath=filepath)
    else:
        raise error_unsupported_type(ext)

    if extract_option.extract_armature:
        armature = [x for x in set(bpy.context.scene.objects)-old_objs if x.type=="ARMATURE"]
        if len(armature)==0:
            raise error_no_armature()
        if len(armature)>1:
            raise error_multiple_armatrues()
        armature = armature[0]
        
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')
        if extract_option.zero_roll:
            for bone in bpy.data.armatures[0].edit_bones:
                bone.roll = 0.
        
        bpy.ops.object.mode_set(mode='OBJECT')
        armature.select_set(False)
        
        bpy.ops.object.select_all(action='DESELECT')
        return armature
    else:
        return None

def get_have_skin(armature) -> dict[str, bool]:
    meshes = []
    for v in bpy.data.objects:
        if v.type == 'MESH':
            meshes.append(v)
    have_skin = defaultdict(bool)
    for obj in meshes:
        obj_verts = obj.data.vertices
        obj_group_names = [g.name for g in obj.vertex_groups]

        for bone in armature.pose.bones:
            have = False
            if bone.name not in obj_group_names:
                continue

            gidx = obj.vertex_groups[bone.name].index
            bone_verts = [v for v in obj_verts if gidx in [g.group for g in v.groups]]
            for v in bone_verts:
                which = [id for id in range(len(v.groups)) if v.groups[id].group==gidx]
                w = v.groups[which[0]].weight
                if abs(w) > 1e-6:
                    have = True
                    break
            have_skin[bone.name] |= have
    return have_skin

def get_arranged_bones(armature, extract_option: ExtractOption) -> Optional[list]:
    if extract_option.extract_armature == False:
        return None
    matrix_world = armature.matrix_world
    arranged_bones = []
    root = armature.pose.bones[0]
    have_skin = get_have_skin(armature=armature)
    while root.parent is not None:
        root = root.parent
    Q = [root]
    rot = np.array(matrix_world)[:3, :3]
    children = defaultdict(list)
    on_final_chain = {}
    for b in armature.pose.bones:
        children[b.parent].append(b)
        on_final_chain[b] = True
    
    for b in armature.pose.bones:
        if len(children[b]) > 1:
            while b is not None:
                on_final_chain[b] = False
                b = b.parent
    
    # remove leaf bones with no skin
    is_dummy = {}
    for b in armature.pose.bones:
        is_dummy[b.name] = True
    for b in armature.pose.bones:
        if have_skin[b.name]:
            while b is not None:
                is_dummy[b.name] = False
                b = b.parent
    
    # dfs and sort
    while len(Q) != 0:
        b = Q.pop(0)
        if extract_option.remove_dummy and is_dummy.get(b.name, False):
            continue
        if not (extract_option.remove_root and b.parent is None and have_skin[b.name]==False and len(children[b])==1): # remove Root, which has no skin and exactly one child
            arranged_bones.append(b)
        children = []
        for cb in b.children:
            head = rot @ np.array(b.head)
            children.append((cb, head[0], head[1], head[2]))
        children = sorted(children, key=lambda x: (x[3], x[1], x[2]))
        _c = [x[0] for x in children]
        Q = _c + Q
    return arranged_bones

def extract_armature(armature, arranged_bones, extract_option: ExtractOption) -> Tuple[Optional[ArmatureInfo], List[ExtractWarning]]:
    """
    Extract joints, tails, parents, names and matrix_local from bpy.
    """
    if extract_option.extract_armature == False:
        return None, []
    warnings = []
    matrix_world = armature.matrix_world
    vis = {}
    is_leaf = {}
    for bone in arranged_bones:
        vis[bone.name] = True
        if bone.parent is None:
            continue
        is_leaf[bone.parent.name] = False
        if vis.get(bone.parent.name) is None and bone.parent in arranged_bones:
            raise error_bad_topoplgy()
    index = {}
    
    for (id, pbone) in enumerate(arranged_bones):
        index[pbone.name] = id
    
    root = armature.pose.bones[0]
    while root.parent is not None:
        root = root.parent
    m = np.array(matrix_world.to_4x4())
    rot = m[:3, :3]
    bias = m[:3, 3]
    scale_inv = np.linalg.inv(np.diag(np.array(matrix_world.to_scale())))
    
    s = []
    bpy.ops.object.editmode_toggle()
    edit_bones = armature.data.edit_bones
    
    J = len(arranged_bones)
    joints = np.zeros((J, 3), dtype=np.float32)
    tails = np.zeros((J, 3), dtype=np.float32)
    parents = []
    name_to_id = {}
    names = []
    matrix_local_stack = np.zeros((J, 4, 4), dtype=np.float32)
    lengths = []
    for (id, pbone) in enumerate(arranged_bones):
        name = pbone.name
        names.append(name)
        matrix_local = np.array(pbone.bone.matrix_local)
        head = rot @ matrix_local[0:3, 3] + bias
        s.append(head)
        edit_bone = edit_bones.get(name)
        tail = rot @ np.array(edit_bone.tail) + bias
        
        name_to_id[name] = id
        joints[id] = head
        tails[id] = tail
        parents.append(
            None if pbone.parent not in arranged_bones else name_to_id[pbone.parent.name]
        )
        matrix_local_stack[id] = matrix_local
        lengths.append(pbone.length)
    bpy.ops.object.editmode_toggle()
    
    if extract_option.extract_track:
        if bpy.data.actions:
            if len(bpy.data.actions) > 1:
                warnings.append(warning_multiple_tracks())
            for action in bpy.data.actions:
                frames = int(action.frame_range.y - action.frame_range.x)
        else:
            raise error_no_track()
        
        J = len(arranged_bones)
        matrix_basis = np.zeros((frames, J, 4, 4))
        matrix_basis[...] = np.eye(4)
        # get matrix_basis
        for frame in range(frames):
            bpy.context.scene.frame_set(frame + 1)
            for (id, pbone) in enumerate(arranged_bones):
                matrix_basis[frame, id] = np.array(pbone.matrix_basis)
    else:
        matrix_basis = None
    
    return ArmatureInfo(
        matrix_local=matrix_local_stack,
        matrix_world=m,
        parents=parents,
        lengths=np.array(lengths),
        bone_names=names,
        matrix_basis=matrix_basis,
        name=armature.name,
    ), warnings

def extract_mesh(arranged_bones: Optional[Any], extract_option: ExtractOption) -> Tuple[Optional[List[MeshInfo]], List[ExtractWarning]]:
    """
    Extract vertices, face_normals, faces and skinning(if possible) from bpy.
    """
    if extract_option.extract_mesh == False:
        return None, []
    warnings = []
    meshes = []
    for v in bpy.data.objects:
        if v.type == 'MESH':
            meshes.append(v)
    
    if arranged_bones is not None:
        index = {}
        # update index first
        for (id, pbone) in enumerate(arranged_bones):
            index[pbone.name] = id
        total_bones = len(arranged_bones)
    else:
        total_bones = None
    
    _dict_mesh = {}
    _dict_skin = {}
    mesh_infos = []
    no_skin = False
    for obj in meshes:
        m = np.array(obj.matrix_world)
        matrix_world_rot = m[:3, :3]
        matrix_world_bias = m[:3, 3]
        rot = matrix_world_rot
        total_vertices = len(obj.data.vertices)
        vertices = np.zeros((3, total_vertices))
        if total_bones:
            skin_weight = np.zeros((total_vertices, total_bones))
        obj_verts = obj.data.vertices
        obj_group_names = [g.name for g in obj.vertex_groups]
        faces = []
        normals = []
        
        for polygon in obj.data.polygons:
            edges = polygon.edge_keys
            nodes = []
            adj = {}
            for edge in edges:
                if adj.get(edge[0]) is None:
                    adj[edge[0]] = []
                adj[edge[0]].append(edge[1])
                if adj.get(edge[1]) is None:
                    adj[edge[1]] = []
                adj[edge[1]].append(edge[0])
                nodes.append(edge[0])
                nodes.append(edge[1])
            normal = polygon.normal
            nodes = list(set(sorted(nodes)))
            first = nodes[0]
            loop = []
            now = first
            vis = {}
            while True:
                loop.append(now)
                vis[now] = True
                if vis.get(adj[now][0]) is None:
                    now = adj[now][0]
                elif vis.get(adj[now][1]) is None:
                    now = adj[now][1]
                else:
                    break
            for (second, third) in zip(loop[1:], loop[2:]):
                faces.append((first, second, third))
                normals.append(rot @ normal)
        
        # extract skin
        if total_bones and extract_option.extract_skin:
            for bone in arranged_bones:
                if bone.name not in obj_group_names:
                    continue

                gidx = obj.vertex_groups[bone.name].index
                bone_verts = [v for v in obj_verts if gidx in [g.group for g in v.groups]]
                for v in bone_verts:
                    which = [id for id in range(len(v.groups)) if v.groups[id].group==gidx]
                    w = v.groups[which[0]].weight
                    vv = rot @ v.co
                    vv = np.array(vv) + matrix_world_bias
                    vertices[0:3, v.index] = vv
                    if total_bones:
                        skin_weight[v.index, index[bone.name]] = w
        correct_faces = []
        for (i, face) in enumerate(faces):
            normal = normals[i]
            v0 = face[0]
            v1 = face[1]
            v2 = face[2]
            v = np.cross(
                vertices[:3, v1] - vertices[:3, v0],
                vertices[:3, v2] - vertices[:3, v0],
            )
            if (v*normal).sum() > 0:
                correct_faces.append(face)
            else:
                correct_faces.append((face[0], face[2], face[1]))
        mesh_info = MeshInfo(
            name=obj.name,
            vertices=vertices.T,
            faces=np.array(correct_faces, dtype=np.int32),
            skin=skin_weight if total_bones else None,
        )
        if total_bones:
            no_skin |= np.any(skin_weight.sum(axis=1) < 1e-8)
        mesh_info.construct(max_f=extract_option.max_f)
        mesh_infos.append(mesh_info)
    
    if no_skin:
        warnings.append(warning_no_skin())
    return mesh_infos, warnings

def extract_asset(filepath: str, extract_option: ExtractOption) -> Asset:
    clean_bpy()
    result = Asset(warnings=None, error=None)
    try:
        armature = load(filepath=filepath, extract_option=extract_option)
        arranged_bones = get_arranged_bones(armature=armature, extract_option=extract_option)
        mesh_infos, warnings = extract_mesh(arranged_bones=arranged_bones, extract_option=extract_option)
        result.add_warning(warnings)
        result.meshes = mesh_infos
        
        armature_info, warnings = extract_armature(armature=armature, arranged_bones=arranged_bones, extract_option=extract_option)
        result.add_warning(warnings)
        result.armature = armature_info
    except ExtractError as e:
        result.error = e
    if extract_option.extract_mesh and extract_option.merge_meshes:
        result.merge_meshes()
    return result
