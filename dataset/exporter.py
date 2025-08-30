import numpy as np
from numpy import ndarray
from typing import List, Union, Tuple
from collections import defaultdict
import os

try:
    import open3d as o3d
    OPEN3D_EQUIPPED = True
except:
    print("do not have open3d")
    OPEN3D_EQUIPPED = False

class Exporter():
    
    def _safe_make_dir(self, path):
        if os.path.dirname(path) == '':
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    def _export_skeleton(self, joints: ndarray, parents: List[Union[int, None]], path: str):
        format = path.split('.')[-1]
        assert format in ['obj']
        name = path.removesuffix('.obj')
        path = name + ".obj"
        self._safe_make_dir(path)
        J = joints.shape[0]
        with open(path, 'w') as file:
            file.write("o spring_joint\n")
            _joints = []
            for id in range(J):
                pid = parents[id]
                if pid is None:
                    continue
                bx, by, bz = joints[id]
                ex, ey, ez = joints[pid]
                _joints.extend([
                    f"v {bx} {bz} {-by}\n",
                    f"v {ex} {ez} {-ey}\n",
                    f"v {ex} {ez} {-ey + 0.00001}\n"
                ])
            file.writelines(_joints)
            
            _faces = [f"f {id*3+1} {id*3+2} {id*3+3}\n" for id in range(J)]
            file.writelines(_faces)
    
    def _export_mesh(self, vertices: ndarray, faces: ndarray, path: str):
        format = path.split('.')[-1]
        assert format in ['obj', 'ply']
        if path.endswith('ply'):
            if not OPEN3D_EQUIPPED:
                raise RuntimeError("open3d is not available")
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            self._safe_make_dir(path)
            o3d.io.write_triangle_mesh(path, mesh)
            return
        name = path.removesuffix('.obj')
        path = name + ".obj"
        self._safe_make_dir(path)
        with open(path, 'w') as file:
            file.write("o mesh\n")
            _vertices = []
            for co in vertices:
                _vertices.append(f"v {co[0]} {co[2]} {-co[1]}\n")
            file.writelines(_vertices)
            _faces = []
            for face in faces:
                _faces.append(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            file.writelines(_faces)
            
    def _export_pc(self, vertices: ndarray, path: str, vertex_normals: Union[ndarray, None]=None, size: float=0.01):
        if path.endswith('.ply'):
            if vertex_normals is not None:
                print("normal result will not be displayed in .ply format")
            name = path.removesuffix('.ply')
            path = name + ".ply"
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(vertices)
            # segment fault when numpy >= 2.0 !! use torch environment
            self._safe_make_dir(path)
            o3d.io.write_point_cloud(path, pc)
            return
        name = path.removesuffix('.obj')
        path = name + ".obj"
        self._safe_make_dir(path)
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
                        f"v {co[0]+x*size} {co[2]+z*size} {-(co[1]+y*size)}\n",
                        f"f {i*3+1} {i*3+2} {i*3+3}\n",
                    ])
                nfile.writelines(_normal)
    
    def _make_armature(
        self,
        vertices: ndarray,
        joints: ndarray,
        skin: ndarray,
        parents: List[Union[int, None]],
        names: list[str],
        faces: Union[ndarray, None]=None,
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        try_connect: bool=True,
        extrude_from_parent: bool=True,
    ):
        import bpy # type: ignore
        from mathutils import Vector # type: ignore
        # make mesh
        mesh = bpy.data.meshes.new('mesh')
        if faces is None:
            faces = []
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        
        # make object from mesh
        object = bpy.data.objects.new('character', mesh)
        
        # make collection
        collection = bpy.data.collections.new('new_collection')
        bpy.context.scene.collection.children.link(collection)
        
        # add object to scene collection
        collection.objects.link(object)
        
        # deselect mesh
        # mesh.select_set(False)
        bpy.ops.object.armature_add(enter_editmode=True)
        armature = bpy.data.armatures.get('Armature')
        edit_bones = armature.edit_bones
        
        J = joints.shape[0]
        tails = joints.copy()
        tails[:, 2] += extrude_size
        connects = [False for _ in range(J)]
        if try_connect:
            children = defaultdict(list)
            for i in range(1, J):
                children[parents[i]].append(i)
            for i in range(J):
                if len(children[i]) == 1:
                    child = children[i][0]
                    tails[i] = joints[child]
                if len(children[i]) != 1 and extrude_from_parent and i != 0:
                    pjoint = joints[parents[i]]
                    joint = joints[i]
                    d = joint - pjoint
                    d = d / np.linalg.norm(d)
                    tails[i] = joint + d * extrude_size
                if parents[i] is not None and len(children[parents[i]]) == 1:
                    connects[i] = True
        
        if add_root:
            bone_root = edit_bones.get('Bone')
            bone_root.name = 'Root'
            bone_root.tail = Vector((joints[0, 0], joints[0, 1], joints[0, 2]))
        else:
            bone_root = edit_bones.get('Bone')
            bone_root.name = names[0]
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
        
        for i in range(J):
            if add_root is False and i==0:
                continue
            edit_bones = armature.edit_bones
            pname = 'Root' if parents[i] is None else names[parents[i]]
            extrude_bone(edit_bones, names[i], pname, joints[i], tails[i], connects[i])
        
        # must set to object mode to enable parent_set
        bpy.ops.object.mode_set(mode='OBJECT')
        objects = bpy.data.objects
        for o in bpy.context.selected_objects:
            o.select_set(False)
        ob = objects['character']
        arm = bpy.data.objects['Armature']
        ob.select_set(True)
        arm.select_set(True)
        bpy.ops.object.parent_set(type='ARMATURE_NAME')
        vis = []
        for x in ob.vertex_groups:
            vis.append(x.name)
        #sparsify
        argsorted = np.argsort(-skin, axis=1)
        vertex_group_reweight = skin[np.arange(skin.shape[0])[..., None], argsorted]
        if group_per_vertex == -1:
            group_per_vertex = vertex_group_reweight.shape[-1]
        if not do_not_normalize:
            vertex_group_reweight = vertex_group_reweight / vertex_group_reweight[..., :group_per_vertex].sum(axis=1)[...,None]

        for v, w in enumerate(skin):
            for ii in range(group_per_vertex):
                i = argsorted[v, ii]
                if i >= J:
                    continue
                n = names[i]
                if n not in vis:
                    continue
                ob.vertex_groups[n].add([v], vertex_group_reweight[v, ii], 'REPLACE')

    def _clean_bpy(self):
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
    
    def _export_fbx(
        self,
        path: str,
        vertices: ndarray,
        joints: ndarray,
        skin: ndarray,
        parents: List[Union[int, None]],
        names: list[str],
        faces: Union[ndarray, None]=None,
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        try_connect: bool=True,
        extrude_from_parent: bool=True,
    ):
        '''
        Requires bpy installed
        '''
        import bpy # type: ignore
        self._clean_bpy()
        self._make_armature(
            vertices=vertices,
            joints=joints,
            skin=skin,
            parents=parents,
            names=names,
            faces=faces,
            extrude_size=extrude_size,
            group_per_vertex=group_per_vertex,
            add_root=add_root,
            do_not_normalize=do_not_normalize,
            try_connect=try_connect,
            extrude_from_parent=extrude_from_parent,
        )
        
        bpy.ops.export_scene.fbx(filepath=path, check_existing=False, add_leaf_bones=False) # the cursed leaf bone of blender
    
    def _export_animation(
        self,
        path: str,
        matrix_basis: ndarray,
        offset: ndarray,
        vertices: ndarray,
        joints: ndarray,
        skin: ndarray,
        parents: List[Union[int, None]],
        names: list[str],
        faces: Union[ndarray, None]=None,
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        try_connect=True,
    ):
        '''
        offset: (frames, 3)
        matrix_basis: (frames, J, 4, 4)
        matrix_local: (J, 4, 4)
        '''
        import bpy # type: ignore
        from mathutils import Matrix # type: ignore
        self._clean_bpy()
        self._make_armature(
            vertices=vertices,
            joints=joints,
            skin=skin,
            parents=parents,
            names=names,
            faces=faces,
            extrude_size=extrude_size,
            group_per_vertex=group_per_vertex,
            add_root=add_root,
            do_not_normalize=do_not_normalize,
            try_connect=try_connect,
        )
        name_to_id = {name: i for (i, name) in enumerate(names)}
        frames = matrix_basis.shape[0]
        armature = bpy.data.objects.get('Armature')
        for bone in bpy.data.armatures[0].edit_bones:
            bone.roll = 0.
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')
        for frame in range(frames):
            bpy.context.scene.frame_set(frame + 1)
            for pbone in armature.pose.bones:
                name = pbone.name
                q = Matrix(matrix_basis[frame, name_to_id[name]]).to_4x4()
                if name == names[0]:
                    q[0][3] = offset[frame, 0]
                    q[1][3] = offset[frame, 1]
                    q[2][3] = offset[frame, 2]
                if pbone.rotation_mode == "QUATERNION":
                    pbone.rotation_quaternion = q.to_quaternion()
                    pbone.keyframe_insert(data_path='rotation_quaternion')
                else:
                    pbone.rotation_euler = q.to_euler()
                    pbone.keyframe_insert(data_path='rotation_euler')
                pbone.location = q.to_translation()
                pbone.keyframe_insert(data_path = 'location')
                pbone.matrix_basis = q
        bpy.ops.export_scene.fbx(filepath=path, check_existing=False, add_leaf_bones=False)
    
    def _render_skeleton(
        self,
        path: str,
        joints: ndarray,
        parents: List[Union[int, None]],
        x_lim: Tuple[float, float]=(-0.5, 0.5),
        y_lim: Tuple[float, float]=(-0.5, 0.5),
        z_lim: Tuple[float, float]=(-0.5, 0.5),
    ):
        self._safe_make_dir(path=path)
        import numpy as np
        from matplotlib import pyplot as plt
        plt.switch_backend('agg')
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='g', marker='o')
        
        # Draw lines between joints and their parents
        for i, parent in enumerate(parents):
            if parent is not None:
                ax.plot(
                    [joints[i, 0], joints[parent, 0]],
                    [joints[i, 1], joints[parent, 1]],
                    [joints[i, 2], joints[parent, 2]],
                    color='r',
                )
        
        ax.set_proj_type('ortho')
        ax.view_init(elev=30, azim=-135)
        ax.set_position([0, 0, 1, 1])
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_zlim(*z_lim)
        ax.set_axis_off()
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        plt.savefig(path, transparent=True, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _render_skin(
        self,
        path: str,
        vertices: ndarray,
        skin: ndarray,
        x_lim: Tuple[float, float]=(-0.5, 0.5),
        y_lim: Tuple[float, float]=(-0.5, 0.5),
        z_lim: Tuple[float, float]=(-0.5, 0.5),
        joint: Union[ndarray, None]=None,
    ):
        '''
        Render a picture of skin for a easier life.
        '''
        self._safe_make_dir(path=path)
        import numpy as np
        from matplotlib import pyplot as plt
        
        plt.switch_backend('agg')
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        weights_normalized = (skin - skin.min()) / (skin.max() - skin.min() + 1e-10)
        sizes = 10 * np.ones_like(weights_normalized)
        colors = np.zeros((len(weights_normalized), 4))  # RGBA
        colors[:, 0] = weights_normalized
        colors[:, 2] = 1 - weights_normalized
        colors[:, 3] = 1.0
        scatter = ax.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            color=colors,
            sizes=sizes,
            marker='o',
        )
        # plot joint
        if joint is not None:
            scatter = ax.scatter(
                joint[0],
                joint[1],
                joint[2],
                color=np.array([0., 1., 0., 1.]),
                sizes=np.array([1.]) * 100,
                marker='x',
            )
        ax.set_proj_type('ortho')
        ax.view_init(elev=30, azim=-135)
        ax.set_position([0, 0, 1, 1])
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_zlim(*z_lim)
        ax.set_axis_off()
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        plt.savefig(path, transparent=True, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _render_pc(
        self,
        path: str,
        vertices: ndarray,
        x_lim: Tuple[float, float]=(-0.5, 0.5),
        y_lim: Tuple[float, float]=(-0.5, 0.5),
        z_lim: Tuple[float, float]=(-0.5, 0.5),
    ):
        self._safe_make_dir(path=path)
        import numpy as np
        from matplotlib import pyplot as plt
        
        plt.switch_backend('agg')
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            color='blue',
            sizes=np.array([10.]),
            marker='o',
        )
        ax.set_proj_type('ortho')
        ax.view_init(elev=30, azim=-135)
        ax.set_position([0, 0, 1, 1])
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_zlim(*z_lim)
        ax.set_axis_off()
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        plt.savefig(path, transparent=True, dpi=300, bbox_inches='tight')
        plt.close(fig)