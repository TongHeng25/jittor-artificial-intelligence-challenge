import math
import numpy as np

from src.bone_inspector.asset import Asset
from src.bone_inspector.extract import extract_asset, ExtractOption
from src.bone_inspector.export import Exporter

mapping = {
    "Hips": "mixamorig:Hips",
    "LeftUpLeg": "mixamorig:LeftUpLeg",
    "LeftLeg": "mixamorig:LeftLeg",
    "LeftFoot": "mixamorig:LeftFoot",
    "RightUpLeg": "mixamorig:RightUpLeg",
    "RightLeg": "mixamorig:RightLeg",
    "RightFoot": "mixamorig:RightFoot",
    "Spine": "mixamorig:Spine",
    "Spine1": "mixamorig:Spine1",
    "Spine2": "mixamorig:Spine2",
    "Neck": "mixamorig:Neck",
    "LeftShoulder": "mixamorig:LeftShoulder",
    "LeftArm": "mixamorig:LeftArm",
    "LeftForeArm": "mixamorig:LeftForeArm",
    "RightShoulder": "mixamorig:RightShoulder",
    "RightArm": "mixamorig:RightArm",
    "RightForeArm": "mixamorig:RightForeArm",
}

# use models from mixamo
# extract target mesh
tgt = extract_asset("asset/Ty.fbx",
    ExtractOption(
        zero_roll=False,
        extract_mesh=True,
        extract_track=False,
    ))
tgt.armature.change_matrix_local()
# use motion from LAFAN1
# extract source animation
src = extract_asset("asset/sprint1_subject2.bvh",
    ExtractOption(
        zero_roll=False,
        extract_mesh=False,
        extract_track=True,
        merge_meshes=False,
    ))
src.keep(mapping.keys())

trans = np.eye(4)
trans[:3, :3] *= 0.01
src.transform(trans)

# map motion's skeleton into target's matrix_local space
m = []
for k1 in src.armature.bone_names:
    g = None
    for (i, k2) in enumerate(tgt.armature.bone_names):
        if k1 in mapping and mapping[k1] == k2:
            g = tgt.armature.matrix_local[i]
    if g is None:
        g = np.eye(4)
    m.append(g)
matrix_local = np.stack(m)
roll = {
    k: -90.0 / 180 * math.pi for k in mapping
}
src.armature.matrix_basis = src.armature.matrix_basis[:100]
src.armature.change_matrix_local(matrix_local=matrix_local, src_orientation="+x+z-y", roll=roll)

tgt.armature = tgt.armature.retarget(src.armature, exact=False, ignore_missing_bone=True, mapping=mapping)

# export animation
tgt.export_animation('test.fbx')