from src.bone_inspector.extract import extract_asset, ExtractOption
from src.bone_inspector.export import Exporter

# use models from mixamo
# extract target mesh
tgt = extract_asset("asset/Ty.fbx",
    ExtractOption(
        zero_roll=False,
        extract_mesh=True,
        extract_track=False,
    ))
# extract source animation
src = extract_asset("asset/jump.fbx",
    ExtractOption(
        zero_roll=False,
        extract_mesh=True,
        extract_track=True,
        merge_meshes=False,
    ))
# retarget
tgt.armature = tgt.armature.retarget(src.armature, exact=False, ignore_missing_bone=True)

# export animation
tgt.export_animation('test.fbx')
# or export glb format, but needs to set `matrix_world` to identity to prevent bug in glb exporter
# and needs to remove `glTF_not_exported` manually in blender to correctly visualize armature
tgt.armature.change_matrix_local()
tgt.export_animation('test.glb')

# apply a pose at frame 10, this operation is irreversible
tgt.armature.apply_pose(tgt.armature.get_frame(10), inplace=True)
# export
tgt.armature.export_skeleton('skeleton.obj')