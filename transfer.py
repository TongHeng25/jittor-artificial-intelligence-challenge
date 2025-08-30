import numpy as np

from dataset.asset import Asset
from dataset.format import retarget_mapping

#from bone_inspector.extract import extract_asset, ExtractOption
from bone_inspector.src.bone_inspector.extract import extract_asset, ExtractOption
'''asset = Asset.load("data/data/train/mixamo/13.npz")
asset.export_fbx("res.fbx")'''
parents = [None, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20, 9, 22, 23, 9, 25, 26, 9, 28, 29, 9, 31, 32, 9, 34, 35, 13, 37, 38, 13, 40, 41, 13, 43, 44, 13, 46, 47, 13, 49, 50,]
# # if you need to use prediction results, write code like:
asset           = Asset.load("data/data/test/mixamo/3188.npz")
asset.vertices  = np.load("/data3/jitu_B/jittor-comp-human-main/predict/mixamo/3188/transformed_vertices.npy")
asset.parents   = parents
asset.joints    = np.load("/data3/jitu_B/jittor-comp-human-main/predict/mixamo/3188/predict_skeleton.npy")
asset.skin      = np.load("/data3/jitu_B/jittor-comp-human-main/predict/mixamo/3188/predict_skin.npy")
asset.export_fbx("3188.fbx")

tgt = extract_asset("3188.fbx",
    ExtractOption(zero_roll=False,extract_mesh=True,extract_track=False,)
)
# load animation
src = extract_asset("data/data/animation/Swing_Dancing_1.fbx",
    ExtractOption(zero_roll=False,extract_mesh=False,extract_track=True,)
)

# remove bones of toes for better visualization
keep = [v for k, v in retarget_mapping.items() if v != 'l_toe_base' and v != 'r_toe_base']
tgt.keep(keep)

roll = {k: 3.1415926 for (i, k) in enumerate(retarget_mapping) if i >= 6}
src.armature.change_matrix_local(roll=roll)
tgt.armature.change_matrix_local()
tgt.armature = tgt.armature.retarget(src.armature, exact=False, mapping=retarget_mapping)

# export animation
tgt.export_animation("3188.fbx")