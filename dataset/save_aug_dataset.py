
import numpy as np
import os

from tqdm import tqdm
import os
from typing import List, Dict, Callable, Union
from asset import Asset

data_root = 'data/data/'
data_list = 'data/data/train_list.txt'
save_root = 'data/data/'
with open(data_list, 'r') as f:
        paths = [line.strip() for line in f.readlines()]



def get_dict(cls,id,vertices,vertex_normals,faces,face_normals,joints,skin,parents,names,matrix_local):
    return {'cls':cls,
            'id':id,
            'vertices':vertices,
            'vertex_normals':vertex_normals,
            'faces':faces,
            'face_normals':face_normals,
            'joints':joints,
            'skin':skin,
            'parents':parents,
            'names':names,
            'matrix_local':matrix_local}

dataset_paths = []

for i in tqdm(paths):
    asset = Asset.load(os.path.join(data_root, i))
    dict_0 = get_dict(asset.cls,asset.id,asset.vertices,asset.vertex_normals,asset.faces,asset.face_normals,asset.joints,asset.skin,asset.parents,asset.names,asset.matrix_local)
    asset.apply_matrix_basis(asset.get_random_matrix_basis(15.0))
    dict_15 = get_dict(asset.cls,asset.id,asset.vertices,asset.vertex_normals,asset.faces,asset.face_normals,asset.joints,asset.skin,asset.parents,asset.names,asset.matrix_local)
    
    asset = Asset.load(os.path.join(data_root, i))
    asset.apply_matrix_basis(asset.get_random_matrix_basis(30.0))
    dict_30 = get_dict(asset.cls,asset.id,asset.vertices,asset.vertex_normals,asset.faces,asset.face_normals,asset.joints,asset.skin,asset.parents,asset.names,asset.matrix_local)

    asset = Asset.load(os.path.join(data_root, i))
    asset.apply_matrix_basis(asset.get_random_matrix_basis(45.0))
    dict_45 = get_dict(asset.cls,asset.id,asset.vertices,asset.vertex_normals,asset.faces,asset.face_normals,asset.joints,asset.skin,asset.parents,asset.names,asset.matrix_local)
    
    asset = Asset.load(os.path.join(data_root, i))
    asset.apply_matrix_basis(asset.get_random_matrix_basis(60.0))
    dict_60 = get_dict(asset.cls,asset.id,asset.vertices,asset.vertex_normals,asset.faces,asset.face_normals,asset.joints,asset.skin,asset.parents,asset.names,asset.matrix_local)

    asset = Asset.load(os.path.join(data_root, i))
    asset.apply_matrix_basis(asset.get_random_matrix_basis(75.0))
    dict_75 = get_dict(asset.cls,asset.id,asset.vertices,asset.vertex_normals,asset.faces,asset.face_normals,asset.joints,asset.skin,asset.parents,asset.names,asset.matrix_local)
    
    name = i.replace('train','train_aug')
    np.savez(save_root + name, data= dict_0)
    dataset_paths.append(name)
    name = i.replace('train','train_aug')
    name = name.split('.')[0]
    np.savez(save_root + name + '_15.npz', data= dict_15)
    np.savez(save_root + name + '_30.npz', data= dict_30)
    np.savez(save_root + name + '_45.npz', data= dict_45)
    np.savez(save_root + name + '_60.npz', data= dict_60)
    np.savez(save_root + name + '_75.npz', data= dict_75)
    
    dataset_paths.append(name + '_15.npz')
    dataset_paths.append(name + '_30.npz')
    dataset_paths.append(name + '_45.npz')
    dataset_paths.append(name + '_60.npz')
    dataset_paths.append(name + '_75.npz')

with open("data/data/train_list_aug.txt", "w") as f:
    for path in dataset_paths:
        f.write(path + "\n")

    