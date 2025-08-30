import jittor as jt
import numpy as np
import os
from jittor.dataset import Dataset

import os
from typing import List, Dict, Callable, Union

from .asset import Asset
from .sampler import Sampler

def transform(asset: Asset):
    """
    Transform the asset data into [-1, 1]^3.
    """
    # Find min and max values for each dimension of points
    min_vals = np.min(asset.vertices, axis=0)
    max_vals = np.max(asset.vertices, axis=0)
    
    # Calculate the center of the bounding box
    center = (min_vals + max_vals) / 2
    
    # Calculate the scale factor to normalize to [-1, 1]
    # We take the maximum range across all dimensions to preserve aspect ratio
    scale = np.max(max_vals - min_vals) / 2
    
    # Normalize points to [-1, 1]^3
    normalized_vertices = (asset.vertices - center) / scale
    
    # Apply the same transformation to joints
    if asset.joints is not None:
        normalized_joints = (asset.joints - center) / scale
    else:
        normalized_joints = None
    
    asset.vertices  = normalized_vertices
    asset.joints    = normalized_joints
    # remember to change matrix_local !
    asset.matrix_local[:, :3, 3] = normalized_joints

class RigDataset(Dataset):
    '''
    A simple dataset class.
    '''
    def __init__(
        self,
        data_root: str,
        paths: List[str],
        train: bool,
        batch_size: int,
        shuffle: bool,
        sampler: Sampler,
        transform: Union[Callable, None] = None,
        return_origin_vertices: bool = False,
        random_pose: bool = False,
    ):
        super().__init__()
        self.data_root  = data_root
        self.paths      = paths.copy()
        self.batch_size = batch_size
        self.train      = train
        self.shuffle    = shuffle
        self._sampler   = sampler # do not use `sampler` to avoid name conflict
        self.transform  = transform
        
        self.random_pose = random_pose
        
        self.return_origin_vertices = return_origin_vertices
        self.set_attrs(
            batch_size=self.batch_size,
            total_len=len(self.paths),
            shuffle=self.shuffle,
        )
    
    def __getitem__(self, index) -> Dict:
        """
        Get a sample from the dataset
        
        Args:
            index (int): Index of the sample
            
        Returns:
            data (Dict): Dictionary containing the following keys:
                - vertices: jt.Var, (B, N, 3) point cloud data
                - normals: jt.Var, (B, N, 3) point cloud normals
                - joints: jt.Var, (B, J, 3) joint positions
                - skin: jt.Var, (B, J, J) skinning weights
        """
        
        path = self.paths[index]

        asset = Asset.load(os.path.join(self.data_root, path))
        #asset = Asset.load_test(os.path.join(self.data_root, path))
        '''t = np.random.rand()
        if self.random_pose and  t < 0.2:
            asset.apply_matrix_basis(asset.get_random_matrix_basis(15.0))
        elif self.random_pose and  t < 0.4:
            asset.apply_matrix_basis(asset.get_random_matrix_basis(30.0))
        elif self.random_pose and  t < 0.5:
            asset.apply_matrix_basis(asset.get_random_matrix_basis(45.0))'''
        

        if self.transform is not None:
            self.transform(asset)
        origin_vertices = jt.array(asset.vertices.copy()).float32()
        
        sampled_asset = asset.sample(sampler=self._sampler)

        vertices    = jt.array(sampled_asset.vertices).float32()
        normals     = jt.array(sampled_asset.normals).float32()

        if sampled_asset.joints is not None:
            joints      = jt.array(sampled_asset.joints).float32()
        else:
            joints      = None

        if sampled_asset.skin is not None:
            skin        = jt.array(sampled_asset.skin).float32()
        else:
            skin        = None

        res = {
            'vertices': vertices,
            'normals': normals,
            'cls': asset.cls,
            'id': asset.id,
        }
        if joints is not None:
            res['joints'] = joints
        if skin is not None:
            res['skin'] = skin
        if self.return_origin_vertices:
            res['origin_vertices'] = origin_vertices
        return res
    
    def collate_batch(self, batch):
        if self.return_origin_vertices:
            max_N = 0
            for b in batch:
                max_N = max(max_N, b['origin_vertices'].shape[0])
            for b in batch:
                N = b['origin_vertices'].shape[0]
                b['origin_vertices'] = np.pad(b['origin_vertices'], ((0, max_N-N), (0, 0)), 'constant', constant_values=0.)
                b['N'] = N
        return super().collate_batch(batch)

# Example usage of the dataset
def get_dataloader(
    data_root: str,
    data_list: str,
    train: bool,
    batch_size: int,
    shuffle: bool,
    sampler: Sampler,
    transform: Union[Callable, None] = None,
    return_origin_vertices: bool = False,
    random_pose: bool = False,
):
    """
    Create a dataloader for point cloud data
    
    Args:
        data_root (str): Root directory for the data files
        data_list (str): Path to the file containing list of data files
        train (bool): Whether the dataset is for training
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the dataset
        sampler (Sampler): Sampler to use for point cloud sampling
        transform (callable, optional): Optional post-transform to be applied on a sample
        return_origin_vertices (bool): Whether to return original vertices
        
    Returns:
        dataset (RigDataset): The dataset
    """
    with open(data_list, 'r') as f:
        paths = [line.strip() for line in f.readlines()]
    dataset = RigDataset(
        data_root=data_root,
        paths=paths,
        train=train,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        transform=transform,
        return_origin_vertices=return_origin_vertices,
        random_pose=random_pose,
    )
    
    return dataset


def get_dataloader_udt(
    data_root: str,
    data_list: str,
    train: bool,
    batch_size: int,
    shuffle: bool,
    sampler: Sampler,
    transform: Union[Callable, None] = None,
    return_origin_vertices: bool = False,
):
    """
    Create a dataloader for point cloud data
    
    Args:
        data_root (str): Root directory for the data files
        data_list (str): Path to the file containing list of data files
        train (bool): Whether the dataset is for training
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the dataset
        sampler (Sampler): Sampler to use for point cloud sampling
        transform (callable, optional): Optional post-transform to be applied on a sample
        return_origin_vertices (bool): Whether to return original vertices
        
    Returns:
        dataset (RigDataset): The dataset
    """
    with open(data_list, 'r') as f:
        paths = [line.strip() for line in f.readlines()]
    dataset = RigDataset_udt(
        data_root=data_root,
        paths=paths,
        train=train,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        transform=transform,
        return_origin_vertices=return_origin_vertices,
    )
    
    return dataset

class RigDataset_udt(Dataset):
    '''
    A simple dataset class.
    '''
    def __init__(
        self,
        data_root: str,
        paths: List[str],
        train: bool,
        batch_size: int,
        shuffle: bool,
        sampler: Sampler,
        transform: Union[Callable, None] = None,
        return_origin_vertices: bool = False
    ):
        super().__init__()
        self.data_root  = data_root
        self.paths      = paths.copy()
        self.batch_size = batch_size
        self.train      = train
        self.shuffle    = shuffle
        self._sampler   = sampler # do not use `sampler` to avoid name conflict
        self.transform  = transform
        self.random_pose = True
        
        self.return_origin_vertices = return_origin_vertices
        self.set_attrs(
            batch_size=self.batch_size,
            total_len=len(self.paths),
            shuffle=self.shuffle,
        )
    
    def __getitem__(self, index) -> Dict:
        """
        Get a sample from the dataset
        
        Args:
            index (int): Index of the sample
            
        Returns:
            data (Dict): Dictionary containing the following keys:
                - vertices: jt.Var, (B, N, 3) point cloud data
                - normals: jt.Var, (B, N, 3) point cloud normals
                - joints: jt.Var, (B, J, 3) joint positions
                - skin: jt.Var, (B, J, J) skinning weights
        """
        
        path = self.paths[index]
        asset = Asset.load(os.path.join(self.data_root, path))
        '''t = np.random.rand()
        if self.random_pose and  t < 0.3:
            asset.apply_matrix_basis(asset.get_random_matrix_basis(15.0))
        elif self.random_pose and  t < 0.6:
            asset.apply_matrix_basis(asset.get_random_matrix_basis(30.0))
        elif self.random_pose and  t < 0.9:
            asset.apply_matrix_basis(asset.get_random_matrix_basis(45.0))
        elif self.random_pose:
            asset.apply_matrix_basis(asset.get_random_matrix_basis(60.0))'''
        if self.transform is not None:
            self.transform(asset)
        origin_vertices = jt.array(asset.vertices.copy()).float32()
        
        sampled_asset = asset.sample(sampler=self._sampler)
        vertices    = jt.array(sampled_asset.vertices).float32()

        normals     = jt.array(sampled_asset.normals).float32()

        if sampled_asset.joints is not None:
            joints      = jt.array(sampled_asset.joints).float32()
        else:
            joints      = None

        if sampled_asset.skin is not None:
            skin        = jt.array(sampled_asset.skin).float32()
        else:
            skin        = None

        res = {
            'vertices': vertices,
            'normals': normals,
            'cls': asset.cls,
            'id': asset.id,
        }

        return res
    
    def collate_batch(self, batch):
        if self.return_origin_vertices:
            max_N = 0
            for b in batch:
                max_N = max(max_N, b['origin_vertices'].shape[0])
            for b in batch:
                N = b['origin_vertices'].shape[0]
                b['origin_vertices'] = np.pad(b['origin_vertices'], ((0, max_N-N), (0, 0)), 'constant', constant_values=0.)
                b['N'] = N
        return super().collate_batch(batch)