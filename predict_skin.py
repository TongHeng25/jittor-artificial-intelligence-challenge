
import os
import jittor as jt
import numpy as np

import argparse

from dataset.asset import Asset
from dataset.dataset import get_dataloader, transform
from dataset.sampler import SamplerMix
from dataset.format import num_joints
from models.skin import create_model

import numpy as np
from scipy.spatial import cKDTree
import random

from tqdm import tqdm

# Set Jittor flags
jt.flags.use_cuda = 1

def predict(args):
    # Create model
    model = create_model(
        model_name=args.model_name,
        model_type=args.model_type,
        num_joints=num_joints,
    )
    
    sampler = SamplerMix(num_samples=1024, vertex_samples=512)
    
    # Load pre-trained model if specified
    if args.pretrained_model:
        model.load(args.pretrained_model)
    
    predict_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.predict_data_list,
        train=False,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        transform=transform,
        return_origin_vertices=True,
    )
    predict_output_dir = args.predict_output_dir
    print("start predicting...")
    for batch_idx, data in tqdm(enumerate(predict_loader)):
        # currently only support batch_size==1 because origin_vertices is not padded
        vertices, cls, id, origin_vertices, N = data['vertices'], data['cls'], data['id'], data['origin_vertices'], data['N']
        
        # load predicted skeleton
        joints = []
        for i in range(len(cls)):
            path = os.path.join(predict_output_dir, cls[i], str(id[i].item()), "predict_skeleton.npy")
            data = np.load(path)
            joints.append(data)
        joints = jt.array(joints)
        
        B = vertices.shape[0]
        outputs = model(vertices, joints)
        for i in range(B):
            # resample
            skin = outputs[i].numpy()
            o_vertices = origin_vertices[i, :N[i]].numpy()

            tree = cKDTree(vertices[i].numpy())
            distances, indices = tree.query(o_vertices, k=3)

            weights = 1 / (distances + 1e-6)
            weights /= weights.sum(axis=1, keepdims=True) # normalize

            # weighted average of skin weights from the 3 nearest joints
            skin_resampled = np.zeros((o_vertices.shape[0], skin.shape[1]))
            for v in range(o_vertices.shape[0]):
                skin_resampled[v] = weights[v] @ skin[indices[v]]
            
            path = os.path.join(predict_output_dir, cls[i], str(id[i].item()))
            os.makedirs(path, exist_ok=True)
            np.save(os.path.join(path, "predict_skin"), skin_resampled)
            np.save(os.path.join(path, "transformed_vertices"), o_vertices)
    print("finished")

def main():
    """Parse arguments and start training"""
    parser = argparse.ArgumentParser(description='Train a point cloud model')
    
    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='data/data',
                        help='Root directory for the data files')
    parser.add_argument('--predict_data_list', type=str, default='data/data/test_list.txt',
                        help='Path to the prediction data list file')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='skin',
                        choices=['pct', 'pct2', 'custom_pct', 'skeleton'],
                        help='Model architecture to use')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'enhanced'],
                        help='Model type for skeleton model')
    parser.add_argument('--pretrained_model', type=str, default='output/skin/best_model.pkl',
                        help='Path to pretrained model')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    
    # Predict parameters
    parser.add_argument('--predict_output_dir', type=str,default='predict',
                        help='Path to store prediction results')
    
    args = parser.parse_args()
    
    predict(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()