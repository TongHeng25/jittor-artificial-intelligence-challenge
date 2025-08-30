import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys

# Import the PCT model components
from PCT.networks.cls.pct import Point_Transformer, Point_Transformer2, Point_Transformer_Last, SA_Layer, Local_op, sample_and_group
from model.sketon import OurTransformerNet
class SimpleSkeletonModel(nn.Module):
    
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim           = feat_dim
        self.output_channels    = output_channels
        
        self.transformer = Point_Transformer(output_channels=feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_channels),
        )
    
    def execute(self, vertices: jt.Var):
        x = self.transformer(vertices)
        return self.mlp(x)

# Factory function to create models
def create_model(model_name='pct', output_channels=66, **kwargs):
    if model_name == "pct":
        return SimpleSkeletonModel(feat_dim=256, output_channels=output_channels)
    elif model_name == 'skeleton':
        return OurTransformerNet()

    raise NotImplementedError()
