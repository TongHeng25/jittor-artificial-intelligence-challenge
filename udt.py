import os

import jittor as jt
import numpy as np
import os
import argparse
import time
import random
import math
from jittor import nn
from jittor import optim

from dataset.dataset import get_dataloader_udt, transform

from dataset.sampler import SamplerMix

from PCT.misc.ops import knn_point, index_points


import jittor as jt
from jittor import nn



#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_B_4(pretrained=False, **kwargs):
    return DiT(depth=7, hidden_size=384, num_heads=12, **kwargs)



    
# ========== DGCNN EdgeConv ==========
class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, N, k=20,downsample = False):
        super().__init__()
        self.k = k
        self.n = N
        self.downsample = downsample
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.linear = None
        if downsample:
            self.linear = nn.Sequential(nn.Linear(self.n, self.n // 2),nn.GELU())
    def execute(self, x, xyz):  # x: [B, C, N]
        B, C, N = x.shape
        idx = knn_point(self.k, xyz, xyz)  # [B, N, K]
        x = x.permute(0, 2, 1)  # [B, N, C]
        neighbor = index_points(x, idx)  # [B, N, K, C]
        x = x.unsqueeze(2).expand(-1, -1, self.k, -1)  # [B, N, K, C]
        edge_feature = jt.concat([x, neighbor - x], dim=-1)  # [B, N, K, 2C]
        edge_feature = edge_feature.permute(0, 3, 1, 2)  # [B, 2C, N, K]
        out = self.conv(edge_feature).max(dim=-1)  # [B, out_channels, N]
        if self.downsample:
            out = self.linear(out)
        return out

class pccencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = EdgeConv(3, 128, 1024, 16)
        self.block2 = EdgeConv(128, 384, 1024, 16,True)

    def execute(self, pcc):
        
        pos = pcc.transpose(0, 2, 1)  # [B, N, 3]
        
        feat1 = self.block1(pcc, pos)  # [B, 64, N]
        feat2 = self.block2(feat1, pos)  # [B, d_model, N]
        feat = feat2.transpose(0, 2, 1)  # [B, N, d_model]
        return feat

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, x):  # x: [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, D]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = nn.softmax(attn, dim=-1)
        weights = attn
        attn = self.attn_drop(attn)

        out = (attn @ v)  # [B, H, N, D]
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, weights


class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.shape[1]

        q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads).transpose(0, 2, 1, 3)
        k = self.k(k).reshape(B, NK, self.num_heads, C // self.num_heads).transpose(0, 2, 1, 3)
        v = self.v(v).reshape(B, NK, self.num_heads, C // self.num_heads).transpose(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = nn.softmax(attn, dim=-1)
        weights = attn
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SkipCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, out_dim, dim_q=None, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0,
                 attn_drop=0.0, drop_path_rate=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(dim, out_dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def execute(self, q, v):
        norm_q = self.norm1(q)
        q_1, _ = self.self_attn(norm_q)
        q = q + q_1
        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        q_2, weights = self.attn(norm_q, norm_v)
        q = q + q_2
        q = q + self.mlp(self.norm2(q))
        return q, weights

class DiT(nn.Module):
    def __init__(self, input_size=32, in_channels=3, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, learn_sigma=False):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.input_size = input_size

        self.pos_embed = self.position_encoding(384, 512)
        self.blocks = nn.ModuleList([
            SkipCrossAttention(dim=384, out_dim=384, num_heads=12, mlp_ratio=4,
                               qkv_bias=True, drop=0.1, attn_drop=0.1) for _ in range(depth)
        ])

        self.x_embedder = pccencoder()
        self.final_layer1 = nn.Sequential(
            nn.Conv1d(384, 128, 1), nn.SiLU(),
            nn.Conv1d(128, 32, 1), nn.SiLU(),
            nn.Conv1d(32, 1, 1)
        )
        self.final_layer2 = nn.Sequential(
            nn.Conv1d(384, 128, 1), nn.SiLU(),
            nn.Conv1d(128, 32, 1), nn.SiLU(),
            nn.Conv1d(32, 1, 1)
        )


    def position_encoding(self, d_model, length):
        pe = jt.zeros((length, d_model))
        position = jt.arange(0, length).unsqueeze(1)
        div_term = jt.exp(jt.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = jt.sin(position.float() * div_term)
        pe[:, 1::2] = jt.cos(position.float() * div_term)
        return pe

    def execute(self, x):
        x = self.x_embedder(x.transpose(1, 2))
        x = x + self.pos_embed.unsqueeze(0)
        for i in range(len(self.blocks)):
            x, _ = self.blocks[i](x, x)
        z_mean = self.final_layer1(x.transpose(1, 2)).squeeze(1)
        z_logvar = self.final_layer2(x.transpose(1, 2)).squeeze(1)
        return z_mean, z_logvar

    
class Decoder(nn.Module):
    def __init__(self, latent_dim=128, num_points=1024):
        super().__init__()
        self.num_points = num_points
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, num_points * 3)
        )

    def execute(self, z):  # [B, latent_dim]
        out = self.mlp(z)  # [B, num_points * 3]
        return out.reshape(-1, self.num_points, 3)  # [B, N, 3]

    
class PointTransformerVAE(nn.Module):
    def __init__(self, in_dim=6, dim=128, latent_dim=64, num_points=1024, depth=4, k=16):
        super().__init__()
        self.encoder = DiT_B_4()
        self.decoder = Decoder(dim,num_points)

    def execute(self, x):
        # x: [B, N, in_dim]
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)  # [B, latent_dim]
        recon = self.decoder(z)
        return recon, z_mean, z_logvar
    
    def reparameterize(self, mu, logvar):
        std = jt.exp(0.5 * logvar)
        eps = jt.randn_like(std)
        return mu + eps * std
    
    def encoder_x(self, x):
        # x: [B, N, in_dim]
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)
        return z  # [B, latent_dim]
    
    
def vae_loss(z_mean, z_logvar):
    # recon, x: [B, N, 6]

    kl_loss = -0.5 * jt.mean(1 + z_logvar - z_mean ** 2 - jt.exp(z_logvar))
    return kl_loss

def chamfer_distance(p1, p2):
    # p1, p2: [B, N, 3]
    dist1 = ((p1.unsqueeze(2) - p2.unsqueeze(1)) ** 2).sum(-1)  # [B, N, N]
    min1 = dist1.min(dim=2)  # [B, N]
    min2 = dist1.min(dim=1)  # [B, N]
    loss = min1.mean(dim=1) + min2.mean(dim=1)  # [B]
    return loss.mean()

def save_points_to_obj(points, output_file):
    with open(output_file, 'w') as f:
        for point in points:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")
            

jt.flags.use_cuda = 1

def train(args):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    
    def log_message(message):
        """Helper function to log messages to file and print to console"""
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    # Log training parameters
    log_message(f"Starting training with parameters: {args}")
    
    # Create model
    model = PointTransformerVAE(in_dim=6, dim=512, latent_dim=128, num_points=1024, depth=5, k=16)
    #model.load("/data3/jitu2025/jittor-comp-human-main/output_VAE_udt/best_model.pkl")
    
    # Load pre-trained model if specified
    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)
    
    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Create loss function
    

    scheduler = jt.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    # Create dataloaders
    train_loader = get_dataloader_udt(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=SamplerMix(num_samples=1024, vertex_samples=512),
        transform=transform,
    )

   
    val_loader = None
    
    # Training loop
    best_loss = 99999999
    all_loss = 0
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss_mse = 0.0
        train_loss_l1 = 0.0
        all_loss = 0
        start_time = time.time()
        vis_gt = None
        vis_pred = None
        for batch_idx, data in enumerate(train_loader):
            # Get data and labels
            vertices = data['vertices']
            normals = data['normals']
            #x = jt.concat([vertices, normals], dim=-1)  # [B, N, 6]
            x = vertices
            vertices: jt.Var
            joints: jt.Var
            skin: jt.Var
            recon, z_mean, z_logvar  = model(x)
            #vis_gt = vertices[0].numpy()
            #vis_pred = recon[0].numpy()
            kl_loss = vae_loss(z_mean, z_logvar)
            cd_loss = chamfer_distance(recon, vertices)
            loss = kl_loss + cd_loss
            # Backward pass and optimize
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            # Calculate statistics
            train_loss_mse += loss.item()
            all_loss += loss.item()

            
            # Print progress
            if jt.rank == 0:
                if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                    log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                            f"Loss: {loss.item():.4f}")
        #save_points_to_obj(vis_gt, args.output_dir+f"gt_{epoch+1}.obj")
        #save_points_to_obj(vis_pred, args.output_dir+f"pred_{epoch+1}.obj")
        # Calculate epoch statistics
        if jt.rank == 0:
            train_loss_mse /= len(train_loader)
            epoch_time = time.time() - start_time
            
            log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Train Loss: {train_loss_mse:.4f} "
                    f"Time: {epoch_time:.2f}s "
                    f"LR: {optimizer.lr:.6f}")

            
                
            # Save best model
            if all_loss < best_loss:
                best_loss = all_loss
                model_path = os.path.join(args.output_dir, 'best_model.pkl')
                model.save(model_path)
                log_message(f"Saved best model with loss {best_loss:.4f} to {model_path}")
            
            '''# Save checkpoint
            if (epoch + 1) % args.save_freq == 0:
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
                model.save(checkpoint_path)
                log_message(f"Saved checkpoint to {checkpoint_path}")'''
            
        scheduler.step()
    
    '''# Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
    model.save(final_model_path)
    log_message(f"Training completed. Saved final model to {final_model_path}")'''
    
    return model, best_loss

def main():
    """Parse arguments and start training"""
    parser = argparse.ArgumentParser(description='Train a point cloud model')
    
    # Dataset parameters
    parser.add_argument('--train_data_list', type=str, default='data/data/all.txt',
                        help='Path to the training data list file')
    parser.add_argument('--val_data_list', type=str, default='data/data/val_list.txt',
                        help='Path to the validation data list file')
    parser.add_argument('--data_root', type=str, default='data/data',
                        help='Root directory for the data files')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='Our_transfomer',
                        choices=['pct', 'pct2', 'custom_pct', 'skeleton'],
                        help='Model architecture to use')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'enhanced'],
                        help='Model type for skeleton model')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='Path to pretrained model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output_VAE_udt_new/',
                        help='Directory to save output files')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='Save frequency')
    parser.add_argument('--val_freq', type=int, default=1,
                        help='Validation frequency')
    
    args = parser.parse_args()
    
    # Start training
    train(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()