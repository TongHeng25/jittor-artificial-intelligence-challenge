import jittor as jt
from jittor import nn
import numpy as np
from PCT.misc.ops import knn_point, index_points

#from pointVAE import PointTransformerVAE
from udt import PointTransformerVAE
# ========== 相对位置编码 ==========
class RelativePositionEncoding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
    def execute(self, rel_pos):  # [B, N, K, 3]
        return self.mlp(rel_pos)  # [B, N, K, out_dim]

# ========== 局部 Self-Attention ==========
class LocalSelfAttention(nn.Module):
    def __init__(self, d_model, k=16):
        super().__init__()
        self.k = k
        self.q_proj = nn.Sequential(nn.Linear(d_model, d_model * 2),
                                    nn.GELU(),nn.Linear(d_model * 2, d_model))
        self.k_proj = nn.Sequential(nn.Linear(d_model, d_model * 2),
                                    nn.GELU(),nn.Linear(d_model * 2, d_model))
        self.v_proj = nn.Sequential(nn.Linear(d_model, d_model * 2),
                                    nn.GELU(),nn.Linear(d_model * 2, d_model))
        self.rel_enc = RelativePositionEncoding(3, d_model)
        self.softmax = nn.Softmax(dim=-1)
    def execute(self, x, xyz):  # x: [B, N, C], xyz: [B, N, 3]
        B, N, C = x.shape
        idx = knn_point(self.k, xyz, xyz)  # [B, N, K]
        neighbor_feat = index_points(x, idx)  # [B, N, K, C]
        neighbor_xyz = index_points(xyz, idx)  # [B, N, K, 3]
        rel_pos = neighbor_xyz - xyz.unsqueeze(2)  # [B, N, K, 3]
        rel_feat = self.rel_enc(rel_pos)  # [B, N, K, C]

        q = self.q_proj(x).unsqueeze(2)  # [B, N, 1, C]
        k = self.k_proj(neighbor_feat + rel_feat)  # [B, N, K, C]
        v = self.v_proj(neighbor_feat + rel_feat)  # [B, N, K, C]

        attn = self.softmax((q * k).sum(-1, keepdims=True))  # [B, N, K, 1]
        out = (attn * v).sum(dim=2)  # [B, N, C]
        return out

# ========== DGCNN EdgeConv ==========
class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    def execute(self, x, xyz):  # x: [B, C, N]
        B, C, N = x.shape
        idx = knn_point(self.k, xyz, xyz)  # [B, N, K]
        x = x.permute(0, 2, 1)  # [B, N, C]
        neighbor = index_points(x, idx)  # [B, N, K, C]
        x = x.unsqueeze(2).expand(-1, -1, self.k, -1)  # [B, N, K, C]
        edge_feature = jt.concat([x, neighbor - x], dim=-1)  # [B, N, K, 2C]
        edge_feature = edge_feature.permute(0, 3, 1, 2)  # [B, 2C, N, K]
        out = self.conv(edge_feature).max(dim=-1)  # [B, out_channels, N]
        return out

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

        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(dim, out_dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def execute(self, q, v):
        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        q_2, weights = self.attn(norm_q, norm_v)
        q = q + q_2
        q = q + self.mlp(self.norm2(q))
        return q, weights

class TransformerLayer(nn.Module):
    def __init__(self, d_model, k=16):
        super().__init__()
        self.attn = SkipCrossAttention(d_model, num_heads=4, out_dim=d_model, dim_q=d_model, mlp_ratio=2)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.attn2 = SkipCrossAttention(d_model, num_heads=4, out_dim=d_model, dim_q=d_model, mlp_ratio=2)

    def execute(self, x, pcfeature):  # x: [B, N, C]
        attn_out,_ = self.attn(x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ffn(x)
        x1 = self.norm2(x + ff_out)
        x , _ = self.attn2(x1, pcfeature)  # pcfeature: [B, N, C]
        return self.norm3(x+x1)

# ========== 主干网络 ==========
class OurTransformerNet(nn.Module):
    def __init__(self, k=16, d_model=256, num_layers=7):
        super().__init__()
        self.dgcnn1 = EdgeConv(3, 128, k)
        self.dgcnn2 = EdgeConv(128, d_model, k)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerLayer(d_model, k=16) for _ in range(num_layers)
        ])
        
        self.joint_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 52)
        )
        self.joint_regressor = nn.Linear(d_model, 3)
        
        self.pcencoder = PointTransformerVAE(in_dim=6, dim=256, latent_dim=128, num_points=1024, depth=5, k=16)
        #self.pcencoder.load("output_VAE_udt/best_model.pkl")
        #self.pcencoder.eval()  # Set to eval mode

    def execute(self, xyz, xyzwnor):  # xyz: [B, 3, N]
        
        B, _, N = xyz.shape
        pos = xyz.transpose(0, 2, 1)  # [B, N, 3]
        pcfeature = self.pcencoder.encoder_x(pos) # [B, 512]
        pcfeature = pcfeature.unsqueeze(2).repeat(1, 1, 256)  # [B, 512, 256]
        feat1 = self.dgcnn1(xyz, pos)  # [B, 64, N]
        feat2 = self.dgcnn2(feat1, pos)  # [B, d_model, N]
        feat = feat2.transpose(0, 2, 1)  # [B, N, d_model]

        for block in self.transformer_blocks:
            feat = block(feat, pcfeature)  # 每层 transformer

        joint_weights = self.joint_head(feat)  # [B, N, 52]
        joint_weights = nn.softmax(joint_weights, dim=1)

        pred_xyz = self.joint_regressor(feat)  # [B, N, 3]
        joint_pos = jt.bmm(joint_weights.transpose(0, 2, 1), pred_xyz)  # [B, 52, 3]
        
        return joint_pos
