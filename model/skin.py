import jittor as jt
from jittor import nn
from udt import PointTransformerVAE
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
    
class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def get_graph_feature(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        inner = -2 * jt.matmul(x, x.transpose(0,2,1))  # [B, N, N]
        xx = jt.sum(x**2, dim=-1, keepdims=True)       # [B, N, 1]
        pairwise_distance = -xx - inner - xx.transpose(0,2,1)  # [B, N, N]

        _, idx = pairwise_distance.argsort(dim=-1)     # fix here
        idx = idx[:, :, :self.k]                       # [B, N, k]

        idx_base = jt.arange(0, B).view(-1, 1, 1) * N
        idx = (idx + idx_base).reshape(-1)
        x = x.reshape(B*N, C)

        feature = x[idx, :].reshape(B, N, self.k, C)   # [B, N, k, C]
        x = x.reshape(B, N, 1, C).repeat(1, 1, self.k, 1)
        feature = jt.concat([feature - x, x], dim=-1)  # [B, N, k, 2C]
        return feature.permute(0, 3, 1, 2)              # [B, 2C, N, k]


    def execute(self, x):
        # x: [B, N, C]
        feature = self.get_graph_feature(x)
        out = self.mlp(feature)
        out = out.max(dim=-1)  # [B, out_channels, N]
        return out.permute(0, 2, 1)  # [B, N, out_channels]

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super(PositionalEncoding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def execute(self, coords):  # coords: [B, Nq, Nk, 3]
        B, Nq, Nk, _ = coords.shape
        coords = coords.reshape(-1, 3)  # [B*Nq*Nk, 3]
        out = self.mlp(coords)         # [B*Nq*Nk, dim]
        return out.reshape(B, Nq, Nk, -1)  # [B, Nq, Nk, dim]


class JointEncoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=128):
        super(JointEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )

    def execute(self, joints):
        B, N, C = joints.shape
        joints = joints.reshape(-1, C)
        encoded = self.encoder(joints)
        return encoded.reshape(B, N, -1)

class MultiheadAttentionRelPos(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.pos_encoder = PositionalEncoding(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def execute(self, query, key, value, rel_pos=None):
        B, Nq, C = query.shape
        Nk = key.shape[1]

        q = self.q_proj(query).reshape(B, Nq, self.num_heads, self.head_dim).transpose(0,2,1,3)
        k = self.k_proj(key).reshape(B, Nk, self.num_heads, self.head_dim).transpose(0,2,1,3)
        v = self.v_proj(value).reshape(B, Nk, self.num_heads, self.head_dim).transpose(0,2,1,3)

        if rel_pos is not None:
            rel = self.pos_encoder(rel_pos)  # [B, Nq, Nk, head_dim]
            rel = rel.unsqueeze(1)  # [B, 1, Nq, Nk, head_dim]
            qk = jt.matmul(q, k.transpose(0,1,3,2)) * self.scale  # [B, H, Nq, Nk]
            q_rel = (q.unsqueeze(3) * rel).sum(-1)  # [B, H, Nq, Nk]
            attn = jt.nn.softmax(qk + q_rel, dim=-1)
        else:
            attn = jt.nn.softmax(jt.matmul(q, k.transpose(0,1,3,2)) * self.scale, dim=-1)

        attn = self.dropout(attn)
        out = jt.matmul(attn, v).transpose(0,2,1,3).reshape(B, Nq, self.embed_dim)
        return self.out_proj(out), attn

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.attn = MultiheadAttentionRelPos(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Conv1d(dim, ff_dim, 1),
            nn.BatchNorm1d(ff_dim),
            nn.GELU(),
            nn.Conv1d(ff_dim, dim, 1)
        )
        self.attn2 = SkipCrossAttention(dim, num_heads=4, out_dim=dim, dim_q=dim, mlp_ratio=2)

    def execute(self, x, cond, rel_pos=None,pcfeature=None):
        x2, _ = self.attn(x, cond, cond, rel_pos)
        x = x + x2
        x = self.norm1(x)
        x2 = self.ff(x.transpose(1,2)).transpose(1,2)
        x = self.norm2((x + x2))
        if pcfeature is not None:
            x = self.norm3(self.attn2(x, pcfeature)[0] + x)
        return x

class Our_transfomer(nn.Module):
    def __init__(self, dim=256, n_blocks=7):
        super().__init__()
        self.edge_conv = EdgeConv(3, 64, k=20)
        self.vertex_enc = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, dim, 1),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )
        self.joint_encoder = JointEncoder(3, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads=4, ff_dim=512) for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 52)
        )
        self.pcencoder = PointTransformerVAE(in_dim=6, dim=256, latent_dim=128, num_points=1024, depth=5, k=16)
        #self.pcencoder.load("output_VAE_udt/best_model.pkl")
        #self.pcencoder.eval()  # Set to eval mode

    def execute(self, vertices, joints):
        B, N, _ = vertices.shape
        pcfeature = self.pcencoder.encoder_x(vertices) # [B, 512]
        pcfeature = pcfeature.unsqueeze(2).repeat(1, 1, 256)  # [B, 512, 256]
        V_feat = self.edge_conv(vertices)  # [B, N, 64]
        V_feat = self.vertex_enc(V_feat.permute(0,2,1)).permute(0,2,1)  # [B, N, dim]
        J_feat = self.joint_encoder(joints)  # [B, 22, dim]
        rel_pos = (vertices.unsqueeze(2) - joints.unsqueeze(1))  # [B, N, 22, 3]
        for block in self.blocks:
            V_feat = block(V_feat, J_feat, rel_pos,pcfeature)
        weights = self.head(V_feat.reshape(-1, V_feat.shape[-1])).reshape(B, N, 52)
        weights = nn.softmax(weights, dim=-1)
        return weights
