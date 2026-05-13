"""
HAT-Match+: Enhanced Hybrid Attention Transformer for Two-View Correspondence Pruning
Journal version (IEEE TPAMI) extending HAT-Match (ECAI 2025)

Key improvements over HAT-Match:
  1. SA+  : Geometry-Biased Multi-Head Attention (replaces LCT)
  2. CSCA : Channel-Spatial Coupled Attention (replaces SEAttention)
  3. MGA  : Multi-hop Graph Attention (replaces GG_Block)
  4. MCA  : Motion Consistency Attention (NEW)
  5. VCA  : View Cross-Attention (NEW)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import batch_episym
from einops import rearrange


# ====================================================================
# Shared utility modules (same as HAT-Match)
# ====================================================================

class FourierPositionEncoding(nn.Module):
    def __init__(self, num_freqs=8):
        super().__init__()
        self.num_freqs = num_freqs
        freqs = 2 ** torch.arange(num_freqs, dtype=torch.float32) * math.pi
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        # x: [B, C, H, W]
        x_expanded = x.unsqueeze(-1) * self.freqs
        sin_f = torch.sin(x_expanded)
        cos_f = torch.cos(x_expanded)
        out = torch.cat([sin_f, cos_f], dim=-1)
        return rearrange(out, 'b c h w f -> b (c f) h w')


class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(nn.Conv2d(inchannel, outchannel, (1, 1)))
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre else x
        out = self.left(x)
        return torch.relu(out + x1)


class trans(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1, self.dim2 = dim1, dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        super().__init__()
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1), trans(1, 2))
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points), nn.ReLU(), nn.Conv2d(points, points, kernel_size=1))
        self.conv3 = nn.Sequential(
            trans(1, 2), nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        return out + (self.shot_cut(x) if self.shot_cut else x)


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        super().__init__()
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3), nn.BatchNorm2d(in_channel), nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x):
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2).squeeze(3)
        return torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        super().__init__()
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3), nn.BatchNorm2d(in_channel), nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        embed = self.conv(x_up)
        S = torch.softmax(embed, dim=1).squeeze(3)
        return torch.matmul(x_down.squeeze(3), S).unsqueeze(3)


def knn(x, k):
    """x: (B, C, N) -> idx: (B, N, k)"""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def gather_neighbors(feat, knn_idx):
    """
    feat: (B, N, C) or (B, N, D)
    knn_idx: (B, N, k)
    Returns: (B, N, k, C)
    """
    B, N, C = feat.shape
    k = knn_idx.shape[-1]
    device = feat.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx_flat = (knn_idx + idx_base).view(-1)
    return feat.contiguous().view(B * N, C)[idx_flat].view(B, N, k, C)


class MaxDGCNN_Block(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super().__init__()
        self.knn_num = knn_num
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, (1, 1)),
            nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, (1, 1)),
            nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True))

    def forward(self, x):
        """x: (B, 2C, N, k) -> (B, C, N, 1)"""
        out = self.conv(x)
        return out.max(dim=-1, keepdim=True)[0]


def build_graph_feature(x, knn_idx):
    """
    x: (B, C, N, 1)
    knn_idx: (B, N, k)
    Returns: (B, 2C, N, k)
    """
    B, C, N, _ = x.shape
    k = knn_idx.shape[-1]
    x_sq = x.squeeze(-1).transpose(1, 2).contiguous()  # (B, N, C)
    neighbors = gather_neighbors(x_sq, knn_idx)  # (B, N, k, C)
    x_rep = x_sq.unsqueeze(2).expand_as(neighbors)   # (B, N, k, C)
    edge = torch.cat([x_rep, x_rep - neighbors], dim=3)  # (B, N, k, 2C)
    return edge.permute(0, 3, 1, 2).contiguous()  # (B, 2C, N, k)


# ====================================================================
# Module 1: SA+ (Geometry-Biased Multi-Head Attention)
# ====================================================================

class GeometryBiasedAttention(nn.Module):
    """
    Replaces original LCT (Self-Attention).
    Key changes:
      - Shared KNN graph (computed once, not 3 separate DGCNN)
      - Only V goes through DGCNN aggregation
      - Relative geometric position encoding as attention bias
      - Local sparse attention (O(N*k) instead of channel-mixing)
    """
    def __init__(self, channels, num_heads=4, k_num=9):
        super().__init__()
        self.k_num = k_num
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_proj = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 1)),
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels), nn.ReLU())
        self.k_proj = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 1)),
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels), nn.ReLU())
        self.v_proj = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 1)),
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels), nn.ReLU())

        # Shared DGCNN only for V
        self.v_gcn = MaxDGCNN_Block(knn_num=k_num, in_channel=channels)

        # Geometric bias: 6-dim relative features -> per-head bias
        self.geo_bias_net = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, num_heads))

        self.project_out = nn.Conv2d(channels, channels, (1, 1))

    def forward(self, x, coords=None):
        """
        x: (B, C, N, 1)
        coords: (B, N, 4) - [x, y, u, v] for geometric bias
        """
        B, C, N, _ = x.shape
        H, d = self.num_heads, self.head_dim

        # Shared KNN
        knn_idx = knn(x.squeeze(-1), k=self.k_num)  # (B, N, k)
        k_num = self.k_num

        # Projections
        q = self.q_proj(x).squeeze(-1).transpose(1, 2)  # (B, N, C)
        k_f = self.k_proj(x).squeeze(-1).transpose(1, 2)  # (B, N, C)

        # V with graph aggregation
        v = self.v_proj(x)
        v = self.v_gcn(build_graph_feature(v, knn_idx))  # (B, C, N, 1)
        v = v.squeeze(-1).transpose(1, 2)  # (B, N, C)

        # Gather K, V neighbors
        k_neigh = gather_neighbors(k_f, knn_idx)  # (B, N, k, C)
        v_neigh = gather_neighbors(v, knn_idx)  # (B, N, k, C)

        # Multi-head reshape
        q_mh = q.view(B, N, H, d).unsqueeze(3)  # (B, N, H, 1, d)
        k_mh = k_neigh.view(B, N, k_num, H, d).permute(0, 1, 3, 2, 4)  # (B,N,H,k,d)
        v_mh = v_neigh.view(B, N, k_num, H, d).permute(0, 1, 3, 2, 4)  # (B,N,H,k,d)

        # Local attention: (B,N,H,1,d) @ (B,N,H,d,k) = (B,N,H,1,k)
        attn = torch.matmul(q_mh, k_mh.transpose(-2, -1)) / math.sqrt(d)
        attn = attn.squeeze(3)  # (B, N, H, k)

        # Add geometric bias
        if coords is not None:
            geo_feat = self._geo_features(coords, knn_idx)  # (B, N, k, 6)
            geo_bias = self.geo_bias_net(geo_feat)  # (B, N, k, H)
            attn = attn + geo_bias.permute(0, 1, 3, 2)  # (B, N, H, k)

        attn = attn.softmax(dim=-1)

        # Weighted sum: (B,N,H,1,k) @ (B,N,H,k,d) = (B,N,H,1,d)
        out = torch.matmul(attn.unsqueeze(3), v_mh).squeeze(3)  # (B,N,H,d)
        out = out.reshape(B, N, C).transpose(1, 2).unsqueeze(-1)  # (B,C,N,1)
        return self.project_out(out) + x

    def _geo_features(self, coords, knn_idx):
        """Compute 6-dim relative geometric features for each KNN pair."""
        B, N, _ = coords.shape
        motion = coords[:, :, 2:] - coords[:, :, :2]  # (B, N, 2)
        neigh_coords = gather_neighbors(coords, knn_idx)  # (B, N, k, 4)
        neigh_motion = gather_neighbors(motion, knn_idx)  # (B, N, k, 2)

        delta_xy = neigh_coords[:, :, :, :2] - coords[:, :, :2].unsqueeze(2)
        delta_mot = neigh_motion - motion.unsqueeze(2)
        disp_mag = torch.norm(delta_xy, dim=-1, keepdim=True).clamp(min=1e-8)
        cos_ang = F.cosine_similarity(
            motion.unsqueeze(2).expand_as(neigh_motion) + 1e-8,
            neigh_motion + 1e-8, dim=-1).unsqueeze(-1)
        return torch.cat([delta_xy, delta_mot, disp_mag, cos_ang], dim=-1)


class SA_Plus_Block(nn.Module):
    """SA+ with FFN, wrapping GeometryBiasedAttention."""
    def __init__(self, channels, num_heads=4, k_num=9):
        super().__init__()
        self.attn = GeometryBiasedAttention(channels, num_heads, k_num)
        self.ffn = ResNet_Block(channels, channels, pre=False)

    def forward(self, x, coords=None):
        x = x + self.attn(x, coords)
        return x + self.ffn(x)


# ====================================================================
# Module 2: CSCA (Channel-Spatial Coupled Attention)
# ====================================================================

class CSCA(nn.Module):
    """
    Replaces original SEAttention.
    Key changes:
      - Dual-branch pooling (Avg + Max) for channel attention
      - Spatial attention gate (per-correspondence)
      - Coupled channel × spatial weighting
    """
    def __init__(self, channel, reduction=2):
        super().__init__()
        mid = channel // reduction
        # Dual-branch channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, channel, bias=False))

        # Spatial attention gate
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1, 1)),
            nn.Sigmoid())

        self.ffn = ResNet_Block(channel, channel, pre=False)

    def forward(self, x):
        """x: (B, C, N, 1)"""
        b, c, n, _ = x.shape

        # Channel attention (dual-branch)
        avg_desc = self.avg_pool(x).view(b, c)
        max_desc = self.max_pool(x).view(b, c)
        ch_w = torch.sigmoid(self.fc(avg_desc) + self.fc(max_desc))  # (B, C)
        ch_w = ch_w.view(b, c, 1, 1)

        # Spatial attention (per-correspondence)
        sp_avg = torch.mean(x, dim=1, keepdim=True)  # (B, 1, N, 1)
        sp_max = torch.max(x, dim=1, keepdim=True)[0]  # (B, 1, N, 1)
        sp_w = self.spatial_conv(torch.cat([sp_avg, sp_max], dim=1))  # (B, 1, N, 1)

        # Coupled: channel × spatial
        out = x * ch_w * sp_w
        return out + self.ffn(out)


# ====================================================================
# Module 3: MGA (Multi-hop Graph Attention)
# ====================================================================

class MGA_Block(nn.Module):
    """
    Replaces original GG_Block.
    Key changes:
      - Gradient-enabled adjacency (removed torch.no_grad())
      - Combines weight-based and feature-based adjacency
      - Proper Laplacian normalization (D^{-1/2} A D^{-1/2})
      - Two-hop graph propagation
    """
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        # Learnable mixing parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # Two-hop propagation weights
        self.W1 = nn.Linear(in_channel, in_channel)
        self.W2 = nn.Linear(in_channel, in_channel)
        self.norm1 = nn.BatchNorm1d(in_channel)
        self.norm2 = nn.BatchNorm1d(in_channel)

    def forward(self, x, w):
        """
        x: (B, C, N, 1)
        w: (B, N) - local confidence weights
        """
        B, C, N, _ = x.shape

        # --- Build adjacency matrix (WITH gradient) ---
        w_act = torch.relu(torch.tanh(w)).unsqueeze(-1)  # (B, N, 1)
        A_w = torch.bmm(w_act, w_act.transpose(1, 2))  # (B, N, N)

        # Feature-based adjacency (detached for stability)
        feat = x.squeeze(-1).transpose(1, 2).contiguous()  # (B, N, C)
        feat_norm = F.normalize(feat.detach(), dim=-1)
        A_f = torch.bmm(feat_norm, feat_norm.transpose(1, 2))  # (B, N, N)

        # Combine
        alpha = torch.sigmoid(self.alpha)
        A = alpha * A_w + (1.0 - alpha) * A_f

        # --- Laplacian normalization ---
        A_tilde = F.softmax(A, dim=-1) + torch.eye(N, device=x.device).unsqueeze(0)
        D_tilde = torch.sum(A_tilde, dim=-1).clamp(min=1e-8)  # (B, N)
        D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(D_tilde))  # (B, N, N)
        L = torch.bmm(torch.bmm(D_inv_sqrt, A_tilde), D_inv_sqrt)  # (B, N, N)

        # --- Two-hop propagation ---
        F_in = feat  # (B, N, C)
        F1 = torch.bmm(L, self.W1(F_in))  # (B, N, C)
        F1 = F.relu(self.norm1(F1.transpose(1, 2)).transpose(1, 2))
        F2 = torch.bmm(L, self.W2(F1))  # (B, N, C)
        F2 = self.norm2(F2.transpose(1, 2)).transpose(1, 2)

        out = F2.transpose(1, 2).unsqueeze(-1)  # (B, C, N, 1)
        return out


# ====================================================================
# Module 4: MCA (Motion Consistency Attention) — NEW
# ====================================================================

class MotionConsistencyAttention(nn.Module):
    """
    Novel attention mechanism exploiting the geometric prior that
    inlier correspondences exhibit locally smooth motion fields.
    Attention weights are derived from motion vector similarity.
    """
    def __init__(self, channels, k_num=9):
        super().__init__()
        self.k_num = k_num
        # Learnable temperature for motion similarity
        self.sigma = nn.Parameter(torch.tensor(1.0))
        # Value projection
        self.v_proj = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 1)),
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels), nn.ReLU())
        # Gate fusion
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, (1, 1)),
            nn.Sigmoid())
        self.ffn = ResNet_Block(channels, channels, pre=False)

    def forward(self, x, coords):
        """
        x: (B, C, N, 1),  coords: (B, N, 4) [x,y,u,v]
        """
        B, C, N, _ = x.shape
        motion = coords[:, :, 2:] - coords[:, :, :2]  # (B, N, 2)

        # Build KNN graph in feature space
        knn_idx = knn(x.squeeze(-1), k=self.k_num)  # (B, N, k)
        neigh_motion = gather_neighbors(motion, knn_idx)  # (B, N, k, 2)

        # Motion consistency scores
        motion_diff = neigh_motion - motion.unsqueeze(2)  # (B, N, k, 2)
        dist_sq = (motion_diff ** 2).sum(dim=-1)  # (B, N, k)
        sigma_sq = (self.sigma ** 2).clamp(min=1e-4)
        attn_w = torch.exp(-dist_sq / (2.0 * sigma_sq))  # (B, N, k)
        attn_w = attn_w / (attn_w.sum(dim=-1, keepdim=True) + 1e-8)

        # Value features
        v = self.v_proj(x).squeeze(-1).transpose(1, 2)  # (B, N, C)
        v_neigh = gather_neighbors(v, knn_idx)  # (B, N, k, C)

        # Weighted aggregation
        agg = (attn_w.unsqueeze(-1) * v_neigh).sum(dim=2)  # (B, N, C)
        agg = agg.transpose(1, 2).unsqueeze(-1)  # (B, C, N, 1)

        # Gate fusion with original features
        gate_val = self.gate(torch.cat([x, agg], dim=1))  # (B, C, N, 1)
        out = gate_val * x + (1.0 - gate_val) * agg
        return out + self.ffn(out)


# ====================================================================
# Module 5: VCA (View Cross-Attention) — NEW
# ====================================================================

class ViewCrossAttention(nn.Module):
    """
    Decomposes correspondence features into view-specific representations
    and applies cross-attention (view1<->view2) using local KNN for efficiency.
    Inspired by SuperGlue/LoFTR but adapted for correspondence pruning.
    """
    def __init__(self, channels, num_heads=4, k_num=9):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.k_num = k_num

        self.view1_embed = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 1)),
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels), nn.ReLU())
        self.view2_embed = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 1)),
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels), nn.ReLU())

        # Cross-attention projections
        self.q_proj = nn.Conv2d(channels, channels, (1, 1))
        self.k_proj = nn.Conv2d(channels, channels, (1, 1))
        self.v_proj = nn.Conv2d(channels, channels, (1, 1))

        self.merge = nn.Sequential(
            nn.Conv2d(channels * 2, channels, (1, 1)),
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels), nn.ReLU())

    def _local_cross_attn(self, q_feat, kv_feat):
        """Local cross-attention using KNN of kv_feat."""
        B, C, N, _ = q_feat.shape
        H, d, k = self.num_heads, self.head_dim, self.k_num

        knn_idx = knn(kv_feat.squeeze(-1), k=k)  # (B, N, k)

        q = self.q_proj(q_feat).squeeze(-1).transpose(1, 2)  # (B, N, C)
        k_f = self.k_proj(kv_feat).squeeze(-1).transpose(1, 2)
        v_f = self.v_proj(kv_feat).squeeze(-1).transpose(1, 2)

        k_neigh = gather_neighbors(k_f, knn_idx)  # (B, N, k, C)
        v_neigh = gather_neighbors(v_f, knn_idx)

        q_mh = q.view(B, N, H, d).unsqueeze(3)  # (B,N,H,1,d)
        k_mh = k_neigh.view(B, N, k, H, d).permute(0, 1, 3, 2, 4)
        v_mh = v_neigh.view(B, N, k, H, d).permute(0, 1, 3, 2, 4)

        attn = torch.matmul(q_mh, k_mh.transpose(-2, -1)) / math.sqrt(d)
        attn = attn.squeeze(3).softmax(dim=-1)
        out = torch.matmul(attn.unsqueeze(3), v_mh).squeeze(3)
        return out.reshape(B, N, C).transpose(1, 2).unsqueeze(-1)

    def forward(self, x):
        """x: (B, C, N, 1)"""
        f1 = self.view1_embed(x)
        f2 = self.view2_embed(x)
        cross_12 = self._local_cross_attn(f1, f2)  # v1 queries v2
        cross_21 = self._local_cross_attn(f2, f1)  # v2 queries v1
        merged = self.merge(torch.cat([cross_12, cross_21], dim=1))
        return merged + x


# ====================================================================
# Adaptive Gate Fusion
# ====================================================================

class AdaptiveGateFusion(nn.Module):
    """Fuses outputs of SA+, CSCA, MCA with per-point learned gates."""
    def __init__(self, channels, num_branches=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * num_branches, num_branches, (1, 1)),
            nn.Softmax(dim=1))
        self.proj = nn.Conv2d(channels, channels, (1, 1))

    def forward(self, *features):
        """features: list of (B, C, N, 1) tensors"""
        cat = torch.cat(features, dim=1)
        weights = self.gate(cat)  # (B, num_branches, N, 1)
        out = sum(weights[:, i:i+1] * features[i] for i in range(len(features)))
        return self.proj(out)


# ====================================================================
# CL_Block_Plus: Enhanced CL_Block with EHAB
# ====================================================================

class CL_Block_Plus(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=9, sampling_rate=0.5, clusters=500, use_fourier=False, ablate_depth=0):
        super(CL_Block_Plus, self).__init__()
        self.initial = initial
        self.use_fourier = use_fourier
        self.ablate_depth = ablate_depth
        
        if initial:
            # 4 coords + 2 motion coords
            self.in_channel = (4 * 2 * 8 + 2) if use_fourier else 6
        else:
            self.in_channel = 8
            
        self.out_channel = out_channel
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate
        
        if use_fourier and initial:
            self.fourier_enc = FourierPositionEncoding(num_freqs=8)

        # Input projection
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, out_channel, (1, 1)),
            nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

        # Coarse graph — Pool/Unpool layer 1
        self.down_1 = diff_pool(out_channel, clusters)
        self.l1 = nn.Sequential(*[OAFilter(out_channel, clusters) for _ in range(2)])
        self.up_1 = diff_unpool(out_channel, clusters)

        # Coarse graph — Pool/Unpool layer 2
        self.down_2 = diff_pool(out_channel, clusters)
        self.l2 = nn.Sequential(*[OAFilter(out_channel, clusters) for _ in range(2)])
        self.up_2 = diff_unpool(out_channel, clusters)

        # Embedding blocks
        self.embed_00 = nn.Sequential(*[ResNet_Block(out_channel, out_channel) for _ in range(4)])
        self.embed_002 = nn.Sequential(*[ResNet_Block(out_channel, out_channel) for _ in range(4)])

        # ===== EHAB Layer 1 =====
        self.vca_1 = ViewCrossAttention(out_channel, num_heads=4, k_num=k_num)
        self.sa_plus_1 = SA_Plus_Block(out_channel, num_heads=4, k_num=k_num)
        self.csca_1 = CSCA(out_channel, reduction=2)
        self.mca_1 = MotionConsistencyAttention(out_channel, k_num=k_num)
        self.gate_1 = AdaptiveGateFusion(out_channel, num_branches=3)

        # ===== EHAB Layer 2 (no VCA — cross-view info already captured in L1) =====
        self.sa_plus_2 = SA_Plus_Block(out_channel, num_heads=4, k_num=k_num)
        self.csca_2 = CSCA(out_channel, reduction=2)
        self.mca_2 = MotionConsistencyAttention(out_channel, k_num=k_num)
        self.gate_2 = AdaptiveGateFusion(out_channel, num_branches=3)

        # ===== MGA (replaces GG_Block at the end) =====
        self.mga = MGA_Block(out_channel)

        self.embed_1 = nn.Sequential(ResNet_Block(out_channel, out_channel))
        self.linear_0 = nn.Conv2d(out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(out_channel, 1, (1, 1))

        if self.predict:
            self.embed_2 = ResNet_Block(out_channel, out_channel)
            self.linear_2 = nn.Conv2d(out_channel, 2, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N, _ = x.size()
        indices = indices[:, :int(N * self.sr)]
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)
            w_out = torch.gather(weights, dim=-1, index=indices)
        indices = indices.view(B, 1, -1, 1)
        if not predict:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1))
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y):
        B, _, N, _ = x.size()
        coords = x[:, 0, :, :4]  # (B, N, 4) — [x, y, u, v]
        
        if self.initial and self.use_fourier:
            # x is [B, 1, N, 6]. out is [B, 6, N, 1].
            out = x.transpose(1, 3).contiguous()
            b, c, n, _ = out.size()
            c_coords = out[:, :4, :, :]
            c_motion = out[:, 4:, :, :]
            f_coords = self.fourier_enc(c_coords)
            out = torch.cat([f_coords, c_motion], dim=1)
            out = self.conv(out)
        else:
            out = x.transpose(1, 3).contiguous()
            out = self.conv(out)  # (B, 128, N, 1)

        # ---- Layer 1: Coarse Graph + EHAB ----
        x_down = self.down_1(out)
        x2 = self.l1(x_down)
        x_up = self.up_1(out, x2)
        out = self.embed_00(x_up)

        # VCA (cross-view enrichment)
        if hasattr(self, 'vca_1') and self.ablate_depth < 2:
            out = self.vca_1(out)

        # Parallel: SA+, CSCA, MCA
        x_sa = self.sa_plus_1(out, coords) if self.ablate_depth < 4 else out
        x_csca = self.csca_1(out) if self.ablate_depth < 1 else out
        x_mca = self.mca_1(out, coords) if self.ablate_depth < 3 else out

        # Adaptive fusion
        out = self.gate_1(x_sa, x_csca, x_mca)

        # ---- Layer 2: Coarse Graph + EHAB ----
        x_down = self.down_2(out)
        x2 = self.l2(x_down)
        x_up = self.up_2(out, x2)
        out = self.embed_002(x_up)

        # Layer 2: SA+, CSCA, MCA (no VCA — already applied in Layer 1)
        x_sa2 = self.sa_plus_2(out, coords) if self.ablate_depth < 4 else out
        x_csca2 = self.csca_2(out) if self.ablate_depth < 1 else out
        x_mca2 = self.mca_2(out, coords) if self.ablate_depth < 3 else out
        out = self.gate_2(x_sa2, x_csca2, x_mca2)

        # ---- Local weight ----
        w0 = self.linear_0(out).view(B, -1)

        # ---- MGA (Multi-hop Graph Attention) ----
        if self.ablate_depth < 5:
            out = out + self.mga(out, w0.detach())

        out = self.embed_1(out)
        w1 = self.linear_1(out).view(B, -1)

        # ---- Progressive pruning ----
        if not self.predict:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices)
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]
        else:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, True)
            out = self.embed_2(out)
            w2 = self.linear_2(out)
            e_hat = weighted_8points(x_ds, w2)
            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat


# ====================================================================
# HATNetPlus: Top-level network
# ====================================================================

class HATNetPlus(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_fourier = getattr(config, 'use_fourier', False)
        self.ablate_depth = getattr(config, 'ablate_depth', 0)
        k_num_coarse = getattr(config, 'k_num', 9)
        k_num_fine   = max(3, k_num_coarse - 3)  # fine-level uses slightly fewer neighbors
        self.ds_0 = CL_Block_Plus(
            initial=True, predict=False, out_channel=128,
            k_num=k_num_coarse, sampling_rate=config.sr, clusters=config.clusters,
            use_fourier=self.use_fourier, ablate_depth=self.ablate_depth)
        self.ds_1 = CL_Block_Plus(
            initial=False, predict=True, out_channel=128,
            k_num=k_num_fine, sampling_rate=config.sr, clusters=config.clusters,
            use_fourier=self.use_fourier, ablate_depth=self.ablate_depth)

    def forward(self, x, y):
        x0 = x
        B, _, N, _ = x.shape
        motion = x[:, :, :, :2] - x[:, :, :, 2:]
        x = torch.cat([x, motion], dim=-1)  # (B, 1, N, 6)

        x1, y1, ws0, w_ds0 = self.ds_0(x, y)

        B1, _, N1, _ = x1.shape
        motions = x1[:, :, :, :2] - x1[:, :, :, 2:]
        x1 = torch.cat([x1, motions], dim=-1)

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)

        x2, y2, ws1, w_ds1, e_hat = self.ds_1(x_, y1)

        with torch.no_grad():
            y_hat = batch_episym(x0[:, 0, :, :2], x0[:, 0, :, 2:], e_hat)

        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat


# ====================================================================
# Weighted 8-point algorithm (same as HAT-Match)
# ====================================================================

def batch_symeig(X):
    device = X.device
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        _, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze())
        bv[batch_idx, :, :] = v
    bv = bv.to(device)
    return bv


def weighted_8points(x_in, logits):
    mask = logits[:, 0, :, 0]
    weights = logits[:, 1, :, 0]
    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()

    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat
