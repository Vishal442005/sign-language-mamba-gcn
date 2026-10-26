"""
models.py
=========
PyTorch implementation of 4 architectures for Sign Language Recognition:
  1. Baseline: Baseline_CNN_BiLSTM
  2. Comparative: Comparative_Backbone (MobileViT-S / EdgeNeXt / ConvNeXt-Femto)
  3. Proposed Model 1: Proposed1_CNN_Mamba (Enhanced with 2D Visual Position Encoding,
     Channel-Spatial Squeeze-and-Excitation / Attention, and Multi-Scale Token Aggregation)
  4. Proposed Model 2: Proposed2_Lightweight_GCN_CNN (Enhanced with Residual/Dense GCN Skip Connections,
     Spatial Joint Attention, and Dynamic Confidence-Aware Gated Cross-Modal Fusion)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm


# ==============================================================================
# Helper Modules: Attention & Encodings
# ==============================================================================
class SqueezeExcitation(nn.Module):
    """Channel attention for visual feature refinement."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        scale = torch.mean(x, dim=1) # (B, C)
        scale = self.fc(scale).unsqueeze(1) # (B, 1, C)
        return x * scale


class JointSpatialAttention(nn.Module):
    """Learnable spatial attention across the 21 hand joints."""
    def __init__(self, num_joints: int = 21, in_features: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.Tanh(),
            nn.Linear(in_features // 4, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 21, C)
        weights = self.attn(x) # (B, 21, 1)
        pooled = torch.sum(x * weights, dim=1) # (B, C)
        return pooled, weights


# ==============================================================================
# 1. Baseline: CNN + Bi-LSTM
# ==============================================================================
class Baseline_CNN_BiLSTM(nn.Module):
    """
    CNN backbone extracting feature map (B, 512, 7, 7) -> flattened to 49 spatial tokens
    -> Bi-LSTM -> Classification head.
    """
    def __init__(self, num_classes: int = 29, hidden_dim: int = 128):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        
        self.proj = nn.Linear(512, hidden_dim)
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, img: torch.Tensor, kpts: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat_map = self.cnn(img)
        B, C, H, W = feat_map.shape
        tokens = feat_map.flatten(2).permute(0, 2, 1)
        x = self.proj(tokens)
        lstm_out, _ = self.bilstm(x)
        pooled = torch.mean(lstm_out, dim=1)
        out = self.fc(pooled)
        return out


# ==============================================================================
# 2. Comparative Backbones (ConvNeXt-Femto, MobileViT-S, MobileNetV3, ViT)
# ==============================================================================
class Comparative_Backbone(nn.Module):
    """
    Comparative modern lightweight backbones loaded via timm.
    Supports: 'convnext_femto', 'mobilevit_s', 'mobilenetv3_large_100', 'vit_tiny_patch16_224'
    """
    def __init__(self, model_name: str = 'convnext_femto', num_classes: int = 29, pretrained: bool = True):
        super().__init__()
        model_map = {
            'convnext_femto': 'convnext_femto',
            'mobilevit_s': 'mobilevit_s',
            'mobilenetv3_large_100': 'mobilenetv3_large_100',
            'vit_tiny': 'vit_tiny_patch16_224',
            'edgenext_small': 'edgenext_small'
        }
        arch = model_map.get(model_name, model_name)
        try:
            self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
        except Exception:
            self.backbone = timm.create_model('resnet34', pretrained=pretrained, num_classes=num_classes)

    def forward(self, img: torch.Tensor, kpts: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.backbone(img)


class Pure_Skeletal_GCN(nn.Module):
    """
    Pure skeletal GCN baseline operating exclusively on 21 MediaPipe hand landmarks.
    Does not require RGB frames: provides modality isolation benchmark.
    """
    def __init__(self, num_classes: int = 29, embed_dim: int = 128):
        super().__init__()
        self.gcn_stream = EnhancedGCNStream(embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, img: Optional[torch.Tensor] = None, kpts: Optional[torch.Tensor] = None) -> torch.Tensor:
        if kpts is None and img is not None:
            kpts = torch.zeros(img.size(0), 21, 3, device=img.device, dtype=img.dtype)
        z_gcn, _ = self.gcn_stream(kpts, use_adaptive=True)
        return self.classifier(z_gcn)


class Standalone_MobileNetV3(nn.Module):
    """
    Standalone lightweight visual CNN baseline (MobileNetV3) without skeletal stream.
    Used to isolate the visual stream contribution from the dual-modal network.
    """
    def __init__(self, num_classes: int = 29, embed_dim: int = 128):
        super().__init__()
        self.cnn_stream = LightweightCNNStream(embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, img: torch.Tensor, kpts: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_cnn = self.cnn_stream(img)
        return self.classifier(z_cnn)



# ==============================================================================
# 3. Proposed Model 1: Enhanced CNN + Bidirectional Mamba (Selective SSM)
# ==============================================================================
class SelectiveSSMBlock(nn.Module):
    """
    Pure PyTorch implementation of Selective State-Space Model (S6) block
    with bidirectional scanning over visual tokens.
    """
    def __init__(self, d_model: int = 192, d_state: int = 16, expand: int = 2, dt_rank: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner
        )
        
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.repeat_interleave(
            torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0),
            repeats=self.d_inner,
            dim=0
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def ssm_scan(self, u: torch.Tensor) -> torch.Tensor:
        B, L, D = u.shape
        N = self.d_state
        A = -torch.exp(self.A_log.float())

        x_dbl = self.x_proj(u)
        dt, B_mat, C_mat = torch.split(x_dbl, [self.dt_rank, N, N], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        y = torch.zeros_like(u)
        h = torch.zeros(B, D, N, device=u.device, dtype=u.dtype)

        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            u_t = u[:, t, :].unsqueeze(-1)
            B_t = B_mat[:, t, :].unsqueeze(1)
            C_t = C_mat[:, t, :].unsqueeze(1)

            dA = torch.exp(dt_t * A.unsqueeze(0))
            dB = dt_t * B_t

            h = dA * h + dB * u_t
            y_t = torch.sum(h * C_t, dim=-1)
            y[:, t, :] = y_t

        y = y + u * self.D.unsqueeze(0).unsqueeze(0)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_branch, z_branch = torch.chunk(xz, 2, dim=-1)

        x_conv = x_branch.transpose(1, 2)
        x_conv = F.silu(self.conv1d(x_conv)).transpose(1, 2)

        y_fwd = self.ssm_scan(x_conv)
        y_bwd = torch.flip(self.ssm_scan(torch.flip(x_conv, dims=[1])), dims=[1])
        y = (y_fwd + y_bwd) * F.silu(z_branch)

        out = self.out_proj(y)
        return out


class Proposed1_CNN_Mamba(nn.Module):
    """
    Enhanced CNN + Bidirectional Mamba (Vision SSM):
      - 2D Learnable Visual Spatial Position Embeddings (49 tokens)
      - Stacked Bidirectional Selective SSM (S6) Blocks with Pre-Norm
      - Squeeze-and-Excitation Channel Attention
      - Dual Global Average & Max Token Aggregation
    """
    def __init__(self, num_classes: int = 29, d_model: int = 192, num_layers: int = 2):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        self.proj = nn.Linear(512, d_model)
        
        # 1. 2D Spatial Position Encoding for the 7x7=49 tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, 49, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 2. Stacked Mamba SSM Blocks
        self.ssm_blocks = nn.ModuleList([
            SelectiveSSMBlock(d_model=d_model, d_state=16, expand=2)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        # 3. Channel Attention (SE block)
        self.se = SqueezeExcitation(d_model, reduction=16)
        
        # 4. Multi-Scale Dual-Pooling (Avg + Max) Classification Head
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, img: torch.Tensor, kpts: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat_map = self.cnn(img) # (B, 512, 7, 7)
        tokens = feat_map.flatten(2).permute(0, 2, 1) # (B, 49, 512)
        x = self.proj(tokens) + self.pos_embed # Add position encodings

        for ssm, norm in zip(self.ssm_blocks, self.norms):
            x = x + ssm(norm(x))

        x = self.se(x) # Channel Attention

        # Dual pooling: Global Average + Global Max
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        pooled = torch.cat([avg_pool, max_pool], dim=-1) # (B, d_model * 2)

        out = self.head(pooled)
        return out


# ==============================================================================
# 4. Proposed Model 2: Enhanced Adaptive GCN + CNN (Dual-Stream)
# ==============================================================================
def build_hand_adjacency() -> torch.Tensor:
    """
    MediaPipe Hand 21-landmark anatomical connectivity graph.
    Returns normalized symmetric adjacency matrix (21, 21).
    """
    edges = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm connections
        (5, 9), (9, 13), (13, 17)
    ]
    A = torch.zeros(21, 21, dtype=torch.float32)
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    A = A + torch.eye(21)
    deg = torch.sum(A, dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D_mat = torch.diag(deg_inv_sqrt)
    A_norm = torch.matmul(torch.matmul(D_mat, A), D_mat)
    return A_norm


class ResidualAdaptiveGCNLayer(nn.Module):
    """
    Adaptive Graph Convolutional Layer with Residual / Skip Connections:
      H^{(l+1)} = ReLU(BatchNorm(A_total @ H^{(l)} @ W)) + Residual(H^{(l)})
    """
    def __init__(self, in_features: int, out_features: int, A_init: torch.Tensor):
        super().__init__()
        self.register_buffer('A_fixed', A_init)
        self.A_learn = nn.Parameter(torch.zeros_like(A_init))
        nn.init.uniform_(self.A_learn, -0.01, 0.01)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(21)
        self.relu = nn.ReLU(inplace=True)

        if in_features != out_features:
            self.res_proj = nn.Linear(in_features, out_features, bias=False)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x: torch.Tensor, use_adaptive: bool = True) -> torch.Tensor:
        if use_adaptive:
            A = self.A_fixed + self.alpha * self.A_learn
        else:
            A = self.A_fixed
            
        res = self.res_proj(x)
        out = torch.matmul(A.unsqueeze(0), x)
        out = self.fc(out)
        out = self.bn(out)
        return self.relu(out + res)


class EnhancedGCNStream(nn.Module):
    """
    3-layer Residual Adaptive GCN with Spatial Joint Attention.
    """
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        A_init = build_hand_adjacency()
        self.gcn1 = ResidualAdaptiveGCNLayer(3, 64, A_init)
        self.gcn2 = ResidualAdaptiveGCNLayer(64, embed_dim, A_init)
        self.gcn3 = ResidualAdaptiveGCNLayer(embed_dim, embed_dim, A_init)
        self.joint_attn = JointSpatialAttention(num_joints=21, in_features=embed_dim)

    def forward(self, kpts: torch.Tensor, use_adaptive: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.gcn1(kpts, use_adaptive=use_adaptive)
        x = self.gcn2(x, use_adaptive=use_adaptive)
        x = self.gcn3(x, use_adaptive=use_adaptive) # (B, 21, embed_dim)
        z_gcn, attn_weights = self.joint_attn(x) # (B, embed_dim), (B, 21, 1)
        return z_gcn, attn_weights


class LightweightCNNStream(nn.Module):
    """Lightweight MobileNetV3-Small backbone for visual stream."""
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(576, embed_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.features(img)
        x = self.pool(x).flatten(1)
        z_cnn = self.proj(x)
        return z_cnn


class Proposed2_Lightweight_GCN_CNN(nn.Module):
    """
    Enhanced Dual-Stream GCN + CNN:
      - Residual Adaptive Graph Stream with Learned Joint Spatial Attention
      - MobileNetV3 Visual Stream
      - Confidence-Aware Dynamic Gated Cross-Modal Fusion
    """
    def __init__(self, num_classes: int = 29, embed_dim: int = 128, fusion_mode: str = 'gated'):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.gcn_stream = EnhancedGCNStream(embed_dim=embed_dim)
        self.cnn_stream = LightweightCNNStream(embed_dim=embed_dim)

        # Dynamic Confidence-Aware Gating Network
        if fusion_mode == 'gated':
            self.gate_fc = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid()
            )
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(embed_dim, num_classes)
            )
        else: # 'concat'
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(embed_dim, num_classes)
            )

    def forward(self, img: torch.Tensor, kpts: Optional[torch.Tensor] = None, use_adaptive: bool = True) -> torch.Tensor:
        if kpts is None:
            kpts = torch.zeros(img.size(0), 21, 3, device=img.device, dtype=img.dtype)
            
        z_gcn, _ = self.gcn_stream(kpts, use_adaptive=use_adaptive) # (B, embed_dim)
        z_cnn = self.cnn_stream(img) # (B, embed_dim)

        if self.fusion_mode == 'gated':
            concat_feat = torch.cat([z_gcn, z_cnn], dim=-1)
            g = self.gate_fc(concat_feat) # (B, embed_dim)
            z_fused = g * z_gcn + (1.0 - g) * z_cnn
            out = self.classifier(z_fused)
        else:
            z_concat = torch.cat([z_gcn, z_cnn], dim=-1)
            out = self.classifier(z_concat)
            
        return out
