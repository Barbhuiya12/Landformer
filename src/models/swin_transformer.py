"""
Swin Transformer for Hydrological Regression
Author: Siddik
Description: Clean Swin Transformer implementation for hydrological surrogate modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition sequence into non-overlapping windows
    Args:
        x: (B, L, C) where L is sequence length
        window_size: window size
    Returns:
        windows: (B*num_windows, window_size, C)
    """
    B, L, C = x.shape
    pad_l = (window_size - L % window_size) % window_size
    if pad_l > 0:
        x = F.pad(x, (0, 0, 0, pad_l))
        L = L + pad_l

    x = x.view(B, L // window_size, window_size, C)
    windows = x.view(-1, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, L: int) -> torch.Tensor:
    """
    Reverse window partition
    Args:
        windows: (B*num_windows, window_size, C)
        window_size: window size
        L: original sequence length
    Returns:
        x: (B, L, C)
    """
    B = int(windows.shape[0] / (math.ceil(L / window_size)))
    x = windows.view(B, -1, window_size, windows.shape[-1])
    x = x.view(B, -1, windows.shape[-1])

    if x.shape[1] > L:
        x = x[:, :L, :]

    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module"""

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1), num_heads))

        coords = torch.arange(window_size)
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords += window_size - 1
        self.register_buffer("relative_position_index", relative_coords)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""

    def __init__(self, dim: int, num_heads: int, window_size: int = 7, shift_size: int = 0,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True, dropout: float = 0.0,
                 drop_path: float = 0.0, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, dropout=dropout)

        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows)
        shifted_x = window_reverse(attn_windows, self.window_size, L)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbedding(nn.Module):
    """Temporal Patch Embedding for time series"""

    def __init__(self, seq_len: int = 270, patch_size: int = 10, in_chans: int = 20,
                 embed_dim: int = 96, norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.patches_resolution = seq_len // patch_size
        self.num_patches = self.patches_resolution
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, C) where L is sequence length, C is input channels
        Returns:
            x: (B, num_patches, embed_dim)
        """
        B, L, C = x.shape
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x


class SwinTransformerLayer(nn.Module):
    """A basic Swin Transformer layer for one stage"""

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True, dropout: float = 0.0,
                 drop_path: float = 0.0, norm_layer: nn.Module = nn.LayerNorm,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, dropout=dropout,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class ImprovedRegressionHead(nn.Module):
    """Regression head with skip connections to preserve low-level features"""

    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim

        self.main_path = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.skip_path = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(feature_dim // 4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_features = self.main_path(x)
        skip_features = self.skip_path(x)
        combined = torch.cat([main_features, skip_features], dim=1)
        output = self.fusion(combined)
        return output


class SwinTransformerHydro(nn.Module):
    """
    Swin Transformer for Hydrological Regression
    Sequence-to-one prediction for meteorological -> hydrological variables
    """

    def __init__(self,
                 seq_len: int = 270,
                 patch_size: int = 10,
                 in_chans: int = 20,
                 num_classes: int = 3,
                 embed_dim: int = 96,
                 depths: list = None,
                 num_heads: list = None,
                 window_size: int = 7,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 dropout: float = 0.0,
                 drop_path_rate: float = 0.1,
                 norm_layer: nn.Module = nn.LayerNorm,
                 patch_norm: bool = True,
                 **kwargs):
        super().__init__()

        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbedding(
            seq_len=seq_len, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.patches_resolution

        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer_num_heads = min(num_heads[i_layer], embed_dim // 32)
            layer_num_heads = max(layer_num_heads, 1)

            layer = SwinTransformerLayer(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=layer_num_heads,
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None)
            self.layers.append(layer)

        self.num_features = embed_dim

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(self.num_features, self.num_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.num_features // 2, num_classes)
        )

        nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, in_chans)
        Returns:
            x: (B, num_classes)
        """
        x = self.patch_embed(x)
        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x

    def get_model_info(self) -> str:
        """Get model information string"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = f"""
=== SwinTransformerHydro Model Info ===
Input: {self.patch_embed.seq_len} timesteps, {self.patch_embed.in_chans} channels
Patch Size: {self.patch_embed.patch_size}
Embed Dimension: {self.embed_dim}
Depths: {[layer.depth for layer in self.layers]}
Window Size: {self.layers[0].blocks[0].window_size if self.layers[0].blocks else 0}
Output Classes: {self.num_classes}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Model Size: ~{total_params * 4 / 1024 / 1024:.2f} MB
"""
        return info


class MultiTaskSwinTransformer(nn.Module):
    """
    Multi-task version of Swin Transformer for hydrological regression
    Compatible with combined input format
    """

    def __init__(self,
                 n_exo: int,
                 n_lag_features: int = 0,
                 seq_len: int = 270,
                 patch_size: int = 10,
                 embed_dim: int = 96,
                 depths: list = None,
                 num_heads: list = None,
                 window_size: int = 7,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 dropout: float = 0.0,
                 drop_path_rate: float = 0.1,
                 task_names: list = None):
        super().__init__()

        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        self.task_names = task_names or ['evap', 'sm', 'Q']
        self.num_tasks = len(self.task_names)

        total_input_features = n_exo + n_lag_features

        self.swin_backbone = SwinTransformerHydro(
            seq_len=seq_len,
            patch_size=patch_size,
            in_chans=total_input_features,
            num_classes=embed_dim * 8,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            dropout=dropout,
            drop_path_rate=drop_path_rate
        )

        feature_dim = self.swin_backbone.num_features
        self.swin_backbone.head = nn.Identity()

        self.task_heads = nn.ModuleDict()
        for task_name in self.task_names:
            self.task_heads[task_name] = ImprovedRegressionHead(
                feature_dim, dropout=dropout
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns tensor for compatibility with existing trainer

        Args:
            x: Input tensor (batch_size, seq_len, features)

        Returns:
            Output tensor (batch_size, num_tasks)
        """
        features = self.swin_backbone(x)

        outputs = []
        for task_name in self.task_names:
            task_output = self.task_heads[task_name](features)
            outputs.append(task_output.squeeze(-1))

        return torch.stack(outputs, dim=1)

    def forward_task_specific(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with task-specific outputs for individual loss calculation

        Args:
            x: Input tensor (batch_size, seq_len, features)

        Returns:
            Dict with task-specific outputs: {'evap': tensor, 'sm': tensor, 'Q': tensor}
        """
        features = self.swin_backbone(x)

        task_outputs = {}
        for task_name in self.task_names:
            task_output = self.task_heads[task_name](features)
            task_outputs[task_name] = task_output.squeeze(-1)

        return task_outputs

    def get_model_info(self) -> str:
        """Get model information string"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = f"""
=== MultiTaskSwinTransformer Model Info ===
Tasks: {self.task_names} ({self.num_tasks} tasks)
Shared Backbone: SwinTransformerHydro
Feature Dimension: {self.swin_backbone.num_features}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Model Size: ~{total_params * 4 / 1024 / 1024:.2f} MB
"""
        return info
