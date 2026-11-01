"""
Swin Transformer for Hydrological Regression
Author: Siddik Barbhuiya
Description: Swin Transformer implementation for meteorological sequence to hydrological variables regression
Adapted for time series regression (sequence-to-one prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any, List

# Try to import einops, fallback if not available
try:
    from einops import rearrange
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False
    def rearrange(tensor, pattern, **axes_lengths):
        # Basic fallback for simple cases
        return tensor


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
    # Pad if necessary
    pad_l = (window_size - L % window_size) % window_size
    if pad_l > 0:
        x = F.pad(x, (0, 0, 0, pad_l))
        L = L + pad_l
    
    x = x.view(B, L // window_size, window_size, C)
    windows = x.view(-1, window_size, C)  # (B*num_windows, window_size, C)
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
    
    # Remove padding if added
    if x.shape[1] > L:
        x = x[:, :L, :]
    
    return x


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module
    """
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        coords = torch.arange(window_size)
        relative_coords = coords[:, None] - coords[None, :]  # (window_size, window_size)
        relative_coords += window_size - 1  # shift to start from 0
        self.register_buffer("relative_position_index", relative_coords)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, window_size, C)
            mask: (0/-inf) mask with shape of (num_windows, window_size, window_size) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)  # window_size, window_size, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, window_size, window_size
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block
    """
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

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (B*num_windows, window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # (B*num_windows, window_size, C)

        # merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, L)  # (B, L, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbedding(nn.Module):
    """
    Temporal Patch Embedding for time series
    """
    def __init__(self, seq_len: int = 270, patch_size: int = 10, in_chans: int = 20, 
                 embed_dim: int = 96, norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.patches_resolution = seq_len // patch_size
        self.num_patches = self.patches_resolution
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Use 1D convolution for temporal patching
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
        
        # Reshape for 1D conv: (B, C, L)
        x = x.transpose(1, 2)
        
        # Apply patch embedding
        x = self.proj(x)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x


class SwinTransformerLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage
    """
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True, dropout: float = 0.0,
                 drop_path: float = 0.0, norm_layer: nn.Module = nn.LayerNorm,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, dropout=dropout,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # Patch merging layer
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


class PatchMerging(nn.Module):
    """
    Patch Merging Layer for temporal downsampling
    Input: (B, L, C) -> Output: (B, L/2, 2*C)
    """
    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # Key fix: reduction should output 2*dim (double the dimension)
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C) -> (B, L/2, 2*C)
        """
        B, L, C = x.shape

        # Pad to even length if necessary
        if L % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
            L = L + 1

        x0 = x[:, 0::2, :]  # (B, L/2, C)
        x1 = x[:, 1::2, :]  # (B, L/2, C)
        x = torch.cat([x0, x1], -1)  # (B, L/2, 2*C)

        x = self.norm(x)
        x = self.reduction(x)  # (B, L/2, 2*C)

        return x


class ImprovedRegressionHead(nn.Module):
    """
    Regression head with skip connections to preserve low-level features
    """
    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Main pathway (feature compression)
        self.main_path = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Skip pathway (preserve original features)
        self.skip_path = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)  # Less dropout for skip path
        )
        
        # Fusion layer (combine main + skip)
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim // 2, feature_dim // 4),  # 2 * (feature_dim // 4)
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(feature_dim // 4, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize final layer with small weights to start predictions near 0
        # The final layer is the last Linear layer in self.fusion
        final_layer = self.fusion[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
        nn.init.constant_(final_layer.bias, 0.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with skip connection
        x: (batch_size, feature_dim)
        returns: (batch_size, 1)
        """
        # Main pathway: compress features
        main_features = self.main_path(x)  # (batch_size, feature_dim // 4)
        
        # Skip pathway: preserve original information
        skip_features = self.skip_path(x)  # (batch_size, feature_dim // 4)
        
        # Combine both pathways
        combined = torch.cat([main_features, skip_features], dim=1)  # (batch_size, feature_dim // 2)
        
        # Final prediction
        output = self.fusion(combined)  # (batch_size, 1)
        
        return output


class EnhancedQHead(nn.Module):
    """
    Enhanced Task Head specifically designed for Runoff (Q) prediction.
    Author: Siddik Barbhuiya
    
    Features:
    - Deeper architecture for complex runoff dynamics
    - Attention mechanism to focus on relevant features
    - Non-negative output (runoff can't be negative)
    - Skip connections for gradient flow
    """
    
    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        hidden_dim = feature_dim // 2
        
        # Layer normalization for input stability
        self.input_norm = nn.LayerNorm(feature_dim)
        
        # Feature attention: learn which features matter for Q
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.Tanh(),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )
        
        # Deeper main pathway (4 layers instead of 2)
        self.main_path = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Skip pathway
        self.skip_path = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),  # Changed to // 2 to match checkpoint (24)
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Final prediction with non-negativity
        # Main path output: hidden_dim // 4 (12)
        # Skip path output: hidden_dim // 2 (24)
        # Combined: 12 + 24 = 36
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim // 4 + hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Q prediction.
        x: (batch_size, feature_dim)
        returns: (batch_size, 1) - always non-negative
        """
        # Normalize input
        x = self.input_norm(x)
        
        # Apply attention weighting
        attention_weights = self.attention(x)
        x_attended = x * attention_weights
        
        # Main pathway
        main_out = self.main_path(x_attended)
        
        # Skip pathway (from original features)
        skip_out = self.skip_path(x)
        
        # Combine
        combined = torch.cat([main_out, skip_out], dim=1)
        
        # Final output (apply softplus for non-negative runoff)
        output = self.output_layer(combined)
        # Note: Softplus removed as data is normalized, can add back if needed
        
        return output


class SwinTransformerHydro(nn.Module):
    """
    Swin Transformer for Hydrological Regression
    Sequence-to-one prediction for meteorological -> hydrological variables
    
    Args:
        use_hierarchical: If True (default), use hierarchical dimensions (doubling per stage).
                         If False, use constant dimensions (legacy mode for backward compatibility).
    """
    
    def __init__(self,
                 seq_len: int = 270,
                 patch_size: int = 10,
                 in_chans: int = 20,
                 num_classes: int = 3,
                 embed_dim: int = 96,
                 depths: list = [2, 2, 6, 2],
                 num_heads: list = [3, 6, 12, 24],
                 window_size: int = 7,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 dropout: float = 0.0,
                 drop_path_rate: float = 0.1,
                 norm_layer: nn.Module = nn.LayerNorm,
                 patch_norm: bool = True,
                 use_hierarchical: bool = True,
                 **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.use_hierarchical = use_hierarchical
        self.mlp_ratio = mlp_ratio
        
        # Calculate final feature dimension based on architecture mode
        if use_hierarchical:
            self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        else:
            # Non-hierarchical: constant dimension throughout
            self.num_features = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            seq_len=seq_len, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.patches_resolution

        # Absolute position embedding (learnable)
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers - mode depends on use_hierarchical flag
        self.layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            if use_hierarchical:
                # HIERARCHICAL: double dims each stage with downsampling
                layer_dim = int(embed_dim * 2 ** i_layer)
                downsample_class = PatchMerging if (i_layer < self.num_layers - 1) else None
            else:
                # NON-HIERARCHICAL: constant dimension, no downsampling (legacy mode)
                layer_dim = embed_dim
                downsample_class = None
            
            # Ensure num_heads divides layer_dim evenly
            layer_num_heads = num_heads[i_layer]
            if layer_dim % layer_num_heads != 0:
                layer_num_heads = max(1, layer_dim // 32)  # At least 32 dims per head
            
            layer = SwinTransformerLayer(
                dim=layer_dim,
                depth=depths[i_layer],
                num_heads=layer_num_heads,
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsample_class)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(self.num_features, self.num_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.num_features // 2, num_classes)
        )

        # Initialize weights
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
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add absolute position embedding
        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # Swin Transformer layers
        for layer in self.layers:
            x = layer(x)  # (B, num_patches_i, embed_dim_i)

        x = self.norm(x)  # (B, num_patches_final, num_features)
        
        # Global average pooling
        x = self.avgpool(x.transpose(1, 2))  # (B, num_features, 1)
        x = torch.flatten(x, 1)  # (B, num_features)
        
        # Regression head
        x = self.head(x)  # (B, num_classes)
        
        return x
    
    def get_model_info(self) -> str:
        """Get model information string"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Get per-stage dimensions
        stage_dims = [layer.dim for layer in self.layers]
        
        hierarchy_mode = 'Hierarchical (doubling dims)' if self.use_hierarchical else 'Non-Hierarchical (constant dim)'
        info = f"""
=== SwinTransformerHydro Model Info ===
Input: {self.patch_embed.seq_len} timesteps, {self.patch_embed.in_chans} channels
Patch Size: {self.patch_embed.patch_size}
Embed Dimension: {self.embed_dim} (base)
Architecture Mode: {hierarchy_mode}
Stage Dimensions: {stage_dims}
Final Feature Dim: {self.num_features}
Depths: {[layer.depth for layer in self.layers]}
Window Size: {self.layers[0].blocks[0].window_size if self.layers[0].blocks else 0}
Downsampling: {'Enabled (PatchMerging)' if any(layer.downsample is not None for layer in self.layers) else 'Disabled'}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Model Size: ~{total_params * 4 / 1024 / 1024:.2f} MB
"""
        return info


class MultiTaskSwinTransformer(nn.Module):
    """
    Multi-task version of Swin Transformer for hydrological regression
    Compatible with combined input format
    
    Args:
        use_hierarchical: If True (default), use hierarchical dimensions (doubling per stage).
                         If False, use constant dimensions (legacy mode for backward compatibility).
    """
    
    def __init__(self,
                 n_exo: int,
                 n_lag_features: int = 0,
                 seq_len: int = 270,
                 patch_size: int = 10,
                 embed_dim: int = 96,
                 depths: list = [2, 2, 6, 2],
                 num_heads: list = [3, 6, 12, 24],
                 window_size: int = 7,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 dropout: float = 0.0,
                 drop_path_rate: float = 0.1,
                 task_names: list = None,
                 use_hierarchical: bool = True,
                 spatial_attention: Optional[Dict] = None):
        super().__init__()

        self.task_names = task_names or ['evap', 'sm', 'Q']
        self.num_tasks = len(self.task_names)
        self.use_hierarchical = use_hierarchical
        
        # Calculate total input features
        total_input_features = n_exo + n_lag_features
        
        # Shared Swin Transformer backbone
        self.swin_backbone = SwinTransformerHydro(
            seq_len=seq_len,
            patch_size=patch_size,
            in_chans=total_input_features,
            num_classes=embed_dim * 8,  # Use as feature extractor
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            use_hierarchical=use_hierarchical
        )
        
        # Remove the original head
        feature_dim = self.swin_backbone.num_features
        self.swin_backbone.head = nn.Identity()

        # Optional spatial graph attention between the pooled Swin features and the heads.
        # Disabled (None) keeps the model and its state_dict identical to earlier checkpoints.
        self.spatial_attention = None
        self.spatial_max_distance = None
        self.spatial_k = None
        sp = dict(spatial_attention or {})
        if sp.get('enabled', False):
            from .spatial_attention import SpatialGraphAttention
            gat_heads = int(sp.get('num_heads', 8))
            if feature_dim % gat_heads != 0:
                raise ValueError(f"spatial_attention.num_heads={gat_heads} must divide feature_dim={feature_dim}")
            self.spatial_max_distance = float(sp.get('max_distance', 2.0))
            # k nearest neighbours within max_distance; None = all cells within max_distance
            self.spatial_k = int(sp['k_neighbors']) if sp.get('k_neighbors') else None
            self.spatial_attention = SpatialGraphAttention(
                in_features=feature_dim, out_features=feature_dim, num_heads=gat_heads,
                dropout=float(sp.get('dropout', 0.1)), concat=True)
        self.reset_graph_stats()

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name in self.task_names:
            # Use ImprovedRegressionHead for all tasks including Q to match checkpoint
            self.task_heads[task_name] = ImprovedRegressionHead(
                feature_dim, dropout=dropout
            )

    @property
    def uses_spatial_attention(self) -> bool:
        return self.spatial_attention is not None

    def reset_graph_stats(self) -> None:
        self._graph_nodes = 0
        self._graph_edges = 0
        self._graph_isolated = 0

    def pop_graph_stats(self) -> Optional[Dict[str, float]]:
        """Mean neighbours per cell and % of cells without a neighbour since the last call."""
        if self._graph_nodes == 0:
            return None
        stats = {'nodes': self._graph_nodes,
                 'mean_neighbours': self._graph_edges / self._graph_nodes,
                 'isolated_pct': 100.0 * self._graph_isolated / self._graph_nodes}
        self.reset_graph_stats()
        return stats

    def _encode(self, x: torch.Tensor, lat: Optional[torch.Tensor] = None,
                lon: Optional[torch.Tensor] = None, graph_key: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Swin features, followed by spatial attention among same-day neighbours when enabled."""
        features = self.swin_backbone(x)  # (batch_size, feature_dim)
        if self.spatial_attention is None:
            return features
        if lat is None or lon is None:
            raise ValueError("spatial_attention is enabled but lat/lon were not passed to the model")
        from .spatial_attention import build_spatial_graph, build_knn_graph
        if self.spatial_k:
            edge_index = build_knn_graph(lat.float(), lon.float(), self.spatial_k, self.spatial_max_distance,
                                         group=graph_key).to(features.device)
        else:
            edge_index = build_spatial_graph(lat.float(), lon.float(), self.spatial_max_distance,
                                             group=graph_key).to(features.device)
        real = edge_index[0] != edge_index[1]
        degree = torch.bincount(edge_index[1][real], minlength=features.shape[0])
        self._graph_nodes += int(features.shape[0])
        self._graph_edges += int(real.sum())
        self._graph_isolated += int((degree == 0).sum())
        # GATConv is run in float32 for numerical stability under autocast
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = self.spatial_attention(features.float(), edge_index)
        return features

    def forward(self, x: torch.Tensor, lat: Optional[torch.Tensor] = None,
                lon: Optional[torch.Tensor] = None, graph_key: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass - returns tensor for compatibility with existing trainer

        Args:
            x: Input tensor (batch_size, seq_len, features)
            lat, lon: Cell coordinates (batch_size,), required when spatial_attention is enabled
            graph_key: Per-sample key; only samples with equal keys (same day and parameter set) are connected

        Returns:
            Output tensor (batch_size, num_tasks)
        """
        # Extract features using Swin backbone (+ spatial attention if enabled)
        features = self._encode(x, lat, lon, graph_key)  # (batch_size, feature_dim)

        # Task-specific predictions
        outputs = []
        for task_name in self.task_names:
            task_output = self.task_heads[task_name](features)  # (batch_size, 1)
            outputs.append(task_output.squeeze(-1))  # (batch_size,)
        
        # Stack to create (batch_size, num_tasks)
        return torch.stack(outputs, dim=1)
    
    def forward_task_specific(self, x: torch.Tensor, lat: Optional[torch.Tensor] = None,
                              lon: Optional[torch.Tensor] = None,
                              graph_key: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with task-specific outputs for individual loss calculation
        
        Args:
            x: Input tensor (batch_size, seq_len, features)
            
        Returns:
            Dict with task-specific outputs: {'evap': tensor, 'sm': tensor, 'Q': tensor}
        """
        # Extract features using Swin backbone
        features = self._encode(x, lat, lon, graph_key)  # (batch_size, feature_dim)
        
        # Task-specific predictions - return as dict for individual loss computation
        task_outputs = {}
        for task_name in self.task_names:
            task_output = self.task_heads[task_name](features)  # (batch_size, 1)
            task_outputs[task_name] = task_output.squeeze(-1)  # (batch_size,)
        
        return task_outputs
    
    def forward_dict(self, x: torch.Tensor, lat: Optional[torch.Tensor] = None,
                     lon: Optional[torch.Tensor] = None,
                     graph_key: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning dictionary - for advanced usage
        """
        # Extract features using Swin backbone
        features = self._encode(x, lat, lon, graph_key)  # (batch_size, feature_dim)
        
        # Task-specific predictions
        outputs = {}
        for task_name in self.task_names:
            task_output = self.task_heads[task_name](features)  # (batch_size, 1)
            outputs[task_name] = task_output.squeeze(-1)  # (batch_size,)
        
        return outputs
    
    def forward_autoregressive(self, x: torch.Tensor, 
                                n_target_lags: int = 0,
                                target_lag_indices: List[int] = None,
                                p_teacher_forcing: float = 1.0,
                                lat: Optional[torch.Tensor] = None,
                                lon: Optional[torch.Tensor] = None,
                                graph_key: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Autoregressive forward pass with scheduled sampling.
        
        Processes the sequence and optionally replaces target lag features
        with model predictions based on teacher forcing probability.
        
        Args:
            x: Input tensor (batch_size, seq_len, features)
               Features = [forcing, params, other_lags, target_lags]
            n_target_lags: Number of target lag features at end of feature dim
            target_lag_indices: Indices in feature dim for target lags (optional)
            p_teacher_forcing: Probability of using ground truth (1.0=always teacher, 0.0=always predict)
            
        Returns:
            Output tensor (batch_size, num_tasks)
        """
        batch_size, seq_len, n_features = x.shape
        
        # If no target lags or always teacher forcing, use standard forward
        if n_target_lags == 0 or p_teacher_forcing >= 1.0:
            return self.forward(x, lat=lat, lon=lon, graph_key=graph_key)
        
        # Determine target lag indices (assume last n_target_lags features)
        if target_lag_indices is None:
            target_lag_indices = list(range(n_features - n_target_lags, n_features))
        
        # Clone input to avoid modifying original
        x_modified = x.clone()
        
        # For fully autoregressive (p=0), we need step-by-step inference
        # For mixed (0 < p < 1), we sample per batch item
        
        if p_teacher_forcing <= 0.0:
            # Fully autoregressive: replace target lags with predictions
            # This is a simplified version - full seq2seq would process step by step
            # Here we use the model's current predictions to create lag-like features
            
            # First forward pass with original input
            with torch.no_grad():
                initial_pred = self.forward(x, lat=lat, lon=lon, graph_key=graph_key)  # (batch, num_tasks)
            
            # Replace target lag features with predictions (shifted)
            # For lag=1: position t gets prediction from position t-1
            for i, lag_idx in enumerate(target_lag_indices):
                if i < self.num_tasks:
                    # Shift predictions to create lagged version
                    shifted_pred = torch.zeros(batch_size, seq_len, device=x.device, dtype=x.dtype)
                    shifted_pred[:, 1:] = initial_pred[:, i].unsqueeze(1).expand(-1, seq_len-1)
                    x_modified[:, :, lag_idx] = shifted_pred
            
            # Forward with modified input
            return self.forward(x_modified, lat=lat, lon=lon, graph_key=graph_key)
        
        else:
            # Mixed: randomly select per sample
            mask = torch.rand(batch_size, device=x.device) < p_teacher_forcing
            
            # Get predictions using original input for samples that need it
            with torch.no_grad():
                full_pred = self.forward(x, lat=lat, lon=lon, graph_key=graph_key)
            
            # For samples where mask=False, replace target lags
            for b in range(batch_size):
                if not mask[b]:
                    for i, lag_idx in enumerate(target_lag_indices):
                        if i < self.num_tasks:
                            shifted_pred = torch.zeros(seq_len, device=x.device, dtype=x.dtype)
                            shifted_pred[1:] = full_pred[b, i]
                            x_modified[b, :, lag_idx] = shifted_pred
            
            # Forward with modified input
            return self.forward(x_modified, lat=lat, lon=lon, graph_key=graph_key)
    
    def get_model_info(self) -> str:
        """Get model information string"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Breakdown by component
        backbone_params = sum(p.numel() for p in self.swin_backbone.parameters())
        task_head_params = sum(p.numel() for p in self.task_heads.parameters())
        if self.spatial_attention is not None:
            gat_params = sum(p.numel() for p in self.spatial_attention.parameters())
            nb = f"k={self.spatial_k} nearest within" if self.spatial_k else "all within"
            gat_line = (f"Spatial GAT Parameters: {gat_params:,} "
                        f"({nb} {self.spatial_max_distance} deg, {self.spatial_attention.num_heads} heads)\n")
        else:
            gat_line = "Spatial GAT: disabled\n"
        
        info = f"""
=== MultiTaskSwinTransformer Model Info ===
Tasks: {self.task_names} ({self.num_tasks} tasks)
Shared Backbone: SwinTransformerHydro
Feature Dimension: {self.swin_backbone.num_features}

--- Parameter Breakdown ---
Backbone Parameters: {backbone_params:,}
{gat_line}Task Heads Parameters: {task_head_params:,}
TOTAL Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Model Size: ~{total_params * 4 / 1024 / 1024:.2f} MB
"""
        return info

    
    def freeze_task_head(self, task_name: str) -> bool:
        """
        Freeze a task-specific head to stop its gradient updates.
        
        Args:
            task_name: Name of the task head to freeze ('evap', 'soil_moisture_total', 'Q')
            
        Returns:
            True if frozen successfully, False if task not found
        """
        if task_name not in self.task_heads:
            return False
        
        for param in self.task_heads[task_name].parameters():
            param.requires_grad = False
        
        return True
    
    def unfreeze_task_head(self, task_name: str) -> bool:
        """
        Unfreeze a task-specific head to resume its gradient updates.
        
        Args:
            task_name: Name of the task head to unfreeze
            
        Returns:
            True if unfrozen successfully, False if task not found
        """
        if task_name not in self.task_heads:
            return False
        
        for param in self.task_heads[task_name].parameters():
            param.requires_grad = True
        
        return True
    
    def get_frozen_status(self) -> Dict[str, bool]:
        """
        Get the frozen/trainable status of each task head.
        
        Returns:
            Dict mapping task names to frozen status (True = frozen)
        """
        status = {}
        for task_name in self.task_names:
            # Check if any parameter requires grad (if not, head is frozen)
            is_frozen = not any(p.requires_grad for p in self.task_heads[task_name].parameters())
            status[task_name] = is_frozen
        return status
    
    def freeze_backbone(self) -> None:
        """Freeze the shared Swin backbone"""
        for param in self.swin_backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze the shared Swin backbone"""
        for param in self.swin_backbone.parameters():
            param.requires_grad = True