"""
Spatial Graph Attention Module for Hydrological Modeling
Author: Siddik Barbhuiya
Description: Graph Attention Network to connect spatially neighboring grids
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

try:
    from torch_geometric.nn import GATConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("WARNING: torch_geometric not available. Install with: pip install torch_geometric")


def build_knn_graph(lat: torch.Tensor, lon: torch.Tensor, k: int, max_distance: float,
                    group: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Connect every cell to its k nearest cells within max_distance degrees (same group only).

    A fixed k keeps the number of neighbours the same whether few cells (training batch,
    validation) or many cells (test set, full domain) are evaluated together; only the
    distance to the neighbours changes.

    Returns:
        edge_index: (2, num_edges) as [source j, target i]; messages flow from j to i
    """
    n = len(lat)
    if n <= 1 or k <= 0:
        return torch.zeros((2, 0), dtype=torch.long, device=lat.device)
    coords = torch.stack([lat, lon], dim=1).float()
    d = torch.cdist(coords, coords, p=2)
    invalid = (d > max_distance) | torch.eye(n, dtype=torch.bool, device=lat.device)
    if group is not None:
        invalid |= group.view(-1, 1) != group.view(1, -1)
    d = d.masked_fill(invalid, float('inf'))
    kk = min(k, n - 1)
    dist, nbr = torch.topk(d, kk, dim=1, largest=False)          # (n, kk) nearest per target
    tgt = torch.arange(n, device=lat.device).view(-1, 1).expand(-1, kk)
    keep = torch.isfinite(dist)
    return torch.stack([nbr[keep], tgt[keep]], dim=0)


def build_spatial_graph(lat: torch.Tensor, lon: torch.Tensor, max_distance: float = 1.0,
                        group: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Build edge_index connecting grids within max_distance degrees.
    Optimized with torch.cdist for speed and memory efficiency.

    Args:
        lat: Latitude coordinates of grids (n_grids,)
        lon: Longitude coordinates of grids (n_grids,)
        max_distance: Maximum distance in degrees to connect neighbors
        group: Optional key per sample (n_grids,). Only samples with the same key are
            connected, e.g. the same day and parameter set, so a cell never receives
            information from a neighbour on a different day.

    Returns:
        edge_index: (2, num_edges) tensor of edges
    """
    n_grids = len(lat)
    if n_grids <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=lat.device)

    # Stack coordinates: (n, 2)
    coords = torch.stack([lat, lon], dim=1).float()

    # Use torch.cdist for fast pairwise distance computation (Euclidean, p=2)
    # This is much faster and more memory-efficient than manual broadcasting
    distances = torch.cdist(coords, coords, p=2) # (n, n)

    # Find connected pairs (within max_distance, excluding self-loops)
    mask = (distances <= max_distance) & (distances > 0)
    if group is not None:
        mask &= group.view(-1, 1) == group.view(1, -1)
    edge_index = mask.nonzero(as_tuple=False).T # (2, num_edges)
    
    if edge_index.shape[1] == 0:
        # No neighbors found, create self-loops as fallback
        edge_index = torch.stack([torch.arange(n_grids), torch.arange(n_grids)], dim=0).to(lat.device)
    
    return edge_index


class SpatialGraphAttention(nn.Module):
    """
    Graph Attention layer for spatial communication between grids.
    
    Uses GATConv from PyTorch Geometric to implement attention-based
    message passing between spatially neighboring grids.
    """
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 concat: bool = False):
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required for SpatialGraphAttention")
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        
        # GAT layer
        self.gat = GATConv(
            in_channels=in_features,
            out_channels=out_features // num_heads if concat else out_features,
            heads=num_heads,
            dropout=dropout,
            concat=concat
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(out_features)
        
        # Residual projection if dimensions differ
        if in_features != out_features:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Apply graph attention.
        
        Args:
            x: Node features (n_grids, in_features)
            edge_index: Graph connectivity (2, num_edges)
            
        Returns:
            Updated features (n_grids, out_features)
        """
        # Graph attention with residual
        residual = self.residual(x)
        x = self.gat(x, edge_index)
        x = x + residual
        x = self.norm(x)
        
        return x


class SpatioTemporalSwinTransformer(nn.Module):
    """
    Swin Transformer with Spatial Graph Attention.
    
    Combines:
    1. Temporal Swin Transformer (existing) for time series processing
    2. Spatial Graph Attention (new) for grid-to-grid communication
    
    Flow:
        Input: (batch_grids, seq_len, features)
        → Temporal Swin: Extract temporal features per grid
        → Spatial GAT: Share information between neighboring grids
        → Task Heads: Predict evap, SM, Q
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
                 max_distance: float = 1.0,
                 spatial_heads: int = 4,
                 task_head_configs: dict = None):
        super().__init__()
        
        self.task_names = task_names or ['evap', 'sm', 'Q']
        self.num_tasks = len(self.task_names)
        self.max_distance = max_distance
        
        # Import existing Swin Transformer
        from .swin_transformer import SwinTransformerHydro, ImprovedRegressionHead
        
        # Calculate total input features
        total_input_features = n_exo + n_lag_features
        
        # Shared Temporal Swin Transformer backbone
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
        
        # Remove original head
        feature_dim = self.swin_backbone.num_features
        self.swin_backbone.head = nn.Identity()
        
        # Spatial Graph Attention layer
        self.spatial_attention = SpatialGraphAttention(
            in_features=feature_dim,
            out_features=feature_dim,
            num_heads=spatial_heads,
            dropout=dropout
        )
        
        # Configurable Task-specific heads
        self.task_heads = nn.ModuleDict()
        task_head_configs = task_head_configs or {}
        
        for task_name in self.task_names:
            # Get specific config for this task if available
            h_cfg = task_head_configs.get(task_name, {
                'hidden_dims': [feature_dim // 2],
                'activation': 'relu',
                'dropout': dropout,
                'output_activation': 'none'
            })
            
            # Create the head
            layers = []
            curr_dim = feature_dim
            
            # Add hidden layers
            h_dims = h_cfg.get('hidden_dims', [feature_dim // 2])
            for h_dim in h_dims:
                layers.append(nn.Linear(curr_dim, h_dim))
                
                # Activation
                act = h_cfg.get('activation', 'relu').lower()
                if act == 'relu':
                    layers.append(nn.ReLU())
                elif act == 'gelu':
                    layers.append(nn.GELU())
                elif act == 'tanh':
                    layers.append(nn.Tanh())
                
                drop = h_cfg.get('dropout', dropout)
                if drop > 0:
                    layers.append(nn.Dropout(drop))
                
                curr_dim = h_dim
            
            # Final projection
            layers.append(nn.Linear(curr_dim, 1))
            
            # Output activation (for physical bounds)
            out_act = h_cfg.get('output_activation', 'none').lower()
            if out_act == 'softplus':
                layers.append(nn.Softplus())
            elif out_act == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            self.task_heads[task_name] = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, 
                lat: Optional[torch.Tensor] = None,
                lon: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional spatial attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, features)
            lat: Latitude of each grid in batch (batch_size,)
            lon: Longitude of each grid in batch (batch_size,)
            
        Returns:
            Output tensor (batch_size, num_tasks)
        """
        # 1. Temporal processing with Swin
        features = self.swin_backbone(x)  # (batch_size, feature_dim)
        
        # 2. Spatial attention if coordinates provided
        if lat is not None and lon is not None and len(lat) > 1:
            # Build spatial graph
            edge_index = build_spatial_graph(lat, lon, self.max_distance)
            edge_index = edge_index.to(x.device)
            
            # Apply spatial attention
            features = self.spatial_attention(features, edge_index)
        
        # 3. Task-specific predictions
        outputs = []
        for task_name in self.task_names:
            task_output = self.task_heads[task_name](features)
            outputs.append(task_output.squeeze(-1))
        
        return torch.stack(outputs, dim=1)
    
    def forward_autoregressive(self, x: torch.Tensor, 
                                 n_target_lags: int = 0,
                                 p_teacher_forcing: float = 0.0,
                                 lat: Optional[torch.Tensor] = None,
                                 lon: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Autoregressive forward pass for inference or scheduled sampling.
        
        Args:
            x: Input tensor (batch_size, seq_len, features)
            n_target_lags: Number of target variables that have lag features
            p_teacher_forcing: Probability of using ground truth instead of predictions
            lat: Latitude coordinates
            lon: Longitude coordinates
            
        Returns:
            Predictions (batch_size, num_tasks)
        """
        # This model implements a simplified sequence-to-one autoregressive pass
        # matching the logic in MultiTaskSwinTransformer.
        
        if p_teacher_forcing >= 1.0:
            return self.forward(x, lat, lon)
            
        # For p < 1.0, we would ideally replace lag features with previous predictions.
        # However, since this is a sequence-to-one model, we only predict the last step.
        # The trainer handles the sequence-to-one logic.
        
        # We pass lat/lon to ensure spatial attention is applied during autoregressive steps
        return self.forward(x, lat, lon)
    
    def get_model_info(self) -> str:
        """Get model information string with full parameter breakdown"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Breakdown by component
        backbone_params = sum(p.numel() for p in self.swin_backbone.parameters())
        spatial_attn_params = sum(p.numel() for p in self.spatial_attention.parameters())
        task_head_params = sum(p.numel() for p in self.task_heads.parameters())
        
        info = f"""
=== SpatioTemporalSwinTransformer Model Info ===
Tasks: {self.task_names} ({self.num_tasks} tasks)
Max Distance (spatial): {self.max_distance}°

--- Parameter Breakdown ---
Swin Backbone:      {backbone_params:,}
Spatial Attention:  {spatial_attn_params:,}
Task Heads:         {task_head_params:,}
----------------------------
TOTAL Parameters:   {total_params:,}
Trainable:          {trainable_params:,}
Model Size: ~{total_params * 4 / 1024 / 1024:.2f} MB
"""
        return info

    
    @classmethod
    def from_config(cls, config) -> 'SpatioTemporalSwinTransformer':
        """
        Create model from YAML config object.
        
        Args:
            config: Config object with model and data settings
            
        Returns:
            SpatioTemporalSwinTransformer instance
        """
        import logging
        logger = logging.getLogger('SpatioTemporalSwinTransformer')
        
        # Get model config
        model_cfg = config.model
        embed_dim = getattr(model_cfg, 'embed_dim', 96)
        feature_dim = embed_dim  # In current SwinHydro, feature_dim == embed_dim
        
        # Calculate input features
        n_forcing = len(config.data.variables.forcing) if hasattr(config.data, 'variables') else 8
        n_params = len(config.data.variables.parameters) if hasattr(config.data, 'variables') else 13
        n_exo = n_forcing + n_params
        
        # Get spatial attention config (with defaults)
        spatial_cfg = getattr(model_cfg, 'spatial_attention', {})
        if isinstance(spatial_cfg, dict):
            spatial_enabled = spatial_cfg.get('enabled', True)
            max_distance = spatial_cfg.get('max_distance', 2.0)
            spatial_heads = spatial_cfg.get('num_heads', 4)
            spatial_dropout = spatial_cfg.get('dropout', 0.1)
        else:
            spatial_enabled = getattr(spatial_cfg, 'enabled', True)
            max_distance = getattr(spatial_cfg, 'max_distance', 2.0)
            spatial_heads = getattr(spatial_cfg, 'num_heads', 4)
            spatial_dropout = getattr(spatial_cfg, 'dropout', 0.1)
        
        # Log configuration
        logger.info("=" * 50)
        logger.info("SpatioTemporalSwinTransformer Configuration")
        logger.info("=" * 50)
        logger.info(f"  Input features: {n_exo} (forcing={n_forcing}, params={n_params})")
        logger.info(f"  Sequence length: {getattr(model_cfg, 'sequence_length', 270)}")
        logger.info(f"  Embed dim: {getattr(model_cfg, 'embed_dim', 96)}")
        logger.info(f"  Depths: {getattr(model_cfg, 'depths', [2, 2, 6, 2])}")
        logger.info(f"  Num heads: {getattr(model_cfg, 'num_heads', [4, 8, 8, 8])}")
        logger.info(f"  Window size: {getattr(model_cfg, 'window_size', 12)}")
        logger.info(f"  Spatial attention enabled: {spatial_enabled}")
        logger.info(f"  Max distance: {max_distance}°")
        logger.info(f"  Spatial heads: {spatial_heads}")
        logger.info("=" * 50)
        
        # Get target variable names from config
        task_names = config.data.variables.target if hasattr(config.data, 'variables') else ['evap', 'sm', 'Q']
        
        # Get task-specific head configs
        task_head_configs = {}
        if hasattr(model_cfg, 'task_heads'):
            for task in task_names:
                if hasattr(model_cfg.task_heads, task):
                    t_cfg = getattr(model_cfg.task_heads, task)
                    task_head_configs[task] = {
                        'hidden_dims': getattr(t_cfg, 'hidden_dims', [feature_dim // 2]),
                        'activation': getattr(t_cfg, 'activation', 'relu'),
                        'dropout': getattr(t_cfg, 'dropout', 0.1),
                        'output_activation': getattr(t_cfg, 'output_activation', 'none')
                    }
        
        # Create model
        model = cls(
            n_exo=n_exo,
            n_lag_features=getattr(model_cfg.input_dims, 'n_lag_features', 0) if hasattr(model_cfg, 'input_dims') else 0,
            seq_len=getattr(model_cfg, 'sequence_length', 270),
            patch_size=getattr(model_cfg, 'patch_size', 10),
            embed_dim=getattr(model_cfg, 'embed_dim', 96),
            depths=getattr(model_cfg, 'depths', [2, 2, 6, 2]),
            num_heads=getattr(model_cfg, 'num_heads', [4, 8, 8, 8]),
            window_size=getattr(model_cfg, 'window_size', 12),
            dropout=getattr(model_cfg, 'dropout', 0.4),
            drop_path_rate=getattr(model_cfg, 'drop_path_rate', 0.1),
            task_names=task_names,
            max_distance=max_distance,
            spatial_heads=spatial_heads,
            task_head_configs=task_head_configs  # Pass the new configs
        )
        
        return model


def test_spatial_attention():
    """Test the spatial attention module."""
    print("Testing Spatial Graph Attention...")
    
    # Create dummy data
    batch_size = 16
    seq_len = 270
    n_features = 21
    feature_dim = 96
    
    # Random lat/lon (simulating sparse grids)
    lat = torch.rand(batch_size) * 10 + 6  # 6-16 degrees
    lon = torch.rand(batch_size) * 10 + 73  # 73-83 degrees
    
    # Build graph
    edge_index = build_spatial_graph(lat, lon, max_distance=2.0)
    print(f"Built graph: {batch_size} nodes, {edge_index.shape[1]} edges")
    print(f"Average neighbors per node: {edge_index.shape[1] / batch_size:.1f}")
    
    # Test GAT layer
    if TORCH_GEOMETRIC_AVAILABLE:
        x = torch.randn(batch_size, feature_dim)
        gat = SpatialGraphAttention(feature_dim, feature_dim)
        out = gat(x, edge_index)
        print(f"GAT input: {x.shape}, output: {out.shape}")
        
        # Test full model
        model = SpatioTemporalSwinTransformer(n_exo=21)
        x_full = torch.randn(batch_size, seq_len, n_features)
        pred = model(x_full, lat, lon)
        print(f"Full model output: {pred.shape}")
        print("All tests passed!")
    else:
        print("torch_geometric not available, skipping full test")


if __name__ == "__main__":
    test_spatial_attention()
