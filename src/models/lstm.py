"""
Simple LSTM Benchmark Model for Hydrological Regression
Author: Siddik Barbhuiya
Description: Simple LSTM baseline model for comparison with Swin Transformer
             No attention - just basic LSTM with multi-task heads
"""

import torch
import torch.nn as nn
from typing import List, Optional


class MultiTaskLSTM(nn.Module):
    """
    Simple Multi-task LSTM for hydrological regression.
    
    Interface matches MultiTaskSwinTransformer for easy benchmarking.
    Input: (batch, seq_len, features) 
    Output: (batch, num_tasks)
    
    Architecture:
        LSTM Encoder → Last Hidden State → Task-Specific MLP Heads
    """
    
    def __init__(self,
                 n_exo: int,
                 n_lag_features: int = 0,
                 seq_len: int = 270,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = False,
                 task_names: List[str] = None,
                 **kwargs):  # Accept extra kwargs for compatibility
        """
        Args:
            n_exo: Number of exogenous features (forcing + parameters)
            n_lag_features: Number of lag features
            seq_len: Sequence length (not used but kept for interface compatibility)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
            task_names: List of task names ['evap', 'sm', 'Q']
        """
        super().__init__()
        
        self.task_names = task_names or ['evap', 'soil_moisture_total', 'Q']
        self.num_tasks = len(self.task_names)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Total input features
        in_features = n_exo + n_lag_features
        
        # Simple LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Feature dimension after LSTM (use last hidden state)
        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size
        
        # Simple Task-specific heads (just MLP)
        self.task_heads = nn.ModuleDict()
        for task_name in self.task_names:
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(lstm_out_dim, lstm_out_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_out_dim // 2, 1)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for head in self.task_heads.values():
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns tensor for compatibility with existing trainer
        
        Args:
            x: Input tensor (batch_size, seq_len, features)
            
        Returns:
            Output tensor (batch_size, num_tasks)
        """
        # LSTM encoding - get last hidden state
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last timestep output (simpler than hidden state for bidirectional)
        # lstm_out: (batch, seq_len, hidden*directions)
        last_output = lstm_out[:, -1, :]  # (batch, hidden*directions)
        
        # Task-specific predictions
        outputs = []
        for task_name in self.task_names:
            task_output = self.task_heads[task_name](last_output)
            outputs.append(task_output.squeeze(-1))
        
        return torch.stack(outputs, dim=1)  # (batch, num_tasks)
    
    def forward_autoregressive(self, x: torch.Tensor, 
                               n_target_lags: int = 0,
                               target_lag_indices: List[int] = None,
                               p_teacher_forcing: float = 1.0) -> torch.Tensor:
        """
        Autoregressive forward pass (simplified for LSTM)
        For compatibility with Swin trainer
        """
        return self.forward(x)
    
    def get_model_info(self) -> str:
        """Get model information string"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        lstm_params = sum(p.numel() for p in self.lstm.parameters())
        head_params = sum(p.numel() for p in self.task_heads.parameters())
        
        info = f"""
=== Simple MultiTaskLSTM Model Info ===
Tasks: {self.task_names} ({self.num_tasks} tasks)
Hidden Size: {self.hidden_size}
Num Layers: {self.num_layers}
Bidirectional: {self.bidirectional}

--- Parameter Breakdown ---
LSTM Encoder:       {lstm_params:,}
Task Heads:         {head_params:,}
----------------------------
TOTAL Parameters:   {total_params:,}
Trainable:          {trainable_params:,}
Model Size: ~{total_params * 4 / 1024 / 1024:.2f} MB
"""
        return info
    
    @classmethod
    def from_config(cls, config) -> 'MultiTaskLSTM':
        """
        Create model from YAML config object.
        """
        import logging
        logger = logging.getLogger('MultiTaskLSTM')
        
        model_cfg = config.model
        
        # Calculate input features
        n_forcing = len(config.data.variables.forcing) if hasattr(config.data, 'variables') else 8
        n_params = len(config.data.variables.parameters) if hasattr(config.data, 'variables') else 13
        n_exo = n_forcing + n_params
        
        # Get LSTM config
        hidden_size = getattr(model_cfg, 'hidden_size', 256)
        num_layers = getattr(model_cfg, 'num_layers', 2)
        dropout = getattr(model_cfg, 'dropout', 0.2)
        
        task_names = config.data.variables.target if hasattr(config.data, 'variables') else ['evap', 'sm', 'Q']
        
        logger.info(f"Creating MultiTaskLSTM: hidden={hidden_size}, layers={num_layers}, tasks={task_names}")
        
        return cls(
            n_exo=n_exo,
            seq_len=getattr(model_cfg, 'sequence_length', 270),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            task_names=task_names
        )


if __name__ == "__main__":
    # Quick test
    model = MultiTaskLSTM(n_exo=21, task_names=['evap', 'soil_moisture_total', 'Q'])
    x = torch.randn(4, 270, 21)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    print(model.get_model_info())

