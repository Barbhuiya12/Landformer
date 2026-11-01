"""
Temporal Convolutional Network (TCN) for Hydrological Surrogate Modeling
Author: Siddik Barbhuiya
Description: TCN implementation for VIC surrogate model with dilated causal convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math


class Chomp1d(nn.Module):
    """Remove rightmost elements to ensure causal convolution"""
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Temporal Convolutional Block with dilated causal convolution
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, 
                 dilation: int, padding: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()
        
        # First dilated causal conv
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second dilated causal conv
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Combine layers
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network (TCN) base architecture
    """
    def __init__(self, num_inputs: int, num_channels: list, kernel_size: int = 2, 
                 dropout: float = 0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, 
                                      stride=1, dilation=dilation_size,
                                      padding=(kernel_size-1) * dilation_size, 
                                      dropout=dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class UnifiedTCN(nn.Module):
    """
    TCN model for hydrological surrogate modeling - compatible with combined input format
    Similar to UnifiedLSTM but using TCN instead of LSTM layers
    """
    
    def __init__(self, 
                 n_exo: int,
                 ny: int, 
                 lags: int = 0,
                 n_lag_features: int = 0,
                 hidden_channels: list = None,
                 kernel_size: int = 3,
                 dropout: float = 0.2,
                 lag_config: Optional[Dict] = None):
        """
        Initialize UnifiedTCN
        
        Args:
            n_exo: Number of exogenous variables (forcing + parameters)
            ny: Number of target variables
            lags: Legacy lag parameter (for compatibility)
            n_lag_features: Number of lag features
            hidden_channels: List of channel sizes for TCN layers
            kernel_size: Convolution kernel size
            dropout: Dropout rate
            lag_config: Lag configuration dictionary
        """
        super().__init__()
        
        self.n_exo = n_exo
        self.ny = ny
        self.n_lag_features = n_lag_features
        self.kernel_size = kernel_size
        
        # Default channel configuration
        if hidden_channels is None:
            hidden_channels = [64, 128, 128, 64]
        
        # Calculate total input features
        total_input_features = n_exo + n_lag_features
        
        # TCN backbone
        self.tcn = TCN(
            num_inputs=total_input_features,
            num_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1] // 2, ny)
        )
        
        # Store configuration for model info
        self.config = {
            'hidden_channels': hidden_channels,
            'kernel_size': kernel_size,
            'dropout': dropout,
            'total_params': sum(p.numel() for p in self.parameters())
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, features)
            
        Returns:
            Output tensor (batch_size, ny) - last timestep prediction
        """
        # TCN expects (batch_size, features, seq_len)
        x = x.transpose(1, 2)
        
        # Pass through TCN
        tcn_out = self.tcn(x)  # (batch_size, channels, seq_len)
        
        # Take last timestep
        last_timestep = tcn_out[:, :, -1]  # (batch_size, channels)
        
        # Project to output dimensions
        output = self.output_projection(last_timestep)  # (batch_size, ny)
        
        return output
    
    def get_model_info(self) -> str:
        """Get model information string"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = f"""
=== UnifiedTCN Model Info ===
Input Features: {self.n_exo + self.n_lag_features} (exo: {self.n_exo}, lag: {self.n_lag_features})
Output Features: {self.ny}
Hidden Channels: {self.config['hidden_channels']}
Kernel Size: {self.kernel_size}
Dropout: {self.config['dropout']:.2f}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Model Size: ~{total_params * 4 / 1024 / 1024:.2f} MB
"""
        return info


class MultiTaskTCN(nn.Module):
    """
    Multi-task TCN with separate heads for each target variable
    Compatible with combined input format like MultiTaskVICLSTM
    """
    
    def __init__(self,
                 n_exo: int,
                 n_lag_features: int = 0, 
                 hidden_channels: list = None,
                 kernel_size: int = 3,
                 dropout: float = 0.2,
                 task_names: list = None):
        """
        Initialize MultiTaskTCN
        
        Args:
            n_exo: Number of exogenous variables (forcing + parameters)
            n_lag_features: Number of lag features
            hidden_channels: List of channel sizes for TCN layers
            kernel_size: Convolution kernel size
            dropout: Dropout rate
            task_names: List of task names ['evap', 'sm', 'Q']
        """
        super().__init__()
        
        self.n_exo = n_exo
        self.n_lag_features = n_lag_features
        self.task_names = task_names or ['evap', 'sm', 'Q']
        self.num_tasks = len(self.task_names)
        self.kernel_size = kernel_size
        
        # Default channel configuration
        if hidden_channels is None:
            hidden_channels = [64, 128, 128, 64]
        
        # Calculate total input features
        total_input_features = n_exo + n_lag_features
        
        # Shared TCN backbone
        self.shared_tcn = TCN(
            num_inputs=total_input_features,
            num_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name in self.task_names:
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels[-1] // 2, 1)
            )
        
        # Store configuration
        self.config = {
            'hidden_channels': hidden_channels,
            'kernel_size': kernel_size,
            'dropout': dropout,
            'task_names': self.task_names
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns tensor for compatibility with existing trainer
        
        Args:
            x: Input tensor (batch_size, seq_len, features)
            
        Returns:
            Output tensor (batch_size, num_tasks)
        """
        # TCN expects (batch_size, features, seq_len)
        x = x.transpose(1, 2)
        
        # Shared TCN backbone
        shared_features = self.shared_tcn(x)  # (batch_size, channels, seq_len)
        
        # Take last timestep
        last_timestep = shared_features[:, :, -1]  # (batch_size, channels)
        
        # Task-specific predictions
        outputs = []
        for task_name in self.task_names:
            task_output = self.task_heads[task_name](last_timestep)  # (batch_size, 1)
            outputs.append(task_output.squeeze(-1))  # (batch_size,)
        
        # Stack to create (batch_size, num_tasks)
        return torch.stack(outputs, dim=1)
    
    def forward_dict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning dictionary - for advanced usage
        
        Args:
            x: Input tensor (batch_size, seq_len, features)
            
        Returns:
            Dictionary of task outputs {task_name: tensor}
        """
        # TCN expects (batch_size, features, seq_len)
        x = x.transpose(1, 2)
        
        # Shared TCN backbone
        shared_features = self.shared_tcn(x)  # (batch_size, channels, seq_len)
        
        # Take last timestep
        last_timestep = shared_features[:, :, -1]  # (batch_size, channels)
        
        # Task-specific predictions
        outputs = {}
        for task_name in self.task_names:
            task_output = self.task_heads[task_name](last_timestep)  # (batch_size, 1)
            outputs[task_name] = task_output.squeeze(-1)  # (batch_size,)
        
        return outputs
    
    def get_model_info(self) -> str:
        """Get model information string"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = f"""
=== MultiTaskTCN Model Info ===
Input Features: {self.n_exo + self.n_lag_features} (exo: {self.n_exo}, lag: {self.n_lag_features})
Tasks: {self.task_names} ({self.num_tasks} tasks)
Hidden Channels: {self.config['hidden_channels']}
Kernel Size: {self.kernel_size}
Dropout: {self.config['dropout']:.2f}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Model Size: ~{total_params * 4 / 1024 / 1024:.2f} MB
"""
        return info


class MultiTaskTCNSeparate(nn.Module):
    """
    Multi-task TCN with separate encoders - compatible with MultiTaskLSTM interface
    Uses separate inputs: (forcing, params, lags)
    """
    
    def __init__(self,
                 n_forcing: int,
                 n_params: int,
                 n_lag_features: int = 0,
                 hidden_channels: list = None,
                 kernel_size: int = 3,
                 dropout: float = 0.2,
                 task_names: list = None):
        """
        Initialize MultiTaskTCNSeparate
        
        Args:
            n_forcing: Number of forcing variables
            n_params: Number of parameter variables
            n_lag_features: Number of lag features
            hidden_channels: List of channel sizes for TCN layers
            kernel_size: Convolution kernel size
            dropout: Dropout rate
            task_names: List of task names ['evap', 'sm', 'Q']
        """
        super().__init__()
        
        self.n_forcing = n_forcing
        self.n_params = n_params
        self.n_lag_features = n_lag_features
        self.task_names = task_names or ['evap', 'sm', 'Q']
        self.num_tasks = len(self.task_names)
        self.kernel_size = kernel_size
        
        # Default channel configuration
        if hidden_channels is None:
            hidden_channels = [64, 128, 128, 64]
        
        # Separate encoders for each input type
        self.forcing_tcn = TCN(
            num_inputs=n_forcing,
            num_channels=[32, 64, 64],
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Parameter encoder (simple linear since params don't vary with time)
        self.param_encoder = nn.Sequential(
            nn.Linear(n_params, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32)
        )
        
        # Lag features encoder (if present)
        if n_lag_features > 0:
            self.lag_tcn = TCN(
                num_inputs=n_lag_features,
                num_channels=[16, 32, 32],
                kernel_size=kernel_size,
                dropout=dropout
            )
            combined_features = 64 + 32 + 32  # forcing + params + lags
        else:
            self.lag_tcn = None
            combined_features = 64 + 32  # forcing + params
        
        # Combined processing
        self.combiner = nn.Sequential(
            nn.Linear(combined_features, hidden_channels[-1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name in self.task_names:
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels[-1] // 2, 1)
            )
        
        # Store configuration
        self.config = {
            'hidden_channels': hidden_channels,
            'kernel_size': kernel_size,
            'dropout': dropout,
            'task_names': self.task_names
        }
    
    def forward(self, forcing_seq: torch.Tensor, param_vec: torch.Tensor, 
                lag_seq: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with separate inputs
        
        Args:
            forcing_seq: Forcing sequence (batch_size, seq_len, n_forcing)
            param_vec: Parameter vector (batch_size, n_params)
            lag_seq: Lag sequence (batch_size, seq_len, n_lag_features) or None
            
        Returns:
            Dictionary of task outputs {task_name: tensor}
        """
        batch_size = forcing_seq.size(0)
        
        # Process forcing sequence through TCN
        forcing_seq = forcing_seq.transpose(1, 2)  # (batch_size, n_forcing, seq_len)
        forcing_features = self.forcing_tcn(forcing_seq)  # (batch_size, channels, seq_len)
        forcing_features = forcing_features[:, :, -1]  # Take last timestep (batch_size, channels)
        
        # Process parameters
        param_features = self.param_encoder(param_vec)  # (batch_size, 32)
        
        # Process lag features if present
        if lag_seq is not None and self.lag_tcn is not None:
            lag_seq = lag_seq.transpose(1, 2)  # (batch_size, n_lag_features, seq_len)
            lag_features = self.lag_tcn(lag_seq)  # (batch_size, channels, seq_len)
            lag_features = lag_features[:, :, -1]  # Take last timestep
            
            # Combine all features
            combined = torch.cat([forcing_features, param_features, lag_features], dim=1)
        else:
            # Combine forcing and parameters only
            combined = torch.cat([forcing_features, param_features], dim=1)
        
        # Process combined features
        shared_features = self.combiner(combined)  # (batch_size, hidden_channels[-1])
        
        # Task-specific predictions
        outputs = {}
        for task_name in self.task_names:
            task_output = self.task_heads[task_name](shared_features)  # (batch_size, 1)
            outputs[task_name] = task_output.squeeze(-1)  # (batch_size,)
        
        return outputs
    
    def get_model_info(self) -> str:
        """Get model information string"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = f"""
=== MultiTaskTCNSeparate Model Info ===
Forcing Features: {self.n_forcing}
Parameter Features: {self.n_params}
Lag Features: {self.n_lag_features}
Tasks: {self.task_names} ({self.num_tasks} tasks)
Hidden Channels: {self.config['hidden_channels']}
Kernel Size: {self.kernel_size}
Dropout: {self.config['dropout']:.2f}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Model Size: ~{total_params * 4 / 1024 / 1024:.2f} MB
"""
        return info