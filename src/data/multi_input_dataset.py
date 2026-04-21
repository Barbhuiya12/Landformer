"""
Multi-Input Dataset Adapter for MultiTaskLSTM Architecture
Author: Siddik & Claude
Description: Adapter to convert existing concatenated dataset to separate inputs
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, Any
import logging
from .dataset import UnifiedSurrogateDataset


class MultiInputDatasetAdapter(Dataset):
    """
    Adapter that wraps the existing UnifiedSurrogateDataset to provide
    separate inputs (forcing, params, lags) for the new MultiTaskLSTM architecture.
    
    Converts from:
        - Single concatenated tensor: (batch, seq_len, total_features)
    To:
        - forcing_seq: (batch, seq_len, n_forcing)  
        - param_vec: (batch, n_params)
        - lag_seq: (batch, seq_len, n_lag_features) or None
    """
    
    def __init__(self, 
                 original_dataset: UnifiedSurrogateDataset,
                 n_forcing: int,
                 n_params: int,
                 n_lag_features: int = 0):
        """
        Initialize the adapter
        
        Args:
            original_dataset: The existing UnifiedSurrogateDataset
            n_forcing: Number of forcing variables (e.g., 7)
            n_params: Number of parameter variables (e.g., 13) 
            n_lag_features: Number of lag features (default: 0)
        """
        self.dataset = original_dataset
        self.n_forcing = n_forcing
        self.n_params = n_params
        self.n_lag_features = n_lag_features
        
        # Validate dimensions
        expected_total = n_forcing + n_params + n_lag_features
        if hasattr(original_dataset, 'n_exo') and hasattr(original_dataset, 'n_lag_features'):
            actual_total = original_dataset.n_exo + original_dataset.n_lag_features
            if expected_total != actual_total:
                logging.warning(f"Dimension mismatch: expected {expected_total}, got {actual_total}")
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"MultiInputAdapter: forcing={n_forcing}, params={n_params}, lags={n_lag_features}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], 
                                           torch.Tensor, Dict[str, Any]]:
        """
        Get a single sample with separated inputs
        
        Returns:
            ((forcing_seq, param_vec, lag_seq), targets, metadata)
        """
        # Get original data
        inputs, targets, metadata = self.dataset[idx]
        
        # inputs shape: (seq_len, total_features)
        seq_len = inputs.shape[0]
        
        # Split concatenated inputs
        forcing_end = self.n_forcing
        params_end = forcing_end + self.n_params
        
        # Extract each component
        forcing_seq = inputs[:, :forcing_end]  # (seq_len, n_forcing)
        params_seq = inputs[:, forcing_end:params_end]  # (seq_len, n_params)
        
        # Parameters are repeated across sequence - take first timestep
        param_vec = params_seq[0, :]  # (n_params,)
        
        # Extract lag features if present
        lag_seq = None
        if self.n_lag_features > 0:
            lag_seq = inputs[:, params_end:]  # (seq_len, n_lag_features)
        
        # Package separated inputs
        separated_inputs = (forcing_seq, param_vec, lag_seq)
        
        return separated_inputs, targets, metadata
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions for each input type"""
        return {
            'n_forcing': self.n_forcing,
            'n_params': self.n_params, 
            'n_lag_features': self.n_lag_features
        }
    
    def get_task_names(self) -> list:
        """Get task names from original dataset"""
        if hasattr(self.dataset, 'target_variables'):
            return self.dataset.target_variables
        return ['evap', 'sm', 'Q']  # Default
    
    def get_normalization_stats(self, variable_name: str) -> Tuple[float, float]:
        """Get normalization stats from original dataset"""
        if hasattr(self.dataset, 'get_normalization_stats'):
            return self.dataset.get_normalization_stats(variable_name)
        return 0.0, 1.0  # Default: no normalization


def create_multi_input_dataloaders(config, 
                                  n_forcing: int, 
                                  n_params: int, 
                                  n_lag_features: int = 0) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Convenience function to create train/valid/test dataloaders with the adapter
    
    Args:
        config: UnifiedConfig object
        n_forcing: Number of forcing variables
        n_params: Number of parameter variables
        n_lag_features: Number of lag features
        
    Returns:
        (train_loader, valid_loader, test_loader) with MultiInputDatasetAdapter
    """
    from torch.utils.data import DataLoader
    
    # Create original datasets
    train_dataset = UnifiedSurrogateDataset(config, mode='train')
    valid_dataset = UnifiedSurrogateDataset(config, mode='valid')  
    test_dataset = UnifiedSurrogateDataset(config, mode='test')
    
    # Wrap with adapters
    train_adapter = MultiInputDatasetAdapter(train_dataset, n_forcing, n_params, n_lag_features)
    valid_adapter = MultiInputDatasetAdapter(valid_dataset, n_forcing, n_params, n_lag_features)
    test_adapter = MultiInputDatasetAdapter(test_dataset, n_forcing, n_params, n_lag_features)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_adapter,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.memory.num_workers,
        pin_memory=config.training.memory.pin_memory,
        drop_last=config.training.memory.drop_last
    )
    
    valid_loader = DataLoader(
        valid_adapter,
        batch_size=config.training.batch_size_valid,
        shuffle=False,
        num_workers=config.training.memory.num_workers,
        pin_memory=config.training.memory.pin_memory
    )
    
    test_loader = DataLoader(
        test_adapter,
        batch_size=config.training.batch_size_valid,
        shuffle=False,
        num_workers=config.training.memory.num_workers,
        pin_memory=config.training.memory.pin_memory
    )
    
    return train_loader, valid_loader, test_loader, train_adapter