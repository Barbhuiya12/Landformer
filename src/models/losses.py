"""
Loss Functions for VIC Surrogate Model
Author: Siddik
Description: Implementation of hydrological performance metrics as loss functions

This module implements specialized loss functions for hydrological modeling that
go beyond traditional regression metrics. These loss functions are designed to
capture the specific characteristics and requirements of hydrological processes.

The module includes:
1. Traditional loss functions (MSE, MAE)
2. Hydrological-specific loss functions (NSE, KGE)
3. Multi-output variants for simultaneous prediction of multiple variables
4. Physics-informed constraint losses (mass conservation)
5. A unified loss function factory for easy configuration

These loss functions are crucial for training models that produce physically
consistent and hydrologically meaningful predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict


class LossFunction:
    """
    Unified loss function factory
    Author: Siddik
    
    This factory class provides a unified interface for creating different types
    of loss functions used in hydrological modeling. It supports both traditional
    regression losses and hydrological-specific metrics.
    
    The factory supports:
    - Standard losses: MSE, MAE
    - Hydrological metrics: NSE, KGE
    - Multi-output variants: MultiOutputNSELoss, MultiOutputKGELoss
    
    This approach allows for easy configuration and switching between different
    loss functions based on the training requirements.
    """
    
    @staticmethod
    def get_loss_function(loss_type: str, **kwargs):
        """
        Get loss function by name
        
        Args:
            loss_type: Type of loss ('MSE', 'NSE', 'KGE', 'MAE', 'multi_NSE', 'multi_KGE')
            **kwargs: Additional arguments for loss functions
            
        Returns:
            Loss function
        """
        # Filter kwargs based on loss type
        # Allow passing weights to single-output NSE/KGE (used as weighted mean across outputs)
        single_output_kwargs = {k: v for k, v in kwargs.items() if k != 'target_names'}
        multi_output_kwargs = kwargs
        
        if loss_type.upper() == 'MSE':
            return nn.MSELoss()
        elif loss_type.upper() == 'MAE':
            return nn.L1Loss()
        elif loss_type.upper() == 'NSE':
            return NSELoss(**single_output_kwargs)
        elif loss_type.upper() == 'KGE':
            return KGELoss(**single_output_kwargs)
        elif loss_type.upper() == 'MULTI_NSE':
            return MultiOutputNSELoss(**multi_output_kwargs)
        elif loss_type.upper() == 'MULTI_KGE':
            return MultiOutputKGELoss(**multi_output_kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


class NSELoss(nn.Module):
    """
    Nash-Sutcliffe Efficiency Loss Function
    Author: Siddik
    
    The Nash-Sutcliffe Efficiency (NSE) is a widely used metric in hydrology
    that measures the relative magnitude of the residual variance compared to
    the variance of the observed data.
    
    NSE = 1 - Σ(obs-sim)²/Σ(obs-mean(obs))²
    Loss = 1 - NSE (lower is better)
    
    An NSE value of 1 indicates perfect model performance, 0 indicates performance
    equivalent to the mean of observations, and negative values indicate that
    the model performs worse than the mean.
    
    This loss function is particularly useful for hydrological modeling because
    it is sensitive to extreme values and provides a good overall assessment
    of model performance.
    
    Attributes:
        epsilon (float): Small value to avoid division by zero
        reduce_spatial (bool): Whether to average over spatial dimensions
    """
    
    def __init__(self, epsilon: float = 1e-8, reduce_spatial: bool = True, weights: Optional[List[float]] = None):
        """
        Initialize NSE loss
        
        Args:
            epsilon: Small value to avoid division by zero
            reduce_spatial: Whether to average over spatial dimensions
        """
        super().__init__()
        self.epsilon = epsilon
        self.reduce_spatial = reduce_spatial
        # Optional weights to compute weighted average across the last dimension (e.g., outputs)
        self.weights = weights
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate NSE loss
        
        Args:
            y_pred: Predictions (B, ny) or (B, T, ny)
            y_true: Ground truth (B, ny) or (B, T, ny)
            
        Returns:
            NSE loss
        """
        # Flatten spatial/temporal dimensions if needed
        if y_pred.ndim == 3:
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            y_true = y_true.reshape(-1, y_true.shape[-1])
        
        if self.reduce_spatial:
            # Calculate NSE for each spatial location separately
            nse_values = []
            for i in range(y_pred.shape[1]):
                pred_i = y_pred[:, i]
                true_i = y_true[:, i]
                
                # Skip if all values are the same (no variance)
                if torch.std(true_i) < self.epsilon:
                    # Use MSE loss for this dimension
                    mse = torch.mean((true_i - pred_i) ** 2)
                    # Convert to NSE-like scale (0 is perfect, higher is worse)
                    nse_i = torch.clamp(mse, min=0.0, max=2.0)  # Clamp to reasonable range
                else:
                    mean_obs = torch.mean(true_i)
                    numerator = torch.sum((true_i - pred_i) ** 2)
                    denominator = torch.sum((true_i - mean_obs) ** 2)
                    
                    # Calculate NSE with numerical stability
                    nse_i = numerator / (denominator + self.epsilon)
                    # Clamp NSE to prevent extreme values
                    nse_i = torch.clamp(nse_i, min=-10.0, max=10.0)
                    nse_main = 1 - nse_i
                
                nse_values.append(nse_main)
                # print(nse_values)
            
            nse_tensor = torch.stack(nse_values)
            # Weighted average if weights provided, else simple mean
            if self.weights is not None and len(self.weights) > 0:
                w = torch.tensor(self.weights[: nse_tensor.numel()], device=nse_tensor.device, dtype=nse_tensor.dtype)
                if w.numel() < nse_tensor.numel():
                    # Pad missing weights with 1.0
                    pad = torch.ones(nse_tensor.numel() - w.numel(), device=nse_tensor.device, dtype=nse_tensor.dtype)
                    w = torch.cat([w, pad], dim=0)
                mean_nse = (nse_tensor * w).sum() / (w.sum() + self.epsilon)
            else:
                mean_nse = nse_tensor.mean()
            # Final clamping to prevent extreme loss values
            loss = 1 - mean_nse
            return torch.clamp(loss, min=-10.0, max=10.0)
        else:
            # Global NSE across all dimensions
            # Check for low variance
            if torch.std(y_true) < self.epsilon:
                # Use MSE loss when variance is too low
                mse = torch.mean((y_true - y_pred) ** 2)
                # Convert to NSE-like scale
                loss = torch.clamp(mse, min=0.0, max=2.0)
                return loss
            
            mean_obs = torch.mean(y_true)
            numerator = torch.sum((y_true - y_pred) ** 2)
            denominator = torch.sum((y_true - mean_obs) ** 2)
            
            # Calculate NSE with clamping
            nse_ratio = numerator / (denominator + self.epsilon)
            nse_ratio = torch.clamp(nse_ratio, min=-10.0, max=10.0)
            nse = 1 - nse_ratio
            
            # Final loss with clamping
            loss = 1 - nse
            return torch.clamp(loss, min=-10.0, max=10.0)


class KGELoss(nn.Module):
    """
    Kling-Gupta Efficiency Loss Function
    Author: Siddik
    
    The Kling-Gupta Efficiency (KGE) is a comprehensive hydrological metric
    that decomposes model performance into three components: correlation,
    variability, and bias. It is particularly useful for hydrological models
    as it accounts for the temporal dynamics that are crucial in hydrological
    processes.
    
    KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)
    Loss = 1 - KGE
    
    Where:
    - r: Pearson correlation coefficient (temporal correlation)
    - α: Ratio of standard deviations (variability ratio)
    - β: Ratio of means (bias ratio)
    
    A KGE value of 1 indicates perfect model performance, while values below 0
    indicate that the model performs worse than the mean of observations.
    
    Attributes:
        epsilon (float): Small value to avoid division by zero
        reduce_spatial (bool): Whether to average over spatial dimensions
    """
    
    def __init__(self, epsilon: float = 1e-8, reduce_spatial: bool = True, weights: Optional[List[float]] = None):
        """
        Initialize KGE loss
        
        Args:
            epsilon: Small value to avoid division by zero
            reduce_spatial: Whether to average over spatial dimensions
        """
        super().__init__()
        self.epsilon = epsilon
        self.reduce_spatial = reduce_spatial
        # Optional weights to compute weighted average across the last dimension (e.g., outputs)
        self.weights = weights
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate KGE loss
        
        Args:
            y_pred: Predictions (B, ny) or (B, T, ny)  
            y_true: Ground truth (B, ny) or (B, T, ny)
            
        Returns:
            KGE loss
        """
        # Flatten if needed
        if y_pred.ndim == 3:
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            y_true = y_true.reshape(-1, y_true.shape[-1])
        
        if self.reduce_spatial:
            kge_values = []
            for i in range(y_pred.shape[1]):
                pred_i = y_pred[:, i]
                true_i = y_true[:, i]
                
                # Handle low variance case
                if torch.std(true_i) < self.epsilon:
                    kge_i = 1 - F.mse_loss(pred_i, true_i)
                else:
                    # Calculate KGE components
                    mean_pred = torch.mean(pred_i)
                    mean_true = torch.mean(true_i)
                    std_pred = torch.std(pred_i)
                    std_true = torch.std(true_i)
                    
                    # Correlation coefficient
                    vx = pred_i - mean_pred
                    vy = true_i - mean_true
                    r_num = torch.sum(vx * vy)
                    r_den = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
                    r = r_num / (r_den + self.epsilon)
                    
                    # Variability ratio and bias ratio
                    alpha = std_pred / (std_true + self.epsilon)
                    beta = mean_pred / (mean_true + self.epsilon)
                    
                    # KGE calculation
                    kge_i = 1 - torch.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                
                kge_values.append(kge_i)
            
            kge_tensor = torch.stack(kge_values)
            # Weighted average if weights provided, else simple mean
            if self.weights is not None and len(self.weights) > 0:
                w = torch.tensor(self.weights[: kge_tensor.numel()], device=kge_tensor.device, dtype=kge_tensor.dtype)
                if w.numel() < kge_tensor.numel():
                    pad = torch.ones(kge_tensor.numel() - w.numel(), device=kge_tensor.device, dtype=kge_tensor.dtype)
                    w = torch.cat([w, pad], dim=0)
                mean_kge = (kge_tensor * w).sum() / (w.sum() + self.epsilon)
            else:
                mean_kge = kge_tensor.mean()
            return 1 - mean_kge
        else:
            # Global KGE
            mean_pred = torch.mean(y_pred)
            mean_true = torch.mean(y_true)
            std_pred = torch.std(y_pred)
            std_true = torch.std(y_true)
            
            vx = y_pred - mean_pred
            vy = y_true - mean_true
            r_num = torch.sum(vx * vy)
            r_den = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
            r = r_num / (r_den + self.epsilon)
            
            alpha = std_pred / (std_true + self.epsilon)
            beta = mean_pred / (mean_true + self.epsilon)
            
            kge = 1 - torch.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            return 1 - kge


class MultiOutputNSELoss(nn.Module):
    """
    Multi-output NSE Loss Function with weighted outputs
    Author: Siddik
    
    This class extends the NSE loss function to handle multi-output scenarios
    where multiple hydrological variables are predicted simultaneously.
    
    It calculates the NSE for each output variable separately and then
    combines them using optional weights. This allows for different importance
    to be assigned to different variables during training.
    
    For example, in a VIC surrogate model predicting evapotranspiration,
    runoff, and soil moisture, different weights can be assigned based on
    the relative importance or difficulty of predicting each variable.
    
    Attributes:
        weights (Optional[List[float]]): Weights for each output variable
        target_names (Optional[List[str]]): Names of target variables
        epsilon (float): Small value to avoid division by zero
        nse_loss (NSELoss): Single-output NSE loss instance
    """
    
    def __init__(self, weights: Optional[List[float]] = None, 
                 target_names: Optional[List[str]] = None, epsilon: float = 1e-8):
        """
        Initialize multi-output NSE loss
        
        Args:
            weights: List of weights for each output variable
            target_names: Names of target variables for logging
            epsilon: Small value to avoid division by zero
        """
        super().__init__()
        self.weights = weights
        if target_names is None:
            raise ValueError("target_names must be provided for MultiOutputNSELoss")
        self.target_names = target_names
        self.epsilon = epsilon
        self.nse_loss = NSELoss(epsilon=epsilon, reduce_spatial=True)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted multi-output NSE loss
        
        Args:
            y_pred: Predictions (B, ny)
            y_true: Ground truth (B, ny)
            
        Returns:
            Weighted multi-output NSE loss
        """
        ny = y_pred.shape[1]
        losses = []
        
        # Calculate NSE loss for each output
        for i in range(ny):
            pred_i = y_pred[:, i:i+1]
            true_i = y_true[:, i:i+1]
            
            # Calculate NSE for this output
            loss_i = self.nse_loss(pred_i, true_i)
            losses.append(loss_i)
        
        # Apply weights if provided and calculate weighted average
        if self.weights is not None:
            weighted_sum = 0.0
            total_weight = 0.0
            for i, loss in enumerate(losses):
                weight = self.weights[i] if i < len(self.weights) else 1.0
                weighted_sum += weight * loss
                total_weight += weight
            
            # Return weighted average loss
            if total_weight > 0:
                total_loss = weighted_sum / total_weight
            else:
                total_loss = torch.stack(losses).mean()
        else:
            # Return simple average loss
            total_loss = torch.stack(losses).mean()
        
        return total_loss
    
    def get_individual_losses(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get individual losses for each output"""
        ny = y_pred.shape[1]
        individual_losses = {}
        
        for i in range(ny):
            pred_i = y_pred[:, i:i+1]
            true_i = y_true[:, i:i+1]
            loss_i = self.nse_loss(pred_i, true_i)
            
            var_name = self.target_names[i] if i < len(self.target_names) else f'output_{i}'
            individual_losses[f'nse_{var_name}'] = loss_i
        
        return individual_losses


class MultiOutputKGELoss(nn.Module):
    """
    Multi-output KGE Loss Function with weighted outputs
    Author: Siddik
    
    This class extends the KGE loss function to handle multi-output scenarios
    where multiple hydrological variables are predicted simultaneously.
    
    It calculates the KGE for each output variable separately and then
    combines them using optional weights. This allows for different importance
    to be assigned to different variables during training.
    
    The multi-output KGE loss is particularly useful when different variables
    have different characteristics or when certain variables are more important
    for the specific application.
    
    Attributes:
        weights (Optional[List[float]]): Weights for each output variable
        target_names (Optional[List[str]]): Names of target variables
        epsilon (float): Small value to avoid division by zero
        kge_loss (KGELoss): Single-output KGE loss instance
    """
    
    def __init__(self, weights: Optional[List[float]] = None, 
                 target_names: Optional[List[str]] = None, epsilon: float = 1e-8):
        """
        Initialize multi-output KGE loss
        
        Args:
            weights: List of weights for each output variable
            target_names: Names of target variables for logging
            epsilon: Small value to avoid division by zero
        """
        super().__init__()
        self.weights = weights
        if target_names is None:
            raise ValueError("target_names must be provided for MultiOutputKGELoss")
        self.target_names = target_names
        self.epsilon = epsilon
        self.kge_loss = KGELoss(epsilon=epsilon, reduce_spatial=True)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted multi-output KGE loss
        
        Args:
            y_pred: Predictions (B, ny)
            y_true: Ground truth (B, ny)
            
        Returns:
            Weighted multi-output KGE loss
        """
        ny = y_pred.shape[1]
        losses = []
        
        # Calculate KGE loss for each output
        for i in range(ny):
            pred_i = y_pred[:, i:i+1]
            true_i = y_true[:, i:i+1]
            
            # Calculate KGE for this output
            loss_i = self.kge_loss(pred_i, true_i)
            losses.append(loss_i)
        
        # Apply weights if provided
        if self.weights is not None:
            weighted_sum = 0.0
            total_weight = 0.0
            for i, loss in enumerate(losses):
                weight = self.weights[i] if i < len(self.weights) else 1.0
                weighted_sum += weight * loss
                total_weight += weight
            
            # Return weighted average loss
            if total_weight > 0:
                total_loss = weighted_sum / total_weight
            else:
                total_loss = torch.stack(losses).mean()
        else:
            # Return simple average loss
            total_loss = torch.stack(losses).mean()
        
        return total_loss
    
    def get_individual_losses(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get individual losses for each output"""
        ny = y_pred.shape[1]
        individual_losses = {}
        
        for i in range(ny):
            pred_i = y_pred[:, i:i+1]
            true_i = y_true[:, i:i+1]
            loss_i = self.kge_loss(pred_i, true_i)
            
            var_name = self.target_names[i] if i < len(self.target_names) else f'output_{i}'
            individual_losses[f'kge_{var_name}'] = loss_i
        
        return individual_losses


class MassConservationLoss(nn.Module):
    """
    Mass Conservation Constraint Loss
    Author: Siddik
    
    Ensures physical realism by penalizing negative fluxes and 
    checking mass conservation principles
    
    This constraint loss ensures that hydrological fluxes remain within
    physically realistic bounds:
    - Evapotranspiration (ET) cannot be negative
    - Runoff (Q) cannot be negative
    - Other flux variables should generally be non-negative
    
    These constraints are particularly important for preventing unphysical
    predictions that might occur during early training stages.
    
    Attributes:
        check_negative (bool): Whether to check for negative fluxes
        epsilon (float): Small value for numerical stability
    """
    
    def __init__(self, check_negative: bool = True, epsilon: float = 1e-8):
        """
        Initialize mass conservation loss
        
        Args:
            check_negative: Whether to penalize negative fluxes
            epsilon: Small value for numerical stability
        """
        super().__init__()
        self.check_negative = check_negative
        self.epsilon = epsilon
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate mass conservation loss
        
        Args:
            y_pred: Model predictions (B, ny)
            y_true: Ground truth targets (B, ny)
            
        Returns:
            Mass conservation loss
        """
        losses = []
        
        if self.check_negative:
            # Penalize negative fluxes (ET and runoff should be non-negative)
            for i in range(y_pred.shape[1]):
                # Most hydrological fluxes should be non-negative
                negative_penalty = torch.relu(-y_pred[:, i])  # Penalty for negative values
                losses.append(negative_penalty.mean())
        
        # Additional mass conservation checks could be added here
        # For example, ensuring ET doesn't exceed available water
        
        if len(losses) > 0:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)


class PhysicalConstraintLoss(nn.Module):
    """
    Combined Physical Constraint Loss for VIC Surrogate Model
    Author: Siddik
    
    Implements mass conservation constraints
    
    This class serves as a container for all enabled physical constraints,
    managing their initialization based on configuration and computing
    their combined contribution to the total loss function.
    
    The class dynamically initializes constraint modules based on the
    configuration settings and only includes enabled constraints in
    the forward pass computation.
    
    Attributes:
        config (Dict): Physical constraints configuration
        target_vars (List[str]): List of target variable names
        dataset: Training dataset for normalization statistics
        forcing_vars (List[str]): List of forcing variable names
        mass_conservation_loss (Optional[MassConservationLoss]): Mass conservation constraint
        mc_weight (float): Weight for mass conservation constraint
    """
    
    def __init__(self, config_constraints: Dict, target_variables: List[str], forcing_variables: List[str] = None):
        """
        Initialize combined physical constraint loss
        
        Args:
            config_constraints: Physical constraints configuration from config file
            target_variables: List of target variable names for indexing
            forcing_variables: List of forcing variable names for indexing
        """
        super().__init__()
        
        self.config = config_constraints
        self.target_vars = target_variables
        self.forcing_vars = forcing_variables or []
        
        # Initialize constraint losses
        if self.config.get('mass_conservation', {}).get('enabled', False):
            mc_config = self.config['mass_conservation']
            self.mass_conservation_loss = MassConservationLoss(
                check_negative=mc_config.get('check_negative_fluxes', True)
            )
            self.mc_weight = mc_config.get('weight', 0.05)
        else:
            self.mass_conservation_loss = None
            self.mc_weight = 0.0
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
                forcing_data: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Calculate combined physical constraint losses
        
        Args:
            y_pred: Model predictions (B, ny)
            y_true: Ground truth targets (B, ny)
            forcing_data: Forcing data (B, seq_len, n_forcing)
            
        Returns:
            Dictionary of constraint losses
        """
        constraint_losses = {}
        total_loss = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        
        # Mass conservation loss  
        if self.mass_conservation_loss is not None:
            mc_loss = self.mass_conservation_loss(y_pred, y_true)
            constraint_losses['mass_conservation'] = mc_loss
            total_loss = total_loss + self.mc_weight * mc_loss
        
        constraint_losses['total_constraint'] = total_loss
        return constraint_losses