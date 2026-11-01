"""
Loss Functions for VIC Surrogate Model
Author: Siddik Barbhuiya
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
    Author: Siddik Barbhuiya
    
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
        # target_names is only used by multi-output losses
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
        elif loss_type.upper() == 'GRID_NSE':
            return GridAwareNSELoss(**single_output_kwargs)
        elif loss_type.upper() == 'GRID_KGE':
            return GridAwareKGELoss(**single_output_kwargs)
        elif loss_type.upper() == 'MULTI_NSE':
            return MultiOutputNSELoss(**multi_output_kwargs)
        elif loss_type.upper() == 'MULTI_KGE':
            return MultiOutputKGELoss(**multi_output_kwargs)
        elif loss_type.upper() in ['BIAS_NSE', 'BIAS_CORRECTED_NSE']:
            # Bias-corrected NSE for fixing magnitude issues during fine-tuning
            bias_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ['nse_weight', 'bias_weight', 'mae_weight', 
                                  'epsilon', 'reduce_spatial', 'weights']}
            return BiasCorrectedNSELoss(**bias_kwargs)
        elif loss_type.upper() == 'FOCAL_MSE':
            # Focal loss for extreme events
            focal_kwargs = {k: v for k, v in kwargs.items() 
                           if k in ['gamma', 'reduction', 'normalize', 'epsilon']}
            return FocalMSELoss(**focal_kwargs)
        elif loss_type.upper() == 'LABEL_SMOOTH_MSE':
            # Label smoothing for regularization
            smooth_kwargs = {k: v for k, v in kwargs.items() 
                            if k in ['smoothing', 'reduction', 'use_running_mean', 'momentum']}
            return LabelSmoothingMSE(**smooth_kwargs)
        elif loss_type.upper() == 'ENHANCED':
            # Combined enhanced loss with focal + smoothing
            enhanced_kwargs = {k: v for k, v in kwargs.items() 
                              if k in ['base_loss', 'focal_gamma', 'label_smoothing', 
                                      'use_focal', 'use_smoothing', 'weights']}
            return CombinedEnhancedLoss(**enhanced_kwargs)
        elif loss_type.upper() == 'BIAS_CORRECTED_MSE':
            # Bias-corrected MSE for explicit bias penalty
            bias_mse_kwargs = {k: v for k, v in kwargs.items() 
                              if k in ['bias_weight', 'reduction']}
            return BiasCorrectedMSELoss(**bias_mse_kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


class NSELoss(nn.Module):
    """
    Nash-Sutcliffe Efficiency Loss Function
    Author: Siddik Barbhuiya
    
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
                    # Use MSE loss for this dimension (low variance fallback)
                    mse = torch.mean((true_i - pred_i) ** 2)
                    # Convert to NSE-like scale: 1 is perfect, lower is worse
                    nse_main = 1.0 - torch.clamp(mse, min=0.0, max=2.0)
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
    Author: Siddik Barbhuiya
    
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
    
    def __init__(self, epsilon: float = 1e-8, reduce_spatial: bool = True, weights: Optional[List[float]] = None, **kwargs):
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
        
        # Component weights for KGE calculation
        # Default to 1.0 (standard KGE)
        self.corr_weight = kwargs.get('corr_weight', 1.0)
        self.var_weight = kwargs.get('var_weight', 1.0)
        self.bias_weight = kwargs.get('bias_weight', 1.0)
    
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
                    kge_i = 1.0 - torch.clamp(F.mse_loss(pred_i, true_i), min=0.0, max=2.0)
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
                    # KGE calculation with component weights
                    # KGE = 1 - sqrt(w_r*(r-1)² + w_a*(α-1)² + w_b*(β-1)²)
                    kge_term = self.corr_weight * (r - 1)**2 + \
                               self.var_weight * (alpha - 1)**2 + \
                               self.bias_weight * (beta - 1)**2
                    kge_i = 1 - torch.sqrt(kge_term)
                
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
            
            kge_term = (self.corr_weight * (r - 1))**2 + \
                       (self.var_weight * (alpha - 1))**2 + \
                       (self.bias_weight * (beta - 1))**2
            kge = 1 - torch.sqrt(kge_term)
            return 1 - kge


class MultiOutputNSELoss(nn.Module):
    """
    Multi-output NSE Loss Function with weighted outputs
    Author: Siddik Barbhuiya
    
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
    Author: Siddik Barbhuiya
    
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


class BiasCorrectedNSELoss(nn.Module):
    """
    Bias-Corrected NSE Loss Function for Fixing Magnitude Prediction Issues
    Author: Siddik Barbhuiya
    
    This loss function combines NSE with explicit bias penalty and MAE to fix
    the common issue where models capture trends but fail on magnitude.
    
    Loss = nse_weight * NSE_loss + bias_weight * |mean(pred) - mean(obs)| + mae_weight * MAE
    
    The bias component directly penalizes systematic over/underestimation,
    which pure NSE/KGE losses don't adequately address.
    
    Recommended for fine-tuning when:
    - Model captures correlation/trend well but magnitudes are off
    - System shows consistent bias (e.g., +22% overestimation)
    - You want to explicitly correct scale without affecting pattern learning
    
    Attributes:
        nse_weight (float): Weight for NSE component (pattern matching)
        bias_weight (float): Weight for bias penalty (magnitude correction)
        mae_weight (float): Weight for MAE (absolute error)
        epsilon (float): Small value for numerical stability
    """
    
    def __init__(self, nse_weight: float = 0.4, bias_weight: float = 0.4, 
                 mae_weight: float = 0.2, epsilon: float = 1e-8,
                 reduce_spatial: bool = True, weights: Optional[List[float]] = None):
        """
        Initialize Bias-Corrected NSE loss
        
        Args:
            nse_weight: Weight for NSE loss component (0.0-1.0)
            bias_weight: Weight for bias penalty (0.0-1.0) - CRITICAL for magnitude fix
            mae_weight: Weight for MAE loss component (0.0-1.0)
            epsilon: Small value for numerical stability
            reduce_spatial: Whether to calculate per-output then average
            weights: Optional weights for multi-output averaging
        """
        super().__init__()
        self.nse_weight = nse_weight
        self.bias_weight = bias_weight
        self.mae_weight = mae_weight
        self.epsilon = epsilon
        self.reduce_spatial = reduce_spatial
        self.weights = weights
        
        # Use existing NSE loss for the NSE component
        self.nse_loss = NSELoss(epsilon=epsilon, reduce_spatial=reduce_spatial, weights=weights)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate bias-corrected NSE loss
        
        Args:
            y_pred: Predictions (B, ny) or (B, T, ny)
            y_true: Ground truth (B, ny) or (B, T, ny)
            
        Returns:
            Combined loss value
        """
        # Flatten if needed
        if y_pred.ndim == 3:
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            y_true = y_true.reshape(-1, y_true.shape[-1])
        
        # 1. NSE component (pattern matching)
        nse_loss = self.nse_loss(y_pred, y_true)
        
        # 2. Bias component (magnitude correction) - KEY for fixing overestimation
        if self.reduce_spatial and y_pred.shape[1] > 1:
            # Calculate bias per output
            bias_losses = []
            for i in range(y_pred.shape[1]):
                bias_i = torch.mean(y_pred[:, i]) - torch.mean(y_true[:, i])
                bias_losses.append(torch.abs(bias_i))
            bias_tensor = torch.stack(bias_losses)
            
            # Weighted average if weights provided
            if self.weights is not None and len(self.weights) > 0:
                w = torch.tensor(self.weights[:bias_tensor.numel()], 
                               device=bias_tensor.device, dtype=bias_tensor.dtype)
                if w.numel() < bias_tensor.numel():
                    pad = torch.ones(bias_tensor.numel() - w.numel(), 
                                   device=bias_tensor.device, dtype=bias_tensor.dtype)
                    w = torch.cat([w, pad], dim=0)
                bias_loss = (bias_tensor * w).sum() / (w.sum() + self.epsilon)
            else:
                bias_loss = bias_tensor.mean()
        else:
            # Global bias
            bias = torch.mean(y_pred) - torch.mean(y_true)
            bias_loss = torch.abs(bias)
        
        # 3. MAE component (absolute magnitude accuracy)
        mae_loss = torch.mean(torch.abs(y_pred - y_true))
        
        # Combine all components
        total_loss = (self.nse_weight * nse_loss + 
                     self.bias_weight * bias_loss + 
                     self.mae_weight * mae_loss)
        
        return total_loss
    
    def get_components(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get individual loss components for debugging"""
        if y_pred.ndim == 3:
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            y_true = y_true.reshape(-1, y_true.shape[-1])
            
        nse_loss = self.nse_loss(y_pred, y_true)
        bias = torch.mean(y_pred) - torch.mean(y_true)
        mae_loss = torch.mean(torch.abs(y_pred - y_true))
        
        return {
            'nse_loss': nse_loss,
            'bias': bias,
            'bias_loss': torch.abs(bias),
            'mae_loss': mae_loss,
            'total_loss': self.forward(y_pred, y_true)
        }


class FocalMSELoss(nn.Module):
    """
    Focal Loss for Regression - Focuses on Hard Samples (Extreme Events)
    Author: Siddik Barbhuiya
    
    This loss function applies a focal weighting to MSE, making the model
    focus more on samples with large errors (extreme events) and less on
    easy samples where predictions are already accurate.
    
    Loss = |error|^gamma * MSE
    
    For gamma=0: Standard MSE (equal weight to all samples)
    For gamma>0: Up-weight hard samples (large errors), down-weight easy ones
    
    Benefits:
    - Better handling of extreme hydrological events (floods, droughts)
    - Prevents model from ignoring rare but important events
    - More balanced learning across different conditions
    
    Attributes:
        gamma (float): Focusing parameter (0-5, recommended 2.0)
        reduction (str): 'mean' or 'sum'
        normalize (bool): Whether to normalize weights
    """
    
    def __init__(self, gamma: float = 2.0, reduction: str = 'mean', 
                 normalize: bool = True, epsilon: float = 1e-8):
        """
        Initialize Focal MSE Loss
        
        Args:
            gamma: Focusing parameter. Higher = more focus on hard samples
                   0.0 = standard MSE
                   1.0 = linear focus
                   2.0 = quadratic focus (recommended)
                   3.0+ = very aggressive focusing
            reduction: 'mean' or 'sum'
            normalize: If True, normalize focal weights to sum to 1
            epsilon: Small value for numerical stability
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.normalize = normalize
        self.epsilon = epsilon
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal MSE loss
        
        Args:
            y_pred: Predictions (B, ny) or (B, T, ny)
            y_true: Ground truth (B, ny) or (B, T, ny)
            
        Returns:
            Focal MSE loss
        """
        # Flatten if needed
        if y_pred.ndim == 3:
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            y_true = y_true.reshape(-1, y_true.shape[-1])
        
        # Calculate per-sample squared error
        se = (y_pred - y_true) ** 2  # (B, ny)
        
        # Calculate absolute error for focal weights
        abs_error = torch.abs(y_pred - y_true)  # (B, ny)
        
        # Normalize errors to [0, 1] range for stable focal weights
        max_error = abs_error.max() + self.epsilon
        normalized_error = abs_error / max_error
        
        # Focal weights: w = |error|^gamma
        focal_weights = (normalized_error + self.epsilon) ** self.gamma
        
        # Optionally normalize weights
        if self.normalize:
            focal_weights = focal_weights / (focal_weights.mean() + self.epsilon)
        
        # Apply focal weights to squared errors
        focal_se = focal_weights * se
        
        # Reduction
        if self.reduction == 'mean':
            return focal_se.mean()
        elif self.reduction == 'sum':
            return focal_se.sum()
        else:
            return focal_se


class LabelSmoothingMSE(nn.Module):
    """
    Label Smoothing for Regression - Regularization to Prevent Overconfidence
    Author: Siddik Barbhuiya
    
    This loss function applies label smoothing to regression targets during
    training. It perturbs the target values slightly towards the global mean,
    which acts as regularization and prevents the model from becoming
    overconfident in its predictions.
    
    Smoothed target = target * (1 - smoothing) + mean * smoothing
    
    Benefits:
    - Prevents overfitting to training data
    - More robust to label noise in training data
    - Better generalization to unseen grids/conditions
    - Reduces overconfident predictions
    
    Attributes:
        smoothing (float): Smoothing factor (0-1, recommended 0.05-0.1)
        reduction (str): 'mean' or 'sum'
    """
    
    def __init__(self, smoothing: float = 0.05, reduction: str = 'mean',
                 use_running_mean: bool = True, momentum: float = 0.99):
        """
        Initialize Label Smoothing MSE Loss
        
        Args:
            smoothing: Smoothing factor (0.0 = no smoothing, 1.0 = predict mean)
                       Recommended range: 0.01-0.1
            reduction: 'mean' or 'sum'
            use_running_mean: If True, use EMA of batch means as smooth target
            momentum: Momentum for running mean update
        """
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.use_running_mean = use_running_mean
        self.momentum = momentum
        
        # Running mean for more stable smoothing target
        self.register_buffer('running_mean', torch.zeros(1))


        self.register_buffer('num_batches', torch.zeros(1))
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate label smoothed MSE loss
        
        Args:
            y_pred: Predictions (B, ny) or (B, T, ny)
            y_true: Ground truth (B, ny) or (B, T, ny)
            
        Returns:
            Label smoothed MSE loss
        """
        # Flatten if needed
        if y_pred.ndim == 3:
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            y_true = y_true.reshape(-1, y_true.shape[-1])
        
        # Calculate smoothing target
        if self.training and self.smoothing > 0:
            # Get batch mean
            batch_mean = y_true.mean()
            
            # Update running mean if using EMA
            if self.use_running_mean:
                if self.num_batches == 0:
                    self.running_mean = batch_mean.detach()
                else:
                    self.running_mean = (self.momentum * self.running_mean + 
                                        (1 - self.momentum) * batch_mean.detach())
                self.num_batches += 1
                smooth_target = self.running_mean
            else:
                smooth_target = batch_mean
            
            # Apply label smoothing
            y_true_smooth = y_true * (1 - self.smoothing) + smooth_target * self.smoothing
        else:
            # No smoothing during evaluation
            y_true_smooth = y_true
        
        # Calculate MSE with smoothed targets
        se = (y_pred - y_true_smooth) ** 2
        
        # Reduction
        if self.reduction == 'mean':
            return se.mean()
        elif self.reduction == 'sum':
            return se.sum()
        else:
            return se


class CombinedEnhancedLoss(nn.Module):
    """
    Combined Loss with Focal Weighting and Label Smoothing
    Author: Siddik Barbhuiya
    
    Combines multiple loss components:
    - Primary loss (NSE/KGE/Grid-aware)
    - Focal weighting for extreme events
    - Label smoothing for regularization
    
    This provides a unified interface for using all enhancements together.
    """
    
    def __init__(self, 
                 base_loss: str = 'GRID_KGE',
                 focal_gamma: float = 2.0,
                 label_smoothing: float = 0.05,
                 use_focal: bool = True,
                 use_smoothing: bool = True,
                 **kwargs):
        """
        Initialize Combined Enhanced Loss
        
        Args:
            base_loss: Base loss type ('NSE', 'KGE', 'GRID_NSE', 'GRID_KGE')
            focal_gamma: Focal loss gamma parameter
            label_smoothing: Label smoothing factor
            use_focal: Whether to use focal weighting
            use_smoothing: Whether to use label smoothing
        """
        super().__init__()
        
        # Base loss function
        self.base_loss = LossFunction.get_loss_function(base_loss, **kwargs)
        
        # Focal and smoothing components
        self.use_focal = use_focal
        self.use_smoothing = use_smoothing
        
        if use_focal:
            self.focal = FocalMSELoss(gamma=focal_gamma)
        if use_smoothing:
            self.smoothing = LabelSmoothingMSE(smoothing=label_smoothing)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
                grid_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate combined enhanced loss
        """
        # Apply label smoothing to targets if training
        if self.use_smoothing and self.training:
            y_true_smooth = self._apply_smoothing(y_true)
        else:
            y_true_smooth = y_true
        
        # Calculate base loss
        if hasattr(self.base_loss, 'forward') and 'grid_indices' in str(type(self.base_loss)):
            base_loss_val = self.base_loss(y_pred, y_true_smooth, grid_indices)
        else:
            base_loss_val = self.base_loss(y_pred, y_true_smooth)
        
        # Add focal component if enabled
        if self.use_focal:
            focal_loss = self.focal(y_pred, y_true)
            # Weighted combination (focal as auxiliary loss)
            return base_loss_val + 0.1 * focal_loss
        
        return base_loss_val
    
    def _apply_smoothing(self, y_true: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing to targets"""
        if y_true.ndim == 3:
            y_true = y_true.reshape(-1, y_true.shape[-1])
        batch_mean = y_true.mean()
        return y_true * (1 - self.smoothing.smoothing) + batch_mean * self.smoothing.smoothing

class MassConservationLoss(nn.Module):
    """
    Mass Conservation Constraint Loss
    Author: Siddik Barbhuiya
    
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
    Author: Siddik Barbhuiya
    
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


class GridAwareNSELoss(nn.Module):
    """
    Grid-Aware Nash-Sutcliffe Efficiency Loss Function
    Author: Siddik Barbhuiya
    
    This loss computes NSE for each grid separately and then averages them,
    ensuring that each grid contributes equally to the loss regardless of
    how many samples it contains in the batch.
    
    This is crucial for spatial generalization as it prevents grids with more
    samples from dominating the training signal.
    
    Attributes:
        epsilon (float): Small value to avoid division by zero
        weights (list): Optional per-variable weights
    """
    
    def __init__(self, epsilon: float = 1e-8, weights: Optional[List[float]] = None):
        super().__init__()
        self.epsilon = epsilon
        self.weights = weights
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
                grid_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate grid-aware NSE loss.
        
        Args:
            y_pred: Predictions (B, ny)
            y_true: Ground truth (B, ny)
            grid_indices: (B,) grid index for each sample
            
        Returns:
            Mean of per-grid NSE losses
        """
        n_vars = y_pred.shape[-1]
        
        if grid_indices is None:
            # Fallback to standard NSE if no grid indices
            return self._compute_standard_nse(y_pred, y_true)
        
        # Get unique grids
        unique_grids = torch.unique(grid_indices)
        
        grid_nse_losses = []
        for grid_idx in unique_grids:
            mask = (grid_indices == grid_idx)
            if mask.sum() < 5:  # Skip grids with too few samples
                continue
            
            pred_grid = y_pred[mask]
            true_grid = y_true[mask]
            
            # Compute NSE for each variable in this grid
            var_nse_values = []
            for var_idx in range(n_vars):
                pred_var = pred_grid[:, var_idx]
                true_var = true_grid[:, var_idx]
                
                if torch.std(true_var) < self.epsilon:
                    nse_var = torch.tensor(0.0, device=y_pred.device)
                else:
                    mean_obs = torch.mean(true_var)
                    ss_res = torch.sum((true_var - pred_var) ** 2)
                    ss_tot = torch.sum((true_var - mean_obs) ** 2)
                    nse_var = 1 - (ss_res / (ss_tot + self.epsilon))
                    nse_var = torch.clamp(nse_var, min=-10.0, max=1.0)
                
                var_nse_values.append(nse_var)
            
            # Average across variables (with optional weights)
            var_nse_tensor = torch.stack(var_nse_values)
            if self.weights is not None:
                w = torch.tensor(self.weights[:n_vars], device=y_pred.device, dtype=y_pred.dtype)
                grid_nse = (var_nse_tensor * w).sum() / (w.sum() + self.epsilon)
            else:
                grid_nse = var_nse_tensor.mean()
            
            grid_nse_losses.append(grid_nse)
        
        if len(grid_nse_losses) == 0:
            return self._compute_standard_nse(y_pred, y_true)
        
        # Average NSE across all grids (each grid contributes equally)
        mean_nse = torch.stack(grid_nse_losses).mean()
        
        # Return loss (1 - NSE)
        return 1 - mean_nse
    
    def _compute_standard_nse(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Fallback standard NSE computation."""
        n_vars = y_pred.shape[-1]
        nse_values = []
        
        for var_idx in range(n_vars):
            pred_var = y_pred[:, var_idx]
            true_var = y_true[:, var_idx]
            
            if torch.std(true_var) < self.epsilon:
                nse_var = torch.tensor(0.0, device=y_pred.device)
            else:
                mean_obs = torch.mean(true_var)
                ss_res = torch.sum((true_var - pred_var) ** 2)
                ss_tot = torch.sum((true_var - mean_obs) ** 2)
                nse_var = 1 - (ss_res / (ss_tot + self.epsilon))
                nse_var = torch.clamp(nse_var, min=-10.0, max=1.0)
            
            nse_values.append(nse_var)
        
        return 1 - torch.stack(nse_values).mean()


class GridAwareKGELoss(nn.Module):
    """
    Grid-Aware Kling-Gupta Efficiency Loss Function
    Author: Siddik Barbhuiya
    
    This loss computes KGE for each grid separately and then averages them,
    ensuring that each grid contributes equally to the loss regardless of
    how many samples it contains in the batch.
    
    Attributes:
        epsilon (float): Small value to avoid division by zero
        weights (list): Optional per-variable weights
    """
    
    def __init__(self, epsilon: float = 1e-8, weights: Optional[List[float]] = None):
        super().__init__()
        self.epsilon = epsilon
        self.weights = weights
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                grid_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate grid-aware KGE loss.
        
        Args:
            y_pred: Predictions (B, ny)
            y_true: Ground truth (B, ny)
            grid_indices: (B,) grid index for each sample
            
        Returns:
            Mean of per-grid KGE losses
        """
        n_vars = y_pred.shape[-1]
        
        if grid_indices is None:
            return self._compute_standard_kge(y_pred, y_true)
        
        unique_grids = torch.unique(grid_indices)
        
        grid_kge_losses = []
        for grid_idx in unique_grids:
            mask = (grid_indices == grid_idx)
            if mask.sum() < 5:
                continue
            
            pred_grid = y_pred[mask]
            true_grid = y_true[mask]
            
            var_kge_values = []
            for var_idx in range(n_vars):
                pred_var = pred_grid[:, var_idx]
                true_var = true_grid[:, var_idx]
                
                if torch.std(true_var) < self.epsilon:
                    kge_var = torch.tensor(0.0, device=y_pred.device)
                else:
                    mean_pred = torch.mean(pred_var)
                    mean_true = torch.mean(true_var)
                    std_pred = torch.std(pred_var)
                    std_true = torch.std(true_var)
                    
                    # Correlation
                    vx = pred_var - mean_pred
                    vy = true_var - mean_true
                    r_num = torch.sum(vx * vy)
                    r_den = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
                    r = r_num / (r_den + self.epsilon)
                    
                    # Variability and bias ratios
                    alpha = std_pred / (std_true + self.epsilon)
                    beta = mean_pred / (mean_true + self.epsilon)
                    
                    kge_var = 1 - torch.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                    kge_var = torch.clamp(kge_var, min=-10.0, max=1.0)
                
                var_kge_values.append(kge_var)
            
            var_kge_tensor = torch.stack(var_kge_values)
            if self.weights is not None:
                w = torch.tensor(self.weights[:n_vars], device=y_pred.device, dtype=y_pred.dtype)
                grid_kge = (var_kge_tensor * w).sum() / (w.sum() + self.epsilon)
            else:
                grid_kge = var_kge_tensor.mean()
            
            grid_kge_losses.append(grid_kge)
        
        if len(grid_kge_losses) == 0:
            return self._compute_standard_kge(y_pred, y_true)
        
        mean_kge = torch.stack(grid_kge_losses).mean()
        return 1 - mean_kge
    
    def _compute_standard_kge(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Fallback standard KGE computation."""
        n_vars = y_pred.shape[-1]
        kge_values = []
        
        for var_idx in range(n_vars):
            pred_var = y_pred[:, var_idx]
            true_var = y_true[:, var_idx]
            
            if torch.std(true_var) < self.epsilon:
                kge_var = torch.tensor(0.0, device=y_pred.device)
            else:
                mean_pred = torch.mean(pred_var)
                mean_true = torch.mean(true_var)
                std_pred = torch.std(pred_var)
                std_true = torch.std(true_var)
                
                vx = pred_var - mean_pred
                vy = true_var - mean_true
                r_num = torch.sum(vx * vy)
                r_den = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
                r = r_num / (r_den + self.epsilon)
                
                alpha = std_pred / (std_true + self.epsilon)
                beta = mean_pred / (mean_true + self.epsilon)
                
                kge_var = 1 - torch.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                kge_var = torch.clamp(kge_var, min=-10.0, max=1.0)
            
            kge_values.append(kge_var)
        
        return 1 - torch.stack(kge_values).mean()


class BiasCorrectedMSELoss(nn.Module):
    """
    Bias-Corrected MSE Loss Function
    Author: Siddik Barbhuiya
    
    This loss function adds an explicit bias penalty to the standard MSE loss.
    Standard MSE implicitly includes bias (MSE = Variance + Bias^2), but this
    allows for weighting the bias component more heavily to force the model
    to correct systematic over/underestimation.
    
    Loss = MSE + bias_weight * (mean_pred - mean_true)^2
    
    Attributes:
        bias_weight (float): Weight for the bias penalty term
        reduction (str): 'mean' or 'sum'
    """
    
    def __init__(self, bias_weight: float = 1.0, reduction: str = 'mean'):
        """
        Initialize Bias-Corrected MSE Loss
        
        Args:
            bias_weight: Weight for bias penalty. 
                         0.0 = standard MSE
                         >0.0 = additional penalty for mean shift
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.bias_weight = bias_weight
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate Bias-Corrected MSE loss
        
        Args:
            y_pred: Predictions
            y_true: Ground truth
            
        Returns:
            Combined loss
        """
        # Flatten if needed
        if y_pred.ndim == 3:
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            y_true = y_true.reshape(-1, y_true.shape[-1])
            
        # Standard MSE
        mse_loss = self.mse(y_pred, y_true)
        
        # Bias component (Mean Squared Bias)
        # Calculate mean over batch dimension
        # If multi-output, calculate mean per output then average
        if y_pred.ndim > 1:
            # Per-variable bias
            bias = torch.mean(y_pred, dim=0) - torch.mean(y_true, dim=0)
            bias_sq = torch.mean(bias ** 2)
        else:
            # Single variable bias
            bias = torch.mean(y_pred) - torch.mean(y_true)
            bias_sq = bias ** 2
            
        # Total loss
        total_loss = mse_loss + self.bias_weight * bias_sq
        
        return total_loss