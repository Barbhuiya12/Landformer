"""
Hydrological metrics for model evaluation.

This module implements standard hydrological performance metrics
including Kling-Gupta Efficiency (KGE), correlation, bias, and variability.
These metrics are specifically designed for hydrological model evaluation
and provide more meaningful insights than generic regression metrics.

The module includes:
1. Individual metric functions (KGE, NSE, correlation, etc.)
2. A comprehensive metrics calculator that computes all metrics at once
3. A metrics tracker for accumulating and computing metrics over multiple batches

These metrics are used during both training and evaluation to monitor
model performance and guide the training process.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def kling_gupta_efficiency(y_pred: torch.Tensor, y_true: torch.Tensor, 
                          return_components: bool = False) -> torch.Tensor:
    """
    Calculate Kling-Gupta Efficiency (KGE).
    
    KGE is a comprehensive hydrological metric that decomposes model performance
    into three components: correlation, variability, and bias. It is particularly
    useful for hydrological models as it accounts for the temporal dynamics
    that are crucial in hydrological processes.
    
    KGE = 1 - sqrt((r - 1)^2 + (α - 1)^2 + (β - 1)^2)
    
    Where:
    - r: Pearson correlation coefficient (temporal correlation)
    - α: Ratio of standard deviations (variability ratio)
    - β: Ratio of means (bias ratio)
    
    A KGE value of 1 indicates perfect model performance, while values below 0
    indicate that the model performs worse than the mean of observations.
    
    Args:
        y_pred: Predicted values [batch_size, ...]
        y_true: True values [batch_size, ...]
        return_components: If True, return (KGE, r, α, β)
    
    Returns:
        KGE value (higher is better, perfect = 1)
    """
    # Flatten tensors
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()
    
    # Calculate means
    mean_pred = torch.mean(y_pred_flat)
    mean_true = torch.mean(y_true_flat)
    
    # Calculate standard deviations
    std_pred = torch.std(y_pred_flat, unbiased=False)
    std_true = torch.std(y_true_flat, unbiased=False)
    
    # Calculate correlation coefficient
    cov = torch.mean((y_pred_flat - mean_pred) * (y_true_flat - mean_true))
    r = cov / (std_pred * std_true + 1e-8)
    
    # Calculate variability ratio (α)
    alpha = std_pred / (std_true + 1e-8)
    
    # Calculate bias ratio (β)
    beta = mean_pred / (mean_true + 1e-8)
    
    # Calculate KGE
    kge_components = (r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2
    kge = 1 - torch.sqrt(kge_components)
    
    if return_components:
        return kge, r, alpha, beta
    return kge


def nash_sutcliffe_efficiency(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculate Nash-Sutcliffe Efficiency (NSE).
    
    NSE is a widely used metric in hydrology that measures the relative magnitude
    of the residual variance compared to the variance of the observed data.
    It is sensitive to extreme values and provides a good overall assessment
    of model performance.
    
    NSE = 1 - Σ(y_true - y_pred)^2 / Σ(y_true - mean(y_true))^2
    
    An NSE value of 1 indicates perfect model performance, 0 indicates performance
    equivalent to the mean of observations, and negative values indicate that
    the model performs worse than the mean.
    
    Args:
        y_pred: Predicted values
        y_true: True values
    
    Returns:
        NSE value (higher is better, perfect = 1)
    """
    # Flatten tensors
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()
    
    # Calculate mean of true values
    mean_true = torch.mean(y_true_flat)
    
    # Calculate numerator (sum of squared errors)
    numerator = torch.sum((y_true_flat - y_pred_flat) ** 2)
    
    # Calculate denominator (variance of true values)
    denominator = torch.sum((y_true_flat - mean_true) ** 2)
    
    # Calculate NSE
    nse = 1 - numerator / (denominator + 1e-8)
    
    return nse


def pearson_correlation(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculate Pearson correlation coefficient.
    
    The Pearson correlation coefficient measures the linear correlation between
    predicted and observed values. In hydrological modeling, it indicates how
    well the model captures the temporal patterns of the observed data.
    
    Values range from -1 to 1, where 1 indicates perfect positive correlation,
    0 indicates no linear correlation, and -1 indicates perfect negative correlation.
    
    Args:
        y_pred: Predicted values
        y_true: True values
    
    Returns:
        Correlation coefficient (-1 to 1, perfect = 1)
    """
    # Flatten tensors
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()
    
    # Calculate means
    mean_pred = torch.mean(y_pred_flat)
    mean_true = torch.mean(y_true_flat)
    
    # Calculate covariance
    cov = torch.mean((y_pred_flat - mean_pred) * (y_true_flat - mean_true))
    
    # Calculate standard deviations
    std_pred = torch.std(y_pred_flat, unbiased=False)
    std_true = torch.std(y_true_flat, unbiased=False)
    
    # Calculate correlation
    corr = cov / (std_pred * std_true + 1e-8)
    
    return corr


def percent_bias(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculate percent bias (PBIAS).
    
    PBIAS measures the average tendency of the simulated data to be larger or
    smaller than their observed counterparts. It indicates whether the model
    systematically overestimates or underestimates the observed values.
    
    PBIAS = 100 * Σ(y_true - y_pred) / Σ(y_true)
    
    A PBIAS value of 0 indicates perfect model performance with no bias.
    Positive values indicate overestimation, and negative values indicate
    underestimation.
    
    Args:
        y_pred: Predicted values
        y_true: True values
    
    Returns:
        Percent bias (0 is perfect, negative = underestimation, positive = overestimation)
    """
    # Flatten tensors
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()
    
    # Calculate percent bias
    pbias = 100 * torch.sum(y_true_flat - y_pred_flat) / (torch.sum(y_true_flat) + 1e-8)
    
    return pbias


def root_mean_squared_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    RMSE is a commonly used metric that measures the square root of the average
    of squared differences between predicted and observed values. It provides
    a measure of the magnitude of the error in the same units as the data.
    
    RMSE is sensitive to outliers and gives higher weight to large errors,
    making it useful for identifying models that consistently perform well.
    
    Args:
        y_pred: Predicted values
        y_true: True values
    
    Returns:
        RMSE value (lower is better, perfect = 0)
    """
    # Calculate MSE
    mse = torch.mean((y_pred - y_true) ** 2)
    
    # Calculate RMSE
    rmse = torch.sqrt(mse)
    
    return rmse


def calculate_all_metrics(y_pred: torch.Tensor, y_true: torch.Tensor,
                         variable_names: Optional[list] = None) -> Dict[str, float]:
    """
    Calculate all hydrological metrics.
    
    Args:
        y_pred: Predicted values [batch, n_outputs] or [batch]
        y_true: True values [batch, n_outputs] or [batch]
        variable_names: Names of output variables for multi-output case
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Check if multi-output
    if y_pred.dim() > 1 and y_pred.shape[1] > 1:
        # Multi-output case
        n_outputs = y_pred.shape[1]
        
        # Calculate metrics for each output
        per_output_nse = []
        per_output_kge = []
        for i in range(n_outputs):
            var_name = variable_names[i] if variable_names else f"var_{i}"
            
            # Extract single output
            pred_i = y_pred[:, i]
            true_i = y_true[:, i]
            
            # Calculate metrics
            kge, r, alpha, beta = kling_gupta_efficiency(pred_i, true_i, return_components=True)
            nse = nash_sutcliffe_efficiency(pred_i, true_i)
            pbias = percent_bias(pred_i, true_i)
            rmse = root_mean_squared_error(pred_i, true_i)
            
            # Store metrics
            metrics[f'{var_name}_kge'] = kge.item()
            metrics[f'{var_name}_correlation'] = r.item()
            metrics[f'{var_name}_variability'] = alpha.item()
            metrics[f'{var_name}_bias_ratio'] = beta.item()
            metrics[f'{var_name}_nse'] = nse.item()
            metrics[f'{var_name}_pbias'] = pbias.item()
            metrics[f'{var_name}_rmse'] = rmse.item()
            per_output_nse.append(nse.item())
            per_output_kge.append(kge.item())
        
        # Equal-weight average of per-output NSEs
        if len(per_output_nse) > 0:
            metrics['overall_nse_mean'] = float(np.mean(per_output_nse))
        
        # Equal-weight average of per-output KGEs
        if len(per_output_kge) > 0:
            metrics['overall_kge_mean'] = float(np.mean(per_output_kge))
        
        # Calculate aggregate metrics
        kge_all, r_all, alpha_all, beta_all = kling_gupta_efficiency(y_pred, y_true, return_components=True)
        nse_all = nash_sutcliffe_efficiency(y_pred, y_true)
        pbias_all = percent_bias(y_pred, y_true)
        rmse_all = root_mean_squared_error(y_pred, y_true)
        
        metrics['overall_kge'] = kge_all.item()
        metrics['overall_correlation'] = r_all.item()
        metrics['overall_variability'] = alpha_all.item()
        metrics['overall_bias_ratio'] = beta_all.item()
        metrics['overall_nse'] = nse_all.item()
        metrics['overall_pbias'] = pbias_all.item()
        metrics['overall_rmse'] = rmse_all.item()
        
    else:
        # Single output case
        kge, r, alpha, beta = kling_gupta_efficiency(y_pred, y_true, return_components=True)
        nse = nash_sutcliffe_efficiency(y_pred, y_true)
        pbias = percent_bias(y_pred, y_true)
        rmse = root_mean_squared_error(y_pred, y_true)
        
        metrics['kge'] = kge.item()
        metrics['correlation'] = r.item()
        metrics['variability'] = alpha.item()
        metrics['bias_ratio'] = beta.item()
        metrics['nse'] = nse.item()
        metrics['pbias'] = pbias.item()
        metrics['rmse'] = rmse.item()
    
    return metrics


class HydrologicalMetricsTracker:
    """
    Track and aggregate hydrological metrics during training.
    
    This class accumulates predictions and targets over multiple batches
    and computes comprehensive metrics at the end of each epoch. It is
    particularly useful for distributed training or when computing metrics
    over large datasets that don't fit in memory.
    
    The tracker supports both single-output and multi-output scenarios,
    automatically computing per-variable metrics when variable names are provided.
    
    Attributes:
        variable_names (list): Names of output variables for multi-output case
        predictions (list): Accumulated predictions
        targets (list): Accumulated targets
    """
    
    def __init__(self, variable_names: Optional[list] = None):
        """
        Initialize metrics tracker.
        
        Args:
            variable_names: Names of output variables
        """
        self.variable_names = variable_names
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.predictions = []
        self.targets = []
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Update tracker with new predictions and targets.
        
        Args:
            y_pred: Predicted values
            y_true: True values
        """
        self.predictions.append(y_pred.detach().cpu())
        self.targets.append(y_true.detach().cpu())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated predictions and targets.
        
        Returns:
            Dictionary of computed metrics
        """
        if len(self.predictions) == 0:
            return {}
        
        # Concatenate all predictions and targets
        y_pred = torch.cat(self.predictions, dim=0)
        y_true = torch.cat(self.targets, dim=0)
        
        # Calculate metrics
        metrics = calculate_all_metrics(y_pred, y_true, self.variable_names)
        
        return metrics
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = "", logger: Optional[logging.Logger] = None):
        """
        Log metrics to console.
        
        Args:
            metrics: Dictionary of metrics
            prefix: Prefix for metric names (e.g., "train", "val")
            logger: Logger instance
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        
        # Log overall metrics
        if 'overall_kge' in metrics:
            logger.info(f"{prefix} Overall Metrics:")
            logger.info(f"  KGE: {metrics['overall_kge']:.4f}")
            logger.info(f"  NSE: {metrics['overall_nse']:.4f}")
            logger.info(f"  Correlation: {metrics['overall_correlation']:.4f}")
            logger.info(f"  PBIAS: {metrics['overall_pbias']:.2f}%")
            logger.info(f"  RMSE: {metrics['overall_rmse']:.4f}")
        elif 'kge' in metrics:
            logger.info(f"{prefix} Metrics:")
            logger.info(f"  KGE: {metrics['kge']:.4f}")
            logger.info(f"  NSE: {metrics['nse']:.4f}")
            logger.info(f"  Correlation: {metrics['correlation']:.4f}")
            logger.info(f"  PBIAS: {metrics['pbias']:.2f}%")
            logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        
        # Log per-variable metrics if available
        if self.variable_names:
            for var_name in self.variable_names:
                if f'{var_name}_kge' in metrics:
                    logger.info(f"{prefix} {var_name} Metrics:")
                    logger.info(f"  KGE: {metrics[f'{var_name}_kge']:.4f}")
                    logger.info(f"  NSE: {metrics[f'{var_name}_nse']:.4f}")
                    logger.info(f"  Correlation: {metrics[f'{var_name}_correlation']:.4f}")
                    logger.info(f"  PBIAS: {metrics[f'{var_name}_pbias']:.2f}%")
                    logger.info(f"  RMSE: {metrics[f'{var_name}_rmse']:.4f}")
