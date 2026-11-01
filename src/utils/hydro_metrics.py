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
import tempfile
import os

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
        
        # Equal-weight average AND median of per-output NSEs
        if len(per_output_nse) > 0:
            metrics['overall_nse_mean'] = float(np.mean(per_output_nse))
            metrics['overall_nse_median'] = float(np.median(per_output_nse))
        
        # Equal-weight average AND median of per-output KGEs
        if len(per_output_kge) > 0:
            metrics['overall_kge_mean'] = float(np.mean(per_output_kge))
            metrics['overall_kge_median'] = float(np.median(per_output_kge))
        
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
    and computes comprehensive metrics at the end of each epoch. It uses
    disk-backed storage (memory-mapped files) to avoid OOM errors when
    processing large datasets.
    
    The tracker supports both single-output and multi-output scenarios,
    automatically computing per-variable metrics when variable names are provided.
    
    Attributes:
        variable_names (list): Names of output variables for multi-output case
        use_disk_storage (bool): Whether to use disk-backed storage
        temp_dir (str): Directory for temporary files
    """
    
    def __init__(self, variable_names: Optional[list] = None, use_disk_storage: bool = False,
                 temp_dir: Optional[str] = None, max_memory_mb: float = 8000.0):
        """
        Initialize metrics tracker.
        
        Args:
            variable_names: Names of output variables
            use_disk_storage: (Deprecated) No longer needed with online stats
            temp_dir: (Deprecated) No longer needed
            max_memory_mb: (Deprecated) No longer needed
        """
        self.variable_names = variable_names
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics and running statistics."""
        # Overall statistics per variable
        # Structure: {var_idx: {'count': 0, 'sum_p': 0, 'sum_t': 0, 'sum_p2': 0, 'sum_t2': 0, 'sum_pt': 0, 'sum_sq_err': 0}}
        self._stats = {}
        
        # Per-grid statistics
        # Structure: {grid_idx: {var_idx: {'count': 0, 'sum_p': 0, 'sum_t': 0, 'sum_p2': 0, 'sum_t2': 0, 'sum_pt': 0, 'sum_sq_err': 0}}}
        self._grid_stats = {}
        
        # Counters
        self._total_samples = 0
        
        # Placeholders for diagnostic reports (store only a small sample)
        self.last_predictions = None
        self.last_targets = None
        self.last_grid_indices = None
        self.last_predictions_normalized = None
        self.last_targets_normalized = None
        
        # Legacy attributes for backward compatibility
        self.predictions = []
        self.targets = []
        self.grid_indices = []
        self.predictions_normalized = []
        self.targets_normalized = []
        
        import gc
        gc.collect()
    
    def _update_stats_dict(self, stats_dict, var_idx, p, t):
        """Update a single statistics dictionary with new prediction and target."""
        if var_idx not in stats_dict:
            stats_dict[var_idx] = {
                'count': 0, 'sum_p': 0.0, 'sum_t': 0.0, 
                'sum_p2': 0.0, 'sum_t2': 0.0, 'sum_pt': 0.0, 'sum_sq_err': 0.0
            }
        
        s = stats_dict[var_idx]
        s['count'] += p.shape[0]
        s['sum_p'] += np.sum(p)
        s['sum_t'] += np.sum(t)
        s['sum_p2'] += np.sum(p**2)
        s['sum_t2'] += np.sum(t**2)
        s['sum_pt'] += np.sum(p * t)
        s['sum_sq_err'] += np.sum((p - t)**2)

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor, grid_indices: Optional[torch.Tensor] = None,
               y_pred_normalized: Optional[torch.Tensor] = None, y_true_normalized: Optional[torch.Tensor] = None):
        """
        Update tracker with new batch of predictions and targets using online statistics.
        """
        # Convert to numpy for efficient accumulation
        p_np = y_pred.detach().cpu().numpy()
        t_np = y_true.detach().cpu().numpy()
        
        if len(p_np.shape) == 1:
            p_np = p_np.reshape(-1, 1)
            t_np = t_np.reshape(-1, 1)
            
        n_samples = p_np.shape[0]
        n_vars = p_np.shape[1]
        self._total_samples += n_samples
        
        # Update overall stats
        for i in range(n_vars):
            self._update_stats_dict(self._stats, i, p_np[:, i], t_np[:, i])
            
        # Update per-grid stats if provided
        if grid_indices is not None:
            g_np = grid_indices.detach().cpu().numpy() if hasattr(grid_indices, 'detach') else np.array(grid_indices)
            unique_grids = np.unique(g_np)
            
            for grid_idx in unique_grids:
                if grid_idx not in self._grid_stats:
                    self._grid_stats[grid_idx] = {}
                
                mask = (g_np == grid_idx)
                for i in range(n_vars):
                    self._update_stats_dict(self._grid_stats[grid_idx], i, p_np[mask, i], t_np[mask, i])
        
        # Store a small sample for diagnostic reports (max 1000 samples)
        if self.last_predictions is None or self.last_predictions.shape[0] < 1000:
            sample_size = min(n_samples, 1000)
            self.last_predictions = torch.from_numpy(p_np[:sample_size])
            self.last_targets = torch.from_numpy(t_np[:sample_size])
            if grid_indices is not None:
                self.last_grid_indices = torch.from_numpy(g_np[:sample_size])
            if y_pred_normalized is not None:
                self.last_predictions_normalized = y_pred_normalized[:sample_size].detach().cpu()
                self.last_targets_normalized = y_true_normalized[:sample_size].detach().cpu()

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated online statistics.
        """
        if not self._stats:
            return {}
            
        metrics = {}
        n_vars = len(self._stats)
        
        # 1. Compute overall metrics per variable
        per_var_nse = []
        per_var_kge = []
        
        for i in range(n_vars):
            var_name = self.variable_names[i] if self.variable_names and i < len(self.variable_names) else f"var_{i}"
            var_metrics = self._calculate_metrics_from_stats(self._stats[i])
            
            for k, v in var_metrics.items():
                metrics[f"{var_name}_{k}"] = v
            
            per_var_nse.append(var_metrics['nse'])
            per_var_kge.append(var_metrics['kge'])
            
            # For backward compatibility with single-output logging
            if n_vars == 1:
                for k, v in var_metrics.items():
                    metrics[k] = v
        
        # 2. Compute aggregate metrics
        if n_vars > 1:
            metrics['overall_nse_mean'] = float(np.mean(per_var_nse))
            metrics['overall_nse_median'] = float(np.median(per_var_nse))
            metrics['overall_kge_mean'] = float(np.mean(per_var_kge))
            metrics['overall_kge_median'] = float(np.median(per_var_kge))
            
            # Flattened averages
            metrics['overall_kge'] = metrics['overall_kge_mean']
            metrics['overall_nse'] = metrics['overall_nse_mean']
            
        # 3. Compute grid-wise metrics
        if self._grid_stats:
            grid_metrics = self._compute_grid_wise_metrics_online()
            metrics.update(grid_metrics)
            
        return metrics

    def _calculate_metrics_from_stats(self, s: Dict) -> Dict[str, float]:
        """Calculate KGE, NSE, etc. from a stats dictionary."""
        count = s['count']
        if count < 2:
            return {'kge': 0.0, 'nse': 0.0, 'correlation': 0.0, 'pbias': 0.0, 'rmse': 0.0}
            
        mean_p = s['sum_p'] / count
        mean_t = s['sum_t'] / count
        
        var_p = max(0, (s['sum_p2'] / count) - (mean_p ** 2))
        var_t = max(0, (s['sum_t2'] / count) - (mean_t ** 2))
        
        std_p = np.sqrt(var_p)
        std_t = np.sqrt(var_t)
        
        # Correlation
        cov = (s['sum_pt'] / count) - (mean_p * mean_t)
        corr = cov / (std_p * std_t + 1e-8)
        corr = np.clip(corr, -1.0, 1.0)
        
        # KGE components
        alpha = std_p / (std_t + 1e-8)
        beta = mean_p / (mean_t + 1e-8)
        kge = 1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        
        # NSE
        ss_res = s['sum_sq_err']
        ss_tot = count * var_t
        nse = 1 - (ss_res / (ss_tot + 1e-8))
        
        # PBIAS and RMSE
        pbias = 100 * (s['sum_t'] - s['sum_p']) / (s['sum_t'] + 1e-8)
        rmse = np.sqrt(s['sum_sq_err'] / count)
        
        return {
            'kge': float(kge),
            'nse': float(nse),
            'correlation': float(corr),
            'variability': float(alpha),
            'bias_ratio': float(beta),
            'pbias': float(pbias),
            'rmse': float(rmse)
        }

    def _compute_grid_wise_metrics_online(self) -> Dict[str, float]:
        """Compute per-grid metrics from online grid statistics."""
        n_vars = len(self._stats)
        
        per_grid_nse = []
        per_grid_kge = []
        grid_nse_per_var = {i: [] for i in range(n_vars)}
        grid_kge_per_var = {i: [] for i in range(n_vars)}
        
        for grid_idx, vars_stats in self._grid_stats.items():
            grid_var_nse = []
            grid_var_kge = []
            
            for i in range(n_vars):
                if i not in vars_stats or vars_stats[i]['count'] < 5:
                    continue
                    
                m = self._calculate_metrics_from_stats(vars_stats[i])
                
                # Clip for stability in grid-wise averaging
                nse = np.clip(m['nse'], -10.0, 1.0)
                kge = np.clip(m['kge'], -10.0, 1.0)
                
                grid_var_nse.append(nse)
                grid_var_kge.append(kge)
                grid_nse_per_var[i].append(nse)
                grid_kge_per_var[i].append(kge)
            
            if grid_var_nse:
                per_grid_nse.append(np.mean(grid_var_nse))
            if grid_var_kge:
                per_grid_kge.append(np.mean(grid_var_kge))
        
        metrics = {}
        if per_grid_nse:
            metrics['grid_nse_mean'] = float(np.mean(per_grid_nse))
            metrics['grid_nse_median'] = float(np.median(per_grid_nse))
        if per_grid_kge:
            metrics['grid_kge_mean'] = float(np.mean(per_grid_kge))
            metrics['grid_kge_median'] = float(np.median(per_grid_kge))
            
        # Per-variable medians
        per_var_nse_medians = []
        per_var_kge_medians = []
        
        for i in range(n_vars):
            var_name = self.variable_names[i] if self.variable_names and i < len(self.variable_names) else f"var_{i}"
            if grid_nse_per_var[i]:
                v_nse_med = np.median(grid_nse_per_var[i])
                metrics[f'{var_name}_grid_nse'] = float(np.mean(grid_nse_per_var[i]))
                metrics[f'{var_name}_grid_nse_median'] = float(v_nse_med)
                per_var_nse_medians.append(v_nse_med)
            if grid_kge_per_var[i]:
                v_kge_med = np.median(grid_kge_per_var[i])
                metrics[f'{var_name}_grid_kge'] = float(np.mean(grid_kge_per_var[i]))
                metrics[f'{var_name}_grid_kge_median'] = float(v_kge_med)
                per_var_kge_medians.append(v_kge_med)
                
        if per_var_nse_medians:
            metrics['overall_nse_median'] = float(np.mean(per_var_nse_medians))
        if per_var_kge_medians:
            metrics['overall_kge_median'] = float(np.mean(per_var_kge_medians))
            
        return metrics

    def _cleanup_temp_files(self):
        """No-op for backward compatibility."""
        pass

    def _estimate_memory_usage_mb(self) -> float:
        """No-op for backward compatibility."""
        return 0.0

    def _should_flush_to_disk(self) -> bool:
        """No-op for backward compatibility."""
        return False

    def _flush_to_disk(self):
        """No-op for backward compatibility."""
        pass
    
    def _compute_grid_wise_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
                                    grid_indices: torch.Tensor) -> Dict[str, float]:
        """
        Compute per-grid metrics and return their mean.
        
        Calculation approach:
        1. For each grid: Compute NSE/KGE for each variable
        2. For each grid: Take mean across variables → ONE NSE/KGE per grid
        3. Mean of per-grid metrics → final overall grid metric
        
        Args:
            y_pred: Predictions (N, n_vars)
            y_true: Ground truth (N, n_vars)
            grid_indices: Grid index for each sample (N,)
            
        Returns:
            Dictionary with grid-wise mean metrics
        """
        unique_grids = torch.unique(grid_indices)
        n_vars = y_pred.shape[-1] if len(y_pred.shape) > 1 else 1
        
        # Per-grid metrics (one value per grid, averaged across variables)
        per_grid_nse = []  # One NSE per grid (mean of variables)
        per_grid_kge = []  # One KGE per grid (mean of variables)
        
        # Per-variable grid metrics (for per-variable logging)
        grid_nse_per_var = {i: [] for i in range(n_vars)}
        grid_kge_per_var = {i: [] for i in range(n_vars)}
        
        for grid_idx in unique_grids:
            mask = (grid_indices == grid_idx)
            if mask.sum() < 5:
                continue
            
            pred_grid = y_pred[mask]
            true_grid = y_true[mask]
            
            # Compute NSE/KGE for each variable in this grid
            grid_var_nse = []
            grid_var_kge = []
            
            for var_idx in range(n_vars):
                pred_var = pred_grid[:, var_idx] if n_vars > 1 else pred_grid
                true_var = true_grid[:, var_idx] if n_vars > 1 else true_grid
                
                if torch.std(true_var) < 1e-8:
                    continue
                
                # NSE
                mean_obs = torch.mean(true_var)
                ss_res = torch.sum((true_var - pred_var) ** 2)
                ss_tot = torch.sum((true_var - mean_obs) ** 2)
                nse = 1 - (ss_res / (ss_tot + 1e-8))
                nse = torch.clamp(nse, min=-10.0, max=1.0)
                grid_var_nse.append(nse.item())
                grid_nse_per_var[var_idx].append(nse.item())
                
                # KGE
                mean_pred = torch.mean(pred_var)
                std_pred = torch.std(pred_var)
                std_true = torch.std(true_var)
                
                vx = pred_var - mean_pred
                vy = true_var - mean_obs
                r_num = torch.sum(vx * vy)
                r_den = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
                r = r_num / (r_den + 1e-8)
                
                alpha = std_pred / (std_true + 1e-8)
                beta = mean_pred / (mean_obs + 1e-8)
                kge = 1 - torch.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                kge = torch.clamp(kge, min=-10.0, max=1.0)
                grid_var_kge.append(kge.item())
                grid_kge_per_var[var_idx].append(kge.item())
            
            # Step 2: Average across variables for this grid → ONE NSE/KGE per grid
            if grid_var_nse:
                per_grid_nse.append(np.mean(grid_var_nse))
            if grid_var_kge:
                per_grid_kge.append(np.mean(grid_var_kge))
        
        # Compute final metrics
        metrics = {}
        
        # Step 3: Mean of per-grid metrics
        if per_grid_nse:
            metrics['grid_nse_mean'] = np.mean(per_grid_nse)
            metrics['grid_nse_mean_of_vars'] = np.mean(per_grid_nse)
        if per_grid_kge:
            metrics['grid_kge_mean'] = np.mean(per_grid_kge)
            metrics['grid_kge_mean_of_vars'] = np.mean(per_grid_kge)
        
        # Per-variable median, then mean across variables
        # This is more robust: median removes outlier grids per variable,
        # then mean balances across variables
        per_var_nse_medians = []
        per_var_kge_medians = []
        
        if self.variable_names:
            for var_idx, var_name in enumerate(self.variable_names):
                if grid_nse_per_var[var_idx]:
                    var_median_nse = np.median(grid_nse_per_var[var_idx])
                    per_var_nse_medians.append(var_median_nse)
                    metrics[f'{var_name}_grid_nse'] = np.mean(grid_nse_per_var[var_idx])
                    metrics[f'{var_name}_grid_nse_median'] = var_median_nse
                if grid_kge_per_var[var_idx]:
                    var_median_kge = np.median(grid_kge_per_var[var_idx])
                    per_var_kge_medians.append(var_median_kge)
                    metrics[f'{var_name}_grid_kge'] = np.mean(grid_kge_per_var[var_idx])
                    metrics[f'{var_name}_grid_kge_median'] = var_median_kge
        
        # overall_nse_median = mean(per_var_medians) - YOUR LOGIC
        if per_var_nse_medians:
            metrics['overall_nse_median'] = np.mean(per_var_nse_medians)
            metrics['overall_nse_mean'] = np.mean(per_grid_nse) if per_grid_nse else 0.0
            metrics['grid_nse_median'] = np.median(per_grid_nse) if per_grid_nse else 0.0
        
        # overall_kge_median = mean(per_var_medians) - YOUR LOGIC
        if per_var_kge_medians:
            metrics['overall_kge_median'] = np.mean(per_var_kge_medians)
            metrics['overall_kge_mean'] = np.mean(per_grid_kge) if per_grid_kge else 0.0
            metrics['grid_kge_median'] = np.median(per_grid_kge) if per_grid_kge else 0.0
        
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
    def generate_detailed_report(self) -> str:
        """
        Generate a detailed report of grid-wise performance using online statistics.
        
        Returns:
            Formatted report string
        """
        if not self._grid_stats:
            return "No grid statistics available for detailed report."
            
        n_vars = len(self._stats)
        
        # Storage for per-grid metrics
        grid_metrics = {i: {'nse': [], 'kge': [], 'corr': [], 'ids': []} for i in range(n_vars)}
        
        for grid_idx, vars_stats in self._grid_stats.items():
            for i in range(n_vars):
                if i not in vars_stats or vars_stats[i]['count'] < 5:
                    continue
                
                m = self._calculate_metrics_from_stats(vars_stats[i])
                
                grid_metrics[i]['nse'].append(m['nse'])
                grid_metrics[i]['kge'].append(m['kge'])
                grid_metrics[i]['corr'].append(m['correlation'])
                grid_metrics[i]['ids'].append(grid_idx)
        
        # Build Report
        report = ["\n" + "="*80]
        report.append("DETAILED GRID PERFORMANCE REPORT (ONLINE STATS)")
        report.append("="*80)
        
        for var_idx in range(n_vars):
            var_name = self.variable_names[var_idx] if self.variable_names and var_idx < len(self.variable_names) else f"Var {var_idx}"
            nses = np.array(grid_metrics[var_idx]['nse'])
            kges = np.array(grid_metrics[var_idx]['kge'])
            corrs = np.array(grid_metrics[var_idx]['corr'])
            ids = np.array(grid_metrics[var_idx]['ids'])
            
            if len(ids) == 0:
                report.append(f"\n--- {var_name} (No valid grids) ---")
                continue
                
            report.append(f"\n--- {var_name} ({len(ids)} grids) ---")
            
            # Threshold Counts
            for metric_name, vals in [('NSE', nses), ('KGE', kges), ('Corr', corrs)]:
                n = len(vals)
                if metric_name == 'Corr':
                    p9 = np.sum(vals > 0.9) / n * 100
                    p8 = np.sum(vals > 0.8) / n * 100
                    p7 = np.sum(vals > 0.7) / n * 100
                    report.append(f"{metric_name} > 0.9: {p9:5.1f}% | > 0.8: {p8:5.1f}% | > 0.7: {p7:5.1f}%")
                else:
                    p7 = np.sum(vals > 0.7) / n * 100
                    p5 = np.sum(vals > 0.5) / n * 100
                    p2 = np.sum(vals > 0.2) / n * 100
                    report.append(f"{metric_name} > 0.7: {p7:5.1f}% | > 0.5: {p5:5.1f}% | > 0.2: {p2:5.1f}%")
            
            # Best/Worst
            best_nse_idx = np.argmax(nses)
            worst_nse_idx = np.argmin(nses)
            report.append(f"Max NSE: {nses[best_nse_idx]:.4f} (Grid {ids[best_nse_idx]})")
            report.append(f"Min NSE: {nses[worst_nse_idx]:.4f} (Grid {ids[worst_nse_idx]})")
            
            best_kge_idx = np.argmax(kges)
            worst_kge_idx = np.argmin(kges)
            report.append(f"Max KGE: {kges[best_kge_idx]:.4f} (Grid {ids[best_kge_idx]})")
            report.append(f"Min KGE: {kges[worst_kge_idx]:.4f} (Grid {ids[worst_kge_idx]})")
            
        report.append("="*80)
        
        # Add a note about normalized space metrics
        if self.last_predictions_normalized is not None:
            report.append("\nNOTE: Detailed normalized space metrics are only available for the last 1000 samples.")
        
        report.append("="*80 + "\n")
        return "\n".join(report)
