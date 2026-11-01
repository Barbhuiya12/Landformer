"""
Epoch Diagnostics Visualization Module
Author: Siddik Barbhuiya
Description: Generate time series plots for best/worst grids and spatial metric maps after each epoch.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_per_grid_metrics(predictions: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute NSE and KGE for each grid cell (averaged across all variables).
    
    Args:
        predictions: (n_grids, n_time, n_vars) or (n_samples, n_vars)
        targets: Same shape as predictions
        
    Returns:
        nse_per_grid: (n_grids,) NSE values
        kge_per_grid: (n_grids,) KGE values
    """
    n_grids = predictions.shape[0]
    nse_per_grid = np.zeros(n_grids)
    kge_per_grid = np.zeros(n_grids)
    
    for i in range(n_grids):
        pred = predictions[i].flatten()
        targ = targets[i].flatten()
        
        # Remove NaN values
        mask = ~np.isnan(pred) & ~np.isnan(targ)
        if mask.sum() < 10:  # Not enough valid data
            nse_per_grid[i] = np.nan
            kge_per_grid[i] = np.nan
            continue
            
        pred = pred[mask]
        targ = targ[mask]
        
        # NSE
        ss_res = np.sum((targ - pred) ** 2)
        ss_tot = np.sum((targ - np.mean(targ)) ** 2)
        nse_per_grid[i] = 1 - (ss_res / (ss_tot + 1e-10))
        
        # KGE
        r = np.corrcoef(pred, targ)[0, 1] if len(pred) > 1 else 0
        alpha = np.std(pred) / (np.std(targ) + 1e-10)
        beta = np.mean(pred) / (np.mean(targ) + 1e-10)
        kge_per_grid[i] = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return nse_per_grid, kge_per_grid


def compute_per_grid_metrics_per_var(predictions: np.ndarray, targets: np.ndarray, 
                                      var_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute NSE, KGE, and Correlation for each grid cell for a specific variable.
    
    Args:
        predictions: (n_grids, n_time, n_vars)
        targets: (n_grids, n_time, n_vars)
        var_idx: Index of the variable to compute metrics for
        
    Returns:
        nse_per_grid: (n_grids,) NSE values for this variable
        kge_per_grid: (n_grids,) KGE values for this variable
        corr_per_grid: (n_grids,) Correlation values for this variable
    """
    n_grids = predictions.shape[0]
    nse_per_grid = np.zeros(n_grids)
    kge_per_grid = np.zeros(n_grids)
    corr_per_grid = np.zeros(n_grids)
    
    for i in range(n_grids):
        if len(predictions.shape) == 3:
            pred = predictions[i, :, var_idx].flatten()
            targ = targets[i, :, var_idx].flatten()
        else:
            pred = predictions[i, var_idx].flatten()
            targ = targets[i, var_idx].flatten()
        
        # Remove NaN values
        mask = ~np.isnan(pred) & ~np.isnan(targ)
        if mask.sum() < 5:  # Not enough valid data
            nse_per_grid[i] = np.nan
            kge_per_grid[i] = np.nan
            corr_per_grid[i] = np.nan
            continue
            
        pred = pred[mask]
        targ = targ[mask]
        
        # NSE
        ss_res = np.sum((targ - pred) ** 2)
        ss_tot = np.sum((targ - np.mean(targ)) ** 2)
        nse_per_grid[i] = 1 - (ss_res / (ss_tot + 1e-10))
        
        # KGE and Correlation
        if len(pred) > 1 and np.std(targ) > 1e-10:
            r = np.corrcoef(pred, targ)[0, 1]
            corr_per_grid[i] = r
            alpha = np.std(pred) / (np.std(targ) + 1e-10)
            beta = np.mean(pred) / (np.mean(targ) + 1e-10)
            kge_per_grid[i] = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        else:
            kge_per_grid[i] = np.nan
            corr_per_grid[i] = np.nan
    
    return nse_per_grid, kge_per_grid, corr_per_grid


def plot_timeseries_for_grid(predictions: np.ndarray, targets: np.ndarray, 
                              grid_idx: int, variable_names: List[str],
                              title: str, save_path: str,
                              lat: float = None, lon: float = None):
    """
    Plot time series for a single grid (all variables as subplots).
    
    Args:
        predictions: (n_grids, n_time, n_vars) predictions
        targets: (n_grids, n_time, n_vars) targets
        grid_idx: Index of the grid to plot
        variable_names: List of variable names
        title: Plot title
        save_path: Path to save the figure
        lat, lon: Optional coordinates to display
    """
    n_vars = len(variable_names)
    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 3 * n_vars), sharex=True)
    
    if n_vars == 1:
        axes = [axes]
    
    coord_str = f" (Lat: {lat:.2f}, Lon: {lon:.2f})" if lat is not None else ""
    fig.suptitle(f"{title}{coord_str}", fontsize=14, fontweight='bold')
    
    pred_grid = predictions[grid_idx]  # (n_time, n_vars)
    targ_grid = targets[grid_idx]
    
    time_steps = np.arange(pred_grid.shape[0])
    
    for var_idx, (ax, var_name) in enumerate(zip(axes, variable_names)):
        ax.plot(time_steps, targ_grid[:, var_idx], 'b-', label='Observed', linewidth=1.5, alpha=0.8)
        ax.plot(time_steps, pred_grid[:, var_idx], 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
        ax.set_ylabel(var_name, fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Compute metrics for this variable
        p = pred_grid[:, var_idx]
        t = targ_grid[:, var_idx]
        mask = ~np.isnan(p) & ~np.isnan(t)
        if mask.sum() > 10:
            ss_res = np.sum((t[mask] - p[mask]) ** 2)
            ss_tot = np.sum((t[mask] - np.mean(t[mask])) ** 2)
            nse = 1 - (ss_res / (ss_tot + 1e-10))
            ax.set_title(f"{var_name} | NSE: {nse:.3f}", fontsize=10)
    
    axes[-1].set_xlabel('Time Steps', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved time series plot: {save_path}")


def plot_spatial_metric_map(lats: np.ndarray, lons: np.ndarray, 
                            metric_values: np.ndarray, metric_name: str,
                            save_path: str, epoch: int, var_name: str = None):
    """
    Create a spatial scatter plot of a metric (NSE or KGE).
    
    Args:
        lats: (n_grids,) latitude values
        lons: (n_grids,) longitude values
        metric_values: (n_grids,) metric values
        metric_name: Name of the metric (e.g., 'NSE', 'KGE')
        save_path: Path to save the figure
        epoch: Current epoch number
        var_name: Optional variable name for title
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Handle NaN values
    valid_mask = ~np.isnan(metric_values)
    
    if valid_mask.sum() == 0:
        logger.warning(f"No valid values for {metric_name} map")
        plt.close(fig)
        return
    
    # Color normalization: vmin=0, vmax=1 for NSE/KGE/Correlation
    # These metrics vary between 0 and 1 for good predictions
    vmin, vmax = 0, 1
    
    scatter = ax.scatter(lons[valid_mask], lats[valid_mask], 
                         c=metric_values[valid_mask], 
                         cmap='RdYlGn', vmin=vmin, vmax=vmax,
                         s=50, alpha=0.8, edgecolors='black', linewidths=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(metric_name, fontsize=12)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    title = f'Spatial {metric_name}'
    if var_name:
        title += f' - {var_name}'
    title += f' - Epoch {epoch}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    valid_vals = metric_values[valid_mask]
    stats_text = f"Mean: {np.mean(valid_vals):.3f}\nMedian: {np.median(valid_vals):.3f}\nStd: {np.std(valid_vals):.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved spatial map: {save_path}")


def save_epoch_diagnostics(predictions: np.ndarray, targets: np.ndarray,
                           lats: np.ndarray, lons: np.ndarray,
                           variable_names: List[str], epoch: int,
                           output_dir: str):
    """
    Main function to save all epoch diagnostics.
    
    Args:
        predictions: (n_grids, n_time, n_vars) model predictions
        targets: (n_grids, n_time, n_vars) ground truth
        lats: (n_grids,) latitude values
        lons: (n_grids,) longitude values
        variable_names: List of variable names
        epoch: Current epoch number
        output_dir: Base output directory
    """
    # Create epoch-specific directory
    epoch_dir = os.path.join(output_dir, 'epoch_plots', f'epoch_{epoch:03d}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    # First compute OVERALL summary metrics (matching training logs)
    n_vars = len(variable_names)
    overall_summary = {'epoch': epoch}
    for var_idx, var_name in enumerate(variable_names):
        if len(predictions.shape) == 3:
            pred_var = predictions[:, :, var_idx].flatten()
            targ_var = targets[:, :, var_idx].flatten()
        else:
            pred_var = predictions[:, var_idx].flatten()
            targ_var = targets[:, var_idx].flatten()
        
        mask = ~np.isnan(pred_var) & ~np.isnan(targ_var)
        if mask.sum() > 10:
            pred_var = pred_var[mask]
            targ_var = targ_var[mask]
            
            # NSE
            ss_res = np.sum((targ_var - pred_var) ** 2)
            ss_tot = np.sum((targ_var - np.mean(targ_var)) ** 2)
            nse = 1 - (ss_res / (ss_tot + 1e-10))
            
            # KGE
            r = np.corrcoef(pred_var, targ_var)[0, 1]
            alpha = np.std(pred_var) / (np.std(targ_var) + 1e-10)
            beta = np.mean(pred_var) / (np.mean(targ_var) + 1e-10)
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            
            overall_summary[f'{var_name}_NSE'] = nse
            overall_summary[f'{var_name}_KGE'] = kge
            logger.info(f"Epoch {epoch} Overall - {var_name}: NSE={nse:.3f}, KGE={kge:.3f}")
    
    # Save overall summary
    summary_path = os.path.join(epoch_dir, 'overall_summary.csv')
    pd.DataFrame([overall_summary]).to_csv(summary_path, index=False)
    
    # NOTE: Per-grid metrics below are computed on reshaped data that may not be
    # correctly aligned by grid. The "best/worst" grids shown are based on sample
    # indices, NOT actual grid locations. For accurate per-grid metrics, grid indices
    # need to be tracked during data loading.
    
    # 1. Compute overall per-grid metrics (warning: may not be correctly aligned)
    nse_per_grid, kge_per_grid = compute_per_grid_metrics(predictions, targets)
    
    # 2. Find best and worst grids (by average of NSE and KGE)
    combined_metric = (nse_per_grid + kge_per_grid) / 2
    valid_mask = ~np.isnan(combined_metric)
    
    if valid_mask.sum() == 0:
        logger.warning(f"No valid grids for epoch {epoch}, skipping diagnostics")
        return
    
    best_idx = np.nanargmax(combined_metric)
    worst_idx = np.nanargmin(combined_metric)
    
    logger.info(f"Epoch {epoch}: Best sample idx={best_idx} (NSE={nse_per_grid[best_idx]:.3f}, KGE={kge_per_grid[best_idx]:.3f})")
    logger.info(f"Epoch {epoch}: Worst sample idx={worst_idx} (NSE={nse_per_grid[worst_idx]:.3f}, KGE={kge_per_grid[worst_idx]:.3f})")
    
    # 3. Plot time series for OVERALL best/worst grids
    plot_timeseries_for_grid(predictions, targets, best_idx, variable_names,
                              f"Overall Best Grid - Epoch {epoch}", 
                              os.path.join(epoch_dir, 'best_grid_timeseries_overall.png'),
                              lat=lats[best_idx], lon=lons[best_idx])
    
    plot_timeseries_for_grid(predictions, targets, worst_idx, variable_names,
                              f"Overall Worst Grid - Epoch {epoch}",
                              os.path.join(epoch_dir, 'worst_grid_timeseries_overall.png'),
                              lat=lats[worst_idx], lon=lons[worst_idx])

    # 3b. Plot time series for PER-VARIABLE best/worst grids
    for var_idx, var_name in enumerate(variable_names):
        safe_var_name = var_name.replace(' ', '_').replace('/', '_')
        
        # Compute metrics just for this variable (now returns 3 values)
        nse_var, kge_var, corr_var = compute_per_grid_metrics_per_var(predictions, targets, var_idx)
        
        # Use KGE as the primary metric for selection (more robust)
        metric_to_use = kge_var
        valid_mask_var = ~np.isnan(metric_to_use)
        
        if valid_mask_var.sum() > 0:
            best_idx_var = np.nanargmax(metric_to_use)
            worst_idx_var = np.nanargmin(metric_to_use)
            
            # Plot BEST for this variable
            plot_timeseries_for_grid(predictions, targets, best_idx_var, variable_names,
                                      f"Best {var_name} Grid (KGE={metric_to_use[best_idx_var]:.3f})", 
                                      os.path.join(epoch_dir, f'best_grid_{safe_var_name}.png'),
                                      lat=lats[best_idx_var], lon=lons[best_idx_var])
            
            # Plot WORST for this variable
            plot_timeseries_for_grid(predictions, targets, worst_idx_var, variable_names,
                                      f"Worst {var_name} Grid (KGE={metric_to_use[worst_idx_var]:.3f})", 
                                      os.path.join(epoch_dir, f'worst_grid_{safe_var_name}.png'),
                                      lat=lats[worst_idx_var], lon=lons[worst_idx_var])
    
    # 4. Create overall spatial maps
    plot_spatial_metric_map(lats, lons, nse_per_grid, 'NSE',
                            os.path.join(epoch_dir, 'spatial_NSE_overall.png'), epoch)
    plot_spatial_metric_map(lats, lons, kge_per_grid, 'KGE',
                            os.path.join(epoch_dir, 'spatial_KGE_overall.png'), epoch)
    
    # 5. Create per-variable spatial maps (NSE, KGE, and Correlation)
    n_vars = len(variable_names)
    metrics_data = {'grid_idx': np.arange(len(lats)), 'lat': lats, 'lon': lons,
                    'NSE_overall': nse_per_grid, 'KGE_overall': kge_per_grid}
    
    for var_idx, var_name in enumerate(variable_names):
        nse_var, kge_var, corr_var = compute_per_grid_metrics_per_var(predictions, targets, var_idx)
        
        # Safe filename
        safe_var_name = var_name.replace(' ', '_').replace('/', '_')
        
        # NSE spatial map
        plot_spatial_metric_map(lats, lons, nse_var, 'NSE',
                                os.path.join(epoch_dir, f'spatial_NSE_{safe_var_name}.png'), 
                                epoch, var_name=var_name)
        
        # KGE spatial map
        plot_spatial_metric_map(lats, lons, kge_var, 'KGE',
                                os.path.join(epoch_dir, f'spatial_KGE_{safe_var_name}.png'), 
                                epoch, var_name=var_name)
        
        # Correlation spatial map (NEW!)
        plot_spatial_metric_map(lats, lons, corr_var, 'Correlation',
                                os.path.join(epoch_dir, f'spatial_Corr_{safe_var_name}.png'), 
                                epoch, var_name=var_name)
        
        # Add to metrics data
        metrics_data[f'NSE_{safe_var_name}'] = nse_var
        metrics_data[f'KGE_{safe_var_name}'] = kge_var
        metrics_data[f'Corr_{safe_var_name}'] = corr_var
    
    # 6. Save metrics CSV
    metrics_df = pd.DataFrame(metrics_data)
    csv_path = os.path.join(epoch_dir, 'grid_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"Saved grid metrics CSV: {csv_path}")

