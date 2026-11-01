#!/usr/bin/env python3
"""
Standalone Inference Module — Follows EXACT training framework patterns.
Author: Siddik Barbhuiya
Usage:
    python run_inference.py --config test.yml --checkpoint_dir /path/to/Result --device cuda:0

This script:
1. Loads config via load_config() — same as trainer
2. Creates grid splits via create_grid_aware_datasets() — same as trainer  
3. Builds dataset via UnifiedSurrogateDataset(grid_indices=splits['test']) — same as trainer
4. Auto-detects model architecture (hierarchical, embed_dim, etc.) from checkpoint
5. Denormalizes via exact _denormalize_targets logic
6. Computes metrics via HydrologicalMetricsTracker — same as trainer
7. Saves DETAILED per-grid CSV (NSE, KGE, correlation, RMSE, PBIAS per variable)
"""

import argparse
import gc
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Project imports ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.config import UnifiedConfig, load_config
from src.data.dataset import UnifiedSurrogateDataset
from src.data.grid_splitter import create_grid_aware_datasets
from src.utils.hydro_metrics import (
    HydrologicalMetricsTracker,
    calculate_all_metrics,
)
from src.utils.memory import DataLoaderOptimizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference")


# ═══════════════════════════════════════════════════════════════
#  1. Auto-detect model architecture from checkpoint
# ═══════════════════════════════════════════════════════════════

def build_model_from_checkpoint(checkpoint_path: str, config, device: str = "cpu", checkpoint_content=None):
    """
    Build model by auto-detecting architecture from checkpoint tensors.
    Returns (model, info). If architecture mismatches config.model_type, returns (None, None).
    """
    from src.models.swin_transformer import MultiTaskSwinTransformer
    from src.models.lstm import MultiTaskLSTM

    if checkpoint_content is not None:
        cp = checkpoint_content
    else:
        cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
    state_dict = cp.get("model_state_dict", cp.get("state_dict", cp))

    # Strip _orig_mod. prefix (from torch.compile) if present
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "")
        cleaned[new_k] = v
    state_dict = cleaned

    # ── Check for Model Type (LSTM vs Swin) ──
    # Priority 1: Config (if available)
    model_type = getattr(config.model, "model_type", None)
    
    # Priority 2: Infer from checkpoint keys
    if not model_type:
        if any("lstm" in k for k in state_dict.keys()):
            model_type = "lstm"
        else:
            model_type = "swin" # Default fallback
    
    logger.info(f"  Detected model_type: {model_type}")

    if model_type == "lstm":
        # ── Compatibility Check ──
        # If user requests LSTM, ensure checkpoint has LSTM keys
        is_lstm_ckpt = any("lstm." in k for k in state_dict.keys())
        if not is_lstm_ckpt:
            logger.warning(f"  Config requests 'lstm', but checkpoint seems to be Swin/Other (no 'lstm.' keys). SKIPPING.")
            return None, None

        # ── Build LSTM ──
        # Extract params from config or defaults
        n_exo = getattr(config.model, "input_dims", {}).get("n_forcing", 8) + \
                getattr(config.model, "input_dims", {}).get("n_params", 13)
        
        # Override n_exo from checkpoint if possible (check input layer weight shape)
        # LSTM input weight: weight_ih_l0 (hidden_size, input_size)
        for k, v in state_dict.items():
            if "lstm.weight_ih_l0" in k:
                n_exo_ckpt = v.shape[1]
                if n_exo != n_exo_ckpt:
                    logger.warning(f"  LSTM n_exo mismatch: config={n_exo}, ckpt={n_exo_ckpt}. Using checkpoint value.")
                    n_exo = n_exo_ckpt
                break
        
        task_names = list(config.data.variables.get("target", ["evap", "soil_moisture_total", "Q"]))
        
        # Instantiate LSTM
        model = MultiTaskLSTM(
            n_exo=n_exo,
            n_lag_features=0, # Assuming no lag features for now or handled in n_exo
            seq_len=getattr(config.model, "sequence_length", 270),
            hidden_size=getattr(config.model, "hidden_size", 256),
            num_layers=getattr(config.model, "num_layers", 2),
            dropout=getattr(config.model, "dropout", 0.0),
            bidirectional=getattr(config.model, "bidirectional", False),
            task_names=task_names
        )
        
        # Load state dict
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False) # Strict=False to be safe with aux layers
            if missing:
                logger.warning(f"  Missing keys: {len(missing)} (first 3: {missing[:3]})")
            if unexpected:
                logger.warning(f"  Unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
        except RuntimeError as e:
             logger.error(f"  LSTM load failed: {e}")
             raise

        model.to(device)
        model.eval()
        return model, {"n_exo": n_exo, "use_hierarchical": False}

    # ── Auto-detect everything from tensor shapes (Swin) ──
    # Keys may have "swin_backbone." prefix (from MultiTaskSwinTransformer)

    # embed_dim from patch_embed.proj (Conv1d: shape [embed_dim, in_chans, kernel_size])
    embed_dim = None
    for k, v in state_dict.items():
        if "patch_embed" in k and k.endswith("proj.weight"):
            # Conv1d weight: (out_channels=embed_dim, in_channels, kernel_size)
            embed_dim = v.shape[0]
            logger.info(f"  embed_dim={embed_dim} from {k} shape={list(v.shape)}")
            break
    if embed_dim is None:
        # Fallback: first layer norm weight has shape [embed_dim]
        for k, v in state_dict.items():
            if "layers.0.blocks.0.norm1.weight" in k and v.dim() == 1:
                embed_dim = v.shape[0]
                logger.info(f"  embed_dim={embed_dim} from {k}")
                break
    if embed_dim is None:
        embed_dim = getattr(config.model, "embed_dim", 96)

    # depths from counting blocks per layer (with or without swin_backbone prefix)
    depths = []
    layer_idx = 0
    while True:
        block_count = sum(
            1 for k in state_dict
            if (f"layers.{layer_idx}.blocks." in k or f"swin_backbone.layers.{layer_idx}.blocks." in k)
            and ".attn.qkv.weight" in k
        )
        if block_count == 0:
            break
        depths.append(block_count)
        layer_idx += 1
    if not depths:
        depths = getattr(config.model, "depths", [2, 2, 6, 2])

    # num_heads from relative_position_bias_table
    num_heads = []
    for i in range(len(depths)):
        found = False
        for prefix in ["swin_backbone.", ""]:
            key = f"{prefix}layers.{i}.blocks.0.attn.relative_position_bias_table"
            if key in state_dict:
                num_heads.append(int(state_dict[key].shape[1]))
                found = True
                break
        if not found:
            num_heads.append(3)

    # use_hierarchical — auto-detect from downsample layers
    use_hierarchical = any("downsample" in k for k in state_dict)

    # n_exo from patch_embed.proj (Conv1d: shape [embed_dim, in_chans, kernel_size])
    n_exo = None
    patch_size = getattr(config.model, "patch_size", 10)
    for k, v in state_dict.items():
        if "patch_embed" in k and k.endswith("proj.weight"):
            # Conv1d: (out_channels, in_channels=n_exo, kernel_size=patch_size)
            n_exo = v.shape[1]  # in_channels = n_exo directly
            logger.info(f"  n_exo={n_exo} from {k} shape={list(v.shape)}")
            break
    # Fallback: compute n_exo from config (n_forcing + n_params)
    if n_exo is None:
        input_dims = getattr(config.model, "input_dims", {})
        if isinstance(input_dims, dict) and input_dims:
            n_exo = input_dims.get("n_forcing", 8) + input_dims.get("n_params", 13)
            logger.info(f"  n_exo from config: {n_exo} (n_forcing + n_params)")
        else:
            n_exo = 21  # Safe default
            logger.warning(f"  n_exo fallback to default: {n_exo}")

    # task_names from config
    task_names = list(config.data.variables.get("target", ["evap", "soil_moisture_total", "Q"]))

    # Auto-detect window_size from relative_position_bias_table shape if possible
    window_size = None
    for k, v in state_dict.items():
        if "relative_position_bias_table" in k:
            # Shape is (2*window_size - 1, num_heads) for 1D or ( (2*window_size-1)**2, num_heads ) for 2D
            # Based on error log (23, 3) vs (27, 3), it's 1D with 2*window_size - 1
            if v.dim() == 2:
                dim0 = v.shape[0]
                # Check if it matches 1D pattern: 2*w - 1
                w_1d = (dim0 + 1) // 2
                # Check if it matches 2D pattern: (2*w - 1)^2
                w_2d = int(np.sqrt(dim0))
                if w_2d * w_2d == dim0:
                    w_2d = (w_2d + 1) // 2
                
                # Heuristic: verify against config or assume 1D if small
                # This code seems to use 1D sequence (Swin Transformer for Time Series)
                # Let's assume 1D for now as per log (23 -> 12, 27 -> 14)
                window_size = w_1d
                logger.info(f"  window_size={window_size} detected from {k} (shape={dim0})")
                break
    
    if window_size is None:
        window_size = getattr(config.model, "window_size", 12)
        logger.info(f"  window_size={window_size} from config (fallback)")

    seq_len = getattr(config.model, "sequence_length", 270)

    # Spatial GAT — present only if the checkpoint holds its weights
    spatial_attention = None
    if any(k.startswith("spatial_attention.gat.") for k in state_dict):
        ck_cfg = cp.get("config") if isinstance(cp, dict) else None
        sp = getattr(getattr(ck_cfg, "model", None), "spatial_attention", None) or \
             getattr(config.model, "spatial_attention", None) or {}
        att = state_dict["spatial_attention.gat.att_src"]          # (1, heads, out_per_head)
        spatial_attention = {"enabled": True, "num_heads": int(att.shape[1]),
                             "max_distance": float(sp.get("max_distance", 2.0)), "dropout": 0.0,
                             "k_neighbors": sp.get("k_neighbors")}
        logger.info(f"  Spatial GAT detected in checkpoint: {spatial_attention}")

    logger.info(f"  Auto-detected architecture:")
    logger.info(f"    embed_dim={embed_dim}, depths={depths}, num_heads={num_heads}")
    logger.info(f"    hierarchical={use_hierarchical}, n_exo={n_exo}, patch_size={patch_size}")
    logger.info(f"    window_size={window_size}, seq_len={seq_len}, tasks={task_names}")

    model = MultiTaskSwinTransformer(
        n_exo=n_exo, n_lag_features=0, seq_len=seq_len, patch_size=patch_size,
        embed_dim=embed_dim, depths=depths, num_heads=num_heads,
        window_size=window_size, dropout=0.0, drop_path_rate=0.0,
        task_names=task_names,
        use_hierarchical=use_hierarchical,
        spatial_attention=spatial_attention,
    )

    # Load state dict — strict=True as requested for rigorous validation
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing:
            logger.warning(f"  Missing keys: {len(missing)} (first 3: {missing[:3]})")
        if unexpected:
            logger.warning(f"  Unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    except RuntimeError as e:
        if "size mismatch" in str(e):
            raise RuntimeError(f"Architecture mismatch — {e}")
        raise

    model.to(device)
    model.eval()
    return model, {"n_exo": n_exo, "use_hierarchical": use_hierarchical}


# ═══════════════════════════════════════════════════════════════
#  2. Denormalization — exact trainer._denormalize_targets()
# ═══════════════════════════════════════════════════════════════

def denormalize_targets(outputs, targets, dataset, config, device):
    """Replicates trainer._denormalize_targets() exactly."""
    target_names = config.data.variables.get("target", [])
    normalization_method = getattr(dataset, "normalization_method", "zscore")

    means, stds = [], []
    for var in target_names:
        mean, std = dataset.get_normalization_stats(var)
        means.append(mean)
        stds.append(std)

    mean_t = torch.tensor(means, device=device, dtype=outputs.dtype).unsqueeze(0)
    std_t = torch.tensor(stds, device=device, dtype=outputs.dtype).unsqueeze(0)

    if normalization_method in ("zscore", "minmax"):
        denorm_out = outputs * std_t + mean_t
        denorm_tgt = targets * std_t + mean_t
    elif normalization_method == "log_zscore":
        log_out = outputs * std_t + mean_t
        log_tgt = targets * std_t + mean_t
        denorm_out = torch.expm1(log_out)
        denorm_tgt = torch.expm1(log_tgt)
    else:
        denorm_out = outputs * std_t + mean_t
        denorm_tgt = targets * std_t + mean_t

    # relu for physical consistency — same as trainer
    denorm_out = torch.relu(denorm_out)
    denorm_tgt = torch.relu(denorm_tgt)
    return denorm_out, denorm_tgt


# ═══════════════════════════════════════════════════════════════
#  3. Compute per-grid metrics from online stats
# ═══════════════════════════════════════════════════════════════

def _metrics_from_stats(s):
    """Calculate KGE, NSE, correlation, RMSE, PBIAS from online stats dict."""
    count = s['count']
    if count < 2:
        return {k: np.nan for k in ['kge', 'nse', 'correlation', 'rmse', 'pbias',
                                     'variability', 'bias_ratio', 'n_samples']}

    mean_p = s['sum_p'] / count
    mean_t = s['sum_t'] / count
    var_p = max(0, (s['sum_p2'] / count) - mean_p ** 2)
    var_t = max(0, (s['sum_t2'] / count) - mean_t ** 2)
    std_p, std_t = np.sqrt(var_p), np.sqrt(var_t)

    cov = (s['sum_pt'] / count) - (mean_p * mean_t)
    corr = np.clip(cov / (std_p * std_t + 1e-8), -1.0, 1.0)
    alpha = std_p / (std_t + 1e-8)   # variability ratio
    beta = mean_p / (mean_t + 1e-8)  # bias ratio
    kge = 1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    ss_res = s['sum_sq_err']
    ss_tot = count * var_t
    nse = 1 - (ss_res / (ss_tot + 1e-8))

    pbias = 100 * (s['sum_p'] - s['sum_t']) / (s['sum_t'] + 1e-8)
    rmse = np.sqrt(ss_res / count)

    return {'kge': float(kge), 'nse': float(nse), 'correlation': float(corr),
            'variability': float(alpha), 'bias_ratio': float(beta),
            'pbias': float(pbias), 'rmse': float(rmse), 'n_samples': int(count)}


# ═══════════════════════════════════════════════════════════════
#  4. Evaluation loop — exact trainer.validate_epoch()
# ═══════════════════════════════════════════════════════════════

def _spatial_kwargs(model, metadata, device):
    """lat, lon and (parameter set, day) key for models with a spatial GAT; {} otherwise."""
    if not getattr(model, "uses_spatial_attention", False):
        return {}
    if not isinstance(metadata, dict) or "lat" not in metadata:
        raise ValueError("Spatial GAT model needs lat/lon metadata in each batch")
    key = metadata["file_iter_val"].long() * 1_000_000 + metadata["time_idx"].long()
    return {"lat": metadata["lat"].to(device).float(), "lon": metadata["lon"].to(device).float(),
            "graph_key": key.to(device)}


def evaluate_model(model, loader, dataset, config, device):
    """
    Run evaluation exactly like trainer.validate_epoch().
    Returns global metrics, text report, sample count, and detailed per-grid rows.
    """
    target_names = config.data.variables.get("target", [])
    tracker = HydrologicalMetricsTracker(variable_names=target_names)
    grid_coords = {}  # grid_idx → {lat, lon}
    all_time_series_data = [] # Collect time series data

    model.eval()
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            if len(batch) == 6:
                _, _, _, inputs, targets, metadata = batch
            elif len(batch) == 3:
                inputs, targets, metadata = batch
            else:
                inputs, targets = batch
                metadata = None

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs, **_spatial_kwargs(model, metadata, device))
            targets_last = targets[:, -1, :]

            denorm_out, denorm_tgt = denormalize_targets(
                outputs, targets_last, dataset, config, device
            )

            # Extract grid indices + coords from metadata — same as trainer
            grid_indices = None
            bs = inputs.size(0)
            if metadata is not None:
                if isinstance(metadata, (list, tuple)):
                    gids = []
                    for m in metadata:
                        if isinstance(m, dict) and "grid_idx" in m:
                            gid = int(m["grid_idx"].item() if hasattr(m["grid_idx"], "item") else m["grid_idx"])
                            gids.append(gid)
                            if gid not in grid_coords and "lat" in m and "lon" in m:
                                lat = m["lat"].item() if hasattr(m["lat"], "item") else float(m["lat"])
                                lon = m["lon"].item() if hasattr(m["lon"], "item") else float(m["lon"])
                                grid_coords[gid] = {"lat": lat, "lon": lon}
                        else:
                            gids.append(-1)
                    grid_indices = torch.tensor(gids)
                elif isinstance(metadata, dict) and "grid_idx" in metadata:
                    idx = metadata["grid_idx"]
                    # Extract grid indices
                    if hasattr(idx, "tolist"):
                        grid_indices = torch.tensor(idx.tolist())
                    elif isinstance(idx, (list, tuple)):
                        grid_indices = torch.tensor(idx)
                    else:
                        grid_indices = torch.tensor([int(idx)] * bs)
                    
                    # Extract lat/lon for these grids
                    if "lat" in metadata and "lon" in metadata:
                        lats = metadata["lat"]
                        lons = metadata["lon"]
                        
                        # Handle tensor or list inputs
                        if hasattr(lats, "tolist"): lats = lats.tolist()
                        if hasattr(lons, "tolist"): lons = lons.tolist()
                        
                        # Ensure we have lists
                        if not isinstance(lats, (list, tuple)): lats = [lats] * bs
                        if not isinstance(lons, (list, tuple)): lons = [lons] * bs
                        
                        # Populate grid_coords
                        current_gids = grid_indices.tolist()
                        for i, gid in enumerate(current_gids):
                            if gid not in grid_coords:
                                grid_coords[gid] = {
                                    "lat": float(lats[i]), 
                                    "lon": float(lons[i])
                                }

                    # Extract time indices
                    if "time_idx" in metadata:
                        t_idxs = metadata["time_idx"]
                        if hasattr(t_idxs, "tolist"): t_idxs = t_idxs.tolist()
                        elif isinstance(t_idxs, torch.Tensor): t_idxs = t_idxs.tolist()
                        if not isinstance(t_idxs, (list, tuple)): t_idxs = [t_idxs] * bs
                        time_indices = t_idxs
                    else:
                        time_indices = [-1] * bs

            tracker.update(
                denorm_out.detach(), denorm_tgt.detach(), grid_indices,
                y_pred_normalized=outputs.detach(),
                y_true_normalized=targets_last.detach(),
            )
            total_samples += bs

            # Collect time series data (Vectorized)
            p_np = denorm_out.detach().cpu().numpy()
            t_np = denorm_tgt.detach().cpu().numpy()
            
            # Ensure 2D
            if len(p_np.shape) == 1: p_np = p_np.reshape(-1, 1)
            if len(t_np.shape) == 1: t_np = t_np.reshape(-1, 1)

            # Get grid indices
            if grid_indices is not None:
                g_np = grid_indices.detach().cpu().numpy()
            else:
                g_np = np.zeros(bs)
            
            # Get time indices
            if 'time_indices' not in locals(): 
                time_indices = np.full(bs, -1)
            else:
                # Ensure time_indices is numpy array
                if isinstance(time_indices, list):
                    time_indices = np.array(time_indices)
                elif isinstance(time_indices, torch.Tensor):
                    time_indices = time_indices.cpu().numpy()
            
            # Store batches (much faster than row-by-row)
            all_time_series_data.append({
                "pred": p_np,
                "target": t_np,
                "grid": g_np,
                "time": time_indices
            })

    # Compute aggregated metrics
    metrics = tracker.compute()
    report = tracker.generate_detailed_report()

    # Extract DETAILED per-grid metrics before reset
    grid_details = []
    if tracker._grid_stats:
        for grid_idx, vars_stats in tracker._grid_stats.items():
            row = {
                "grid_idx": int(grid_idx),
                "lat": grid_coords.get(int(grid_idx), {}).get("lat", np.nan),
                "lon": grid_coords.get(int(grid_idx), {}).get("lon", np.nan),
            }
            for i, var_name in enumerate(target_names):
                if i in vars_stats and vars_stats[i]["count"] >= 2:
                    m = _metrics_from_stats(vars_stats[i])
                    for mk, mv in m.items():
                        row[f"{var_name}_{mk}"] = mv
                else:
                    for mk in ['kge', 'nse', 'correlation', 'rmse', 'pbias',
                                'variability', 'bias_ratio', 'n_samples']:
                        row[f"{var_name}_{mk}"] = np.nan
            grid_details.append(row)

    tracker.reset()
    # Process time series data
    if all_time_series_data:
        # Concatenate all batches
        all_preds = np.concatenate([b["pred"] for b in all_time_series_data], axis=0)
        all_targets = np.concatenate([b["target"] for b in all_time_series_data], axis=0)
        all_grids = np.concatenate([b["grid"] for b in all_time_series_data], axis=0)
        all_times = np.concatenate([b["time"] for b in all_time_series_data], axis=0)
        
        # Create dictionary for DataFrame
        data = {
            "grid_idx": all_grids,
            "time_idx": all_times
        }
        for i, name in enumerate(target_names):
            data[f"pred_{name}"] = all_preds[:, i]
            data[f"target_{name}"] = all_targets[:, i]
            
        # Create list of dicts efficiently (or return DataFrame)
        # Main function expects list of dicts? Let's check `main`.
        # `df_ts = pd.DataFrame(all_time_series_data)`
        # If we return a list of dicts, `pd.DataFrame` works. 
        # If we return dict of arrays, `pd.DataFrame` works too!
        all_time_series_data = data # Return dict of arrays (pd.DataFrame(data) works)

    return metrics, report, total_samples, grid_details, all_time_series_data


# ═══════════════════════════════════════════════════════════════
#  5. Discover checkpoints
# ═══════════════════════════════════════════════════════════════

def discover_checkpoints(result_dir: str):
    """Find all model checkpoints in the result directory."""
    entries = []
    result_path = Path(result_dir)

    for run_dir in sorted(result_path.iterdir()):
        if not run_dir.is_dir():
            continue
        model_dir = run_dir / "models"
        if not model_dir.exists():
            continue

        checkpoints = sorted(model_dir.glob("model_epoch_*.pth"))
        if not checkpoints:
            continue

        # COLLECT ALL CHECKPOINTS (for ensemble analysis)
        for cp in checkpoints:
            try:
                ep = int(re.search(r"epoch_(\d+)", cp.name).group(1))
                entries.append({
                    "run_name": f"{run_dir.name}_ep{ep}", # Unique name for each epoch
                    "model_path": str(cp),
                    "epoch": ep,
                })
            except:
                pass
    
    # Sort by run_name then epoch
    entries.sort(key=lambda x: (x["run_name"], x["epoch"]))

    logger.info(f"Found {len(entries)} checkpoints in {result_dir}")
    return entries


# ═══════════════════════════════════════════════════════════════
#  6. Main — follows trainer's exact pattern
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Standalone inference — follows EXACT training framework patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all checkpoints (uses grid_sampling from config YAML):
  python run_inference.py --config test.yml \\
    --checkpoint_dir /path/to/Result --device cuda:0

  # Single checkpoint:
  python run_inference.py --config test.yml \\
    --checkpoint /path/to/model_epoch_7.pth --device cuda:0

  # Use specific grid pickle (skip grid_sampling):
  python run_inference.py --config test.yml \\
    --checkpoint_dir /path/to/Result --grid_pkl /path/to/grid_locations_test.pkl
        """,
    )
    parser.add_argument("--config", required=True, help="Path to config YAML (e.g. test.yml)")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Path to Result directory containing run folders")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a single checkpoint (.pth file)")
    parser.add_argument("--device", default="cuda:0", help="Device (cuda:0, cpu)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size (default: from config)")
    parser.add_argument("--grid_pkl", default=None,
                        help="Path to grid_locations_test.pkl (overrides grid_sampling in YAML)")
    parser.add_argument("--output_dir", default="inference_output",
                        help="Directory to save all CSV results (default: inference_output)")
    args = parser.parse_args()

    if not args.checkpoint_dir and not args.checkpoint:
        parser.error("At least one of --checkpoint_dir or --checkpoint must be provided")

    # If user provided a file in --checkpoint_dir, treat it as --checkpoint
    if args.checkpoint_dir and os.path.isfile(args.checkpoint_dir):
        if not args.checkpoint:
            args.checkpoint = args.checkpoint_dir
            args.checkpoint_dir = None
            logger.info(f"Arguments: Treating --checkpoint_dir as single checkpoint: {args.checkpoint}")
        else:
            logger.warning("Both --checkpoint and --checkpoint_dir (file) provided. Using --checkpoint.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load config (same as trainer) ──
    logger.info(f"Loading config: {args.config}")
    config = load_config(args.config)
    target_names = config.data.variables.get("target", [])
    logger.info(f"Target variables: {target_names}")

    # ── Grid selection — SAME AS TRAINER ──
    # Priority: 1) --grid_pkl (explicit pickle), 2) create_grid_aware_datasets() from YAML config
    grid_pkl_path = args.grid_pkl
    grid_indices_test = None

    if grid_pkl_path and os.path.exists(grid_pkl_path):
        logger.info(f"Using provided grid pickle: {grid_pkl_path}")
    else:
        # ALWAYS create fresh grid splits from YAML config — same as trainer
        logger.info("Creating grid splits from YAML config (same as trainer)...")
        grid_config = config.data.grid_sampling
        logger.info(f"  use_exact_counts: {grid_config.get('use_exact_counts', True)}")
        logger.info(f"  test_grids: {grid_config.get('test_grids', 'N/A')}")
        logger.info(f"  test_percentage: {grid_config.get('test_percentage', 'N/A')}")

        splits = create_grid_aware_datasets(
            config,
            use_existing_splits=False,  # FORCE new splits from YAML
            create_stratified=False,
        )
        grid_indices_test = splits.get("test")
        if grid_indices_test is not None:
            logger.info(f"Grid splits created: test={len(grid_indices_test)} grids")
        else:
            logger.warning("No test split created! Check grid_sampling in config.")
        grid_pkl_path = None  # Don't use any old pickle

    # ── Discover checkpoints ──
    if args.checkpoint:
        entries = [{"run_name": Path(args.checkpoint).parent.parent.name,
                     "model_path": args.checkpoint, "epoch": 0}]
    else:
        entries = discover_checkpoints(args.checkpoint_dir)

    if not entries:
        logger.error("No checkpoints found!")
        return

    # ── PRE-FLIGHT CHECK: Sync config with FIRST checkpoint to avoid rebuilds ──
    # This prevents the "Load -> Mismatch -> Rebuild" cycle that causes stalls
    try:
        first_cp_path = entries[0]["model_path"]
        logger.info(f"Pre-loading config from: {first_cp_path} ...")
        cp_content = torch.load(first_cp_path, map_location="cpu", weights_only=False)
        
        # 1. Sync Lags
        ckpt_lags = {}
        if "config" in cp_content:
            c_obj = cp_content["config"]
            if hasattr(c_obj, "model"):
                if hasattr(c_obj.model, "lag_config") and c_obj.model.lag_config:
                        ckpt_lags = c_obj.model.lag_config
                elif hasattr(c_obj.model, "lags") and c_obj.model.lags:
                        ckpt_lags = {"target": {v: c_obj.model.lags for v in task_names}}
            elif isinstance(c_obj, dict):
                if "model" in c_obj:
                        m_cfg = c_obj["model"]
                        if "lag_config" in m_cfg:
                            ckpt_lags = m_cfg["lag_config"]
                        elif "lags" in m_cfg:
                            ckpt_lags = {"target": {v: m_cfg["lags"] for v in task_names}}
                elif "data" in c_obj and "lags" in c_obj["data"]:
                        ckpt_lags = {"target": {v: c_obj["data"]["lags"] for v in task_names}}
        
        # Apply lags to current config
        if ckpt_lags:
            logger.info(f"  Syncing lags from checkpoint: {ckpt_lags}")
            config.model.lag_config = ckpt_lags
            config.model.enable_lags = True
        else:
             # If checkpoint has NO lags (empty dict), explicit disable might be needed
             # But usually empty dict implies no lags. 
             pass

        # 2. Sync Window Size (using state_dict shape)
        state_dict = cp_content.get("model_state_dict", cp_content.get("state_dict", cp_content))
        detected_window_size = None
        for k, v in state_dict.items():
            if "relative_position_bias_table" in k:
                if v.dim() == 2:
                    dim0 = v.shape[0]
                    # Assume 1D for now as per previous fix: 2*w - 1 = dim0
                    w_1d = (dim0 + 1) // 2
                    detected_window_size = w_1d
                    break
        
        if detected_window_size:
            logger.info(f"  Syncing window_size from checkpoint: {detected_window_size}")
            config.model.window_size = detected_window_size

        # Cleanup memory from peek
        del cp_content
        del state_dict
        gc.collect()

    except Exception as e:
        logger.warning(f"Failed to pre-sync config from checkpoint: {e}. Inference will proceed but may trigger rebuilds.")

    # ── Build test dataset — SAME as trainer ──
    logger.info("Building test dataset...")
    test_dataset = UnifiedSurrogateDataset(
        config=config,
        mode="test",
        grid_indices=grid_indices_test,
        grid_locations_path=grid_pkl_path,
    )
    logger.info(f"Test dataset: {len(test_dataset)} samples, nx={test_dataset.nx}, ngrid={test_dataset.ngrid}")

    # Show normalization stats
    logger.info("Normalization stats:")
    for var in target_names:
        mean, std = test_dataset.get_normalization_stats(var)
        logger.info(f"  {var}: mean={mean:.6f}, std={std:.6f}")

    # ── Build DataLoader — SAME as trainer using DataLoaderOptimizer ──
    batch_size_test = args.batch_size or getattr(config.training, "batch_size_valid", 256) or 256
    optimal_params = DataLoaderOptimizer.get_optimal_params(
        batch_size_test, len(test_dataset)
    )
    # Override with user config if specified
    memory_config = config.training.memory
    valid_keys = {'num_workers', 'pin_memory', 'persistent_workers', 'prefetch_factor', 'drop_last', 'timeout'}
    for key, value in memory_config.items():
        if value is not None and key in valid_keys:
            optimal_params[key] = value

    logger.info(f"DataLoader params: {optimal_params}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        **{k: v for k, v in optimal_params.items() if k != 'drop_last'},
    )
    logger.info(f"DataLoader: batch_size={batch_size_test}, batches={len(test_loader)}")


    # ── Evaluate each checkpoint ──
    DATA_NX = test_dataset.nx
    summary_rows = []
    all_grid_rows = []
    summary_rows = []
    all_grid_rows = []
    failed = []
    
    # ── Ensemble Accumulator ──
    ensemble_sum_df = None
    ensemble_count = 0
    ensemble_summary_rows = []

    logger.info(f"\n{'═' * 80}")
    logger.info(f"  EVALUATING {len(entries)} CHECKPOINTS")
    logger.info(f"  Test grids: {test_dataset.ngrid} | Test samples: {len(test_dataset)}")
    logger.info(f"  Output: {os.path.abspath(args.output_dir)}")
    logger.info(f"{'═' * 80}\n")

    for entry in entries:
        rn = entry["run_name"]
        logger.info(f"\n{'─' * 60}")
        logger.info(f"Model: {rn} (epoch {entry['epoch']})")

        try:
            # Load checkpoint content (used for both lag detection and model building)
            cp_content = torch.load(entry["model_path"], map_location="cpu", weights_only=False)
            
            # ── Dynamic Lag Handling ──
            # Extract lags from checkpoint config - Robust check
            ckpt_lags = {}
            if "config" in cp_content:
                c_obj = cp_content["config"]
                # Check for UnifiedConfig object structure
                if hasattr(c_obj, "model"):
                    if hasattr(c_obj.model, "lag_config") and c_obj.model.lag_config:
                        ckpt_lags = c_obj.model.lag_config
                    elif hasattr(c_obj.model, "lags") and c_obj.model.lags:
                        ckpt_lags = {"target": {v: c_obj.model.lags for v in task_names}}
                # Check for dict structure
                elif isinstance(c_obj, dict):
                    if "model" in c_obj:
                        m_cfg = c_obj["model"]
                        if "lag_config" in m_cfg:
                            ckpt_lags = m_cfg["lag_config"]
                        elif "lags" in m_cfg:
                            ckpt_lags = {"target": {v: m_cfg["lags"] for v in task_names}}
                    # Fallback to legacy data.lags
                    elif "data" in c_obj and "lags" in c_obj["data"]:
                        ckpt_lags = {"target": {v: c_obj["data"]["lags"] for v in task_names}}
            
            # Normalize current usage
            current_lags = config.model.lag_config or {}
            
            # Rebuild dataset if lags differ
            if ckpt_lags != current_lags:
                logger.info(f"  Lags changed to {ckpt_lags}. Rebuilding dataset...")
                config.model.lag_config = ckpt_lags
                config.model.enable_lags = bool(ckpt_lags)
                
                test_dataset = UnifiedSurrogateDataset(
                    config=config,
                    mode="test",
                    grid_indices=grid_indices_test,
                    grid_locations_path=grid_pkl_path,
                )
                
                # Rebuild DataLoader
                optimal_params_dyn = DataLoaderOptimizer.get_optimal_params(
                    batch_size_test, len(test_dataset)
                )
                for key, value in memory_config.items():
                    if value is not None and key in valid_keys:
                        optimal_params_dyn[key] = value
                
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size_test,
                    shuffle=False,
                    **{k: v for k, v in optimal_params_dyn.items() if k != 'drop_last'},
                )
                logger.info(f"  Dataset rebuilt. Samples: {len(test_dataset)}")
            
            # Build model passing the already loaded content
            model, info = build_model_from_checkpoint(entry["model_path"], config, device=device, checkpoint_content=cp_content)
            
            if model is None:
                msg = f"Skipped: Architecture mismatch (Context: {config.model.model_type})"
                logger.warning(f" {msg}")
                continue

            model_nx = info.get("n_exo")
            current_data_nx = test_dataset.nx if hasattr(test_dataset, "nx") else DATA_NX

            if model_nx is not None and model_nx != current_data_nx:
                # Attempt to fix mismatch if it looks like lag mismatch
                # Case: Model wants 21 (no lags), Data has 23 (lags enabled)
                # Calculate base dimensions (no lags)
                n_forcing = len(config.data.variables['forcing']) if 'forcing' in config.data.variables else 8
                n_params = len(config.data.variables['parameters']) if 'parameters' in config.data.variables else 13
                base_nx = n_forcing + n_params
                
                if model_nx == base_nx and current_data_nx > base_nx and config.model.enable_lags:
                    logger.warning(f"  Dimension mismatch (Model: {model_nx}, Data: {current_data_nx}). Detected model requires NO LAGS. Disabling lags and rebuilding dataset...")
                    
                    # Disable lags
                    config.model.enable_lags = False
                    config.model.lag_config = {}
                    config.model.lags = None
                    
                    # Rebuild dataset
                    test_dataset = UnifiedSurrogateDataset(
                        config=config,
                        mode="test",
                        grid_indices=grid_indices_test,
                        grid_locations_path=grid_pkl_path,
                    )
                    
                    # Rebuild DataLoader
                    optimal_params_dyn = DataLoaderOptimizer.get_optimal_params(
                        batch_size_test, len(test_dataset)
                    )
                    for key, value in memory_config.items():
                        if value is not None and key in valid_keys:
                            optimal_params_dyn[key] = value
                    
                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=batch_size_test,
                        shuffle=False,
                        **{k: v for k, v in optimal_params_dyn.items() if k != 'drop_last'},
                    )
                    logger.info(f"  Dataset rebuilt without lags. New feature count: {test_dataset.nx}")
                    current_data_nx = test_dataset.nx
                
                # Check again
                if model_nx != current_data_nx:
                    msg = f"n_exo mismatch: model={model_nx} data={current_data_nx}"
                    logger.warning(f" {msg} — SKIPPED")
                    failed.append((rn, msg))
                    del model; torch.cuda.empty_cache(); gc.collect()
                    continue

            t0 = time.time()
            metrics, report, n_samples, grid_details, time_series_list = evaluate_model(
                model, test_loader, test_dataset, config, device
            )
            elapsed = time.time() - t0

            logger.info(report)

            # ── Summary ──
            logger.info(f"\n  Summary ({n_samples} samples, {len(grid_details)} grids, {elapsed:.1f}s):")
            row = {"run_name": rn, "epoch": entry["epoch"],
                   "n_samples": n_samples, "n_grids": len(grid_details)}

            for var in target_names:
                kge = metrics.get(f"{var}_kge", np.nan)
                nse = metrics.get(f"{var}_nse", np.nan)
                corr = metrics.get(f"{var}_correlation", np.nan)
                rmse = metrics.get(f"{var}_rmse", np.nan)
                pbias = metrics.get(f"{var}_pbias", np.nan)
                gkge = metrics.get(f"{var}_grid_kge", np.nan)
                gkge_m = metrics.get(f"{var}_grid_kge_median", np.nan)
                gnse = metrics.get(f"{var}_grid_nse", np.nan)
                gnse_m = metrics.get(f"{var}_grid_nse_median", np.nan)

                logger.info(f"    {var:25s}  KGE={kge:+.4f}  NSE={nse:+.4f}  "
                            f"r={corr:+.4f}  RMSE={rmse:.4f}  PBIAS={pbias:+.2f}%")
                logger.info(f"    {'':25s}  Grid: KGE_mean={gkge:+.4f} KGE_med={gkge_m:+.4f} "
                            f"NSE_mean={gnse:+.4f} NSE_med={gnse_m:+.4f}")

                row[f"{var}_kge_global"] = kge
                row[f"{var}_nse_global"] = nse
                row[f"{var}_corr_global"] = corr
                row[f"{var}_rmse_global"] = rmse
                row[f"{var}_pbias_global"] = pbias
                row[f"{var}_kge_grid_mean"] = gkge
                row[f"{var}_kge_grid_median"] = gkge_m
                row[f"{var}_nse_grid_mean"] = gnse
                row[f"{var}_nse_grid_median"] = gnse_m

            overall_kge = metrics.get("overall_kge_mean", metrics.get("overall_kge", np.nan))
            overall_nse = metrics.get("overall_nse_mean", metrics.get("overall_nse", np.nan))
            overall_kge_med = metrics.get("overall_kge_median", np.nan)

            logger.info(f"\n    Overall KGE (global): {overall_kge:+.4f}  |  "
                        f"Grid KGE median: {overall_kge_med:+.4f}")

            row["overall_kge_global"] = overall_kge
            row["overall_nse_global"] = overall_nse
            row["overall_kge_grid_median"] = overall_kge_med
            summary_rows.append(row)

            # ── Save per-grid CSV for this model ──
            if grid_details:
                df_grid = pd.DataFrame(grid_details).sort_values("grid_idx")
                grid_csv = os.path.join(args.output_dir, f"grid_metrics_{rn}.csv")
                df_grid.to_csv(grid_csv, index=False)
                logger.info(f"  Per-grid CSV: {grid_csv} ({len(df_grid)} grids)")

                for gd in grid_details:
                    gd_copy = gd.copy()
                    gd_copy["run_name"] = rn
                    all_grid_rows.append(gd_copy)

            # ── Save time series for every grid ──
            if time_series_list:
                ts_output_dir = os.path.join(args.output_dir, "time_series")
                os.makedirs(ts_output_dir, exist_ok=True)
                
                df_ts = pd.DataFrame(time_series_list)
                unique_grids = df_ts["grid_idx"].unique()
                
                logger.info(f"  Saving time series for {len(unique_grids)} grids to {ts_output_dir}...")
                
                for grid_idx in unique_grids:
                     group = df_ts[df_ts["grid_idx"] == grid_idx].sort_values("time_idx")
                     
                     # Construct filename
                     fname = f"grid_{grid_idx}_{rn}.csv"
                     fpath = os.path.join(ts_output_dir, fname)
                     
                     group.to_csv(fpath, index=False)
                
                logger.info(f"  Saved {len(unique_grids)} time series CSVs")

            del model; torch.cuda.empty_cache(); gc.collect()

            # ── Ensemble Calculation ──
            if time_series_list:
                df_current = pd.DataFrame(time_series_list)
                
                # Ensure correct types
                df_current['grid_idx'] = df_current['grid_idx'].astype(int)
                df_current['time_idx'] = df_current['time_idx'].astype(int)
                
                # Set index for alignment
                df_current = df_current.set_index(['grid_idx', 'time_idx']).sort_index()
                
                # Isolate predictions and targets
                pred_cols = [c for c in df_current.columns if c.startswith('pred_')]
                target_cols = [c for c in df_current.columns if c.startswith('target_')]
                
                # 1. Update Accumulator
                if ensemble_sum_df is None:
                    ensemble_sum_df = df_current[pred_cols].copy()
                else:
                    # Align and add
                    ensemble_sum_df = ensemble_sum_df.add(df_current[pred_cols], fill_value=0)
                
                ensemble_count += 1
                
                # 2. Compute Ensemble Mean
                ensemble_mean_df = ensemble_sum_df / ensemble_count
                
                # 3. Join with Targets (targets are same for all models, use current)
                df_eval = ensemble_mean_df.join(df_current[target_cols], how='inner')
                
                # 4. Compute Metrics for Ensemble
                logger.info(f"\n  Ensemble (N={ensemble_count}) Evaluation:")
                ensemble_row = {"ensemble_count": ensemble_count}
                
                # Re-use tracker logic or calculate manually? 
                # Converting back to tensors for HydrologicalMetricsTracker is cleanest
                # But we need grid_indices. Index has grid_idx.
                
                y_pred_ens = torch.tensor(df_eval[pred_cols].values, device=device, dtype=torch.float32)
                y_true_ens = torch.tensor(df_eval[target_cols].values, device=device, dtype=torch.float32)
                
                # Extract grid indices from MultiIndex
                grid_indices_ens = torch.tensor(df_eval.index.get_level_values('grid_idx').values, device=device, dtype=torch.int64)
                
                # Create a temporary tracker
                ens_tracker = HydrologicalMetricsTracker(variable_names=target_names)
                ens_tracker.update(y_pred_ens, y_true_ens, grid_indices_ens)
                ens_metrics = ens_tracker.compute()
                
                # Log Summary
                overall_kge = ens_metrics.get("overall_kge_mean", ens_metrics.get("overall_kge", np.nan))
                overall_nse = ens_metrics.get("overall_nse_mean", ens_metrics.get("overall_nse", np.nan))
                
                logger.info(f"    {'Overall KGE':25s}: {overall_kge:+.4f}")
                logger.info(f"    {'Overall NSE':25s}: {overall_nse:+.4f}")
                
                ensemble_row["overall_kge"] = overall_kge
                ensemble_row["overall_nse"] = overall_nse
                
                # Per-variable
                for var in target_names:
                    kge = ens_metrics.get(f"{var}_kge", np.nan)
                    nse = ens_metrics.get(f"{var}_nse", np.nan)
                    ensemble_row[f"{var}_kge"] = kge
                    ensemble_row[f"{var}_nse"] = nse
                    logger.info(f"    {var:25s}: KGE={kge:+.4f} NSE={nse:+.4f}")
                
                ensemble_summary_rows.append(ensemble_row)

        except Exception as e:
            import traceback; traceback.print_exc()
            logger.error(f"{rn}: {e}")
            failed.append((rn, str(e)))
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════
    #  Final summary + CSV saves
    # ═══════════════════════════════════════════════════════════
    logger.info(f"\n\n{'═' * 80}")
    logger.info(f"  FINAL: {len(summary_rows)} succeeded, {len(failed)} failed")
    logger.info(f"{'═' * 80}")

    if failed:
        logger.info("\nFailed:")
        for name, reason in failed:
            logger.info(f"  • {name}: {reason[:120]}")

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary = df_summary.sort_values("overall_kge_global", ascending=False)

        logger.info(f"\n  {'Rank':>4}  {'Run':28s}  {'KGE_global':>10}  {'KGE_grid_med':>12}  {'NSE_global':>10}")
        logger.info(f"  {'─' * 70}")
        for rank, (_, r) in enumerate(df_summary.iterrows(), 1):
            logger.info(f"  {rank:4d}  {r['run_name']:28s}  {r['overall_kge_global']:+10.4f}  "
                        f"{r.get('overall_kge_grid_median', np.nan):+12.4f}  "
                        f"{r['overall_nse_global']:+10.4f}")

        summary_csv = os.path.join(args.output_dir, "model_summary.csv")
        df_summary.to_csv(summary_csv, index=False)
        logger.info(f"\nModel summary: {summary_csv}")

    if all_grid_rows:
        df_all = pd.DataFrame(all_grid_rows).sort_values(["run_name", "grid_idx"])
        master_csv = os.path.join(args.output_dir, "all_grids_all_models.csv")
        df_all.to_csv(master_csv, index=False)
        logger.info(f"Master grid CSV: {master_csv} ({len(df_all)} rows)")

    if ensemble_summary_rows:
        ens_csv = os.path.join(args.output_dir, "ensemble_progress.csv")
        pd.DataFrame(ensemble_summary_rows).to_csv(ens_csv, index=False)
        logger.info(f"Ensemble progress CSV: {ens_csv}")

    logger.info(f"\nAll results: {os.path.abspath(args.output_dir)}/")
    logger.info("Done! ")


if __name__ == "__main__":
    main()
