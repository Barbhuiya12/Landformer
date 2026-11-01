#!/usr/bin/env python3
"""Training-resolution experiment (Section 4.5, Figures 10 and 11).

Evaluates the models trained at 1, 0.5 and 0.25 deg at the native 0.0625 deg resolution, and their multi-resolution
ensembles. Output layout as in ensemble_coverage.py.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')))
from landformer_paths import EXT_DIR, ROOT  # noqa: E402
import sys, os, gc, re, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.core.config import load_config
from src.data.dataset import UnifiedSurrogateDataset
from src.data.grid_splitter import create_grid_aware_datasets
from src.utils.hydro_metrics import HydrologicalMetricsTracker
from src.utils.memory import DataLoaderOptimizer
from run_inference import build_model_from_checkpoint, evaluate_model, _metrics_from_stats

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CONFIG_PATH = f"{ROOT}/configs/evaluation/test_resolution.yml"
DEVICE = "cuda:0"
BATCH_SIZE = 512
OUTPUT_DIR = "outputs/ensemble_resolution"

CHECKPOINT_FILES = [
f"{EXT_DIR}/extended_results/1_degree_lag/models/best_model.pth",
f"{EXT_DIR}/extended_results/05_degree_LAG/models/best_model.pth",
f'{EXT_DIR}/extended_results/25_degree_lag/models/best_model.pth'

]


def main():
    print(f"Ensemble Inference: {len(CHECKPOINT_FILES)} models")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"   Device: {DEVICE}")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    indiv_dir = os.path.join(OUTPUT_DIR, "individual_grid_metrics")
    ens_grid_dir = os.path.join(OUTPUT_DIR, "ensemble_grid_metrics")
    ens_sum_dir = os.path.join(OUTPUT_DIR, "ensemble_summary")
    os.makedirs(indiv_dir, exist_ok=True)
    os.makedirs(ens_grid_dir, exist_ok=True)
    os.makedirs(ens_sum_dir, exist_ok=True)
    
    # ── 1. Setup ──
    print("\nLoading config & data...")
    config = load_config(CONFIG_PATH)
    target_names = config.data.variables.get("target", [])
    
    # Sync config from first checkpoint
    if os.path.exists(CHECKPOINT_FILES[0]):
        first_cp = torch.load(CHECKPOINT_FILES[0], map_location="cpu", weights_only=False)
        if "config" in first_cp:
            c_obj = first_cp["config"]
            if hasattr(c_obj, "model") and hasattr(c_obj.model, "lag_config") and c_obj.model.lag_config:
                config.model.lag_config = c_obj.model.lag_config
                config.model.enable_lags = True
                print(f"  Synced lags from checkpoint")
        state_dict = first_cp.get("model_state_dict", first_cp.get("state_dict", first_cp))
        for k, v in state_dict.items():
            if "relative_position_bias_table" in k and v.dim() == 2:
                w = (v.shape[0] + 1) // 2
                config.model.window_size = w
                print(f"  Synced window_size: {w}")
                break
        del first_cp, state_dict; gc.collect()
    
    # Build dataset & loader
    splits = create_grid_aware_datasets(config, use_existing_splits=False)
    test_dataset = UnifiedSurrogateDataset(
        config=config, mode="test",
        grid_indices=splits.get("test"),
        grid_locations_path=None
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=8, pin_memory=True, prefetch_factor=2
    )
    device = torch.device(DEVICE)
    print(f"Data: {len(test_dataset)} samples, {test_dataset.ngrid} grids")
    
    # ── 2. Ensemble Loop ──
    ensemble_sum_df = None
    targets_df = None
    ensemble_count = 0
    ensemble_progress = []
    grid_coords_map = {}  # grid_idx -> {lat, lon}
    
    print(f"\n{'═'*80}")
    print(f"  RUNNING {len(CHECKPOINT_FILES)} MODELS + INCREMENTAL ENSEMBLE")
    print(f"{'═'*80}")
    
    for idx, cp_path in enumerate(CHECKPOINT_FILES):
        model_name = f"model_{idx+1}"
        
        if not os.path.exists(cp_path):
            print(f"\n[{idx+1}/{len(CHECKPOINT_FILES)}] NOT FOUND: {cp_path}")
            continue
        
        print(f"\n{'─'*60}")
        print(f"[{idx+1}/{len(CHECKPOINT_FILES)}] {os.path.basename(os.path.dirname(os.path.dirname(cp_path)))}")
        
        try:
            # Build model
            model, info = build_model_from_checkpoint(cp_path, config, device=device)
            if model is None:
                print("  Build failed, skipping")
                continue
            
            # Run inference
            t0 = time.time()
            metrics, report, n_samples, grid_details, ts_data = evaluate_model(
                model, test_loader, test_dataset, config, device
            )
            elapsed = time.time() - t0
            
            overall_kge = metrics.get("overall_kge_mean", metrics.get("overall_kge", np.nan))
            overall_nse = metrics.get("overall_nse_mean", metrics.get("overall_nse", np.nan))
            print(f"  Individual: KGE={overall_kge:+.4f}  NSE={overall_nse:+.4f}  ({elapsed:.1f}s)")
            
            # ────────────────────────────────────────────
            # SAVE INDIVIDUAL GRID METRICS (from evaluate_model)
            # grid_details has: grid_idx, lat, lon, {var}_kge, {var}_nse,
            #   {var}_correlation, {var}_rmse, {var}_pbias, {var}_variability,
            #   {var}_bias_ratio, {var}_n_samples
            # ────────────────────────────────────────────
            if grid_details:
                df_grid = pd.DataFrame(grid_details)
                out_path = os.path.join(indiv_dir, f"{model_name}_grid_metrics.csv")
                df_grid.to_csv(out_path, index=False)
                print(f"  Saved {model_name}_grid_metrics.csv ({len(df_grid)} grids)")
                
                # Cache lat/lon for ensemble grid metrics
                for _, row in df_grid.iterrows():
                    gid = int(row["grid_idx"])
                    if gid not in grid_coords_map:
                        grid_coords_map[gid] = {"lat": row.get("lat", np.nan), "lon": row.get("lon", np.nan)}
            
            # ────────────────────────────────────────────
            # ENSEMBLE ACCUMULATION
            # ────────────────────────────────────────────
            if ts_data:
                df_curr = pd.DataFrame(ts_data)
                df_curr["grid_idx"] = df_curr["grid_idx"].astype(int)
                df_curr["time_idx"] = df_curr["time_idx"].astype(int)
                df_curr = df_curr.set_index(["grid_idx", "time_idx"]).sort_index()
                
                pred_cols = [c for c in df_curr.columns if c.startswith("pred_")]
                target_cols = [c for c in df_curr.columns if c.startswith("target_")]
                
                if ensemble_sum_df is None:
                    ensemble_sum_df = df_curr[pred_cols].copy()
                    targets_df = df_curr[target_cols].copy()
                else:
                    ensemble_sum_df = ensemble_sum_df.add(df_curr[pred_cols], fill_value=0)
                
                ensemble_count += 1
                
                # ── Compute Ensemble Metrics ──
                print(f"  Computing ensemble (N={ensemble_count})...")
                ens_mean = ensemble_sum_df / ensemble_count
                df_eval = ens_mean.join(targets_df, how="inner")
                
                # Overall ensemble metrics via HydrologicalMetricsTracker
                y_pred = torch.tensor(df_eval[pred_cols].values, dtype=torch.float32)
                y_true = torch.tensor(df_eval[target_cols].values, dtype=torch.float32)
                g_idx = torch.tensor(df_eval.index.get_level_values("grid_idx").values, dtype=torch.int64)
                
                ens_tracker = HydrologicalMetricsTracker(variable_names=target_names)
                ens_tracker.update(y_pred, y_true, g_idx)
                em = ens_tracker.compute()
                
                ens_kge = em.get("overall_kge_mean", em.get("overall_kge", np.nan))
                ens_nse = em.get("overall_nse_mean", em.get("overall_nse", np.nan))
                print(f"  Ensemble (N={ensemble_count}): KGE={ens_kge:+.4f}  NSE={ens_nse:+.4f}")
                
                # ── Save ENSEMBLE PER-GRID METRICS ──
                ens_grid_rows = []
                if ens_tracker._grid_stats:
                    for gid, vars_stats in ens_tracker._grid_stats.items():
                        row = {
                            "grid_idx": int(gid),
                            "lat": grid_coords_map.get(int(gid), {}).get("lat", np.nan),
                            "lon": grid_coords_map.get(int(gid), {}).get("lon", np.nan),
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
                        ens_grid_rows.append(row)
                
                if ens_grid_rows:
                    df_ens_grid = pd.DataFrame(ens_grid_rows)
                    df_ens_grid.to_csv(
                        os.path.join(ens_grid_dir, f"grid_em_{ensemble_count}.csv"),
                        index=False
                    )
                    print(f"  Saved grid_em_{ensemble_count}.csv ({len(df_ens_grid)} grids)")
                
                # ── Save ENSEMBLE SUMMARY ──
                ens_summary = {"N": ensemble_count, "overall_kge": ens_kge, "overall_nse": ens_nse}
                for v in target_names:
                    ens_summary[f"{v}_kge"] = em.get(f"{v}_kge", np.nan)
                    ens_summary[f"{v}_nse"] = em.get(f"{v}_nse", np.nan)
                    ens_summary[f"{v}_rmse"] = em.get(f"{v}_rmse", np.nan)
                    ens_summary[f"{v}_pbias"] = em.get(f"{v}_pbias", np.nan)
                    ens_summary[f"{v}_correlation"] = em.get(f"{v}_correlation", np.nan)
                pd.DataFrame([ens_summary]).to_csv(
                    os.path.join(ens_sum_dir, f"em_n_{ensemble_count}.csv"),
                    index=False
                )
                print(f"  Saved em_n_{ensemble_count}.csv")
                
                ens_tracker.reset()
                
                # ── Progress log ──
                row = {
                    "N": ensemble_count, "model": model_name,
                    "individual_kge": overall_kge, "individual_nse": overall_nse,
                    "ensemble_kge": ens_kge, "ensemble_nse": ens_nse
                }
                for v in target_names:
                    row[f"ens_{v}_kge"] = em.get(f"{v}_kge", np.nan)
                    row[f"ens_{v}_nse"] = em.get(f"{v}_nse", np.nan)
                ensemble_progress.append(row)
                
                pd.DataFrame(ensemble_progress).to_csv(
                    os.path.join(OUTPUT_DIR, "ensemble_progress.csv"), index=False
                )
            
            del model; torch.cuda.empty_cache(); gc.collect()
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback; traceback.print_exc()
            torch.cuda.empty_cache(); gc.collect()
    
    print(f"\n{'═'*80}")
    print(f"  DONE: {ensemble_count} models evaluated")
    print(f"  Results in: {OUTPUT_DIR}/")
    print(f"{'═'*80}")


if __name__ == "__main__":
    main()
