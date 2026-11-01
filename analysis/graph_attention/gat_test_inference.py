"""Test-set inference of LandFormer-AR-All with the spatial graph attention module (Section 4.7).

Uses the evaluation configuration of run_inference.py (configs/evaluation/test.yml: held-out cells, unseen parameter
set, 2001-2006). Each batch contains all test cells of one day, so that every graph is complete.

Usage:  python gat_test_inference.py <checkpoint> <tag>
Writes per-cell daily series to outputs/time_series/grid_<cell>_<tag>.csv and the graph statistics to
outputs/<tag>_run_info.json.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..')))
from landformer_paths import ROOT  # noqa: E402
import os, sys, gc, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT, TAG = os.path.abspath(sys.argv[1]), sys.argv[2]   # resolved before changing directory
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from src.core.config import load_config
from src.data.dataset import UnifiedSurrogateDataset
from src.data.grid_splitter import create_grid_aware_datasets
from run_inference import build_model_from_checkpoint, evaluate_model

OUT = os.path.join(HERE, "outputs")
TS = os.path.join(OUT, "time_series"); os.makedirs(TS, exist_ok=True)
device = torch.device("cuda:0")

config = load_config(os.path.join(ROOT, "configs/evaluation/test.yml"))                          # evaluation configuration of the paper
splits = create_grid_aware_datasets(config, use_existing_splits=False, create_stratified=False)
cp = torch.load(CKPT, map_location="cpu", weights_only=False)
config.model.lag_config = cp["config"].model.lag_config  # lag configuration stored in the checkpoint
config.model.enable_lags = True
for k, v in cp["model_state_dict"].items():
    if "relative_position_bias_table" in k and v.dim() == 2:
        config.model.window_size = (v.shape[0] + 1) // 2
        break
print("epoch", cp.get("epoch"), "| lags", config.model.lag_config, flush=True)
del cp; gc.collect()

ds = UnifiedSurrogateDataset(config=config, mode="test", grid_indices=splits.get("test"), grid_locations_path=None)
print(f"test dataset: {len(ds):,} samples, nx={ds.nx}, ngrid={ds.ngrid}", flush=True)
# one batch = one day of all test cells, so no day's graph is split across batches
loader = DataLoader(ds, batch_size=ds.ngrid, shuffle=False, num_workers=8, pin_memory=True)

model, info = build_model_from_checkpoint(CKPT, config, device=device)
print("GAT in model:", model.uses_spatial_attention, "| n_exo", info.get("n_exo"), flush=True)
t0 = time.time()
metrics, report, n_samples, grid_details, ts_list = evaluate_model(model, loader, ds, config, device)
print(f"inference {time.time() - t0:.0f}s, {n_samples:,} samples", flush=True)
stats = model.pop_graph_stats()
print("graph stats:", stats, flush=True)

df = pd.DataFrame(ts_list)
for gid, grp in df.groupby("grid_idx"):
    grp.to_csv(os.path.join(TS, f"grid_{int(gid)}_{TAG}.csv"), index=False)
json.dump({"checkpoint": CKPT, "tag": TAG, "graph_stats": stats, "n_samples": int(n_samples),
           "n_cells": int(df.grid_idx.nunique())}, open(os.path.join(OUT, f"{TAG}_run_info.json"), "w"), indent=2, default=float)
print("saved", df.grid_idx.nunique(), "cells to", TS, flush=True)
