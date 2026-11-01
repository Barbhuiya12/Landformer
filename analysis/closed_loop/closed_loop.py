"""Closed-loop (free-running) evaluation of LandFormer-AR-All (Section 4.4, Figure 9).

In closed loop the lagged inputs are the model's own previous predictions instead of VIC outputs. Each run is
initialised with VIC values for the 269 days that precede the first prediction day; afterwards no VIC information
is used. With lagged inputs taken from VIC (open loop), the day-by-day engine `free_run` reproduces the batch
inference of run_inference.py.

Steps
  step0   check that the engine reproduces the open-loop predictions written by run_inference.py
  step1   continuous runs over the test period (27 Sep 2001 - 31 Dec 2006) on all test cells: closed loop, open
          loop, LandFormer-NAR and persistence; per-cell metrics and summary statistics (Wilcoxon signed-rank test,
          rank-biserial correlation), cumulative bias and partitioning of precipitation
The lead-time analysis is in error_growth.py.

Usage:  python closed_loop.py [step0|step1|all]
Outputs are written to this folder.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..')))
from landformer_paths import EXT_DIR, RESULTS_DIR, ROOT  # noqa: E402
import os, sys, json, time, pickle, shutil, logging
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
import torch

OUT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
RES = f"{RESULTS_DIR}"
AR_CKPT = f"{RES}/Large_full_lag/models/best_model.pth"          # LandFormer-AR-All
NAR_CKPT = f"{RES}/TEST_HYDRODL_4/models/best_model.pth"         # LandFormer-NAR (109M)
SCALER = {"inference": f"{RES}/Test_0.90_inf/train_data_scaler.bin",   # statistics used by run_inference.py with configs/evaluation/test.yml
          "AR": f"{RES}/Large_full_lag/train_data_scaler.bin",          # AR-All training statistics
          "NAR": f"{RES}/TEST_HYDRODL_4/train_data_scaler.bin"}         # NAR training statistics
PUBLISHED_TS = os.path.join(ROOT, "inference_output", "time_series")   # written by run_inference.py
PUB_TAG = "Large_full_lag"            # run name used by run_inference.py (results folder name)
FORCING = ["prcp", "tasmax", "tasmin", "vp", "swdown", "lwdown", "sp", "wind"]
PARAMS = ["infilt", "Dsmax", "Ds", "Ws", "d1", "d2", "d3", "Ksat1", "Ksat2", "Ksat3", "expt1", "expt2", "expt3"]
TARGETS = ["evap", "soil_moisture_total", "Q"]
SHORT = {"evap": "ET", "soil_moisture_total": "SM", "Q": "Q"}
L = 270                     # input sequence length
T0 = 269                    # first predicted day index (2001-09-27)
LAG_VARS = [0, 2]           # lagged-input channels of LandFormer-AR-All (indices in target order)
DEVICE = torch.device("cuda:0")
CHUNK = 2500                # cells per forward pass
# Step-0 check: the engine runs in full FP32. Batch inference with TF32 arithmetic (the PyTorch default for
# convolutions) changes predictions by up to a few hundredths of mm/day depending on batch size, so an exact match is
# not expected. The check passes when, for every variable, the largest difference from the run_inference.py output
# does not exceed the TF32 batch-size variation measured on the same 50 cells and 60 days.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
log = logging.getLogger("closed_loop")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


# ------------------------------------------------------------------ data
def load_scaler(name):
    return pickle.load(open(SCALER[name], "rb"))


def load_test_data():
    """Test arrays normalised with the inference scaler, exactly as the published inference saw them."""
    f = os.path.join(CACHE, "test_arrays.npz")
    if os.path.exists(f):
        d = np.load(f, allow_pickle=True)
        return {k: d[k] for k in d.files}
    from src.core.config import load_config
    from src.data.dataset import UnifiedSurrogateDataset
    from src.data.grid_splitter import create_grid_aware_datasets
    cfg = load_config(os.path.join(ROOT, "configs/evaluation/test.yml"))
    ds_out = os.path.join(CACHE, "ds_out"); os.makedirs(ds_out, exist_ok=True)
    shutil.copy(SCALER["inference"], os.path.join(ds_out, "train_data_scaler.bin"))
    for a in ("out_dir", "out_model_dir", "tensorb_dir", "figure_dir"):
        setattr(cfg, a, ds_out)                                   # keep all side files out of Result/
    splits = create_grid_aware_datasets(cfg, use_existing_splits=False, create_stratified=False)
    cp = torch.load(AR_CKPT, map_location="cpu", weights_only=False)
    cfg.model.lag_config = cp["config"].model.lag_config; cfg.model.enable_lags = True
    del cp
    ds = UnifiedSurrogateDataset(config=cfg, mode="test", grid_indices=splits.get("test"), grid_locations_path=None)
    assert list(ds.fvarLst) == FORCING and list(ds.pvarLst) == PARAMS and list(ds.ovarLst) == TARGETS
    F = np.asarray(ds.forcings[0], np.float32)                   # (T, G, 8)
    P = np.asarray(ds.pars[0], np.float32)                       # (G, 13)
    Y = np.asarray(ds.target[0], np.float32)                     # (T, G, 3)
    out = dict(F=F, P=P, Y=Y, lat=np.asarray(ds.selected_lats_np), lon=np.asarray(ds.selected_lons_np))
    np.savez(f, **out)
    return out


def renorm(z, names, s_from, s_to):
    """z-scores under scaler s_from -> z-scores under scaler s_to (last axis = variables)."""
    z = z.astype(np.float64)
    m0 = np.array([s_from[f"{n}_mean"] for n in names], float); d0 = np.array([s_from[f"{n}_std"] for n in names], float)
    m1 = np.array([s_to[f"{n}_mean"] for n in names], float); d1 = np.array([s_to[f"{n}_std"] for n in names], float)
    return ((z * d0 + m0 - m1) / d1).astype(np.float32)


def physical(z, names, s):
    m = np.array([s[f"{n}_mean"] for n in names], np.float32); d = np.array([s[f"{n}_std"] for n in names], np.float32)
    return z * d + m


def build_model(ckpt):
    from src.core.config import load_config
    from run_inference import build_model_from_checkpoint
    cfg = load_config(os.path.join(ROOT, "configs/evaluation/test.yml"))
    model, info = build_model_from_checkpoint(ckpt, cfg, device=DEVICE)
    assert not model.uses_spatial_attention
    return model, info


# ------------------------------------------------------------------ engine
@torch.no_grad()
def free_run(model, Fz, Pz, Yz, cells, starts, horizon, lag_source, out_scaler, use_lags=True):
    """
    Run the model day by day for members m = (cell[m], start[m]).
    Fz, Pz, Yz: normalised arrays (with the model's scaler) on the GPU: (T,G,8), (G,13), (T,G,3).
    Window for day t covers days t-269..t; lag channel at window position j holds y(day_j - 1), zero at j = 0.
    lag_source "vic": lags from VIC always (open loop). "model": VIC before the start (warm start), then the
    model's own clipped, re-normalised predictions.
    Returns predictions in physical units, shape (horizon, M, 3), clipped at zero.
    """
    M = len(cells)
    cells_t = torch.as_tensor(cells, device=DEVICE)
    starts_t = torch.as_tensor(starts, device=DEVICE)
    off = torch.arange(-(L - 1), 1, device=DEVICE)                       # -269..0
    mo = torch.tensor([out_scaler[f"{n}_mean"] for n in TARGETS], device=DEVICE)
    sd = torch.tensor([out_scaler[f"{n}_std"] for n in TARGETS], device=DEVICE)
    lag_idx = torch.as_tensor(LAG_VARS, device=DEVICE)
    if use_lags:
        # initial lag window: position 0 -> 0; positions 1..269 -> y(start-269 .. start-1) from VIC
        days0 = starts_t[:, None] + torch.arange(-(L - 1), 0, device=DEVICE)[None, :]       # (M, 269)
        R = torch.zeros(M, L, len(LAG_VARS), device=DEVICE)
        R[:, 1:, :] = Yz[days0, cells_t[:, None]][:, :, lag_idx]
    preds = torch.empty(horizon, M, 3, device=DEVICE)
    Pm = Pz[cells_t]                                                     # (M, 13)
    for h in range(horizon):
        t = starts_t + h                                                 # (M,)
        days = t[:, None] + off[None, :]                                 # (M, 270)
        out = torch.empty(M, 3, device=DEVICE)
        for a in range(0, M, CHUNK):
            b = min(M, a + CHUNK)
            xf = Fz[days[a:b], cells_t[a:b, None]]                       # (m, 270, 8)
            xp = Pm[a:b, None, :].expand(-1, L, -1)                      # (m, 270, 13)
            x = torch.cat([xf, xp, R[a:b]], dim=2) if use_lags else torch.cat([xf, xp], dim=2)
            out[a:b] = model(x)
        y = torch.relu(out * sd + mo)                                     # physical, clipped
        preds[h] = y
        if use_lags and h < horizon - 1:
            if lag_source == "model":
                new = (y[:, lag_idx] - mo[lag_idx]) / sd[lag_idx]
            else:
                new = Yz[t, cells_t][:, lag_idx]
            R = torch.cat([torch.zeros(M, 1, len(LAG_VARS), device=DEVICE), R[:, 2:], new[:, None, :]], dim=1)
    return preds.cpu().numpy()


# ------------------------------------------------------------------ metrics
def metrics(o, s):
    o = np.asarray(o, float); s = np.asarray(s, float)
    with np.errstate(all="ignore"):
        r = np.corrcoef(o, s)[0, 1] if o.std() > 0 and s.std() > 0 else np.nan
        a = s.std() / o.std() if o.std() > 0 else np.nan
        b = s.mean() / o.mean() if o.mean() != 0 else np.nan
        return {"KGE": 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2),
                "NSE": 1 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum() if o.std() > 0 else np.nan,
                "r": r, "RMSE": np.sqrt(((s - o) ** 2).mean())}


def published_cells():
    import glob
    return sorted(int(os.path.basename(p).split("_")[1]) for p in glob.glob(f"{PUBLISHED_TS}/grid_*_{PUB_TAG}.csv"))


# ------------------------------------------------------------------ steps
def step0(data, model):
    sI = load_scaler("inference")
    cells = published_cells()[:50]
    Fz = torch.as_tensor(data["F"], device=DEVICE); Pz = torch.as_tensor(data["P"], device=DEVICE)
    Yz = torch.as_tensor(data["Y"], device=DEVICE)
    t1 = time.time()
    p = free_run(model, Fz, Pz, Yz, cells, [T0] * len(cells), 60, "vic", sI)
    gpu_s = time.time() - t1
    rows = []
    vic_phys = np.maximum(physical(data["Y"], TARGETS, sI), 0)
    for m, g in enumerate(cells):
        d = pd.read_csv(f"{PUBLISHED_TS}/grid_{g}_{PUB_TAG}.csv").set_index("time_idx").loc[T0:T0 + 59]
        for k, v in enumerate(TARGETS):
            rows.append({"cell": g, "var": SHORT[v],
                         "max_abs_pred_diff": float(np.abs(p[:, m, k] - d[f"pred_{v}"].values).max()),
                         "max_abs_target_diff": float(np.abs(vic_phys[T0:T0 + 60, g, k] - d[f"target_{v}"].values).max())})
    df = pd.DataFrame(rows)
    # TF32 batch-size variation on the same cells and days: batch of 50 against one cell at a time
    global CHUNK
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    chunk0 = CHUNK
    a = free_run(model, Fz, Pz, Yz, cells, [T0] * len(cells), 60, "vic", sI)
    CHUNK = 1
    b = free_run(model, Fz, Pz, Yz, cells, [T0] * len(cells), 60, "vic", sI)
    CHUNK = chunk0
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    noise = {SHORT[v]: float(np.abs(a - b)[:, :, k].max()) for k, v in enumerate(TARGETS)}
    diff = df.groupby("var").max_abs_pred_diff.max().to_dict()
    res = {"cells": 50, "days": 60, "gpu_seconds": gpu_s, "precision": "float32, TF32 off",
           "max_abs_pred_diff_by_var": diff,
           "tf32_batch_noise_by_var": noise,
           "max_abs_target_diff_by_var": df.groupby("var").max_abs_target_diff.max().to_dict(),
           "exact_match_within_1e-4": bool(df.max_abs_pred_diff.max() < 1e-4)}
    res["passed"] = bool(all(diff[v] <= noise[v] for v in diff))
    json.dump(res, open(os.path.join(OUT, "step0_check.json"), "w"), indent=2)
    log.info(f"STEP 0: {res}")
    return res


def step1(data, ar_model, nar_model):
    sI, sA, sN = load_scaler("inference"), load_scaler("AR"), load_scaler("NAR")
    cells = published_cells()
    G = len(cells); H = data["F"].shape[0] - T0                           # 1922 days
    timing = {}
    runs = {}
    for name, model, scaler, src, lags in [("open_loop", ar_model, sA, "vic", True),
                                           ("closed_loop", ar_model, sA, "model", True),
                                           ("nar", nar_model, sN, "vic", False)]:
        f = os.path.join(OUT, f"pred_{name}.npy")
        if os.path.exists(f):
            runs[name] = np.load(f); continue
        Fz = torch.as_tensor(renorm(data["F"], FORCING, sI, scaler), device=DEVICE)
        Pz = torch.as_tensor(renorm(data["P"], PARAMS, sI, scaler), device=DEVICE)
        Yz = torch.as_tensor(renorm(data["Y"], TARGETS, sI, scaler), device=DEVICE)
        t1 = time.time()
        runs[name] = free_run(model, Fz, Pz, Yz, cells, [T0] * G, H, src, scaler, use_lags=lags)
        timing[name] = time.time() - t1
        log.info(f"STEP 1 {name}: {timing[name]:.0f} s")
        np.save(f, runs[name])
        del Fz, Pz, Yz; torch.cuda.empty_cache()
    vic = np.maximum(physical(data["Y"], TARGETS, sI), 0)[:, cells, :]     # (T, G, 3) physical
    runs["persistence"] = vic[T0 - 1:T0 - 1 + H]                          # y_hat_t = VIC y_{t-1}
    obs = vic[T0:T0 + H]
    np.save(os.path.join(OUT, "vic_obs.npy"), obs)
    json.dump({"cells": cells, "gpu_seconds": timing}, open(os.path.join(OUT, "step1_info.json"), "w"))
    return cells, obs, runs


def per_cell_table(cells, obs, runs):
    rows = []
    for name, pr in runs.items():
        for m, g in enumerate(cells):
            for k, v in enumerate(TARGETS):
                rows.append({"cell": g, "variable": SHORT[v], "case": name, **metrics(obs[:, m, k], pr[:, m, k])})
    return pd.DataFrame(rows)


def summary_table(pc):
    from scipy.stats import wilcoxon
    out = []
    piv = pc.pivot_table(index=["cell", "variable"], columns="case", values=["KGE", "NSE", "r", "RMSE"])
    for var in ["ET", "SM", "Q"]:
        sub = piv.xs(var, level="variable")
        for met in ["KGE", "NSE", "r", "RMSE"]:
            row = {"variable": var, "metric": met}
            for case in ["closed_loop", "open_loop", "nar", "persistence"]:
                v = sub[(met, case)]; v = v[np.isfinite(v)]
                row[f"{case}_median"] = v.median(); row[f"{case}_q25"] = v.quantile(.25); row[f"{case}_q75"] = v.quantile(.75)
            for ref in ["open_loop", "nar", "persistence"]:
                a, b = sub[(met, "closed_loop")], sub[(met, ref)]
                ok = np.isfinite(a) & np.isfinite(b)
                d = (a - b)[ok]
                better = (d < 0) if met == "RMSE" else (d > 0)
                worse = (d > 0) if met == "RMSE" else (d < 0)
                row[f"pct_cells_closed_beats_{ref}"] = 100 * better.mean()
                if ref in ("open_loop", "nar"):
                    row[f"wilcoxon_p_closed_vs_{ref}"] = wilcoxon(a[ok], b[ok]).pvalue
                    row[f"rank_biserial_closed_vs_{ref}"] = (better.sum() - worse.sum()) / max(1, (d != 0).sum())
            row["n_cells"] = int((np.isfinite(sub[(met, "closed_loop")])).sum())
            out.append(row)
    return pd.DataFrame(out)


def step3_4(data, cells, obs, runs):
    sI = load_scaler("inference")
    H = obs.shape[0]
    res = {}
    # cumulative bias
    cb = {}
    for case in ["closed_loop", "open_loop"]:
        for k, var in [(0, "ET"), (2, "Q")]:
            c = np.cumsum(runs[case][:, :, k] - obs[:, :, k], axis=0)          # (H, G)
            cb[(case, var)] = c
            med = np.median(c, axis=1)
            x = np.arange(H); sel = x >= 270
            slope = np.polyfit(x[sel], med[sel], 1)[0]
            per_cell = np.polyfit(x[sel], c[sel], 1)[0]                          # (G,)
            res[f"cumbias_slope_median_curve_{case}_{var}_mm_per_day"] = float(slope)
            res[f"cumbias_slope_per_cell_median_{case}_{var}_mm_per_day"] = float(np.median(per_cell))
            res[f"cumbias_final_median_{case}_{var}_mm"] = float(med[-1])
            res[f"cumbias_final_iqr_{case}_{var}_mm"] = [float(np.percentile(c[-1], 25)), float(np.percentile(c[-1], 75))]
    np.savez(os.path.join(OUT, "cumulative_bias.npz"),
             **{f"{a}_{b}": np.percentile(v, [25, 50, 75], axis=1) for (a, b), v in cb.items()})
    # ET/P and Q/P
    prcp = np.maximum(physical(data["F"][..., :1], ["prcp"], sI)[..., 0], 0)[T0:T0 + H][:, cells]   # (H, G)
    Psum = prcp.sum(0)
    for case, arr in [("vic", obs), ("open_loop", runs["open_loop"]), ("closed_loop", runs["closed_loop"])]:
        res[f"ET_over_P_median_{case}"] = float(np.median(arr[:, :, 0].sum(0) / Psum))
        res[f"Q_over_P_median_{case}"] = float(np.median(arr[:, :, 2].sum(0) / Psum))
    # stability
    for k, var in enumerate(["ET", "SM", "Q"]):
        rc = np.sqrt(((runs["closed_loop"][:, :, k] - obs[:, :, k]) ** 2).mean(0))
        ro = np.sqrt(((runs["open_loop"][:, :, k] - obs[:, :, k]) ** 2).mean(0))
        res[f"pct_cells_closed_rmse_gt_2x_open_{var}"] = float(100 * np.mean(rc > 2 * ro))
    for case in ["closed_loop", "open_loop"]:
        zero = (runs[case][:, :, 2] == 0)
        res[f"pct_cells_Q_collapse_gt90pct_zero_{case}"] = float(100 * np.mean(zero.mean(0) > 0.9))
        res[f"mean_fraction_zero_Q_days_{case}"] = float(zero.mean())
    res["mean_fraction_zero_Q_days_vic"] = float((obs[:, :, 2] == 0).mean())
    json.dump(res, open(os.path.join(OUT, "drift_stability.json"), "w"), indent=2)
    return res


def figures(pc):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cb = np.load(os.path.join(OUT, "cumulative_bias.npz"))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for a, var in zip(ax, ["ET", "Q"]):
        for case, c in [("closed_loop", "#c0504d"), ("open_loop", "#3a6ea5")]:
            q = cb[f"{case}_{var}"]
            x = np.arange(q.shape[1])
            a.fill_between(x, q[0], q[2], color=c, alpha=0.2, lw=0)
            a.plot(x, q[1], color=c, lw=1.2, label=case.replace("_", " "))
        a.axhline(0, color="k", lw=0.6); a.axvline(270, color="grey", lw=0.6, ls="--")
        a.set_title(f"{var}: cumulative bias (median, IQR)", fontsize=8); a.set_xlabel("days since 2001-09-27"); a.set_ylabel("mm")
    ax[0].legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_b_cumulative_bias.png"), dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
    cases = ["open_loop", "closed_loop", "nar", "persistence"]
    for a, var in zip(ax, ["ET", "SM", "Q"]):
        data = [pc[(pc.variable == var) & (pc.case == c)].KGE.replace([np.inf, -np.inf], np.nan).dropna().clip(-1, 1) for c in cases]
        a.boxplot(data, showfliers=False, widths=0.6)
        a.set_xticks(range(1, 5)); a.set_xticklabels(["open", "closed", "NAR", "persist."], fontsize=7)
        a.set_title(f"{var}", fontsize=8); a.set_ylabel("KGE per cell (clipped at -1)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_c_kge_distributions.png"), dpi=300); plt.close(fig)


def main(which):
    data = load_test_data()
    ar_model, ar_info = build_model(AR_CKPT)
    log.info(f"AR model: {ar_info}")
    if which in ("step0", "all"):
        r0 = step0(data, ar_model)
        if not r0["passed"]:
            log.error("STEP 0 FAILED - stopping."); return
    if which in ("step1", "all"):
        nar_model, nar_info = build_model(NAR_CKPT)
        log.info(f"NAR model: {nar_info}")
        cells, obs, runs = step1(data, ar_model, nar_model)
        pc = per_cell_table(cells, obs, runs)
        pc.to_csv(os.path.join(OUT, "per_cell_metrics.csv"), index=False)
        summary_table(pc).to_csv(os.path.join(OUT, "summary.csv"), index=False)
        res = step3_4(data, cells, obs, runs)
        figures(pc)
        log.info(f"STEP 3/4: {res}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
