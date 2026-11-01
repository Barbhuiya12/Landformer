"""Growth of closed-loop errors with lead time (Section 4.4, Figure 9a-c).

500 test cells, stratified by Koppen-Geiger main class (Beck et al., 2023; 0.1 deg). Runs of 365 days start every
STEP days from 27 September 2001. Each run is initialised with VIC lagged inputs and then uses its own predictions.
References on the same cells and days: open loop and LandFormer-NAR (arrays from closed_loop.py step1) and
persistence of the initial state (the last VIC value before the start, held constant).

Writes RMSE per lead day, KGE per lead-time window and the lead time at which closed-loop RMSE first exceeds that
of LandFormer-NAR.

Usage (after closed_loop.py step1):  python error_growth.py [STEP_DAYS=30]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..')))
from landformer_paths import EXT_DIR, ROOT  # noqa: E402
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import xarray as xr
from closed_loop import (OUT, DEVICE, T0, TARGETS, SHORT, FORCING, PARAMS, load_test_data, load_scaler, renorm,
                         physical, build_model, free_run, metrics, AR_CKPT, log)

HORIZON = 365
N_CELLS = 500
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 30
GROUP = 10                       # starts per batch (10 x 500 = 5,000 members)
KG = f"{EXT_DIR}/Climate_zones/1991_2020/koppen_geiger_0p1.nc"
MAIN = {1: "A", 2: "A", 3: "A", 4: "B", 5: "B", 6: "B", 7: "B", **{c: "C" for c in range(8, 17)},
        **{c: "D" for c in range(17, 29)}, 29: "E", 30: "E"}


def pick_cells(data, cells):
    kg = xr.open_dataset(KG).kg_class
    lat = data["lat"][cells]; lon = data["lon"][cells]
    cls = kg.sel(lat=xr.DataArray(lat), lon=xr.DataArray(lon), method="nearest").values.astype(int)
    main = np.array([MAIN.get(c, "NA") for c in cls])
    rng = np.random.default_rng(0)
    counts = pd.Series(main).value_counts()
    quota = (counts / counts.sum() * N_CELLS).round().astype(int)
    quota.iloc[0] += N_CELLS - quota.sum()                         # make the total exactly N_CELLS
    pick = []
    for k, q in quota.items():
        idx = np.flatnonzero(main == k)
        pick += list(rng.choice(idx, size=min(q, len(idx)), replace=False))
    pick = np.sort(np.array(pick))
    pd.DataFrame({"cell": np.asarray(cells)[pick], "lat": lat[pick], "lon": lon[pick], "kg_class": cls[pick],
                  "kg_main": main[pick]}).to_csv(os.path.join(OUT, "error_growth_cells.csv"), index=False)
    return pick, counts.to_dict()


def main():
    t_all = time.time()
    info = json.load(open(os.path.join(OUT, "step1_info.json")))
    cells_all = info["cells"]
    data = load_test_data()
    T = data["F"].shape[0]
    pick, kg_counts = pick_cells(data, cells_all)
    cells = [cells_all[i] for i in pick]
    starts = list(range(T0, T - HORIZON + 1, STEP))
    log.info(f"Step 2: {len(cells)} cells, {len(starts)} starts every {STEP} days, horizon {HORIZON}")
    sI, sA = load_scaler("inference"), load_scaler("AR")
    Fz = torch.as_tensor(renorm(data["F"], FORCING, sI, sA), device=DEVICE)
    Pz = torch.as_tensor(renorm(data["P"], PARAMS, sI, sA), device=DEVICE)
    Yz = torch.as_tensor(renorm(data["Y"], TARGETS, sI, sA), device=DEVICE)
    model, _ = build_model(AR_CKPT)
    vic = np.maximum(physical(data["Y"], TARGETS, sI), 0)[:, cells, :]           # (T, C, 3)
    open_ = np.load(os.path.join(OUT, "pred_open_loop.npy"))[:, pick, :]          # (H1, C, 3), day index t - T0
    nar = np.load(os.path.join(OUT, "pred_nar.npy"))[:, pick, :]
    C = len(cells)
    se = {k: np.zeros((HORIZON, 3)) for k in ("closed", "open", "nar", "persistence")}
    n = np.zeros(HORIZON)
    windows = {"1-30": (0, 30), "31-90": (30, 90), "91-180": (90, 180), "181-365": (180, 365)}
    pooled = {(w, c): [[] for _ in range(C)] for w in windows for c in ("closed", "open", "obs")}
    gpu_s = 0.0
    for g0 in range(0, len(starts), GROUP):
        grp = starts[g0:g0 + GROUP]
        mem_cells = np.repeat(np.arange(C), 1)[None, :].repeat(len(grp), 0).ravel()    # cell position per member
        mem_start = np.repeat(grp, C)
        t1 = time.time()
        pred = free_run(model, Fz, Pz, Yz, [cells[i] for i in mem_cells], mem_start.tolist(), HORIZON, "model", sA)
        gpu_s += time.time() - t1
        for j, s in enumerate(grp):
            sl = slice(j * C, (j + 1) * C)
            days = s + np.arange(HORIZON)
            ob = vic[days]                                                         # (H, C, 3)
            cl = pred[:, sl, :]
            op = open_[days - T0]; na = nar[days - T0]
            pe = np.broadcast_to(vic[s - 1][None], ob.shape)
            for k_, arr in (("closed", cl), ("open", op), ("nar", na), ("persistence", pe)):
                se[k_] += ((arr - ob) ** 2).sum(1)
            n += C
            for w, (a, b) in windows.items():
                for c in range(C):
                    pooled[(w, "closed")][c].append(cl[a:b, c, :]); pooled[(w, "open")][c].append(op[a:b, c, :])
                    pooled[(w, "obs")][c].append(ob[a:b, c, :])
        log.info(f"  starts {grp[0]}..{grp[-1]} done, GPU {gpu_s:.0f} s")
    rows = []
    for k, v in enumerate(TARGETS):
        for h in range(HORIZON):
            rows.append({"lead_day": h + 1, "variable": SHORT[v],
                         **{f"rmse_{c}": np.sqrt(se[c][h, k] / n[h]) for c in ("closed", "open", "nar", "persistence")}})
    eg = pd.DataFrame(rows)
    eg["rmse_closed_over_open"] = eg.rmse_closed / eg.rmse_open
    eg.to_csv(os.path.join(OUT, "error_growth.csv"), index=False)
    cross = {}
    for var in ["ET", "SM", "Q"]:
        d = eg[eg.variable == var]
        hit = d[d.rmse_closed > d.rmse_nar]
        cross[var] = int(hit.lead_day.iloc[0]) if len(hit) else "never within 365 days"
    kw = []
    for w in windows:
        for k, v in enumerate(TARGETS):
            for case in ("closed", "open"):
                kges = [metrics(np.concatenate(pooled[(w, "obs")][c])[:, k], np.concatenate(pooled[(w, case)][c])[:, k])["KGE"]
                        for c in range(C)]
                kges = np.asarray(kges); kges = kges[np.isfinite(kges)]
                kw.append({"window_days": w, "variable": SHORT[v], "case": case, "median_KGE": float(np.median(kges)),
                           "q25": float(np.percentile(kges, 25)), "q75": float(np.percentile(kges, 75))})
    pd.DataFrame(kw).to_csv(os.path.join(OUT, "error_growth_kge_windows.csv"), index=False)
    info2 = {"cells": C, "kg_main_counts_all_test_cells": kg_counts, "starts": len(starts), "step_days": STEP,
             "first_start_day_index": starts[0], "last_start_day_index": starts[-1], "horizon": HORIZON,
             "nar_crossover_lead_day": cross, "gpu_seconds": gpu_s, "wall_seconds": time.time() - t_all,
             "persistence_definition": "y_hat(start + lead - 1) = VIC y(start - 1), the last value known at the start"}
    json.dump(info2, open(os.path.join(OUT, "error_growth_info.json"), "w"), indent=2, default=str)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for a, var in zip(ax, ["ET", "SM", "Q"]):
        d = eg[eg.variable == var]
        a.plot(d.lead_day, d.rmse_closed, color="#c0504d", lw=1.2, label="closed loop")
        a.plot(d.lead_day, d.rmse_open, color="#3a6ea5", lw=1.0, label="open loop")
        a.plot(d.lead_day, d.rmse_nar, color="#555555", lw=1.0, ls="--", label="NAR")
        a.plot(d.lead_day, d.rmse_persistence, color="#2f8a57", lw=1.0, ls=":", label="persistence from start")
        a.set_title(var, fontsize=8); a.set_xlabel("lead time (days)"); a.set_ylabel("RMSE")
    ax[0].legend(fontsize=6, frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_a_error_growth.png"), dpi=300); plt.close(fig)
    log.info(f"Step 2 done: {info2}")


if __name__ == "__main__":
    main()
