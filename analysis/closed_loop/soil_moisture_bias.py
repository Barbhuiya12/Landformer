"""Soil-moisture bias (model minus VIC, m3 m-3) over the continuous test-period run (Figure 9e).
Daily 25th, 50th and 75th percentiles across test cells for closed loop, open loop and LandFormer-NAR.
Uses the arrays written by closed_loop.py step1.
"""
import os, json, numpy as np
OUT = os.path.dirname(os.path.abspath(__file__))
o = np.load(os.path.join(OUT, "vic_obs.npy"), mmap_mode="r")[:, :, 1]
res, yr = {}, {}
for k in ("closed_loop", "open_loop", "nar"):
    b = np.load(os.path.join(OUT, f"pred_{k}.npy"), mmap_mode="r")[:, :, 1] - o
    res[k] = np.percentile(b, [25, 50, 75], axis=1)
    yr[k] = [float(np.median(b[i:i + 365].mean(0))) for i in range(0, b.shape[0], 365)]
np.savez(os.path.join(OUT, "sm_bias.npz"), **res)
json.dump({"median_per_cell_mean_bias_by_365d_block_m3m3": yr, "vic_median_mean_sm": float(np.median(o.mean(0))),
           "note": "blocks of 365 days from 2001-09-27; the last block is 97 days"}, open(os.path.join(OUT, "sm_bias_by_year.json"), "w"), indent=2)
print("ok")
