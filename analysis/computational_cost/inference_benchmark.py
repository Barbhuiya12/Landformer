"""Inference throughput and peak GPU memory of LandFormer (Section 5).

All test cells over 30 consecutive days, after 3 warm-up days, with 270-day input windows and batches of 2,500 cells,
in full single precision (FP32) and with TF32 arithmetic. Throughput is given in cell-days per second (one daily
prediction of evapotranspiration, total runoff and soil moisture for one cell).
"""
import os, sys, json, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "closed_loop"))
import closed_loop as C
d = C.load_test_data(); cells = C.published_cells()
F, P, Y = (torch.as_tensor(d[k], device=C.DEVICE) for k in "FPY")
out = {"gpu": torch.cuda.get_device_name(0), "n_cells": len(cells), "days_timed": 30, "batch": C.CHUNK}
for name, ck, lags in [("AR-All", C.AR_CKPT, True), ("NAR-109M", C.NAR_CKPT, False)]:
    m, _ = C.build_model(ck)
    out[name] = {"parameters_M": sum(p.numel() for p in m.parameters()) / 1e6}
    for mode in ("fp32", "tf32"):
        on = mode == "tf32"
        torch.backends.cuda.matmul.allow_tf32 = on; torch.backends.cudnn.allow_tf32 = on
        s = C.load_scaler("inference")
        C.free_run(m, F, P, Y, cells, [C.T0] * len(cells), 3, "vic", s, use_lags=lags)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t = time.time()
        C.free_run(m, F, P, Y, cells, [C.T0] * len(cells), 30, "vic", s, use_lags=lags)
        torch.cuda.synchronize(); dt = time.time() - t
        out[name][mode] = {"seconds": dt, "cell_days_per_s": 30 * len(cells) / dt,
                           "peak_gpu_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
    del m; torch.cuda.empty_cache()
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_inference.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
