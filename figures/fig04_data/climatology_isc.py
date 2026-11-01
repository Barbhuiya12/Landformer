"""Figure 4 input: 1990-2006 climatology of the ERA5 forcing over the VIC grid (0.0625 deg, 460 x 507 cells), masked to
the 104,841 cells of the model domain. Writes climatology_isc_1990_2006.npz.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..')))
from landformer_paths import EXT_DIR, ROOT  # noqa: E402
import os, numpy as np, xarray as xr, dask
dask.config.set(scheduler="threads", num_workers=8)
B = f"{EXT_DIR}"
f = xr.open_zarr(f"{B}/forcing/Final_forcing/forcings.zarr").sel(time=slice("1990-01-01", "2006-12-31"))
v = xr.open_dataset(f"{B}/Param/isc_params_with_all_param.nc")
assert np.allclose(f.lat.values, v.lat.values) and np.allclose(f.lon.values, v.lon.values)
m = v["mask"].values == 1
P = (f.prcp.mean("time") * 365.25).compute().values; TX = f.tasmax.mean("time").compute().values
TN = f.tasmin.mean("time").compute().values
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "climatology_isc_1990_2006.npz"),
         lat=f.lat.values, lon=f.lon.values, prcp=np.where(m, P, np.nan), tmax=np.where(m, TX, np.nan),
         tmin=np.where(m, TN, np.nan), prcp_raw=P, tmax_raw=TX, tmin_raw=TN, n_days=int(f.sizes["time"]))
print("domain cells", int(m.sum()), "days", f.sizes["time"], "P", np.nanpercentile(np.where(m, P, np.nan), [1, 50, 99]).round(0),
      "Tmax", np.nanpercentile(np.where(m, TX, np.nan), [1, 50, 99]).round(1))
