"""Figure 4 input: 1990-2006 climatology of the ERA5 forcing on the grid of the surrogate data (forcing.zarr), used for
the southern rows outside the VIC parameter grid. Writes climatology_1990_2006.npz.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..')))
from landformer_paths import DATA_DIR, ROOT  # noqa: E402
import os, numpy as np, xarray as xr, netCDF4 as nc
D = f"{DATA_DIR}"
f = xr.open_zarr(f"{D}/Surr_forcing/forcing.zarr").sel(time=slice("1990-01-01", "2006-12-31"))
P = (f.prcp.mean("time") * 365.25).compute().values
TX = f.tasmax.mean("time").compute().values
TN = f.tasmin.mean("time").compute().values
p = nc.Dataset(f"{D}/Surr_forcing/pram_150.nc")
mask = np.asarray(p["mask"][0], float); elev = np.asarray(p["elev"][0], float)
assert np.allclose(np.asarray(p["lat"][:]), f.lat.values) and np.allclose(np.asarray(p["lon"][:]), f.lon.values)
land = mask == 1
out = dict(lat=f.lat.values, lon=f.lon.values, prcp_raw=P, tmax_raw=TX, tmin_raw=TN, prcp=np.where(land, P, np.nan), tmax=np.where(land, TX, np.nan),
           tmin=np.where(land, TN, np.nan), elev=np.where(land, elev, np.nan), land=land)
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "climatology_1990_2006.npz"), **out)
print("land cells", int(land.sum()), "P range", np.nanmin(out["prcp"]), np.nanmax(out["prcp"]))
