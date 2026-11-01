"""Figure 4 input: GMTED2010 mean elevation (30 arc-seconds; Danielson & Gesch, 2011) averaged to the 0.0625 deg grid.
Writes elev_gmted_vicgrid.npz.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..')))
from landformer_paths import EXT_DIR, ROOT, VICFLOW_DIR  # noqa: E402
import os, glob, numpy as np, xarray as xr, rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
G = f"{VICFLOW_DIR}/Data/1km_param/sources/gmted2010"
tiles = [rasterio.open(f) for f in sorted(glob.glob(f"{G}/*_gmted_mea300.tif")) if os.path.basename(f)[:7] in ("10S060E", "10N060E", "30N060E", "10S090E", "10N090E", "30N090E")]
mos, tr = merge(tiles, bounds=(65.5, 7.5, 98.5, 38.5)); src_crs = tiles[0].crs
v = xr.open_dataset(f"{EXT_DIR}/Param/isc_params_with_all_param.nc")
lat, lon = v.lat.values, v.lon.values
dst = np.full((lat.size, lon.size), np.nan, np.float32)
dtr = from_origin(lon.min() - 0.03125, lat.max() + 0.03125, 0.0625, 0.0625)       # north-up
reproject(mos[0].astype(np.float32), dst, src_transform=tr, src_crs=src_crs, dst_transform=dtr, dst_crs="EPSG:4326",
          resampling=Resampling.average, src_nodata=-32768, dst_nodata=np.nan)
if lat[0] < lat[-1]:
    dst = dst[::-1]                                                                  # to the ascending latitude of the VIC grid
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "elev_gmted_vicgrid.npz"), lat=lat, lon=lon, elev=dst)
vm = v["mask"].values == 1; ve = v["elev"].values
print("GMTED vs VIC elevation inside domain: median |diff| m", float(np.nanmedian(np.abs(dst[vm] - ve[vm]))), "r", float(np.corrcoef(dst[vm & np.isfinite(dst)], ve[vm & np.isfinite(dst)])[0, 1]))
