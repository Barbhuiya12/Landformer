"""Figure 4 input: common display grid for all panels (0.0625 deg, extended southwards to 7.9 N).
Fields: forcing climatology (climatology_isc.py, completed with climatology.py for the southern rows), GMTED2010
elevation and Koppen-Geiger classes (nearest pixel). The footprint is the VIC model domain together with the national
territory of India, Nepal, Bhutan and Bangladesh and the Pakistani territory adjoining the domain.
Writes fig4_grid.npz.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..')))
from landformer_paths import ROOT, VICFLOW_DIR  # noqa: E402
import os, glob, numpy as np, xarray as xr, rasterio, geopandas as gpd
from rasterio import features
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt

H = os.path.dirname(os.path.abspath(__file__))
B = f"{VICFLOW_DIR}"
ci = np.load(os.path.join(H, "climatology_isc_1990_2006.npz"))            # VIC grid, ascending lat
cs = np.load(os.path.join(H, "climatology_1990_2006.npz"))                # surrogate grid, ascending lat
vlat, vlon = ci["lat"], ci["lon"]
NADD = 16
lat = np.concatenate([vlat[0] - 0.0625 * np.arange(NADD, 0, -1), vlat]); lon = vlon
ny, nx = lat.size, lon.size
tr = from_origin(lon.min() - 0.03125, lat.max() + 0.03125, 0.0625, 0.0625)  # north-up transform of the display grid

# forcing climatology: VIC grid rows + surrogate-grid rows for the southern extension
def field(k):
    out = np.full((ny, nx), np.nan, np.float32)
    out[NADD:] = ci[k + "_raw"]
    ii = np.round((lat[:NADD] - cs["lat"][0]) / 0.0625).astype(int); jj = np.round((lon - cs["lon"][0]) / 0.0625).astype(int)
    ok = (jj >= 0) & (jj < cs["lon"].size)
    assert np.allclose(cs["lat"][ii], lat[:NADD]) and np.allclose(cs["lon"][jj[ok]], lon[ok])
    out[:NADD][:, ok] = cs[k + "_raw"][ii][:, jj[ok]]
    return out
P, TX, TN = field("prcp"), field("tmax"), field("tmin")
def nearest_fill(a):          # a few coastal/island cells lack forcing: take the nearest forcing cell (same footprint in all panels)
    idx = distance_transform_edt(~np.isfinite(a), return_distances=False, return_indices=True)
    return a[tuple(idx)]

# GMTED elevation
G = f"{B}/Data/1km_param/sources/gmted2010"
tiles = [rasterio.open(f) for f in sorted(glob.glob(f"{G}/*_gmted_mea300.tif"))
         if os.path.basename(f)[:7] in ("10S060E", "10N060E", "30N060E", "10S090E", "10N090E", "30N090E")]
mos, mtr = merge(tiles, bounds=(60.0, 6.0, 99.0, 39.0))
EL = np.full((ny, nx), np.nan, np.float32)
reproject(mos[0].astype(np.float32), EL, src_transform=mtr, src_crs=tiles[0].crs, dst_transform=tr, dst_crs="EPSG:4326",
          resampling=Resampling.average, src_nodata=-32768, dst_nodata=np.nan)
EL = EL[::-1]

# footprint: VIC domain + national boundaries
vm = np.zeros((ny, nx), bool)
vm[NADD:] = xr.open_dataset(f"{B}/Data_ext/Param/isc_params_with_all_param.nc")["mask"].values == 1
c = gpd.read_file(f"{B}/Data_ext/shape_file/Counry boundary/isc_country.shp").to_crs(4326)
# official boundaries of India, Nepal, Bhutan and Bangladesh are added to the model domain (completes Kashmir/Ladakh, the
# southern tip, the north-east and the coasts); Pakistan is shown exactly as the model domain (its natural western edge)
cm = features.rasterize([(g, 1) for g, n in zip(c.geometry, c.NAME) if n != "Pakistan"], out_shape=(ny, nx), transform=tr,
                        all_touched=True).astype(bool)[::-1]
# Pakistan: model domain plus Pakistani territory within ~0.3 deg (5 cells) of it, which closes the small gaps along
# the Sindh coast and the north-western border; far-western Balochistan (no forcing west of 66.16 E) stays unfilled
from scipy.ndimage import binary_dilation
pk = features.rasterize([(g, 1) for g, n in zip(c.geometry, c.NAME) if n == "Pakistan"], out_shape=(ny, nx), transform=tr,
                        all_touched=True).astype(bool)[::-1]
FP = vm | cm | (pk & binary_dilation(vm, iterations=1))   # Pakistan: model domain (1-cell edge smoothing only)

# Koppen
with rasterio.open(f"{B}/Data_ext/Climate_zones/ISC/isc_clipped_fixed.tif") as s:
    K = s.read(1).astype(float); kb = s.bounds
K[(K == 0)] = np.nan
K = K[tuple(distance_transform_edt(~np.isfinite(K), return_distances=False, return_indices=True))]
r = np.clip(((kb.top - lat) / (kb.top - kb.bottom) * K.shape[0]).astype(int), 0, K.shape[0] - 1)
cc = np.clip(((lon - kb.left) / (kb.right - kb.left) * K.shape[1]).astype(int), 0, K.shape[1] - 1)
KG = K[r][:, cc]

P, TX, TN = nearest_fill(P), nearest_fill(TX), nearest_fill(TN)
np.savez(os.path.join(H, "fig4_grid.npz"), lat=lat, lon=lon, footprint=FP, vic_domain=vm,
         prcp=np.where(FP, P, np.nan), tmax=np.where(FP, TX, np.nan), tmin=np.where(FP, TN, np.nan),
         elev=np.where(FP, EL, np.nan), koppen=np.where(FP, KG, np.nan))
for k, a in (("prcp", P), ("tmax", TX), ("tmin", TN), ("elev", EL), ("koppen", KG)):
    print(f"{k:7s} footprint cells without data: {int((FP & ~np.isfinite(a)).sum())}")
print("footprint cells", int(FP.sum()), "| southernmost footprint row", float(lat[FP.any(1)].min()))
