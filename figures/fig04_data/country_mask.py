"""Figure 4 input: mask of India, Pakistan, Nepal, Bangladesh and Bhutan on the 0.0625 deg grid. Writes country_mask.npz.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..')))
from landformer_paths import EXT_DIR, ROOT  # noqa: E402
import os, numpy as np, geopandas as gpd
from rasterio import features
from rasterio.transform import from_origin
H = os.path.dirname(os.path.abspath(__file__))
g = np.load(os.path.join(H, "elev_gmted_vicgrid.npz")); lat, lon = g["lat"], g["lon"]
c = gpd.read_file(f"{EXT_DIR}/shape_file/Counry boundary/isc_country.shp").to_crs(4326)
tr = from_origin(lon.min() - 0.03125, lat.max() + 0.03125, 0.0625, 0.0625)
m = features.rasterize([(geom, 1) for geom in c.geometry], out_shape=(lat.size, lon.size), transform=tr, all_touched=True).astype(bool)
if lat[0] < lat[-1]:
    m = m[::-1]
np.savez(os.path.join(H, "country_mask.npz"), lat=lat, lon=lon, mask=m)
print("country cells", int(m.sum()), list(c.NAME))
