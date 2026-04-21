"""
Utility functions for dataset processing
"""
import numpy as np
import xarray as xr
from typing import Optional

def rolling_any_true(arr: np.ndarray, window_size: int, axis: int = 0) -> np.ndarray:
    """Helper function to check if any value in a rolling window is True"""
    if not arr.dtype == bool:
        raise ValueError("Input array for rolling_any_true must be boolean.")
    if arr.ndim == 1:
        arr_proc = arr[:, np.newaxis]
    else:
        arr_proc = arr
    pad_width = [(0, 0)] * arr_proc.ndim
    pad_width[axis] = (window_size - 1, 0)
    padded_arr = np.pad(arr_proc, pad_width, mode='constant', constant_values=False)
    result_shape = list(arr.shape)
    result = np.zeros(result_shape, dtype=bool)
    for i in range(arr.shape[axis]):
        current_slices = [slice(None)] * padded_arr.ndim
        current_slices[axis] = slice(i, i + window_size)
        window = padded_arr[tuple(current_slices)]
        if arr.ndim == 1:
            result[i] = window.any()
        else:
            result[tuple([i if j == axis else slice(None) for j in range(arr.ndim)])] = window.any(axis=axis)
    return result

def subset_xr(ds: xr.Dataset, lat_name: str, lon_name: str, 
              lat0: float, lat1: float, lon0: float, lon1: float, 
              step: Optional[int] = None) -> xr.Dataset:
    """Subset xarray dataset spatially"""
    lat_slice = slice(lat0, lat1, step) if step else slice(lat0, lat1)
    lon_slice = slice(lon0, lon1, step) if step else slice(lon0, lon1)
    return ds.sel({lat_name: lat_slice, lon_name: lon_slice})
