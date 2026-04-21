"""
Unified Dataset Module for VIC Surrogate Model
Author: Siddik
Description: Clean, memory-efficient dataset implementation with grid sampling for train/test splits

This module implements a specialized PyTorch Dataset for the VIC surrogate modeling task
that efficiently handles large-scale spatiotemporal data with multiple scenarios and grid cells.

Key features:
1. Memory-efficient data loading with chunked processing and caching
2. Multi-file target support for different hydrological variables
3. Grid-aware sampling with mask-based selection
4. Automatic normalization with scenario-consistent statistics
5. Lagged target feature creation for autoregressive modeling
6. NaN handling and sample filtering
7. Scenario tracking and metadata management

The dataset is designed to work with the grid splitting system to ensure reproducible
train/validation/test splits that maintain spatial generalization capabilities.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import pickle
import os
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any
import logging
from ..core.config import UnifiedConfig
from ..utils.memory import memory_efficient_function, ChunkedProcessor
from .utils import rolling_any_true, subset_xr


class UnifiedSurrogateDataset(Dataset):
    """
    Unified dataset implementation with memory optimization and grid sampling
    Author: Siddik
    
    This dataset class handles the complex data loading requirements for the VIC
    surrogate model, including:
    
    1. Multiple data sources:
       - Forcing variables (meteorological inputs)
       - Parameter variables (soil/vegetation parameters)
       - Target variables (hydrological outputs)
    
    2. Multi-scenario support:
       - Different climate scenarios or ensemble members
       - Configurable file ranges for train/valid/test
    
    3. Spatiotemporal processing:
       - Grid cell selection with masking
       - Sequence extraction for LSTM modeling
       - Lagged target feature creation
    
    4. Memory optimization:
       - Chunked data processing
       - Data caching with size limits
       - Dask-based parallel computation
    
    5. Data quality assurance:
       - NaN detection and filtering
       - Normalization with consistent statistics
       - Sample validation
    
    The dataset produces input tensors with the structure:
    [exogenous_features, parameter_features, lagged_target_features]
    
    And target tensors with:
    [target_variables]
    
    Attributes:
        config (UnifiedConfig): Configuration object
        mode (str): Dataset mode ('train', 'valid', 'test')
        seq_length (int): Sequence length for LSTM inputs
        lags (List[int]): Lag values for autoregressive features
        nx (int): Total input feature dimension
        ny (int): Target variable dimension
        ngrid (int): Number of selected grid cells
        nt (int): Number of time steps
        lookup_table (Dict): Mapping from sample index to (file, grid, time)
    """
    
    def __init__(self, 
                 config: UnifiedConfig,
                 mode: str = 'train',  # 'train', 'valid', 'test'
                 grid_indices: Optional[np.ndarray] = None):
        """
        Initialize dataset
        
        Args:
            config: UnifiedConfig object
            mode: Dataset mode ('train', 'valid', 'test')
            grid_indices: Optional pre-selected grid indices for reproducible splits
        """
        super().__init__()
        
        self.config = config
        self.mode = mode # Training /Testing /Validation
        
        # Setup logging FIRST
        self.logger = logging.getLogger(__name__)
        
        # Set normalization method from config (default to zscore)
        self.normalization_method = getattr(config.data.processing, 'normalization', 'zscore')
        if self.normalization_method not in ['zscore', 'minmax']:
            self.logger.warning(f"Unknown normalization method: {self.normalization_method}. Using zscore.")
            self.normalization_method = 'zscore'
            
        self.seq_length = config.model.sequence_length
        
        # NEW: Flexible lag configuration system
        self.lag_config = config.get_lag_configuration()
        self.enable_lags = config.model.enable_lags
        
        # Legacy compatibility
        self.lags = config.model.lags or []
        
        self.is_train = (mode == 'train')
        
        # Initialize statistics dictionary
        self.stat = {}
        
        # Initialize data cache
        self._data_cache = {}
        self._cache_size_limit = self.config.data.cache_size_limit
        
        # Get mode-specific configuration
        self._setup_mode_config()
        
        # Load datasets
        self._load_datasets()
        
        # Setup grid selection
        self._setup_grid_selection(grid_indices)
        
        # Load data after grid selection if using memory-efficient loader
        if hasattr(self, 'use_memory_efficient') and self.use_memory_efficient:
            self._load_data_memory_efficient()
        
        # Process data
        self._process_data()
        
        # Create lookup table
        self._create_lookup_table()
        
        self.logger.info(f"Dataset initialized: {mode} mode with {len(self)} samples")
    # mode of training and testing function is done here 
    def _setup_mode_config(self):
        """Setup configuration based on mode"""
        if self.mode == 'train':
            self.file_range = self.config.data.file_indices['train_files']
            self.time_range = self.config.data.temporal['train_range']
        elif self.mode == 'valid':
            self.file_range = self.config.data.file_indices['valid_files']
            self.time_range = self.config.data.temporal['valid_range']
        elif self.mode == 'test':
            self.file_range = self.config.data.file_indices['test_files']
            self.time_range = self.config.data.temporal['test_range']
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
    
    def _load_datasets(self):
        """Load and subset datasets - supports multiple target files"""
        self.logger.info(f"Loading datasets for {self.mode} mode...")
        
        # Load datasets with configurable chunking
        optimizations = getattr(self.config.training, 'optimizations', {}) or {}
        use_optimal_chunking = optimizations.get('optimal_chunking', True)
        
        # Load forcing and parameters (same as before)
        if use_optimal_chunking:
            self.logger.info("Loading forcing with optimized chunking...")
            self.df_xr = xr.open_zarr(self.config.data.paths['forcing'], 
                                      chunks={'time': 1000, 'lat': 50, 'lon': 50})
        else:
            self.df_xr = xr.open_zarr(self.config.data.paths['forcing'])
            
        self.dp_xr = xr.open_dataset(self.config.data.paths['parameters'])
        
        # Dynamic target loading - support multiple files
        self._load_target_datasets(use_optimal_chunking)
        
        # Spatial subsetting
        spatial_config = self.config.data.spatial
        lat0, lat1 = spatial_config['lat_min'], spatial_config['lat_max']
        lon0, lon1 = spatial_config['lon_min'], spatial_config['lon_max']
        step = spatial_config['step']
        
        self.df_xr = subset_xr(self.df_xr, 'lat', 'lon', lat0, lat1, lon0, lon1, step)
        self.dp_xr = subset_xr(self.dp_xr, 'lat', 'lon', lat0, lat1, lon0, lon1, step)
        
        # Subset target datasets
        self._subset_target_datasets(lat0, lat1, lon0, lon1, step)
        
        # Variable lists
        self.fvarLst = self.config.data.variables['forcing']
        self.pvarLst = self.config.data.variables['parameters']
        self.ovarLst = self.config.data.variables['target']
        
        self.logger.info(f"Loaded datasets with target variables: {self.ovarLst}")

    def _load_target_datasets(self, use_optimal_chunking: bool):
        """Load target datasets from multiple files or single file"""
        # Check if multiple target files are specified
        target_files = self.config.data.paths.get('target_files', {})
        
        if target_files:
            # Multi-file mode
            self.logger.info("Loading target data from multiple zarr files...")
            self.target_datasets = {}
            self.multi_file_mode = True
            
            for var_name in self.config.data.variables['target']:
                if var_name not in target_files:
                    raise ValueError(f"Target variable '{var_name}' not found in target_files config")
                
                file_path = target_files[var_name]
                self.logger.info(f"Loading {var_name} from {file_path}")
                
                if use_optimal_chunking:
                    ds = xr.open_zarr(file_path, chunks={'time': 1000, 'latitude': 50, 'longitude': 50})
                else:
                    ds = xr.open_zarr(file_path)
                
                self.target_datasets[var_name] = ds
                
        else:
            # Single file mode (legacy)
            self.logger.info("Loading target data from single zarr file...")
            self.multi_file_mode = False
            
            if use_optimal_chunking:
                self.do_xr = xr.open_zarr(self.config.data.paths['target'],
                                          chunks={'time': 1000, 'latitude': 50, 'longitude': 50})
            else:
                self.do_xr = xr.open_zarr(self.config.data.paths['target'])

    def _subset_target_datasets(self, lat0: float, lat1: float, lon0: float, lon1: float, step: Optional[int]):
        """Subset target datasets spatially"""
        if self.multi_file_mode:
            # Subset each target dataset individually
            for var_name, ds in self.target_datasets.items():
                # Determine coordinate names (try both common conventions)
                lat_dim = 'latitude' if 'latitude' in ds.dims else 'lat'
                lon_dim = 'longitude' if 'longitude' in ds.dims else 'lon'
                
                # Subset each dataset
                self.target_datasets[var_name] = subset_xr(
                    ds, lat_dim, lon_dim, lat1, lat0, lon0, lon1, step
                )
                self.logger.info(f"Subsetted {var_name} dataset to spatial domain")
        else:
            # Single file mode
            self.do_xr = subset_xr(self.do_xr, 'latitude', 'longitude', lat1, lat0, lon0, lon1, step)
    
    def _load_data_memory_efficient(self):
        """Load data using memory-efficient loader after grid selection"""
        if hasattr(self, 'memory_loader') and hasattr(self, 'selected_grids'):
            self.logger.info("Loading data with memory-efficient loader...")
            
            # Use the selected grid indices
            data = self.memory_loader.create_pytorch_dataset(self.selected_grids)
            
            # Store the loaded data
            if 'forcing' in data:
                self.forcing_data = data['forcing']
                self.logger.info(f"Loaded forcing data: {self.forcing_data.shape}")
            
            if 'parameters' in data:
                self.param_data = data['parameters']
                self.logger.info(f"Loaded parameter data: {self.param_data.shape}")
            
            if 'targets' in data:
                self.target_data = data['targets']
                self.logger.info(f"Loaded target data: {self.target_data.shape}")
    
    def _setup_grid_selection(self, grid_indices: Optional[np.ndarray] = None):
        """Setup grid selection with mask-based sampling"""
        self.lats_full_dp = self.dp_xr['lat'].values
        self.lons_full_dp = self.dp_xr['lon'].values
        self.full_grid_shape_dp = (len(self.lats_full_dp), len(self.lons_full_dp))
        
        if 'mask' in self.dp_xr:
            mask_da = self.dp_xr['mask']
            mask_data_raw = mask_da.data
            
            # Handle different mask dimensionalities
            if mask_data_raw.ndim == 3 and mask_data_raw.shape[0] > 1:
                mask_2d = mask_data_raw[0, :, :]
            elif mask_data_raw.ndim == 2:
                mask_2d = mask_data_raw
            elif 'nfiles' in mask_da.dims:
                mask_2d = mask_da.isel(nfiles=1).data
            else:
                raise ValueError(f"Cannot reconcile mask shape {mask_data_raw.shape}")
            
            valid_indices_flat = np.where(mask_2d.flatten() == 1)[0]
            
            # Use provided grid indices (from grid splitter)
            if grid_indices is not None:
                # Use pre-selected grid indices from YAML-controlled splitting
                self.selected_flat_indices_from_dp = grid_indices
                self.ngrid = len(grid_indices)
                self.logger.info(f"Using {len(grid_indices)} pre-selected grids from YAML config")
            else:
                # Fallback: use all valid grids (not recommended)
                self.logger.warning("No pre-selected grids provided. Using all valid grids.")
                self.logger.warning("Consider using grid_splitter for better control.")
                self.selected_flat_indices_from_dp = valid_indices_flat
                self.ngrid = len(valid_indices_flat)
            
            # Create coordinate arrays
            i_indices, j_indices = np.unravel_index(self.selected_flat_indices_from_dp, self.full_grid_shape_dp)
            self.selected_lats_coords = xr.DataArray(self.lats_full_dp[i_indices], dims="points")
            self.selected_lons_coords = xr.DataArray(self.lons_full_dp[j_indices], dims="points")
            
            # Create visualization if requested
            if self.config.output.create_plots:
                self._plot_selected_grids(mask_2d)
        else:
            # No mask available - use all grid points
            self.selected_lats_coords = xr.DataArray(self.lats_full_dp, dims="points")
            self.selected_lons_coords = xr.DataArray(self.lons_full_dp, dims="points")
            self.ngrid = len(self.lats_full_dp) * len(self.lons_full_dp)
        
        # Save grid information for this dataset
        self._save_grid_information()
        
        self.logger.info(f"Selected {self.ngrid} grid cells for {self.mode} mode")
    
    def _plot_selected_grids(self, mask_2d: np.ndarray):
        """Create visualization of selected grid points"""
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot mask
            plt.subplot(1, 2, 1)
            plt.imshow(mask_2d, cmap='viridis',
                      extent=[self.lons_full_dp.min(), self.lons_full_dp.max(),
                             self.lats_full_dp.min(), self.lats_full_dp.max()],
                      origin='lower')
            plt.title(f'Mask ({self.mode} mode)')
            plt.colorbar(label='Mask Value')
            
            # Plot selected points
            plt.subplot(1, 2, 2)
            display_selection = np.zeros_like(mask_2d, dtype=float)
            if hasattr(self, 'selected_flat_indices_from_dp'):
                i_coords, j_coords = np.unravel_index(self.selected_flat_indices_from_dp, mask_2d.shape)
                display_selection[i_coords, j_coords] = 1
            
            plt.imshow(display_selection, cmap='Reds', alpha=0.7,
                      extent=[self.lons_full_dp.min(), self.lons_full_dp.max(),
                             self.lats_full_dp.min(), self.lats_full_dp.max()],
                      origin='lower')
            plt.title(f'Selected {self.ngrid} Grid Points ({self.mode})')
            plt.tight_layout()
            
            save_path = os.path.join(self.config.figure_dir, f'selected_grids_{self.mode}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved grid visualization to {save_path}")
        except Exception as e:
            self.logger.warning(f"Could not create grid visualization: {e}")
    
    def _save_grid_information(self):
        """Save grid location information for easy reference"""
        try:
            # Create grid information dictionary
            grid_info = self._create_grid_info_dict()
            
            # Save as CSV
            self._save_grid_csv(grid_info)
            
            # Save as NetCDF
            self._save_grid_netcdf(grid_info)
            
            # Save as pickle for fast loading
            self._save_grid_pickle(grid_info)
            
            self.logger.info(f"Grid information saved for {self.mode} mode ({self.ngrid} grids)")
            
        except Exception as e:
            self.logger.warning(f"Could not save grid information: {e}")
    
    def _create_grid_info_dict(self) -> Dict[str, Any]:
        """Create comprehensive grid information dictionary"""
        # Get lat/lon coordinates
        if hasattr(self, 'selected_flat_indices_from_dp'):
            # Convert flat indices back to 2D indices
            i_indices, j_indices = np.unravel_index(
                self.selected_flat_indices_from_dp, 
                self.full_grid_shape_dp
            )
            
            lats = self.lats_full_dp[i_indices]
            lons = self.lons_full_dp[j_indices]
            flat_indices = self.selected_flat_indices_from_dp
        else:
            # All grid points case
            lats = self.selected_lats_coords.values
            lons = self.selected_lons_coords.values
            
            # Create flat indices
            lat_indices, lon_indices = np.meshgrid(
                np.arange(len(self.lats_full_dp)),
                np.arange(len(self.lons_full_dp)),
                indexing='ij'
            )
            flat_indices = np.ravel_multi_index(
                (lat_indices.flatten(), lon_indices.flatten()),
                self.full_grid_shape_dp
            )
            i_indices = lat_indices.flatten()
            j_indices = lon_indices.flatten()
        
        # Create comprehensive grid information
        grid_info = {
            'mode': self.mode,
            'ngrid': self.ngrid,
            'grid_data': []
        }
        
        for idx in range(self.ngrid):
            grid_data = {
                'dataset_grid_index': idx,              # Index within this dataset (0 to ngrid-1)
                'global_flat_index': int(flat_indices[idx]),  # Flat index in full spatial array
                'i_index': int(i_indices[idx]),         # Row index in 2D spatial array
                'j_index': int(j_indices[idx]),         # Column index in 2D spatial array
                'latitude': float(lats[idx]),           # Latitude coordinate
                'longitude': float(lons[idx]),          # Longitude coordinate
                'file_range': self.file_range,          # Scenario file range
                'time_range': self.time_range,          # Time range
                'author': 'Siddik'
            }
            grid_info['grid_data'].append(grid_data)
        
        # Add metadata
        grid_info['metadata'] = {
            'created_by': 'Siddik',
            'dataset_mode': self.mode,
            'total_grids': self.ngrid,
            'spatial_bounds': {
                'lat_min': float(lats.min()),
                'lat_max': float(lats.max()),
                'lon_min': float(lons.min()),
                'lon_max': float(lons.max())
            },
            'file_range': self.file_range,
            'time_range': self.time_range,
            'full_spatial_shape': self.full_grid_shape_dp,
            'config_file': 'config.yaml'
        }
        
        return grid_info
    
    def _save_grid_csv(self, grid_info: Dict[str, Any]):
        """Save grid information as CSV file"""
        import pandas as pd
        
        # Convert to DataFrame
        df_data = []
        for grid in grid_info['grid_data']:
            row = {
                'dataset_mode': grid_info['mode'],
                'dataset_grid_index': grid['dataset_grid_index'],
                'global_flat_index': grid['global_flat_index'],
                'i_index': grid['i_index'],
                'j_index': grid['j_index'],
                'latitude': grid['latitude'],
                'longitude': grid['longitude'],
                'scenario_start': grid['file_range'][0],
                'scenario_end': grid['file_range'][1],
                'time_start': grid['time_range'][0],
                'time_end': grid['time_range'][1]
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save CSV
        csv_path = os.path.join(self.config.out_dir, f'grid_locations_{self.mode}.csv')
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Grid locations saved to CSV: {csv_path}")
    
    def _save_grid_netcdf(self, grid_info: Dict[str, Any]):
        """Save grid information as NetCDF file"""
        # Extract data
        grid_data = grid_info['grid_data']
        n_grids = len(grid_data)
        
        # Create coordinate arrays
        dataset_indices = [g['dataset_grid_index'] for g in grid_data]
        flat_indices = [g['global_flat_index'] for g in grid_data]
        i_indices = [g['i_index'] for g in grid_data]
        j_indices = [g['j_index'] for g in grid_data]
        lats = [g['latitude'] for g in grid_data]
        lons = [g['longitude'] for g in grid_data]
        
        # Create xarray Dataset
        ds = xr.Dataset({
            'global_flat_index': (['grid'], flat_indices),
            'i_index': (['grid'], i_indices), 
            'j_index': (['grid'], j_indices),
            'latitude': (['grid'], lats),
            'longitude': (['grid'], lons)
        }, coords={
            'grid': (['grid'], dataset_indices)
        })
        
        # Add attributes
        ds.attrs.update({
            'title': f'Grid Locations for {self.mode.title()} Dataset',
            'author': 'Siddik',
            'description': f'Spatial locations of {n_grids} grid cells used for {self.mode} mode',
            'mode': self.mode,
            'total_grids': n_grids,
            'scenario_range': f"{self.file_range[0]}-{self.file_range[1]}",
            'time_range': f"{self.time_range[0]} to {self.time_range[1]}",
            'spatial_bounds': f"lat({min(lats):.2f},{max(lats):.2f}), lon({min(lons):.2f},{max(lons):.2f})",
            'full_spatial_shape': str(self.full_grid_shape_dp),
            'created_with': 'Unified VIC Surrogate Model by Siddik'
        })
        
        # Add variable attributes
        ds['latitude'].attrs = {'units': 'degrees_north', 'long_name': 'Latitude'}
        ds['longitude'].attrs = {'units': 'degrees_east', 'long_name': 'Longitude'}
        ds['global_flat_index'].attrs = {'long_name': 'Flat index in full spatial array'}
        ds['i_index'].attrs = {'long_name': 'Row index in 2D spatial array'}
        ds['j_index'].attrs = {'long_name': 'Column index in 2D spatial array'}
        
        # Save NetCDF
        nc_path = os.path.join(self.config.out_dir, f'grid_locations_{self.mode}.nc')
        ds.to_netcdf(nc_path)
        self.logger.info(f"Grid locations saved to NetCDF: {nc_path}")
    
    def _save_grid_pickle(self, grid_info: Dict[str, Any]):
        """Save grid information as pickle file for fast loading"""
        import pickle
        
        pickle_path = os.path.join(self.config.out_dir, f'grid_locations_{self.mode}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(grid_info, f)
        self.logger.info(f"Grid locations saved to pickle: {pickle_path}")
    
    def __len__(self) -> int:
        """Return dataset length"""
        return getattr(self, '_dataset_length', 0)
    
    def _manage_cache_size(self):
        """Manage cache size to prevent memory issues"""
        if len(self._data_cache) > self._cache_size_limit:
            # Remove oldest cached items
            keys_to_remove = list(self._data_cache.keys())[:len(self._data_cache) - self._cache_size_limit]
            for key in keys_to_remove:
                del self._data_cache[key]
            self.logger.debug(f"Cache cleaned, removed {len(keys_to_remove)} items")
    
    def __del__(self):
        """Cleanup resources when dataset is destroyed"""
        # Close Dask client if it exists
        if hasattr(self, '_dask_client'):
            try:
                self._dask_client.close()
                self.logger.info("Dask client closed successfully")
            except Exception as e:
                self.logger.debug(f"Error closing Dask client: {e}")
            
            # Close cluster if we created it
            try:
                if hasattr(self._dask_client, 'cluster') and self._dask_client.cluster:
                    self._dask_client.cluster.close()
                    self.logger.info("Dask cluster closed successfully")
            except Exception as e:
                self.logger.debug(f"Error closing Dask cluster: {e}")
    
    def _process_data(self):
        """Process and normalize all data"""
        # Load or initialize statistics
        scaler_file = os.path.join(self.config.out_dir, "train_data_scaler.bin")
        if self.is_train:
            self.stat = {}
        else:
            if os.path.exists(scaler_file):
                with open(scaler_file, 'rb') as fp:
                    self.stat = pickle.load(fp)
                self.logger.info(f"Loaded normalization statistics from {scaler_file}")
            else:
                raise FileNotFoundError(f"Scaler file not found at {scaler_file}. Please run training first to generate it.")
        
        # Calculate dimensions
        self.nfile_config = self.file_range[1] - self.file_range[0] + 1
        
        # NEW: Calculate lag features from flexible configuration
        self.n_lag_features = self._calculate_lag_features()
        
        # nx represents total input features: exogenous (forcing + parameters) + all lag features
        self.nx = len(self.fvarLst) + len(self.pvarLst) + self.n_lag_features
        # n_exo represents only exogenous features (forcing + parameters) for model initialization
        self.n_exo = len(self.fvarLst) + len(self.pvarLst)
        self.ny = len(self.ovarLst)
        
        # Get time dimension
        time_slice = slice(self.time_range[0], self.time_range[1])
        try:
            self.nt = len(self.do_xr.time.sel(time=time_slice))
        except Exception:
            self.nt = len(self.df_xr.time.sel(time=time_slice))
        
        # Process datasets
        self.logger.info("Processing parameter data...")
        self.pars = self._process_data_generic(
            self.dp_xr, self.pvarLst, 'lat', 'lon',
            is_param=True, use_flat_indices=self.selected_flat_indices_from_dp
        )
        
        self.logger.info("Processing forcing data...")
        self.forcings = self._process_data_generic(
            self.df_xr, self.fvarLst, 'lat', 'lon',
            is_forcing=True, sel_lats=self.selected_lats_coords,
            sel_lons=self.selected_lons_coords
        )
        
        self.logger.info("Processing target data...")
        if self.multi_file_mode:
            self.target = self._process_multi_target_data()
        else:
            self.target = self._process_data_generic(
                self.do_xr, self.ovarLst, 'latitude', 'longitude',
                sel_lats=self.selected_lats_coords,
                sel_lons=self.selected_lons_coords
            )
        
        # Save normalization statistics if training
        if self.is_train:
            scaler_file = os.path.join(self.config.out_dir, "train_data_scaler.bin")
            os.makedirs(os.path.dirname(scaler_file), exist_ok=True)
            with open(scaler_file, 'wb') as fp:
                pickle.dump(self.stat, fp)
            self.logger.info(f"Saved normalization statistics to {scaler_file}")

    def _process_multi_target_data(self) -> np.ndarray:
        """Process target data from multiple zarr files"""
        self.logger.info("Processing multiple target files...")
        
        all_target_data = []
        
        # Process each target variable from its own file
        for var_name in self.ovarLst:
            if var_name not in self.target_datasets:
                raise ValueError(f"Target variable '{var_name}' not found in loaded datasets")
            
            ds = self.target_datasets[var_name]
            
            # Determine coordinate names
            lat_dim = 'latitude' if 'latitude' in ds.dims else 'lat'
            lon_dim = 'longitude' if 'longitude' in ds.dims else 'lon'
            
            self.logger.info(f"Processing target variable: {var_name}")
            
            # Process this variable using existing generic method
            var_data = self._process_data_generic(
                ds, [var_name], lat_dim, lon_dim,
                sel_lats=self.selected_lats_coords,
                sel_lons=self.selected_lons_coords
            )
            
            all_target_data.append(var_data)
        
        # Concatenate all target variables along the last dimension
        # Shape: (nfiles, nt, ngrid, n_target_vars)
        combined_target = np.concatenate(all_target_data, axis=-1)
        
        self.logger.info(f"Combined target data shape: {combined_target.shape}")
        self.logger.info(f"Target variables processed: {self.ovarLst}")
        
        return combined_target
    
    @memory_efficient_function(cleanup_after=True)
    def _process_data_generic(self, ds_xr: xr.Dataset, var_list: List[str],
                            lat_dim: str, lon_dim: str,
                            is_param: bool = False, is_forcing: bool = False,
                            use_flat_indices: Optional[np.ndarray] = None,
                            sel_lats: Optional[xr.DataArray] = None,
                            sel_lons: Optional[xr.DataArray] = None) -> np.ndarray:
        """Generic data processing with memory optimization and chunked processing"""
        
        # Check if we have cached results
        cache_key = f"{ds_xr.encoding.get('source', 'unknown')}_{var_list}_{is_param}_{is_forcing}"
        if hasattr(self, '_data_cache') and cache_key in self._data_cache:
            self.logger.info(f"Using cached data for {cache_key}")
            return self._data_cache[cache_key]
        
        time_slice = slice(self.time_range[0], self.time_range[1])
        ds_has_nfiles_dim = 'nfiles' in ds_xr.dims
        
        # Get file slice
        if ds_has_nfiles_dim:
            start_idx = self.file_range[0]
            end_idx = min(self.file_range[1], ds_xr.sizes['nfiles'] - 1)
            file_slice = slice(start_idx, end_idx + 1)
            nfile_eff = end_idx - start_idx + 1
        else:
            file_slice = None
            nfile_eff = self.nfile_config
        
        current_nt = self.nt if not is_param else 1
        all_var_data = []
        
        # Process variables in chunks to reduce memory pressure
        chunk_size = max(1, len(var_list) // 4)  # Process in 4 chunks
        for i in range(0, len(var_list), chunk_size):
            var_chunk = var_list[i:i+chunk_size]
            self.logger.info(f"Processing variable chunk {i//chunk_size + 1}/{(len(var_list)-1)//chunk_size + 1}")
            
            for var_name_orig in var_chunk:
                # Handle layer variables (e.g., 'd1', 'Ksat2')
                actual_var_name, layer_idx = var_name_orig, None
                if var_name_orig[-1].isdigit() and len(var_name_orig) > 1:
                    potential_base = var_name_orig[:-1]
                    layer_num_str = var_name_orig[-1]
                    map_dict = {'d': 'depth', 'expt': 'expt', 'Ksat': 'Ksat'}
                    if potential_base in map_dict:
                        actual_var_name = map_dict[potential_base]
                        layer_idx = int(layer_num_str) - 1
                
                if actual_var_name not in ds_xr:
                    raise KeyError(f"Variable '{actual_var_name}' not found")
                
                # Extract variable
                var_da = ds_xr[actual_var_name]
                if not is_param:
                    var_da = var_da.sel(time=time_slice)
                if ds_has_nfiles_dim and file_slice:
                    var_da = var_da.isel(nfiles=file_slice)
                if layer_idx is not None:
                    var_da = var_da.isel(nlayer=layer_idx)
                
                # Spatial selection
                if use_flat_indices is not None:
                    var_da_selected = var_da.stack(points=(lat_dim, lon_dim)).isel(points=use_flat_indices)
                elif sel_lats is not None and sel_lons is not None:
                    sel_dict = {lat_dim: sel_lats, lon_dim: sel_lons}
                    var_da_selected = var_da.sel(**sel_dict, method="nearest", tolerance=5)
                    if 'points' not in var_da_selected.dims:
                        if lat_dim in var_da_selected.dims and lon_dim in var_da_selected.dims:
                            var_da_selected = var_da_selected.stack(points=(lat_dim, lon_dim))
                else:
                    raise ValueError("Spatial selection method not specified")
                
                # For large arrays, process in smaller time chunks
                var_values = var_da_selected.data
                if hasattr(var_values, 'nbytes') and var_values.nbytes > 200 * 1024 * 1024:  # 200MB threshold
                    self.logger.info(f"Very large array detected ({var_values.nbytes / 1024**2:.1f}MB), using chunked processing...")
                    # Process in time chunks to reduce memory pressure
                    if hasattr(var_values, 'chunks') and len(var_values.chunks) > 1:
                        time_chunks = var_values.chunks[1] if len(var_values.chunks) > 1 else [var_values.shape[1]]
                        self.logger.info(f"Processing in {len(time_chunks)} time chunks")
                
                # Compute if needed with optimized processing
                if hasattr(var_values, 'compute'):
                    # For dask arrays, optimize computation
                    if hasattr(var_values, 'nbytes') and var_values.nbytes > 100 * 1024 * 1024:  # 100MB threshold
                        self.logger.info(f"Large array detected ({var_values.nbytes / 1024**2:.1f}MB), using optimized computation...")
                        # Use optimized dask computation with configurable threads
                        optimizations = getattr(self.config.training, 'optimizations', {}) or {}
                        use_dask = optimizations.get('use_dask_threads', False)  # Respect config setting
                        
                        if use_dask:
                            import dask
                            from dask.distributed import Client, LocalCluster
                            
                            # Only create cluster once and reuse it
                            if not hasattr(self, '_dask_client'):
                                try:
                                    # Try to connect to existing dask cluster
                                    self._dask_client = Client(timeout='2s')
                                    self.logger.info(f"Connected to existing Dask cluster: {self._dask_client.cluster}")
                                except:
                                    # Create local cluster if none exists with configurable params
                                    dask_workers = optimizations.get('dask_num_workers', 4)  # Increased
                                    threads_per_worker = optimizations.get('dask_threads_per_worker', 2)  # Increased
                                    memory_limit = optimizations.get('dask_memory_limit', '2GB')  # Increased default
                                    
                                    self.logger.info(f"Starting Dask LocalCluster: {dask_workers} workers, {threads_per_worker} threads/worker, {memory_limit} memory")
                                    cluster = LocalCluster(
                                        n_workers=dask_workers,
                                        threads_per_worker=threads_per_worker,
                                        memory_limit=memory_limit,
                                        processes=False,  # Use threads for less overhead
                                        silence_logs=True,  # Reduce logging
                                        dashboard_address=None  # Disable dashboard for speed
                                    )
                                    self._dask_client = Client(cluster)
                                    self.logger.info("Dask cluster created successfully")
                            
                            var_values = var_values.compute(scheduler=self._dask_client)
                        else:
                            var_values = var_values.compute()
                    else:
                        var_values = var_values.compute()
                
                # Reshape to expected dimensions
                expected_shape = []
                if ds_has_nfiles_dim:
                    expected_shape.append(nfile_eff)
                elif is_forcing:
                    expected_shape.append(1)
                elif is_param:
                    expected_shape.append(nfile_eff)
                
                if not is_param:
                    expected_shape.append(current_nt)
                expected_shape.append(self.ngrid)
                
                # Handle reshaping for different cases
                if is_param and not ds_has_nfiles_dim and var_values.ndim == 1:
                    var_values = np.tile(var_values, (nfile_eff, 1))
                elif is_forcing and not ds_has_nfiles_dim and var_values.ndim == 2:
                    var_values = var_values[np.newaxis, ...]
                
                var_values_reshaped = var_values.reshape(tuple(expected_shape))
                
                # Efficient normalization on full dataset
                stat_mean_key = f"{var_name_orig}_mean"
                stat_std_key = f"{var_name_orig}_std"
                
                if self.is_train:
                    # Compute normalization statistics based on method
                    self.logger.debug(f"Computing {self.normalization_method} normalization stats for {var_name_orig}...")
                    
                    # Convert to float32 contiguous array for faster operations
                    var_flat = np.ascontiguousarray(var_values_reshaped, dtype=np.float32)
                    
                    if self.normalization_method == 'zscore':
                        # Z-score normalization (mean=0, std=1) - BEST for soil moisture in mm!
                        mean = np.nanmean(var_flat).astype(np.float32)
                        std = np.nanstd(var_flat).astype(np.float32)
                        
                        if std < 1e-6:
                            std = 1.0
                        self.stat[stat_mean_key] = mean
                        self.stat[stat_std_key] = std
                        self.logger.debug(f"Z-score stats for {var_name_orig}: mean={mean:.4f}, std={std:.4f}")
                    
                    elif self.normalization_method == 'minmax':
                        # Min-max normalization to [0, 1]
                        min_val = np.nanmin(var_flat).astype(np.float32)
                        max_val = np.nanmax(var_flat).astype(np.float32)
                        range_val = max_val - min_val
                        
                        if range_val < 1e-6:
                            range_val = 1.0
                        self.stat[stat_mean_key] = min_val  # Store min as "mean" 
                        self.stat[stat_std_key] = range_val  # Store range as "std"
                        self.logger.debug(f"MinMax stats for {var_name_orig}: min={min_val:.4f}, max={max_val:.4f}")
                
                else:
                    # Load pre-computed statistics
                    mean_or_min = self.stat.get(stat_mean_key, 0.0)
                    std_or_range = self.stat.get(stat_std_key, 1.0)
                
                # Apply normalization based on method
                if self.normalization_method == 'zscore':
                    if self.is_train:
                        normalized_data = np.divide(np.subtract(var_values_reshaped.astype(np.float32), mean), std)
                    else:
                        normalized_data = np.divide(np.subtract(var_values_reshaped.astype(np.float32), mean_or_min), std_or_range)
                
                elif self.normalization_method == 'minmax':
                    if self.is_train:
                        normalized_data = np.divide(np.subtract(var_values_reshaped.astype(np.float32), mean), std)  # (x - min) / range
                    else:
                        normalized_data = np.divide(np.subtract(var_values_reshaped.astype(np.float32), mean_or_min), std_or_range)
                all_var_data.append(normalized_data)
        
        final_data = np.stack(all_var_data, axis=-1).astype(np.float32)
        
        # Cache the result and manage cache size
        self._data_cache[cache_key] = final_data
        self._manage_cache_size()
        
        return final_data
    
    def _calculate_lag_features(self) -> int:
        """Calculate total number of lag features from flexible configuration"""
        if not self.enable_lags:
            return 0
        
        total_lag_features = 0
        
        # Count forcing variable lag features
        forcing_lags = self.lag_config.get('forcing') or {}
        for var_name in self.fvarLst:
            if var_name in forcing_lags:
                n_lags = len(forcing_lags[var_name])
                total_lag_features += n_lags
        
        # Count parameter variable lag features  
        parameter_lags = self.lag_config.get('parameters') or {}
        for var_name in self.pvarLst:
            if var_name in parameter_lags:
                n_lags = len(parameter_lags[var_name])
                total_lag_features += n_lags
        
        # Count target variable lag features
        target_lags = self.lag_config.get('target') or {}
        for var_name in self.ovarLst:
            if var_name in target_lags:
                n_lags = len(target_lags[var_name])
                total_lag_features += n_lags
        
        # Fallback: legacy target lags
        if total_lag_features == 0 and self.lags:
            total_lag_features = len(self.lags) * len(self.ovarLst)
        
        return total_lag_features
    
    def _make_flexible_lag_features(self, forcing_seq: np.ndarray, 
                                   param_seq: np.ndarray,
                                   target_seq: np.ndarray) -> np.ndarray:
        """Create lag features from any variable type based on flexible configuration"""
        if not self.enable_lags:
            return np.array([]).reshape(self.seq_length, 0)
        
        all_lag_features = []
        
        # Create forcing variable lag features
        forcing_lags = self.lag_config.get('forcing') or {}
        for i, var_name in enumerate(self.fvarLst):
            if var_name in forcing_lags:
                var_lags = forcing_lags[var_name]
                var_data = forcing_seq[:, i:i+1]  # (seq_len, 1)
                lagged = self._make_lagged_targets(var_data, var_lags)
                all_lag_features.append(lagged)
        
        # Create parameter variable lag features
        parameter_lags = self.lag_config.get('parameters') or {}
        for i, var_name in enumerate(self.pvarLst):
            if var_name in parameter_lags:
                var_lags = parameter_lags[var_name]
                var_data = param_seq[:, i:i+1]  # (seq_len, 1)
                lagged = self._make_lagged_targets(var_data, var_lags)
                all_lag_features.append(lagged)
        
        # Create target variable lag features
        target_lags = self.lag_config.get('target') or {}
        for i, var_name in enumerate(self.ovarLst):
            if var_name in target_lags:
                var_lags = target_lags[var_name]
                var_data = target_seq[:, i:i+1]  # (seq_len, 1)
                lagged = self._make_lagged_targets(var_data, var_lags)
                all_lag_features.append(lagged)
        
        # Fallback: legacy target lags
        if len(all_lag_features) == 0 and self.lags:
            lagged = self._make_lagged_targets(target_seq, self.lags)
            all_lag_features.append(lagged)
        
        # Concatenate all lag features
        if all_lag_features:
            return np.concatenate(all_lag_features, axis=1)
        else:
            return np.array([]).reshape(self.seq_length, 0)
    
    def _make_lagged_targets(self, y_seq: np.ndarray, lags: List[int]) -> np.ndarray:
        """Create lagged target features - FIXED VERSION without np.roll corruption"""
        seq_length, ny = y_seq.shape
        lagged = []
        
        for lag in lags:
            # Create properly lagged array without wrap-around corruption
            arr = np.zeros_like(y_seq)
            
            if lag > 0:
                # Positive lag: use past values (shift right, pad beginning with zeros)
                if lag < seq_length:
                    arr[lag:, :] = y_seq[:-lag, :]  # Copy earlier values to later positions
                # arr[:lag, :] remains zero (no data available for first 'lag' positions)
                
            elif lag < 0:
                # Negative lag: use future values (shift left, pad end with zeros)  
                abs_lag = abs(lag)
                if abs_lag < seq_length:
                    arr[:-abs_lag, :] = y_seq[abs_lag:, :]  # Copy later values to earlier positions
                # arr[-abs_lag:, :] remains zero (no data available for last 'abs_lag' positions)
                
            else:
                # lag == 0: no shift, copy original
                arr = y_seq.copy()
            
            lagged.append(arr)
            
        return np.concatenate(lagged, axis=1)
    
    def _create_lookup_table(self):
        """Create efficient lookup table excluding NaN samples"""
        self.lookup_table = {}
        
        # Get file counts
        n_files_forcing = self.forcings.shape[0]
        n_files_params = self.pars.shape[0]
        n_files_target = self.target.shape[0]
        
        max_files_iter = max(1, min(self.nfile_config, max(n_files_params, n_files_target)))
        
        self.logger.info(f"File counts - Forcing: {n_files_forcing}, Params: {n_files_params}, Target: {n_files_target}")
        self.logger.info(f"Max files iter: {max_files_iter}")
        self.logger.info(f"Ngrid: {self.ngrid}, Nt: {self.nt}, Seq length: {self.seq_length}")
        
        # Identify NaN samples
        invalid_samples = self._identify_nan_samples()

        # Build lookup table
        self._build_lookup_table(invalid_samples)

    def _identify_nan_samples(self):
        target_any_nan = np.isnan(self.target).any(axis=3)
        forcing_any_var_nan = np.isnan(self.forcings).any(axis=3)
        param_any_nan = np.isnan(self.pars).any(axis=2)
        
        self.logger.info(f"NaN shapes - Target: {target_any_nan.shape}, Forcing: {forcing_any_var_nan.shape}, Param: {param_any_nan.shape}")
        self.logger.info(f"NaN counts - Target: {np.sum(target_any_nan)}, Forcing: {np.sum(forcing_any_var_nan)}, Param: {np.sum(param_any_nan)}")
        
        # Rolling NaN check for forcing sequences
        forcing_seq_has_nan = np.full((self.forcings.shape[0], self.nt, self.ngrid), False, dtype=bool)
        for f_idx in range(self.forcings.shape[0]):
            for g_idx in range(self.ngrid):
                series = forcing_any_var_nan[f_idx, :, g_idx]
                if series.size > 0:
                    forcing_seq_has_nan[f_idx, :, g_idx] = rolling_any_true(
                        series, self.seq_length, axis=0)
                else:
                    forcing_seq_has_nan[f_idx, :, g_idx] = True
        
        self.logger.info(f"Forcing seq has NaN shape: {forcing_seq_has_nan.shape}")
        self.logger.info(f"Forcing seq NaN count: {np.sum(forcing_seq_has_nan)}")

        return target_any_nan, forcing_seq_has_nan, param_any_nan

    def _build_lookup_table(self, invalid_samples):
        target_any_nan, forcing_seq_has_nan, param_any_nan = invalid_samples
        lookup_idx, skipped_nan_count = 0, 0
        valid_target_count = 0
        valid_forcing_count = 0
        valid_param_count = 0
        
        # Always use memory-efficient loader for safety
        if hasattr(self.config.data, 'file_indices'):
            file_indices = self.config.data.file_indices
            if self.mode == 'train':
                num_files = file_indices['train_files'][1] - file_indices['train_files'][0] + 1
            elif self.mode == 'valid':
                num_files = file_indices['valid_files'][1] - file_indices['valid_files'][0] + 1
            else:
                num_files = file_indices['test_files'][1] - file_indices['test_files'][0] + 1
            
            # Always use memory-efficient loader
            self.logger.info(f"Using memory-efficient loader for {num_files} files")
            from src.data.memory_efficient_loader import MemoryEfficientDataLoader
            self.memory_loader = MemoryEfficientDataLoader(self.config, self.mode, max_memory_gb=8.0)
            
            # Defer actual data loading until needed
            self.use_memory_efficient = True
            self.num_files = num_files
        
        max_files_iter = max(1, min(self.nfile_config, max(self.pars.shape[0], self.target.shape[0])))

        for file_iter_idx in range(max_files_iter):
            f_idx_f = 0 if self.forcings.shape[0] == 1 else file_iter_idx
            f_idx_p = 0 if self.pars.shape[0] == 1 else file_iter_idx
            f_idx_t = 0 if self.target.shape[0] == 1 else file_iter_idx
            
            if (f_idx_f >= self.forcings.shape[0] or 
                f_idx_p >= self.pars.shape[0] or 
                f_idx_t >= self.target.shape[0]):
                continue
            
            for grid_k in range(self.ngrid):
                for time_k in range(self.seq_length - 1, self.nt):
                    target_invalid = target_any_nan[f_idx_t, time_k, grid_k]
                    forcing_invalid = forcing_seq_has_nan[f_idx_f, time_k, grid_k]
                    param_invalid = param_any_nan[f_idx_p, grid_k]
                    
                    if target_invalid:
                        valid_target_count += 1
                    if forcing_invalid:
                        valid_forcing_count += 1
                    if param_invalid:
                        valid_param_count += 1
                    
                    is_invalid = target_invalid or forcing_invalid or param_invalid
                    
                    if is_invalid:
                        skipped_nan_count += 1
                        continue
                    
                    self.lookup_table[lookup_idx] = (file_iter_idx, grid_k, time_k)
                    lookup_idx += 1
        
        self.logger.info(f"Skipped {skipped_nan_count} samples due to NaNs")
        self.logger.info(f"Invalid counts - Target: {valid_target_count}, Forcing: {valid_forcing_count}, Param: {valid_param_count}")
        self.logger.info(f"Created lookup table with {len(self.lookup_table)} valid samples")
        self._dataset_length = len(self.lookup_table)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Get dataset item with memory optimization and scenario tracking
        
        Returns:
            Tuple of (input_data, target_data, metadata)
            where metadata contains scenario information
        """
        if idx >= len(self.lookup_table):
            raise IndexError(f"Index {idx} out of bounds")
        
        file_iter_val, grid_idx_val, time_idx_val = self.lookup_table[idx]
        
        # Get actual file indices (these are the SCENARIO indices)
        f_idx_f = 0 if self.forcings.shape[0] == 1 else file_iter_val
        f_idx_p = 0 if self.pars.shape[0] == 1 else file_iter_val
        f_idx_t = 0 if self.target.shape[0] == 1 else file_iter_val
        
        # Calculate actual scenario numbers (adding file range offset)
        scenario_forcing = self.file_range[0] + f_idx_f
        scenario_params = self.file_range[0] + f_idx_p
        scenario_target = self.file_range[0] + f_idx_t
        
        # Extract sequences
        start_idx = time_idx_val - self.seq_length + 1
        end_idx = time_idx_val + 1
        
        forcing_seq = self.forcings[f_idx_f, start_idx:end_idx, grid_idx_val, :]
        param_vec = self.pars[f_idx_p, grid_idx_val, :]
        param_seq = np.tile(param_vec, (self.seq_length, 1))
        
        # NEW: Create flexible lag features from any variable type
        output_seq = self.target[f_idx_t, start_idx:end_idx, grid_idx_val, :]
        lag_features = self._make_flexible_lag_features(forcing_seq, param_seq, output_seq)
        
        # Combine input features: [forcing, parameters, flexible_lag_features]
        if lag_features.shape[1] > 0:
            input_data = np.concatenate([forcing_seq, param_seq, lag_features], axis=1)
        else:
            # No lag features - use only forcing and parameters
            input_data = np.concatenate([forcing_seq, param_seq], axis=1)
        
        # Create metadata with scenario information
        metadata = {
            'scenario_forcing': int(scenario_forcing),
            'scenario_params': int(scenario_params), 
            'scenario_target': int(scenario_target),
            'grid_idx': int(grid_idx_val),
            'time_idx': int(time_idx_val),
            'file_iter_val': int(file_iter_val),
            'mode': self.mode,
            'sample_idx': int(idx)
        }
        
        return (
            torch.from_numpy(forcing_seq).float(),
            torch.from_numpy(param_vec).float(),
            torch.from_numpy(lag_features).float(),
            torch.from_numpy(input_data).float(),
            torch.from_numpy(output_seq).float(),
            metadata
        )
    
    def get_grid_indices(self) -> np.ndarray:
        """Return selected grid indices for reproducible splits"""
        return getattr(self, 'selected_flat_indices_from_dp', np.array([]))
    
    def get_scenario_info(self) -> Dict[str, Any]:
        """
        Get information about scenarios used in this dataset
        
        Returns:
            Dictionary with scenario information
        """
        scenario_info = {
            'mode': self.mode,
            'file_range': self.file_range,
            'scenario_range': f"{self.file_range[0]}-{self.file_range[1]}",
            'num_scenarios': self.file_range[1] - self.file_range[0] + 1,
            'time_range': self.time_range,
            'num_grids': self.ngrid,
            'num_samples': len(self.lookup_table),
            'sequence_length': self.seq_length,
            'lags': self.lags
        }
        
        # Add forcing/parameter/target scenario counts
        scenario_info['forcing_scenarios'] = self.forcings.shape[0] if self.forcings.shape[0] > 1 else 1
        scenario_info['param_scenarios'] = self.pars.shape[0] if self.pars.shape[0] > 1 else 1
        scenario_info['target_scenarios'] = self.target.shape[0] if self.target.shape[0] > 1 else 1
        
        return scenario_info
    
    def get_normalization_stats(self, var_name: str) -> Tuple[float, float]:
        """
        Get normalization statistics (mean, std) for a variable
        
        Args:
            var_name: Name of the variable
            
        Returns:
            Tuple of (mean, std) used for normalization
        """
        mean_key = f"{var_name}_mean"
        std_key = f"{var_name}_std"
        mean = self.stat.get(mean_key, 0.0)
        std = self.stat.get(std_key, 1.0)
        return mean, std
    
    def print_scenario_summary(self):
        """Print a summary of scenarios used in this dataset"""
        info = self.get_scenario_info()
        
        print(f"\n=== SCENARIO SUMMARY ({info['mode'].upper()} MODE) ===")
        print(f"Author: Siddik")
        print(f"Dataset Mode: {info['mode']}")
        print(f"Scenario Range: {info['scenario_range']} ({info['num_scenarios']} scenarios)")
        print(f"Time Range: {info['time_range'][0]} to {info['time_range'][1]}")
        print(f"Grid Cells: {info['num_grids']}")
        print(f"Total Samples: {info['num_samples']}")
        print(f"Sequence Length: {info['sequence_length']}")
        print(f"Lag Features: {info['lags']}")
        print(f"")
        print(f"Data Sources:")
        print(f"  - Forcing scenarios: {info['forcing_scenarios']}")
        print(f"  - Parameter scenarios: {info['param_scenarios']}")
        print(f"  - Target scenarios: {info['target_scenarios']}")
        
        # Show some example scenario mappings
        if len(self.lookup_table) > 0:
            print(f"\nExample Scenario Mappings:")
            for i in [0, min(5, len(self.lookup_table)-1), len(self.lookup_table)-1]:
                if i < len(self.lookup_table):
                    _, _, metadata = self.__getitem__(i)
                    print(f"  Sample {i}: Scenario {metadata['scenario_target']} "
                          f"(Grid {metadata['grid_idx']}, Time {metadata['time_idx']})")
        print("=" * 50)