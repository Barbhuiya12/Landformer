"""
Memory-Efficient Data Loader for Large-Scale Hydrological Datasets
Author: Siddik
Description: Handles 100+ files without memory errors through intelligent batching
"""

import numpy as np
import xarray as xr
import gc
import psutil
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import torch
from pathlib import Path

class MemoryEfficientDataLoader:
    """Memory-efficient data loader that can handle 100+ files"""
    
    def __init__(self, config, mode='train', max_memory_gb=8.0):
        """
        Initialize memory-efficient data loader
        
        Args:
            config: Configuration object
            mode: 'train', 'valid', or 'test'
            max_memory_gb: Maximum memory to use in GB
        """
        self.config = config
        self.mode = mode
        self.max_memory_gb = max_memory_gb
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Get file range
        self.file_range = self._get_file_range()
        self.total_files = self.file_range[1] - self.file_range[0] + 1
        
        # Dynamic chunk sizes based on dataset size
        self._set_dynamic_chunk_sizes()
        
        # Memory tracking
        self.memory_threshold = 0.8  # Use max 80% of available memory
        self.force_gc_interval = 5  # Force garbage collection every 5 files
        
        self.logger.info(f"Initialized for {self.total_files} files with chunks - "
                        f"Files: {self.file_chunk_size}, Time: {self.time_chunk_size}, "
                        f"Grids: {self.grid_chunk_size}")
    
    def _get_file_range(self) -> Tuple[int, int]:
        """Get file range for current mode"""
        file_indices = self.config.data.file_indices
        if self.mode == 'train':
            return tuple(file_indices['train_files'])
        elif self.mode == 'valid':
            return tuple(file_indices['valid_files'])
        else:
            return tuple(file_indices['test_files'])
    
    def _set_dynamic_chunk_sizes(self):
        """Set chunk sizes based on dataset size and available memory"""
        available_gb = self._get_available_memory()
        
        # Ultra memory-efficient settings for large datasets
        if self.total_files > 50:
            self.file_chunk_size = 1  # Process 1 file at a time
            self.time_chunk_size = 180  # Process 6 months at a time
            self.grid_chunk_size = 20  # Very small grid chunks
            self.logger.info(f"Using ultra memory-efficient mode for {self.total_files} files")
        elif self.total_files > 20:
            self.file_chunk_size = 2
            self.time_chunk_size = 365
            self.grid_chunk_size = 30
        else:
            self.file_chunk_size = 3
            self.time_chunk_size = 500
            self.grid_chunk_size = 50
        
        # Further reduce if low memory
        if available_gb < 10:
            self.file_chunk_size = max(1, self.file_chunk_size // 2)
            self.time_chunk_size = min(180, self.time_chunk_size // 2)
            self.grid_chunk_size = min(20, self.grid_chunk_size // 2)
            self.logger.warning(f"Low memory ({available_gb:.1f}GB), using minimal chunks")
    
    def _get_available_memory(self) -> float:
        """Get available system memory in GB"""
        vm = psutil.virtual_memory()
        return vm.available / (1024**3)
    
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within limits"""
        vm = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        process_gb = process.memory_info().rss / (1024**3)
        
        if vm.percent > 85 or process_gb > self.max_memory_gb:
            self.logger.warning(f"High memory usage: System {vm.percent:.1f}%, "
                              f"Process {process_gb:.1f}GB")
            return False
        return True
    
    def _aggressive_cleanup(self):
        """Aggressive memory cleanup"""
        # Clear any caches
        if hasattr(self, '_cache'):
            self._cache.clear()
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Log memory after cleanup
        vm = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        self.logger.info(f"After cleanup - Process: {process.memory_info().rss/(1024**3):.1f}GB, "
                        f"Available: {vm.available/(1024**3):.1f}GB")
    
    def load_data_file_by_file(self, grid_indices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Load data file by file to handle large datasets
        
        Args:
            grid_indices: Indices of grid cells to load
            
        Returns:
            Dictionary with forcing, parameters, and targets arrays
        """
        self.logger.info(f"Loading {self.total_files} files using file-by-file processing")
        
        # Initialize lists to collect data
        all_forcing = []
        all_targets = []
        all_params = None  # Parameters are the same for all files
        
        # Process files in small batches
        for file_idx in range(self.file_range[0], self.file_range[1] + 1):
            # Check memory before processing
            if not self._check_memory_usage():
                self._aggressive_cleanup()
            
            try:
                # Load single file data
                self.logger.info(f"Processing file {file_idx}/{self.file_range[1]}")
                
                # Load forcing for this file
                forcing_chunk = self._load_forcing_chunk(file_idx, file_idx, grid_indices)
                if forcing_chunk is not None:
                    all_forcing.append(forcing_chunk)
                
                # Load targets for this file
                target_chunk = self._load_target_chunk(file_idx, file_idx, grid_indices)
                if target_chunk is not None:
                    all_targets.append(target_chunk)
                
                # Load parameters only once
                if all_params is None:
                    all_params = self._load_parameters(grid_indices)
                
                # Periodic cleanup
                if (file_idx - self.file_range[0]) % self.force_gc_interval == 0:
                    gc.collect()
                    
            except Exception as e:
                self.logger.error(f"Error processing file {file_idx}: {e}")
                # Try to recover by cleaning memory
                self._aggressive_cleanup()
                continue
        
        # Concatenate all data
        self.logger.info("Concatenating all file data...")
        
        result = {}
        
        if all_forcing:
            result['forcing'] = np.concatenate(all_forcing, axis=0)
            self.logger.info(f"Forcing shape: {result['forcing'].shape}")
        
        if all_targets:
            result['targets'] = np.concatenate(all_targets, axis=0)
            self.logger.info(f"Targets shape: {result['targets'].shape}")
        
        if all_params is not None:
            # Repeat parameters for each file
            result['parameters'] = np.repeat(all_params[np.newaxis, ...], 
                                            len(all_forcing), axis=0)
            self.logger.info(f"Parameters shape: {result['parameters'].shape}")
        
        # Final cleanup
        self._aggressive_cleanup()
        
        return result
    
    def _load_forcing_chunk(self, start_file: int, end_file: int, 
                           grid_indices: np.ndarray) -> Optional[np.ndarray]:
        """Load forcing data for a file chunk"""
        try:
            forcing_path = self.config.data.paths['forcing']
            forcing_vars = self.config.data.variables['forcing']
            
            # Open with minimal chunks
            with xr.open_zarr(forcing_path, 
                            chunks={'time': self.time_chunk_size, 
                                   'latitude': self.grid_chunk_size,
                                   'longitude': self.grid_chunk_size}) as ds:
                
                # Get time range for these files
                time_start = start_file * 365  # Assuming daily data
                time_end = min((end_file + 1) * 365, len(ds.time))
                
                # Select time and spatial subset
                ds_subset = ds.isel(time=slice(time_start, time_end))
                
                # Extract variables
                data_list = []
                for var in forcing_vars:
                    if var in ds_subset:
                        var_data = ds_subset[var].values
                        # Select grid points
                        if len(var_data.shape) == 3:  # time, lat, lon
                            var_data_flat = var_data.reshape(var_data.shape[0], -1)
                            var_data_selected = var_data_flat[:, grid_indices]
                            data_list.append(var_data_selected)
                
                if data_list:
                    # Stack variables along last dimension
                    return np.stack(data_list, axis=-1)
                    
        except Exception as e:
            self.logger.error(f"Error loading forcing chunk {start_file}-{end_file}: {e}")
        
        return None
    
    def _load_target_chunk(self, start_file: int, end_file: int,
                          grid_indices: np.ndarray) -> Optional[np.ndarray]:
        """Load target data for a file chunk"""
        try:
            target_vars = self.config.data.variables['target']
            target_files = self.config.data.paths['target_files']
            
            all_targets = []
            
            for var in target_vars:
                if var in target_files:
                    target_path = target_files[var]
                    
                    with xr.open_zarr(target_path,
                                    chunks={'time': self.time_chunk_size}) as ds:
                        
                        # Get time range
                        time_start = start_file * 365
                        time_end = min((end_file + 1) * 365, len(ds.time))
                        
                        # Select subset
                        ds_subset = ds.isel(time=slice(time_start, time_end))
                        
                        if var.lower() in ds_subset:
                            var_data = ds_subset[var.lower()].values
                        elif var in ds_subset:
                            var_data = ds_subset[var].values
                        else:
                            continue
                        
                        # Select grid points
                        if len(var_data.shape) == 3:
                            var_data_flat = var_data.reshape(var_data.shape[0], -1)
                            var_data_selected = var_data_flat[:, grid_indices]
                            all_targets.append(var_data_selected)
            
            if all_targets:
                return np.stack(all_targets, axis=-1)
                
        except Exception as e:
            self.logger.error(f"Error loading target chunk {start_file}-{end_file}: {e}")
        
        return None
    
    def _load_parameters(self, grid_indices: np.ndarray) -> Optional[np.ndarray]:
        """Load parameter data once for all files"""
        try:
            param_path = self.config.data.paths['parameters']
            
            with xr.open_zarr(param_path) as ds:
                data_list = []
                
                for var in ds.data_vars:
                    if var in ds:
                        var_data = ds[var].values
                        
                        # Flatten spatial dimensions and select grids
                        if len(var_data.shape) >= 2:
                            var_data_flat = var_data.reshape(-1)
                            var_data_selected = var_data_flat[grid_indices]
                            data_list.append(var_data_selected)
                
                if data_list:
                    return np.stack(data_list, axis=-1)
                    
        except Exception as e:
            self.logger.error(f"Error loading parameters: {e}")
        
        return None
    
    def create_pytorch_dataset(self, grid_indices: np.ndarray):
        """Create PyTorch dataset with memory-efficient loading"""
        # For very large datasets, use file-by-file loading
        if self.total_files > 50:
            self.logger.info("Using file-by-file loading for large dataset")
            data = self.load_data_file_by_file(grid_indices)
        else:
            # Use regular chunked loading
            data = self.load_data_chunked(self.file_range, grid_indices)
        
        return data
    
    def load_data_chunked(self, file_range: Tuple[int, int], 
                         grid_indices: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        """Original chunked loading method for smaller datasets"""
        # Implementation of original chunked loading
        # (keeping for backwards compatibility)
        result = {}
        
        # Load in chunks
        all_forcing = []
        all_targets = []
        
        for start in range(file_range[0], file_range[1] + 1, self.file_chunk_size):
            end = min(start + self.file_chunk_size - 1, file_range[1])
            
            forcing = self._load_forcing_chunk(start, end, grid_indices)
            if forcing is not None:
                all_forcing.append(forcing)
            
            targets = self._load_target_chunk(start, end, grid_indices)
            if targets is not None:
                all_targets.append(targets)
            
            # Cleanup after each chunk
            gc.collect()
        
        if all_forcing:
            result['forcing'] = np.concatenate(all_forcing, axis=0)
        if all_targets:
            result['targets'] = np.concatenate(all_targets, axis=0)
        
        # Load parameters once
        params = self._load_parameters(grid_indices)
        if params is not None:
            result['parameters'] = np.repeat(params[np.newaxis, ...], 
                                           len(all_forcing), axis=0)
        
        return result
