"""
Grid Splitting Utilities for Train/Test on Different Grid Samples
Author: Siddik
Description: Utilities for creating reproducible train/test splits on different grid locations

This module provides functionality for creating spatial and temporal splits of gridded
hydrological data. The key features include:

1. Spatial splitting: Dividing the geographic domain into train/validation/test regions
2. Stratified splitting: Ensuring representative samples across elevation or other variables
3. Temporal splitting: Dividing time series data into different periods
4. Reproducibility: Using seeds for consistent splits across experiments

The grid splitting approach is crucial for hydrological modeling to ensure that
the model can generalize to unseen spatial locations (spatial generalization) and
time periods (temporal generalization).
"""

import numpy as np
import xarray as xr
from typing import Tuple, Optional, Dict, List
import pickle
import os
import logging
from ..core.config import UnifiedConfig


class GridSplitter:
    """
    Utility class for creating and managing train/test splits on different grid samples
    Author: Siddik
    
    This class provides methods for creating spatially separated train/validation/test splits
    which are essential for evaluating the generalization capability of hydrological models.
    
    The class supports multiple splitting strategies:
    1. Random sampling: Randomly select grid cells for each split
    2. Stratified sampling: Ensure representative samples across a stratification variable
    3. Exact count vs. percentage-based splitting
    4. Overlapping vs. non-overlapping grid splits
    
    The splits are saved to disk to ensure reproducibility across different runs.
    
    Attributes:
        config (UnifiedConfig): Configuration object
        logger (Logger): Logger instance
        split_file (str): Path to save/load split information
    """
    
    def __init__(self, config: UnifiedConfig):
        """
        Initialize GridSplitter
        
        Args:
            config: UnifiedConfig object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.split_file = os.path.join(config.out_dir, 'grid_splits.pkl')
    
    def create_spatial_splits(self, mask_dataset_path: str, 
                            train_count: Optional[int] = None,
                            val_count: Optional[int] = None,
                            test_count: Optional[int] = None,
                            train_ratio: float = 0.7, 
                            val_ratio: float = 0.15,
                            test_ratio: float = 0.15,
                            use_exact_counts: bool = True,
                            ensure_no_overlap: bool = True,
                            seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Create spatial train/validation/test splits ensuring different grid samples
        Author: Siddik
        
        This method creates spatially separated splits of the grid domain, which is
        essential for evaluating spatial generalization capability of the model.
        
        The method supports two modes:
        1. Exact counts: Specify the exact number of grid cells for each split
        2. Percentage-based: Specify the percentage of valid grid cells for each split
        
        Grid cells are selected based on a land mask to ensure only valid locations
        are included in the splits.
        
        Args:
            mask_dataset_path: Path to dataset containing mask
            train_count: Exact number of grids for training
            val_count: Exact number of grids for validation
            test_count: Exact number of grids for testing
            train_ratio: Ratio of grid cells for training (used if not exact counts)
            val_ratio: Ratio of grid cells for validation (used if not exact counts)
            test_ratio: Ratio of grid cells for testing (used if not exact counts)
            use_exact_counts: Whether to use exact counts or ratios
            ensure_no_overlap: Ensure train/val/test grids don't overlap
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train', 'val', 'test' grid indices
        """
        # Allow any percentage combination - user can select any portion of total grids
        if not use_exact_counts:
            total_ratio = train_ratio + val_ratio + test_ratio
            if total_ratio > 1.0 and ensure_no_overlap:
                self.logger.warning(f"Total ratios ({total_ratio:.2f}) exceed 1.0 with no_overlap=True. Grid overlap will occur.")
            self.logger.info(f"Using {total_ratio*100:.1f}% of total available grids")
        
        # Set seed for reproducibility
        if seed is None:
            seed = self.config.evaluation.get('grid_split_seed', 42)
        np.random.seed(seed)
        
        # Load mask dataset
        self.logger.info(f"Loading mask from {mask_dataset_path}")
        ds = xr.open_dataset(mask_dataset_path)
        
        # Apply spatial subsetting as per config
        spatial_config = self.config.data.spatial
        lat0, lat1 = spatial_config['lat_min'], spatial_config['lat_max']
        lon0, lon1 = spatial_config['lon_min'], spatial_config['lon_max']
        step = spatial_config['step']
        
        # Subset spatially
        lat_slice = slice(lat0, lat1, step) if step else slice(lat0, lat1)
        lon_slice = slice(lon0, lon1, step) if step else slice(lon0, lon1)
        ds_subset = ds.sel(lat=lat_slice, lon=lon_slice)
        
        # Extract mask
        if 'mask' not in ds_subset:
            raise ValueError("Dataset must contain 'mask' variable")
        
        mask_da = ds_subset['mask']
        mask_data_raw = mask_da.data
        
        # Handle different mask dimensionalities
        if mask_data_raw.ndim == 3 and mask_data_raw.shape[0] > 1:
            mask_2d = mask_data_raw[0, :, :]
        elif mask_data_raw.ndim == 2:
            mask_2d = mask_data_raw
        elif 'nlayer' in mask_da.dims:
            mask_2d = mask_da.isel(nlayer=1).data
        else:
            raise ValueError(f"Cannot handle mask shape {mask_data_raw.shape}")
        
        # Get valid grid indices
        valid_indices_flat = np.where(mask_2d.flatten() == 1)[0]
        n_valid = len(valid_indices_flat)
        
        self.logger.info(f"Found {n_valid} valid grid cells")
        
        # Calculate split sizes based on configuration
        if use_exact_counts:
            # Use exact counts specified by user
            n_train = train_count if train_count is not None else 0
            n_val = val_count if val_count is not None else 0  
            n_test = test_count if test_count is not None else 0
            
            total_requested = n_train + n_val + n_test
            
            if total_requested > n_valid:
                raise ValueError(f"Requested {total_requested} grids but only {n_valid} valid grids available")
            
            # If ensure_no_overlap is False, we can use overlapping grids
            if not ensure_no_overlap:
                self.logger.info(f"Creating potentially overlapping grid splits:")
                self.logger.info(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
                
                # Sample with potential overlap
                shuffled_indices = np.random.permutation(valid_indices_flat)
                train_indices = np.random.choice(valid_indices_flat, n_train, replace=False)
                val_indices = np.random.choice(valid_indices_flat, n_val, replace=False) 
                test_indices = np.random.choice(valid_indices_flat, n_test, replace=False)
            else:
                self.logger.info(f"Creating non-overlapping grid splits:")
                self.logger.info(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
                self.logger.info(f"  Total used: {total_requested}/{n_valid} ({100*total_requested/n_valid:.1f}%)")
                
                # Sample without overlap
                shuffled_indices = np.random.permutation(valid_indices_flat)
                train_indices = shuffled_indices[:n_train]
                val_indices = shuffled_indices[n_train:n_train + n_val]
                test_indices = shuffled_indices[n_train + n_val:n_train + n_val + n_test]
        else:
            # Use ratios - calculate each percentage independently
            n_train = int(train_ratio * n_valid)
            n_val = int(val_ratio * n_valid)
            n_test = int(test_ratio * n_valid)  # Calculate test percentage properly
            
            self.logger.info(f"Using ratio-based splits:")
            self.logger.info(f"  Train: {n_train} ({100*train_ratio:.1f}%)")
            self.logger.info(f"  Val: {n_val} ({100*val_ratio:.1f}%)")
            self.logger.info(f"  Test: {n_test} ({100*test_ratio:.1f}%)")
            
            # Shuffle and split - sample independently for each mode
            shuffled_indices = np.random.permutation(valid_indices_flat)
            train_indices = shuffled_indices[:n_train]
            val_indices = np.random.choice(valid_indices_flat, n_val, replace=False)
            test_indices = np.random.choice(valid_indices_flat, n_test, replace=False)
        
        splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
            'metadata': {
                'total_valid_cells': n_valid,
                'train_count': len(train_indices),
                'val_count': len(val_indices), 
                'test_count': len(test_indices),
                'seed': seed,
                'mask_shape': mask_2d.shape,
                'spatial_bounds': {
                    'lat_min': lat0, 'lat_max': lat1,
                    'lon_min': lon0, 'lon_max': lon1
                },
                'method': 'exact_counts' if use_exact_counts else 'ratios',
                'ensure_no_overlap': ensure_no_overlap,
                'requested_counts': {
                    'train': n_train, 'val': n_val, 'test': n_test
                } if use_exact_counts else None,
                'author': 'Siddik'
            }
        }
        
        self.logger.info(f"Created splits - Train: {len(train_indices)}, "
                        f"Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        # Save splits
        self._save_splits(splits)
        
        return splits
    
    def create_temporal_splits(self, temporal_split_config: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
        """
        Create temporal splits based on different time ranges
        
        Args:
            temporal_split_config: Dictionary with time ranges for each split
            
        Returns:
            Dictionary with temporal ranges for each split
        """
        return {
            'train': tuple(temporal_split_config.get('train_range', ['1990-01-01', '1992-12-31'])),
            'val': tuple(temporal_split_config.get('val_range', ['1993-01-01', '1993-12-31'])),
            'test': tuple(temporal_split_config.get('test_range', ['1994-01-01', '1995-12-31']))
        }
    
    def create_stratified_spatial_splits(self, mask_dataset_path: str,
                                       stratification_var: str = 'elev',
                                       n_strata: int = 5,
                                       **kwargs) -> Dict[str, np.ndarray]:
        """
        Create stratified spatial splits based on a stratification variable
        
        This method ensures that each split (train/validation/test) contains
        representative samples across the range of a stratification variable
        (e.g., elevation). This is particularly important in hydrological modeling
        where processes can vary significantly with topography.
        
        The method divides the stratification variable into quantiles and ensures
        that each split contains samples from all quantiles, preventing scenarios
        where a split might be biased toward high or low values of the stratification
        variable.
        
        Args:
            mask_dataset_path: Path to dataset containing mask and stratification variable
            stratification_var: Variable name to use for stratification (e.g., 'elev')
            n_strata: Number of strata to create
            **kwargs: Additional arguments passed to create_spatial_splits
            
        Returns:
            Dictionary with stratified splits
        """
        # Load dataset
        ds = xr.open_dataset(mask_dataset_path)
        
        # Apply spatial subsetting
        spatial_config = self.config.data.spatial
        lat0, lat1 = spatial_config['lat_min'], spatial_config['lat_max']
        lon0, lon1 = spatial_config['lon_min'], spatial_config['lon_max']
        step = spatial_config['step']
        
        lat_slice = slice(lat0, lat1, step) if step else slice(lat0, lat1)
        lon_slice = slice(lon0, lon1, step) if step else slice(lon0, lon1)
        ds_subset = ds.sel(lat=lat_slice, lon=lon_slice)
        
        # Get mask and stratification variable
        mask_data = ds_subset['mask'].data
        if mask_data.ndim == 3:
            mask_2d = mask_data[0, :, :]
        else:
            mask_2d = mask_data
        
        valid_indices_flat = np.where(mask_2d.flatten() == 1)[0]
        
        # Get stratification values at valid locations
        if stratification_var not in ds_subset:
            self.logger.warning(f"Stratification variable '{stratification_var}' not found. "
                              f"Falling back to regular spatial split.")
            return self.create_spatial_splits(mask_dataset_path, **kwargs)
        
        strat_data = ds_subset[stratification_var].data
        if strat_data.ndim > 2:
            strat_data = strat_data[0, :, :]  # Take first time/layer if needed
        
        strat_values = strat_data.flatten()[valid_indices_flat]
        
        # Create strata based on quantiles
        quantiles = np.linspace(0, 1, n_strata + 1)
        strata_bounds = np.quantile(strat_values, quantiles)
        strata_labels = np.digitize(strat_values, strata_bounds[1:-1])
        
        # Perform stratified sampling
        train_ratio = kwargs.get('train_ratio', 0.7)
        val_ratio = kwargs.get('val_ratio', 0.15)
        test_ratio = kwargs.get('test_ratio', 0.15)
        seed = kwargs.get('seed', self.config.evaluation.get('grid_split_seed', 42))
        
        np.random.seed(seed)
        
        train_indices, val_indices, test_indices = [], [], []
        
        for stratum in range(n_strata):
            stratum_mask = (strata_labels == stratum)
            stratum_indices = valid_indices_flat[stratum_mask]
            n_stratum = len(stratum_indices)
            
            if n_stratum == 0:
                continue
            
            # Calculate split sizes for this stratum
            n_train_s = max(1, int(train_ratio * n_stratum))
            n_val_s = max(0, int(val_ratio * n_stratum))
            n_test_s = n_stratum - n_train_s - n_val_s
            
            # Shuffle and split
            shuffled = np.random.permutation(stratum_indices)
            
            train_indices.extend(shuffled[:n_train_s])
            val_indices.extend(shuffled[n_train_s:n_train_s + n_val_s])
            test_indices.extend(shuffled[n_train_s + n_val_s:])
        
        splits = {
            'train': np.array(train_indices),
            'val': np.array(val_indices),
            'test': np.array(test_indices),
            'metadata': {
                'stratified': True,
                'stratification_var': stratification_var,
                'n_strata': n_strata,
                'strata_bounds': strata_bounds.tolist(),
                'total_valid_cells': len(valid_indices_flat),
                'train_count': len(train_indices),
                'val_count': len(val_indices),
                'test_count': len(test_indices),
                'seed': seed
            }
        }
        
        self.logger.info(f"Created stratified splits using {stratification_var} - "
                        f"Train: {len(train_indices)}, Val: {len(val_indices)}, "
                        f"Test: {len(test_indices)}")
        
        self._save_splits(splits)
        return splits
    
    def load_splits(self) -> Optional[Dict[str, np.ndarray]]:
        """Load previously saved splits"""
        if os.path.exists(self.split_file):
            with open(self.split_file, 'rb') as f:
                splits = pickle.load(f)
            self.logger.info(f"Loaded splits from {self.split_file}")
            return splits
        return None
    
    def _save_splits(self, splits: Dict[str, np.ndarray]):
        """Save splits to file"""
        os.makedirs(os.path.dirname(self.split_file), exist_ok=True)
        with open(self.split_file, 'wb') as f:
            pickle.dump(splits, f)
        self.logger.info(f"Saved splits to {self.split_file}")
    
    def get_split_info(self, splits: Dict[str, np.ndarray]) -> str:
        """Get readable information about splits"""
        info_lines = ["Grid Split Information:"]
        info_lines.append("=" * 40)
        
        if 'metadata' in splits:
            metadata = splits['metadata']
            info_lines.append(f"Total valid cells: {metadata.get('total_valid_cells', 'Unknown')}")
            info_lines.append(f"Seed used: {metadata.get('seed', 'Unknown')}")
            
            if metadata.get('stratified', False):
                info_lines.append(f"Stratification: {metadata.get('stratification_var', 'Unknown')}")
                info_lines.append(f"Number of strata: {metadata.get('n_strata', 'Unknown')}")
        
        for split_name in ['train', 'val', 'test']:
            if split_name in splits:
                count = len(splits[split_name])
                info_lines.append(f"{split_name.capitalize()}: {count} cells")
        
        return "\n".join(info_lines)


def create_grid_aware_datasets(config: UnifiedConfig, 
                              use_existing_splits: bool = True,
                              create_stratified: bool = False) -> Dict[str, any]:
    """
    Create datasets with grid-aware train/test splits using YAML configuration
    Author: Siddik
    
    Args:
        config: UnifiedConfig object with grid_sampling configuration
        use_existing_splits: Whether to use existing splits if available
        create_stratified: Whether to create stratified splits
        
    Returns:
        Dictionary containing datasets and split information
    """
    logger = logging.getLogger(__name__)
    splitter = GridSplitter(config)
    
    # Try to load existing splits
    splits = None
    if use_existing_splits:
        splits = splitter.load_splits()
    
    # Create new splits if needed
    if splits is None:
        logger.info("Creating new grid splits based on YAML configuration...")
        mask_path = config.data.paths['parameters']  # Assuming parameters file has mask
        
        # Get grid sampling configuration
        grid_config = config.data.grid_sampling
        
        # Extract configuration parameters
        use_exact_counts = grid_config.get('use_exact_counts', True)
        ensure_no_overlap = grid_config.get('ensure_no_overlap', True)
        selection_strategy = grid_config.get('selection_strategy', 'random')
        
        if use_exact_counts:
            # Use exact counts from YAML
            train_count = grid_config.get('train_grids')
            val_count = grid_config.get('validation_grids')
            test_count = grid_config.get('test_grids')
            
            logger.info(f"Using exact grid counts from config:")
            logger.info(f"  Train: {train_count} grids")
            logger.info(f"  Validation: {val_count} grids") 
            logger.info(f"  Test: {test_count} grids")
            logger.info(f"  Overlap allowed: {not ensure_no_overlap}")
            
            if selection_strategy == 'stratified':
                # Use stratified sampling with exact counts
                stratify_var = grid_config.get('stratify_by', 'elev')
                n_strata = grid_config.get('n_strata', 5)
                
                splits = splitter.create_stratified_spatial_splits_exact(
                    mask_path,
                    train_count=train_count,
                    val_count=val_count,
                    test_count=test_count,
                    stratification_var=stratify_var,
                    n_strata=n_strata,
                    ensure_no_overlap=ensure_no_overlap
                )
            else:
                # Use random sampling with exact counts
                splits = splitter.create_spatial_splits(
                    mask_path,
                    train_count=train_count,
                    val_count=val_count,
                    test_count=test_count,
                    use_exact_counts=True,
                    ensure_no_overlap=ensure_no_overlap
                )
        else:
            # Use percentage-based splits - fully controlled by YAML
            train_ratio = grid_config['train_percentage']
            val_ratio = grid_config['validation_percentage']
            test_ratio = grid_config['test_percentage']
            
            logger.info(f"Using percentage-based grid splits:")
            logger.info(f"  Train: {100*train_ratio:.1f}%")
            logger.info(f"  Validation: {100*val_ratio:.1f}%")
            logger.info(f"  Test: {100*test_ratio:.1f}%")
            
            if selection_strategy == 'stratified':
                stratify_var = grid_config.get('stratify_by', 'elev')
                n_strata = grid_config.get('n_strata', 5)
                
                splits = splitter.create_stratified_spatial_splits(
                    mask_path,
                    stratification_var=stratify_var,
                    n_strata=n_strata,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio
                )
            else:
                splits = splitter.create_spatial_splits(
                    mask_path,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    use_exact_counts=False
                )
    
    logger.info(splitter.get_split_info(splits))
    
    return splits