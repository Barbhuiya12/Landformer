"""
Grid-Aware Training Utilities
Author: Siddik Barbhuiya
Description: Implements grid-balanced sampling and spatial regularization for hydrological modeling.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Sampler
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GridBalancedSampler(Sampler):
    """
    Sampler that ensures each batch has balanced representation across grids.
    
    This sampler groups samples by their grid index and samples equally from each grid,
    ensuring that no single grid dominates the training batches. This is particularly
    important for hydrological models where some grids may have more valid samples.
    
    Args:
        dataset: The dataset to sample from
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle within each grid
        drop_last: Whether to drop last incomplete batch
    """
    
    def __init__(self, dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Build grid-to-sample index mapping
        self.grid_to_samples = self._build_grid_mapping()
        self.n_grids = len(self.grid_to_samples)
        
        logger.info(f"GridBalancedSampler: {self.n_grids} grids, {len(dataset)} total samples")
        
        # Calculate samples per grid per batch
        self.samples_per_grid = max(1, batch_size // self.n_grids)
        
    def _build_grid_mapping(self) -> Dict[int, List[int]]:
        """Build mapping from grid index to sample indices using fast numpy operations."""
        grid_to_samples = {}
        
        # Check if dataset has lookup_table as a numpy array (optimized case)
        if hasattr(self.dataset, 'lookup_table') and isinstance(self.dataset.lookup_table, np.ndarray):
            logger.info("GridBalancedSampler: Building mapping using fast numpy operations...")
            grid_indices = self.dataset.lookup_table[:, 1]
            
            # Use argsort to group indices by grid
            sorted_idx = np.argsort(grid_indices)
            sorted_grids = grid_indices[sorted_idx]
            
            # Find boundaries where grid index changes
            diff = np.diff(sorted_grids)
            boundaries = np.where(diff > 0)[0] + 1
            boundaries = np.concatenate(([0], boundaries, [len(sorted_grids)]))
            
            unique_grids = sorted_grids[boundaries[:-1]]
            
            for i, grid_idx in enumerate(unique_grids):
                start, end = boundaries[i], boundaries[i+1]
                # Store as list for compatibility with existing logic
                grid_to_samples[int(grid_idx)] = sorted_idx[start:end].tolist()
                
            logger.info(f"GridBalancedSampler: Built mapping for {len(grid_to_samples)} unique grids")
            
        elif hasattr(self.dataset, 'lookup_table') and isinstance(self.dataset.lookup_table, dict):
            # Legacy dict-based lookup table (still slow for large datasets)
            logger.warning("GridBalancedSampler: Dataset has dict-based lookup_table, building mapping slowly...")
            for sample_idx, (file_idx, grid_idx, time_idx) in enumerate(self.dataset.lookup_table.values()):
                if grid_idx not in grid_to_samples:
                    grid_to_samples[grid_idx] = []
                grid_to_samples[grid_idx].append(sample_idx)
        else:
            # Fallback: iterate through dataset (very slow)
            logger.warning("GridBalancedSampler: Dataset doesn't have lookup_table, building grid mapping slowly...")
            for sample_idx in range(len(self.dataset)):
                try:
                    *_, metadata = self.dataset[sample_idx]
                    if isinstance(metadata, dict) and 'grid_idx' in metadata:
                        grid_idx = int(metadata['grid_idx'])
                        if grid_idx not in grid_to_samples:
                            grid_to_samples[grid_idx] = []
                        grid_to_samples[grid_idx].append(sample_idx)
                except Exception:
                    pass
        
        return grid_to_samples
    
    def __iter__(self):
        # Get list of grid indices
        grid_indices = list(self.grid_to_samples.keys())
        
        # Shuffle grid-sample mappings if requested
        grid_samples = {}
        for grid_idx in grid_indices:
            samples = self.grid_to_samples[grid_idx].copy()
            if self.shuffle:
                np.random.shuffle(samples)
            grid_samples[grid_idx] = samples
        
        # Track current position in each grid's sample list
        grid_positions = {grid_idx: 0 for grid_idx in grid_indices}
        
        # Generate balanced batches
        all_indices = []
        exhausted_grids = set()
        
        while len(exhausted_grids) < len(grid_indices):
            batch = []
            
            # Sample from each grid
            for grid_idx in grid_indices:
                if grid_idx in exhausted_grids:
                    continue
                    
                samples = grid_samples[grid_idx]
                pos = grid_positions[grid_idx]
                
                # Add samples from this grid
                for _ in range(self.samples_per_grid):
                    if pos >= len(samples):
                        exhausted_grids.add(grid_idx)
                        break
                    batch.append(samples[pos])
                    pos += 1
                
                grid_positions[grid_idx] = pos
            
            if batch:
                all_indices.extend(batch)
        
        return iter(all_indices)
    
    def __len__(self):
        return len(self.dataset)


class SpatialSnapshotSampler(Sampler):
    """
    Sampler for the spatial GAT: yields whole snapshots, i.e. all training cells on the
    same day with the same parameter set, so that neighbouring cells appear together in
    a batch. Snapshot order and the order of cells within a snapshot are shuffled each
    epoch. With batch_size >= cells per snapshot, a batch holds complete snapshots.

    Args:
        dataset: dataset with lookup_table rows [file_iter, grid_idx, time_idx]
        shuffle: shuffle snapshots and cells each epoch
        seed: base seed; the epoch counter is added so every epoch differs
    """

    def __init__(self, dataset, shuffle: bool = True, seed: int = 0):
        lut = np.asarray(dataset.lookup_table)
        order = np.lexsort((lut[:, 1], lut[:, 2], lut[:, 0]))      # by file, then day, then cell
        keys = lut[order][:, [0, 2]]
        starts = np.flatnonzero(np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)])
        self.groups = np.split(order, starts[1:])
        self.n = len(lut)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        sizes = np.array([len(g) for g in self.groups])
        logger.info(f"SpatialSnapshotSampler: {len(self.groups):,} snapshots (day x parameter set), "
                    f"{sizes.mean():.0f} cells per snapshot on average (min {sizes.min()}, max {sizes.max()})")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        idx = rng.permutation(len(self.groups)) if self.shuffle else np.arange(len(self.groups))
        for g in idx:
            cells = self.groups[g]
            yield from (rng.permutation(cells) if self.shuffle else cells).tolist()

    def __len__(self):
        return self.n


class SpatialClusterSampler(Sampler):
    """
    Sampler for the spatial GAT that keeps batches temporally diverse.

    Each group is one random (parameter set, day) snapshot and a random seed cell together
    with its nearest training cells on that day (cells_per_group in total). A batch of B
    samples therefore holds about B / cells_per_group different days, like a random batch,
    while every cell still has same-day neighbours for the GAT. Whole-day snapshots
    (SpatialSnapshotSampler) put only a few days in a batch, which changes the batch-wise
    KGE objective and slows learning.

    Args:
        dataset: dataset with lookup_table rows [file_iter, grid_idx, time_idx] and
            selected_lats_np / selected_lons_np per grid index
        cells_per_group: cells drawn around each seed cell (seed included)
        seed: base seed; the epoch counter is added so every epoch differs
    """

    def __init__(self, dataset, cells_per_group: int = 10, seed: int = 0):
        lut = np.asarray(dataset.lookup_table)
        order = np.lexsort((lut[:, 1], lut[:, 2], lut[:, 0]))      # by file, then day, then cell
        keys = lut[order][:, [0, 2]]
        starts = np.flatnonzero(np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)])
        self.groups = np.split(order, starts[1:])
        self.grid_of = lut[:, 1]
        lat = np.asarray(dataset.selected_lats_np, dtype=float)
        lon = np.asarray(dataset.selected_lons_np, dtype=float)
        # nearest-neighbour order of all training cells around every cell (computed once)
        d = np.hypot(lat[:, None] - lat[None, :], lon[:, None] - lon[None, :])
        self.near = np.argsort(d, axis=1)
        self.k = int(cells_per_group)
        self.n = len(lut)
        self.n_groups = max(1, self.n // self.k)
        self.seed = seed
        self.epoch = 0
        logger.info(f"SpatialClusterSampler: {len(self.groups):,} day snapshots, groups of {self.k} nearby cells, "
                    f"{self.n_groups:,} groups per epoch")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for g in rng.integers(0, len(self.groups), size=self.n_groups):
            members = self.groups[g]                       # sorted by grid index
            cells = self.grid_of[members]
            seed_grid = cells[rng.integers(len(cells))]
            cand = self.near[seed_grid]                    # nearest first, seed itself first
            pos = np.minimum(np.searchsorted(cells, cand), len(cells) - 1)
            ok = cells[pos] == cand                        # candidate present on this day
            yield from members[pos[ok][:self.k]].tolist()

    def __len__(self):
        return self.n_groups * self.k


class SpatialRegularizationLoss(nn.Module):
    """
    Spatial regularization loss that encourages smooth predictions across nearby grids.
    
    This loss penalizes large differences between predictions of geographically
    nearby grid cells, encouraging the model to learn spatially coherent patterns.
    
    Args:
        grid_coords: Dict mapping grid_idx to (lat, lon) coordinates
        distance_threshold: Maximum distance (degrees) for grids to be considered neighbors
        weight: Weight for this loss component
    """
    
    def __init__(self, grid_coords: Dict[int, Tuple[float, float]], 
                 distance_threshold: float = 1.0,
                 weight: float = 0.1):
        super().__init__()
        self.grid_coords = grid_coords
        self.distance_threshold = distance_threshold
        self.weight = weight
        
        # Pre-compute neighbor pairs
        self.neighbor_pairs = self._compute_neighbor_pairs()
        logger.info(f"SpatialRegularizationLoss: {len(self.neighbor_pairs)} neighbor pairs, "
                   f"threshold={distance_threshold}°, weight={weight}")
    
    def _compute_neighbor_pairs(self) -> List[Tuple[int, int]]:
        """Compute pairs of neighboring grids based on distance threshold."""
        pairs = []
        grid_indices = list(self.grid_coords.keys())
        
        for i, idx1 in enumerate(grid_indices):
            lat1, lon1 = self.grid_coords[idx1]
            for idx2 in grid_indices[i+1:]:
                lat2, lon2 = self.grid_coords[idx2]
                
                # Simple Euclidean distance in degrees
                dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
                
                if dist <= self.distance_threshold:
                    pairs.append((idx1, idx2))
        
        return pairs
    
    def forward(self, predictions: torch.Tensor, grid_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial regularization loss.
        
        Args:
            predictions: (batch_size, n_vars) model predictions
            grid_indices: (batch_size,) grid index for each sample
            
        Returns:
            Scalar loss value
        """
        if len(self.neighbor_pairs) == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        total_loss = torch.tensor(0.0, device=predictions.device)
        n_pairs_found = 0
        
        for idx1, idx2 in self.neighbor_pairs:
            # Find samples from each grid in the batch
            mask1 = (grid_indices == idx1)
            mask2 = (grid_indices == idx2)
            
            if mask1.any() and mask2.any():
                # Get mean prediction for each grid in this batch
                pred1 = predictions[mask1].mean(dim=0)
                pred2 = predictions[mask2].mean(dim=0)
                
                # MSE between neighbor predictions (encourages smoothness)
                pair_loss = torch.mean((pred1 - pred2)**2)
                total_loss = total_loss + pair_loss
                n_pairs_found += 1
        
        if n_pairs_found > 0:
            total_loss = total_loss / n_pairs_found
        
        return self.weight * total_loss


def create_grid_balanced_dataloader(dataset, batch_size: int, shuffle: bool = True, 
                                    num_workers: int = 0, **kwargs):
    """
    Create a DataLoader with grid-balanced sampling.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        **kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader with GridBalancedSampler
    """
    from torch.utils.data import DataLoader
    
    sampler = GridBalancedSampler(dataset, batch_size, shuffle=shuffle)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        **kwargs
    )


def build_grid_coords_from_dataset(dataset) -> Dict[int, Tuple[float, float]]:
    """
    Extract grid coordinates from a dataset.
    
    Args:
        dataset: Dataset with grid coordinate information
        
    Returns:
        Dict mapping grid_idx to (lat, lon)
    """
    grid_coords = {}
    
    if hasattr(dataset, 'selected_lats_coords') and hasattr(dataset, 'selected_lons_coords'):
        lats = dataset.selected_lats_coords
        lons = dataset.selected_lons_coords
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            grid_coords[i] = (float(lat), float(lon))
    
    return grid_coords
