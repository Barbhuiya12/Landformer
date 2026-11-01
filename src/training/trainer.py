"""
Unified Training Pipeline for Hydrological Surrogate Models
Author: Siddik Barbhuiya
Description: Main training pipeline with physics constraints, metrics tracking, and grid-aware splits

This module implements the complete training pipeline for hydrological surrogate models,
integrating physics-informed constraints, hydrological metrics, and grid-aware data splits.

Key features of the training pipeline:
1. Multi-output model training with configurable loss weights
2. Physics-informed constraints (mass conservation)
3. Hydrological metrics tracking (KGE, NSE, correlation, bias)
4. Grid-aware train/validation/test splits for spatial generalization
5. Mixed precision training for efficiency
6. Gradient clipping for stability
7. Learning rate scheduling
8. Model checkpointing and early stopping
9. Comprehensive logging and TensorBoard integration
10. Memory monitoring and optimization

The pipeline is designed to be robust and efficient while ensuring physically consistent
model predictions through the integration of domain knowledge via constraints.
"""

import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import time
import json
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


from ..core.config import UnifiedConfig
from ..models.losses import LossFunction
from ..models.swin_transformer import SwinTransformerHydro, MultiTaskSwinTransformer
from ..models.tcn import UnifiedTCN, MultiTaskTCN, MultiTaskTCNSeparate
from ..models.lstm import MultiTaskLSTM
from ..models.physics_constraints import PhysicalConstraintLoss
from ..data.dataset import UnifiedSurrogateDataset
from ..data.grid_splitter import create_grid_aware_datasets
from ..utils.memory import MemoryMonitor, DataLoaderOptimizer, optimize_for_memory, cleanup_memory
from ..utils.hydro_metrics import HydrologicalMetricsTracker, calculate_all_metrics
from ..analysis.grid_analyzer import analyze_grid_locations
from ..analysis.epoch_diagnostics import save_epoch_diagnostics


class UnifiedTrainer:
    """
    Unified trainer for hydrological surrogate models with physics constraints
    Author: Siddik Barbhuiya
    
    This class orchestrates the complete training workflow for hydrological surrogate models.
    It integrates multiple components including data loading, model training, validation,
    physics constraints, metrics tracking, and result logging.
    
    The training process follows these key steps:
    1. Initialize configuration and set up logging
    2. Create grid-aware train/validation/test datasets
    3. Initialize model, optimizer, and loss functions
    4. For each epoch:
       a. Training phase with physics constraints
       b. Validation phase with metrics tracking
       c. Learning rate scheduling
       d. Model checkpointing based on validation performance
       e. Early stopping if no improvement
    5. Final test evaluation and reporting
    
    Attributes:
        config (UnifiedConfig): Configuration object
        logger (Logger): Logger instance
        memory_monitor (MemoryMonitor): Memory usage monitor
        physical_constraint_loss (PhysicalConstraintLoss): Physics constraints
        metrics_tracker (HydrologicalMetricsTracker): Metrics tracking
        multi_output_mode (bool): Whether training multi-output model
        training_history (dict): History of training metrics
    """
    
    def __init__(self, config: UnifiedConfig):
        """
        Initialize the unified trainer
        
        Args:
            config: UnifiedConfig object
        """
        self.config = config
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize memory monitoring
        self.memory_monitor = MemoryMonitor()
        self.memory_monitor.log_memory_usage("trainer_init")
        # Log configured memory-related flags
        mem_cfg = getattr(self.config.training, 'memory', {}) or {}
        self.logger.info(
            "DataLoader memory settings: "
            f"num_workers={mem_cfg.get('num_workers')}, pin_memory={mem_cfg.get('pin_memory')}, "
            f"persistent_workers={mem_cfg.get('persistent_workers')}, prefetch_factor={mem_cfg.get('prefetch_factor')}, "
            f"drop_last={mem_cfg.get('drop_last')}, timeout={mem_cfg.get('timeout')}, "
            f"non_blocking={mem_cfg.get('non_blocking', True)}"
        )
        mem_log_enabled = self.config.logging.get('enable_memory_logging', True)
        mem_log_every = self.config.logging.get('memory_log_every_n_batches', 50)
        self.logger.info(
            f"Memory logging: enabled={mem_log_enabled}, every_n_batches={mem_log_every}"
        )
        
        # === GPU OPTIMIZATIONS ===
        hw_cfg = getattr(self.config, 'hardware', {}) or {}
        
        # Enable cuDNN benchmark for faster convolutions (fixed input sizes)
        if hw_cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
            self.logger.info("cuDNN benchmark enabled (faster for fixed input sizes)")
        
        # Enable TF32 for faster matrix multiplications on Ampere GPUs
        if hw_cfg.get('tf32_mode', False):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("TensorFloat-32 (TF32) enabled (faster on Ampere+ GPUs)")
        
        # Store gradient checkpointing flag for model creation
        self.use_gradient_checkpointing = hw_cfg.get('gradient_checkpointing', False)
        if self.use_gradient_checkpointing:
            self.logger.info("Gradient checkpointing enabled (saves VRAM, slower forward)")
        
        # Initialize training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = float('-inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        # Initialize physical constraints if enabled
        self.physical_constraint_loss = None
        if hasattr(config.training, 'physical_constraints'):
            pcfg = config.training.physical_constraints or {}
            if pcfg.get('enabled', False):
                self.physical_constraint_loss = PhysicalConstraintLoss(config)
                # Check if any constraints were actually enabled
                if self.physical_constraint_loss.constraints:
                    self.logger.info(f"Physics constraints initialized with {len(self.physical_constraint_loss.constraints)} constraints")
                else:
                    self.physical_constraint_loss = None
        
        # Initialize spatial regularization loss if enabled
        self.spatial_reg_loss = None
        self.spatial_reg_weight = 0.0
        if hasattr(config.training, 'spatial_regularization'):
            sreg_cfg = config.training.spatial_regularization or {}
            if sreg_cfg.get('enabled', False):
                self.spatial_reg_weight = sreg_cfg.get('weight', 0.1)
                self.spatial_distance_threshold = sreg_cfg.get('distance_threshold', 1.0)
                self.logger.info(f"Spatial regularization enabled: weight={self.spatial_reg_weight}, "
                               f"distance_threshold={self.spatial_distance_threshold}°")
        
        # Initialize hydrological metrics tracker
        target_names = self.config.data.variables.get('target', [])
        self.metrics_tracker = HydrologicalMetricsTracker(variable_names=target_names)
        self.logger.info("Hydrological metrics tracker initialized")
        
        # Initialize multi-output tracking if enabled
        if (self.config.training.multi_output and 
            self.config.training.multi_output.get('enabled', False) and
            self.config.training.multi_output.get('individual_metrics', True)):
            
            target_names = self.config.data.variables['target']
            for target in target_names:
                self.training_history[f'train_{target}_loss'] = []
                self.training_history[f'val_{target}_loss'] = []
        
        # Setup tensorboard if enabled
        if self.config.logging.get('tensorboard', True):
            self.writer = SummaryWriter(log_dir=self.config.tensorb_dir)
        else:
            self.writer = None
        
        self.logger.info("Trainer initialized by Siddik")
        self.logger.info(f"Output directory: {self.config.out_dir}")
        
        # Check if multi-output is enabled
        self.is_multi_output = (self.config.training.multi_output and 
                               self.config.training.multi_output.get('enabled', False))
        if self.is_multi_output:
            self.logger.info(f"Multi-output mode enabled for targets: {self.config.data.variables['target']}")
        
        # Cached target normalization stats for denormalization
        self._denorm_cache = None

    def _convert_tensors_to_scalars(self, scenarios_set):
        """Convert tensor values in a set to Python scalars for sorting."""
        scenarios_list = []
        for scenario in scenarios_set:
            if torch.is_tensor(scenario):
                scenarios_list.append(scenario.item() if scenario.numel() == 1 else scenario.tolist())
            else:
                scenarios_list.append(scenario)
        return sorted(scenarios_list)
    
    def _get_target_denorm_stats(self, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch and cache target means/stds as tensors on the given device/dtype."""
        target_names = self.config.data.variables.get('target', [])
        # Return cache if matches device/dtype
        if self._denorm_cache is not None:
            mean_t, std_t = self._denorm_cache
            if mean_t.device == device and mean_t.dtype == dtype:
                return mean_t, std_t
        means = []
        stds = []
        for var in target_names:
            mean, std = self.train_dataset.get_normalization_stats(var)
            means.append(mean)
            stds.append(std)
        mean_t = torch.tensor(means, device=device, dtype=dtype).unsqueeze(0)
        std_t = torch.tensor(stds, device=device, dtype=dtype).unsqueeze(0)
        self._denorm_cache = (mean_t, std_t)
        return mean_t, std_t

    def _denormalize_targets(self, outputs: torch.Tensor, targets_last_step: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Denormalize predictions and targets to original scale using training stats."""
        mean_or_min_t, std_or_range_t = self._get_target_denorm_stats(outputs.device, outputs.dtype)
        
        # Get normalization method from train dataset
        normalization_method = getattr(self.train_dataset, 'normalization_method', 'zscore')
        
        if normalization_method == 'zscore':
            # Z-score denormalization: x_original = x_norm * std + mean
            denorm_outputs = outputs * std_or_range_t + mean_or_min_t
            denorm_targets = targets_last_step * std_or_range_t + mean_or_min_t
        elif normalization_method == 'minmax':
            # MinMax denormalization: x_original = x_norm * range + min
            denorm_outputs = outputs * std_or_range_t + mean_or_min_t
            denorm_targets = targets_last_step * std_or_range_t + mean_or_min_t
        elif normalization_method == 'log_zscore':
            # Log-Zscore denormalization: 
            # 1. Reverse z-score: x_log = x_norm * std + mean
            # 2. Reverse log1p: x_original = expm1(x_log)
            
            # Reverse z-score
            log_outputs = outputs * std_or_range_t + mean_or_min_t
            log_targets = targets_last_step * std_or_range_t + mean_or_min_t
            
            # Reverse log1p (expm1)
            denorm_outputs = torch.expm1(log_outputs)
            denorm_targets = torch.expm1(log_targets)
            
            # Ensure non-negative (physics constraint)
            denorm_outputs = torch.relu(denorm_outputs)
            denorm_targets = torch.relu(denorm_targets)
        elif normalization_method == 'quantile_gaussian':
            # Quantile-Gaussian denormalization using inverse_transform
            # Need to get the QuantileTransformer objects from training dataset
            import numpy as np
            
            target_names = self.config.data.variables['target']
            ny = outputs.shape[1]
            
            denorm_outputs_list = []
            denorm_targets_list = []
            
            outputs_np = outputs.detach().cpu().numpy()
            targets_np = targets_last_step.detach().cpu().numpy()
            
            for i in range(ny):
                var_name = target_names[i] if i < len(target_names) else f'output_{i}'
                qt_key = f"{var_name}_qt"
                
                # Get the QuantileTransformer from training stats
                qt = self.train_dataset.stat.get(qt_key)
                
                if qt is not None:
                    # Apply inverse transform
                    out_i = qt.inverse_transform(outputs_np[:, i:i+1]).flatten()
                    tgt_i = qt.inverse_transform(targets_np[:, i:i+1]).flatten()
                else:
                    # Fallback: no transformation (rare case)
                    self.logger.warning(f"QuantileTransformer not found for {var_name}, skipping inverse transform")
                    out_i = outputs_np[:, i]
                    tgt_i = targets_np[:, i]
                
                denorm_outputs_list.append(out_i)
                denorm_targets_list.append(tgt_i)
            
            # Stack and convert back to torch
            denorm_outputs = torch.from_numpy(np.stack(denorm_outputs_list, axis=1)).to(outputs.device, outputs.dtype)
            denorm_targets = torch.from_numpy(np.stack(denorm_targets_list, axis=1)).to(targets_last_step.device, targets_last_step.dtype)
            
            # Ensure non-negative for hydrological variables
            denorm_outputs = torch.relu(denorm_outputs)
            denorm_targets = torch.relu(denorm_targets)
        else:
            # Default to zscore
            denorm_outputs = outputs * std_or_range_t + mean_or_min_t
            denorm_targets = targets_last_step * std_or_range_t + mean_or_min_t
        
        # Ensure non-negative for hydrological variables (evap, Q, SM)
        # This is applied to all methods to ensure physical consistency
        denorm_outputs = torch.relu(denorm_outputs)
        denorm_targets = torch.relu(denorm_targets)
        
        return denorm_outputs, denorm_targets
    
    def _check_progressive_freezing(self, model: nn.Module, val_metrics: Dict, epoch: int) -> None:
        """
        Check if any task heads should be frozen based on their performance.
        Implements progressive head freezing strategy for multi-task learning.
        
        Args:
            model: The model with freeze_task_head method
            val_metrics: Validation metrics from current epoch
            epoch: Current epoch number
        """
        # Check if progressive freezing is enabled (use getattr for config object)
        pf_config = getattr(self.config.training, 'progressive_freezing', None)
        if pf_config is None or not getattr(pf_config, 'enabled', False):
            return
        
        # Check if model supports freezing
        if not hasattr(model, 'freeze_task_head'):
            return
        
        nse_threshold = getattr(pf_config, 'nse_threshold', 0.85)
        min_epochs = getattr(pf_config, 'min_epochs_before_freeze', 5)
        
        # Don't freeze in early epochs
        if epoch < min_epochs:
            return
        
        target_vars = self.config.data.variables.get('target', [])
        frozen_status = model.get_frozen_status() if hasattr(model, 'get_frozen_status') else {}
        
        for var_name in target_vars:
            # Skip if already frozen
            if frozen_status.get(var_name, False):
                continue
            
            # Get per-variable NSE from validation metrics
            nse_key = f'{var_name}_nse'
            var_nse = val_metrics.get(nse_key, None)
            
            # Also check for alt key format
            if var_nse is None:
                nse_key_alt = f'nse_{var_name}'
                var_nse = val_metrics.get(nse_key_alt, None)
            
            if var_nse is not None and var_nse > nse_threshold:
                # Freeze this task head
                success = model.freeze_task_head(var_name)
                if success:
                    self.logger.info(f"PROGRESSIVE FREEZING: Frozen {var_name} head (NSE={var_nse:.4f} > {nse_threshold})")
                    self.logger.info(f"   Remaining trainable heads: {[k for k, v in model.get_frozen_status().items() if not v]}")
    
    def _get_scheduled_sampling_prob(self, epoch: int, is_training: bool = True) -> float:
        """
        Get the probability of using ground truth lag for scheduled sampling.
        
        Args:
            epoch: Current epoch number
            is_training: Whether in training mode (False for validation/inference)
            
        Returns:
            Probability p_real ∈ [0, 1] of using ground truth lag
        """
        # Validation/inference: always use predicted (autoregressive)
        if not is_training:
            return 0.0
        
        # Check if scheduled sampling is enabled
        ss_config = getattr(self.config.training, 'scheduled_sampling', None)
        if ss_config is None or not getattr(ss_config, 'enabled', False):
            return 1.0  # Default: always use ground truth (teacher forcing)
        
        decay_epochs = getattr(ss_config, 'decay_epochs', 100)
        min_prob = getattr(ss_config, 'min_probability', 0.0)
        schedule_type = getattr(ss_config, 'schedule_type', 'linear')
        
        # Calculate probability based on schedule type
        if epoch >= decay_epochs:
            return min_prob
        
        progress = epoch / decay_epochs  # 0 → 1 over decay_epochs
        
        if schedule_type == 'linear':
            # Linear decay: p = 1 - progress
            p_real = 1.0 - progress
        elif schedule_type == 'exponential':
            # Exponential decay: p = exp(-5 * progress)
            import math
            p_real = math.exp(-5.0 * progress)
        elif schedule_type == 'inverse_sigmoid':
            # Inverse sigmoid: smoother transition
            import math
            k = 10  # Steepness
            p_real = 1.0 / (1.0 + math.exp(k * (progress - 0.5)))
        else:
            # Default to linear
            p_real = 1.0 - progress
        
        return max(min_prob, p_real)
    
    def _apply_scheduled_sampling(self, lag_features: torch.Tensor, 
                                   predictions: torch.Tensor, 
                                   epoch: int, 
                                   is_training: bool = True) -> torch.Tensor:
        """
        Apply scheduled sampling to lag features.
        Randomly chooses between ground truth and predicted lags based on epoch.
        
        Args:
            lag_features: Ground truth lagged target features (batch, seq_len, n_lag)
            predictions: Model predictions from previous step (batch, n_outputs)
            epoch: Current epoch
            is_training: Whether in training mode
            
        Returns:
            Mixed lag features with scheduled sampling applied
        """
        p_real = self._get_scheduled_sampling_prob(epoch, is_training)
        
        if p_real >= 1.0:
            # Always use ground truth
            return lag_features
        elif p_real <= 0.0:
            # Always use predictions (autoregressive)
            # Replace lag features with predictions
            return self._create_autoregressive_lags(lag_features, predictions)
        else:
            # Mixed: randomly choose per sample
            batch_size = lag_features.shape[0]
            mask = torch.rand(batch_size, 1, 1, device=lag_features.device) < p_real
            pred_lags = self._create_autoregressive_lags(lag_features, predictions)
            return torch.where(mask, lag_features, pred_lags)
    
    def _create_autoregressive_lags(self, lag_template: torch.Tensor, 
                                     predictions: torch.Tensor) -> torch.Tensor:
        """
        Create lag features from model predictions for autoregressive inference.
        
        Args:
            lag_template: Template for lag shape (batch, seq_len, n_lag)
            predictions: Model predictions (batch, n_outputs)
            
        Returns:
            Lag features created from predictions
        """
        # Scheduled sampling is not used for the models of the paper (teacher forcing): the lagged inputs are
        # returned unchanged. Free-running evaluation is implemented in analysis/closed_loop/closed_loop.py.
        return lag_template
    
    def _spatial_kwargs(self, model: nn.Module, metadata) -> Dict[str, torch.Tensor]:
        """
        lat, lon and graph key for models with spatial GAT; empty dict otherwise.
        The graph key (parameter-set file, day) makes the GAT connect only cells
        that share the same day and parameter set.
        """
        m = model.module if hasattr(model, 'module') else model
        if not getattr(m, 'uses_spatial_attention', False):
            return {}
        if not isinstance(metadata, dict) or 'lat' not in metadata:
            raise ValueError("Spatial GAT is enabled but the batch has no lat/lon metadata")
        dev = self.config.device
        key = metadata['file_iter_val'].long() * 1_000_000 + metadata['time_idx'].long()
        return {'lat': metadata['lat'].to(dev).float(),
                'lon': metadata['lon'].to(dev).float(),
                'graph_key': key.to(dev)}

    def _log_graph_stats(self, model: nn.Module, phase: str) -> None:
        m = model.module if hasattr(model, 'module') else model
        stats = m.pop_graph_stats() if hasattr(m, 'pop_graph_stats') else None
        if stats:
            self.logger.info(f"Spatial GAT ({phase}): {stats['mean_neighbours']:.1f} neighbours per cell on average, "
                             f"{stats['isolated_pct']:.1f}% of cells without a neighbour ({stats['nodes']:,} cells)")

    def _get_n_target_lags(self) -> int:
        """
        Get the number of target lag features from config.
        
        Returns:
            Number of target lag features
        """
        lag_config = getattr(self.config.data.processing, 'lag_features', None)
        if lag_config is None:
            return 0
        
        target_lags = getattr(lag_config, 'target', None) or {}
        if isinstance(target_lags, dict):
            # Count total target lags across all target variables
            total = 0
            for var_name, lags in target_lags.items():
                if isinstance(lags, (list, tuple)):
                    total += len(lags)
            return total
        return 0
    
    def calculate_individual_metrics(self, criterion, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Calculate individual output metrics for multi-output models
        
        Args:
            criterion: Loss function (should have get_individual_losses method)
            predictions: Model predictions (B, ny)
            targets: Ground truth targets (B, ny)
            
        Returns:
            Dictionary with individual output metrics
        """
        individual_metrics = {}
        
        if (self.is_multi_output and 
            hasattr(criterion, 'get_individual_losses')):
            
            # Get individual losses from the criterion
            individual_losses = criterion.get_individual_losses(predictions, targets)
            
            # Convert to metrics if needed
            for key, loss_val in individual_losses.items():
                # Convert NSE/KGE losses to metrics (1 - loss)
                if 'nse_' in key or 'kge_' in key:
                    metric_key = key.replace('nse_', '').replace('kge_', '') + '_loss'
                    individual_metrics[metric_key] = 1 - loss_val.item()
                else:
                    individual_metrics[key] = loss_val.item()
        
        return individual_metrics
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_level = getattr(logging, self.config.logging.get('level', 'INFO'))
        log_file = os.path.join(self.config.out_dir, self.config.logging.get('log_file', 'training.log'))
        
        # Clear existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        
        # Console handler  
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        
        # Configure root logger
        logging.root.setLevel(log_level)
        logging.root.addHandler(file_handler)
        logging.root.addHandler(console_handler)
    
    def create_datasets_and_loaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Create datasets and data loaders with grid-aware splits
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        self.logger.info("Creating datasets with grid-aware splits...")
        
        # Create grid splits if grid generalization is enabled
        if self.config.evaluation.get('grid_generalization', {}).get('enabled', True):
            splits = create_grid_aware_datasets(
                self.config,
                use_existing_splits=False,  # Force new splits to use updated YAML percentages
                create_stratified=False
            )
            
            # Create datasets with specific grid indices
            # Check for existing grid locations pickle to avoid subsetting issues
            train_pkl = os.path.join(self.config.out_dir, 'grid_locations_train.pkl')
            train_dataset = UnifiedSurrogateDataset(
                self.config, mode='train', 
                grid_indices=splits.get('train'),
                grid_locations_path=train_pkl if os.path.exists(train_pkl) else None
            )
            val_pkl = os.path.join(self.config.out_dir, 'grid_locations_valid.pkl')
            val_dataset = UnifiedSurrogateDataset(
                self.config, mode='valid',
                grid_indices=splits.get('val'),
                grid_locations_path=val_pkl if os.path.exists(val_pkl) else None
            )
            
            # Create test dataset if test files are configured
            # DISABLED during training for faster startup - create at inference time
            test_dataset = None
            # if (self.config.data.file_indices.get('test_files') and 
            #     self.config.data.temporal.get('test_range')):
            #     test_pkl = os.path.join(self.config.out_dir, 'grid_locations_test.pkl')
            #     test_dataset = UnifiedSurrogateDataset(
            #         self.config, mode='test',
            #         grid_indices=splits.get('test'),
            #         grid_locations_path=test_pkl if os.path.exists(test_pkl) else None
            #     )
        else:
            # Create datasets without grid splits
            train_dataset = UnifiedSurrogateDataset(self.config, mode='train')
            val_dataset = UnifiedSurrogateDataset(self.config, mode='valid')
            test_dataset = None
            
            if (self.config.data.file_indices.get('test_files') and 
                self.config.data.temporal.get('test_range')):
                test_dataset = UnifiedSurrogateDataset(self.config, mode='test')
        
        # Store dataset info
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        # Log scenario summaries to debug (not console)
        self.logger.debug("=" * 60)
        self.logger.debug("DATASET SCENARIO INFORMATION")
        self.logger.debug(f"Train dataset: {len(train_dataset)} samples from {train_dataset.get_scenario_info()['scenario_range']}")
        self.logger.debug(f"Validation dataset: {len(val_dataset)} samples from {val_dataset.get_scenario_info()['scenario_range']}")
        if test_dataset:
            self.logger.debug(f"Test dataset: {len(test_dataset)} samples from {test_dataset.get_scenario_info()['scenario_range']}")
        self.logger.debug("=" * 60)
        
        # Get memory-optimized DataLoader parameters
        optimal_params = DataLoaderOptimizer.get_optimal_params(
            self.config.training.batch_size, len(train_dataset)
        )
        
        # Override with user config if specified, but exclude non-DataLoader params
        memory_config = self.config.training.memory
        valid_dataloader_keys = {'num_workers', 'pin_memory', 'persistent_workers', 
                               'prefetch_factor', 'drop_last', 'timeout'}
        for key, value in memory_config.items():
            if value is not None and key in valid_dataloader_keys:
                optimal_params[key] = value
        
        self.logger.info(f"Using DataLoader params: {optimal_params}")
        
        # Check if grid-balanced sampling is enabled
        use_grid_balanced = getattr(self.config.training, 'grid_balanced_sampling', False)
        spatial_cfg = getattr(self.config.model, 'spatial_attention', None) or {}
        use_spatial_gat = (spatial_cfg.get('enabled', False) and
                           getattr(self.config.model, 'model_type', '').lower() in ['multi_task_swin', 'multi_task_swin_transformer'])

        if use_spatial_gat:
            # The GAT needs neighbouring cells of the same day in one batch. Each batch holds many
            # days with a cluster of nearby cells per day, so the batch-wise KGE still sees seasonal
            # variation as with random batches. Validation and test loaders are not shuffled and the
            # lookup table is ordered by day, so their batches already hold whole days.
            from .grid_aware import SpatialClusterSampler
            if use_grid_balanced:
                self.logger.warning("grid_balanced_sampling is ignored: spatial GAT uses SpatialClusterSampler")
            train_sampler = SpatialClusterSampler(train_dataset,
                                                  cells_per_group=int(spatial_cfg.get('cells_per_group', 10)),
                                                  seed=42)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.training.batch_size,
                sampler=train_sampler,
                **optimal_params
            )
        elif use_grid_balanced:
            from .grid_aware import GridBalancedSampler
            self.logger.info("Using grid-balanced sampling for training")
            train_sampler = GridBalancedSampler(
                train_dataset, 
                batch_size=self.config.training.batch_size,
                shuffle=True
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.training.batch_size,
                sampler=train_sampler,
                **optimal_params
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                **optimal_params
            )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size_valid,
            shuffle=False,
            **{k: v for k, v in optimal_params.items() if k != 'drop_last'}
        )
        
        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.training.batch_size_valid,
                shuffle=False,
                **{k: v for k, v in optimal_params.items() if k != 'drop_last'}
            )
        
        self.logger.info(f"Training batches per epoch: {len(train_loader)}")
        self.logger.info(f"Validation batches per epoch: {len(val_loader)}")
        if test_loader:
            self.logger.info(f"Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def create_model(self) -> nn.Module:
        """
        Create and initialize the model based on configuration
        
        Returns:
            Initialized model (supports multiple architectures)
        """
        model_type = getattr(self.config.model, 'model_type', 'unified').lower()
        self.logger.info(f"Creating model of type: {model_type}")
        
        # Get common parameters
        n_exo = self.train_dataset.n_exo
        ny = self.train_dataset.ny
        n_lag_features = self.train_dataset.n_lag_features
        hidden_size = getattr(self.config.model, 'hidden_size', 128)
        dropout = getattr(self.config.model, 'dropout', 0.3)
        num_layers = getattr(self.config.model, 'num_layers', 2)
        sequence_length = getattr(self.config.model, 'sequence_length', 270)
        
        # Task names for multi-task models
        task_names = self.config.data.variables.get('target', ['evap'])
        
        # Create model based on type
        if model_type in ['unified_tcn', 'tcn']:
            self.logger.info("Creating UnifiedTCN model...")
            hidden_channels = getattr(self.config.model, 'hidden_channels', [64, 128, 128, 64])
            kernel_size = getattr(self.config.model, 'kernel_size', 3)
            model = UnifiedTCN(
                n_exo=n_exo,
                ny=ny,
                n_lag_features=n_lag_features,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                dropout=dropout
            )
            
        elif model_type in ['multi_task_tcn']:
            self.logger.info("Creating MultiTaskTCN model...")
            hidden_channels = getattr(self.config.model, 'hidden_channels', [64, 128, 128, 64])
            kernel_size = getattr(self.config.model, 'kernel_size', 3)
            model = MultiTaskTCN(
                n_exo=n_exo,
                n_lag_features=n_lag_features,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                task_names=task_names
            )
            
        elif model_type in ['multi_task_tcn_separate']:
            self.logger.info("Creating MultiTaskTCNSeparate model...")
            n_forcing = len(self.config.data.variables.get('forcing', []))
            n_params = len(self.config.data.variables.get('parameters', []))
            hidden_channels = getattr(self.config.model, 'hidden_channels', [64, 128, 128, 64])
            kernel_size = getattr(self.config.model, 'kernel_size', 3)
            model = MultiTaskTCNSeparate(
                n_forcing=n_forcing,
                n_params=n_params,
                n_lag_features=n_lag_features,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                task_names=task_names
            )
            
        elif model_type in ['swin_transformer', 'swin']:
            self.logger.info("Creating SwinTransformerHydro model...")
            patch_size = getattr(self.config.model, 'patch_size', 10)
            embed_dim = getattr(self.config.model, 'embed_dim', 96)
            depths = getattr(self.config.model, 'depths', [2, 2, 6, 2])
            num_heads = getattr(self.config.model, 'num_heads', [3, 6, 12, 24])
            window_size = getattr(self.config.model, 'window_size', 7)
            drop_path_rate = getattr(self.config.model, 'drop_path_rate', 0.1)
            
            total_input_features = n_exo + n_lag_features
            model = SwinTransformerHydro(
                seq_len=sequence_length,
                patch_size=patch_size,
                in_chans=total_input_features,
                num_classes=ny,
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                dropout=dropout,
                drop_path_rate=drop_path_rate
            )
            
        elif model_type in ['multi_task_swin', 'multi_task_swin_transformer']:
            self.logger.info("Creating MultiTaskSwinTransformer model...")
            patch_size = getattr(self.config.model, 'patch_size', 10)
            embed_dim = getattr(self.config.model, 'embed_dim', 96)
            depths = getattr(self.config.model, 'depths', [2, 2, 6, 2])
            num_heads = getattr(self.config.model, 'num_heads', [3, 6, 12, 24])
            window_size = getattr(self.config.model, 'window_size', 7)
            drop_path_rate = getattr(self.config.model, 'drop_path_rate', 0.1)
            # Backward compatibility: use_hierarchical defaults to True (current behavior)
            # Set to False for legacy models with constant dimensions
            use_hierarchical = getattr(self.config.model, 'use_hierarchical', True)
            spatial_attention = getattr(self.config.model, 'spatial_attention', None)
            if spatial_attention and spatial_attention.get('enabled', False):
                self.logger.info(f"Spatial GAT ENABLED after the Swin encoder: {spatial_attention}")
            else:
                self.logger.info("Spatial GAT disabled")

            model = MultiTaskSwinTransformer(
                n_exo=n_exo,
                n_lag_features=n_lag_features,
                seq_len=sequence_length,
                patch_size=patch_size,
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                dropout=dropout,
                drop_path_rate=drop_path_rate,
                task_names=task_names,
                use_hierarchical=use_hierarchical,
                spatial_attention=spatial_attention
            )

        elif model_type in ['lstm', 'multi_task_lstm']:
            self.logger.info("Creating MultiTaskLSTM model...")
            model = MultiTaskLSTM(
                n_exo=n_exo,
                n_lag_features=n_lag_features,
                seq_len=sequence_length,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                task_names=task_names
            )
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: lstm, multi_task_lstm, unified_tcn, tcn, multi_task_tcn, "
                           f"multi_task_tcn_separate, swin_transformer, swin, multi_task_swin, multi_task_swin_transformer")
        
        # Move model to device
        model = model.to(self.config.device)
        
        # Log model information
        if hasattr(model, 'get_model_info'):
            self.logger.info(model.get_model_info())
        else:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f"Model created: {total_params:,} total parameters, {trainable_params:,} trainable")
        
        # Model compilation (PyTorch 2.0+)
        if self.config.hardware.get('compile_model', False):
            try:
                model = torch.compile(model)
                self.logger.info("Model compiled successfully")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
        
        return model

    def freeze_unfreeze_heads(self, model: nn.Module, finetune_head: str):
        """
        Freeze/unfreeze task-specific heads based on finetune_head configuration.
        Only applicable for multi-task models with task_heads (e.g., MultiTaskSwinTransformer).

        Args:
            model: The model to freeze/unfreeze parameters
            finetune_head: Head to fine-tune. Options:
                - "none": Train all heads (default behavior)
                - "SM", "Q", "ET", "evap": Fine-tune only specified head, freeze others
        """
        # Check if model has task_heads (multi-task models)
        if not hasattr(model, 'task_heads'):
            if finetune_head.lower() != "none":
                self.logger.warning(
                    f"Model does not have task_heads. finetune_head='{finetune_head}' will be ignored. "
                    f"This option only works with multi-task models."
                )
            return

        # Normalize finetune_head to match actual head names
        # Create a mapping of common input variations to actual head names
        head_mapping = {
            'none': 'none',
            'evap': 'evap',
            'et': 'evap',
            'ET': 'evap',
            'sm': 'sm',
            'SM': 'sm',
            'soil_moisture': 'sm',
            'soil_moisture_total': 'sm',
            'q': 'Q',
            'Q': 'Q',
            'runoff': 'Q',
        }

        normalized_head = head_mapping.get(finetune_head)
        if normalized_head is None:
            raise ValueError(
                f"Invalid finetune_head '{finetune_head}'. "
                f"Valid values are: 'none', 'evap', 'sm', 'Q' "
                f"(also accepts case-insensitive variants like 'q', 'et', 'ET', 'SM')"
            )

        finetune_head = normalized_head

        if finetune_head == "none":
            # Default: keep all heads trainable
            self.logger.info("Fine-tuning mode: DISABLED (all heads trainable)")
            return

        # Validate finetune_head value
        valid_heads = set(model.task_heads.keys())
        if finetune_head not in valid_heads:
            raise ValueError(
                f"Invalid finetune_head '{finetune_head}'. "
                f"Must be 'none', {', '.join(sorted(valid_heads))}"
            )

        # Freeze all heads except the one to fine-tune
        frozen_count = 0
        trainable_count = 0

        for head_name, head_module in model.task_heads.items():
            if head_name != finetune_head:
                # Freeze this head
                for param in head_module.parameters():
                    param.requires_grad = False
                frozen_count += sum(p.numel() for p in head_module.parameters())
            else:
                # Keep this head trainable
                for param in head_module.parameters():
                    param.requires_grad = True
                trainable_count += sum(p.numel() for p in head_module.parameters())

        # Freeze the shared backbone (important for true fine-tuning!)
        backbone_params = 0
        if hasattr(model, 'swin_backbone'):
            backbone = model.swin_backbone
            for param in backbone.parameters():
                param.requires_grad = False
            backbone_params = sum(p.numel() for p in backbone.parameters())
            frozen_count += backbone_params
            self.logger.info(
                f"Fine-tuning mode: HEAD='{finetune_head.upper()}'\n"
                f"  - Shared backbone FROZEN: {backbone_params:,} parameters\n"
                f"  - {finetune_head.upper()} head trainable: {trainable_count:,} parameters\n"
                f"  - Other heads frozen: {frozen_count - backbone_params:,} parameters\n"
                f"  - Total frozen: {frozen_count:,} parameters"
            )
        elif hasattr(model, 'lstm'):
            # For LSTM models
            lstm = model.lstm
            for param in lstm.parameters():
                param.requires_grad = False
            backbone_params = sum(p.numel() for p in lstm.parameters())
            frozen_count += backbone_params
            self.logger.info(
                f"Fine-tuning mode: HEAD='{finetune_head.upper()}'\n"
                f"  - Shared LSTM FROZEN: {backbone_params:,} parameters\n"
                f"  - {finetune_head.upper()} head trainable: {trainable_count:,} parameters\n"
                f"  - Other heads frozen: {frozen_count - backbone_params:,} parameters\n"
                f"  - Total frozen: {frozen_count:,} parameters"
            )
        else:
            self.logger.info(
                f"Fine-tuning mode: HEAD='{finetune_head.upper()}'\n"
                f"  - {finetune_head.upper()} head trainable: {trainable_count:,} parameters\n"
                f"  - Other heads frozen: {frozen_count:,} parameters"
            )

    def compute_head_specific_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                                   finetune_head: str, criterion: nn.Module,
                                   grid_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute loss focusing on the fine-tuned head while still outputting all variables.

        This method allows selective loss computation when fine-tuning a specific head,
        while the model still produces predictions for all heads.

        Args:
            predictions: Full model predictions (B, num_tasks)
            targets: Full target values (B, num_tasks)
            finetune_head: Head being fine-tuned ('none', 'SM', 'Q', 'ET', 'evap')
            criterion: Loss function
            grid_indices: Optional grid indices for grid-aware losses

        Returns:
            Computed loss (scalar tensor)
        """
        # Check if criterion is grid-aware (accepts grid_indices)
        is_grid_aware = hasattr(criterion, 'forward') and 'grid_indices' in str(criterion.forward.__code__.co_varnames)
        
        if finetune_head.lower() == "none":
            # Default: compute loss for all heads
            if is_grid_aware and grid_indices is not None:
                return criterion(predictions, targets, grid_indices)
            return criterion(predictions, targets)

        # Map head names to column indices dynamically from config
        target_names = self.config.data.variables.get('target', ['evap', 'soil_moisture_total', 'Q'])
        head_to_idx = {name.lower(): i for i, name in enumerate(target_names)}
        
        # Add common aliases
        if 'evap' in head_to_idx and 'et' not in head_to_idx:
            head_to_idx['et'] = head_to_idx['evap']
        if 'soil_moisture_total' in head_to_idx and 'sm' not in head_to_idx:
            head_to_idx['sm'] = head_to_idx['soil_moisture_total']
            
        finetune_head_lower = finetune_head.lower()

        if finetune_head_lower not in head_to_idx:
            self.logger.warning(
                f"Unknown head '{finetune_head}'. Computing loss for all heads instead."
            )
            if is_grid_aware and grid_indices is not None:
                return criterion(predictions, targets, grid_indices)
            return criterion(predictions, targets)

        # Extract predictions and targets for the target head
        head_idx = head_to_idx[finetune_head_lower]
        pred_target = predictions[:, head_idx:head_idx+1]  # Keep dimension for loss computation
        target_target = targets[:, head_idx:head_idx+1]

        # Compute loss only for the target head
        # This allows the frozen heads to maintain their learned representations
        if is_grid_aware and grid_indices is not None:
            loss = criterion(pred_target, target_target, grid_indices)
        else:
            loss = criterion(pred_target, target_target)

        self.logger.debug(
            f"Head-specific loss computed for '{finetune_head}' (index {head_idx})"
        )

        return loss

    def create_optimizer_and_criterion(self, model: nn.Module) -> Tuple[torch.optim.Optimizer, nn.Module, Optional[torch.optim.lr_scheduler._LRScheduler]]:
        """
        Create optimizer, loss criterion, and learning rate scheduler

        Args:
            model: The model to optimize

        Returns:
            Tuple of (optimizer, criterion, scheduler)
        """
        # Apply head freezing if finetune_head is specified
        finetune_head = self.config.training.finetune_head
        if finetune_head and finetune_head.lower() != "none":
            self.freeze_unfreeze_heads(model, finetune_head)

        # Create optimizer with only trainable parameters (requires_grad=True)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.config.training.learning_rate,
            weight_decay=1e-5  # Add small regularization
        )
        
        # Create learning rate scheduler if enabled
        scheduler = None
        if self.config.training.scheduler.get('enabled', False):
            scheduler_type = self.config.training.scheduler.get('type', 'cosine')
            warmup_epochs = self.config.training.scheduler.get('warmup_epochs', 10)
            min_lr = self.config.training.scheduler.get('min_lr', 1e-6)
            
            if scheduler_type == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.training.epochs - warmup_epochs,
                    eta_min=min_lr
                )
                self.logger.info(f"Created CosineAnnealingLR scheduler with min_lr={min_lr}")
            elif scheduler_type == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=30,
                    gamma=0.1
                )
                self.logger.info("Created StepLR scheduler")
            elif scheduler_type == 'reduce_on_plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=200,
                    min_lr=min_lr
                )
                self.logger.info("Created ReduceLROnPlateau scheduler")
        
        # Create loss criterion
        loss_kwargs = {
            'epsilon': 1e-8
        }
        
        # Add multi-output specific parameters
        if self.config.training.multi_output and self.config.training.multi_output.get('enabled', False):
            multi_config = self.config.training.multi_output
            loss_kwargs['weights'] = multi_config.get('loss_weights')
            loss_kwargs['target_names'] = self.config.data.variables['target']
            # Multi-output losses don't use reduce_spatial parameter
        else:
            # Only single-output losses use reduce_spatial
            loss_kwargs['reduce_spatial'] = True
        
        criterion = LossFunction.get_loss_function(
            self.config.training.loss_function,
            **loss_kwargs
        )
        
        # Check for Focal Loss configuration (Author: Siddik Barbhuiya)
        focal_config = getattr(self.config.model, 'focal_loss', None)
        self.use_focal_loss = False
        self.focal_loss_fn = None
        
        # Handle both dictionary (from YAML) and object access
        if focal_config:
            if isinstance(focal_config, dict):
                is_enabled = focal_config.get('enabled', False)
                focal_gamma = focal_config.get('gamma', 2.0)
            else:
                is_enabled = getattr(focal_config, 'enabled', False)
                focal_gamma = getattr(focal_config, 'gamma', 2.0)
            
            if is_enabled:
                from src.models.losses import FocalMSELoss
                self.focal_loss_fn = FocalMSELoss(gamma=focal_gamma)
                self.use_focal_loss = True
                self.focal_weight = 0.1  # Weight for focal loss component
                self.logger.info(f"Focal Loss ENABLED (gamma={focal_gamma}, weight={self.focal_weight})")
            else:
                self.logger.info("Focal Loss: DISABLED (enabled=false in config)")
        else:
            self.logger.info("Focal Loss: DISABLED (no config found)")
        
        self.logger.info(f"Optimizer: Adam with lr={self.config.training.learning_rate}")
        self.logger.info(f"Loss function: {self.config.training.loss_function}")
        
        return optimizer, criterion, scheduler
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss criterion
            
        Returns:
            Dictionary with training metrics
        """
        model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_samples = 0
        
        # Setup progress bar
        pbar = tqdm(train_loader, 
                   desc=f"Epoch {self.current_epoch}/{self.config.training.epochs} [Train]",
                   leave=False)
        
        # Mixed precision training
        use_amp = self.config.training.optimizations.get('use_amp', False) and torch.cuda.is_available()
        if not hasattr(self, '_grad_scaler'):
            self._grad_scaler = torch.amp.GradScaler('cuda') if use_amp else None
        scaler = self._grad_scaler
        
        # Track scenarios used in training
        scenarios_used = set()
        
        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch data - handle multiple return formats
            if len(batch_data) == 3:  # Standard format: input, target, metadata
                inputs, targets, metadata = batch_data
                # Track scenarios
                if isinstance(metadata, (list, tuple)):
                    for meta in metadata:
                        if isinstance(meta, dict) and 'scenario_target' in meta:
                            scenarios_used.add(meta['scenario_target'])
                elif isinstance(metadata, dict) and 'scenario_target' in metadata:
                    scenarios_used.add(metadata['scenario_target'])
            else:  # Legacy format without metadata
                inputs, targets = batch_data[:2]
                metadata = None
            del batch_data  # Free reference immediately
            
            # Move to device
            non_blocking = self.config.training.memory.get('non_blocking', True)
            inputs = inputs.to(self.config.device, non_blocking=non_blocking)
            targets = targets.to(self.config.device, non_blocking=non_blocking)
            
            optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if use_amp:
                with torch.amp.autocast('cuda'):
                    # Check for scheduled sampling
                    ss_config = getattr(self.config.training, 'scheduled_sampling', None)
                    use_scheduled_sampling = ss_config and getattr(ss_config, 'enabled', False)
                    
                    # lat/lon/graph key for the spatial GAT (empty when GAT is disabled)
                    spatial = self._spatial_kwargs(model, metadata)

                    if use_scheduled_sampling and hasattr(model, 'forward_autoregressive'):
                        p_teacher = self._get_scheduled_sampling_prob(self.current_epoch, is_training=True)
                        n_target_lags = self._get_n_target_lags()
                        outputs = model.forward_autoregressive(inputs, n_target_lags=n_target_lags, 
                                                                p_teacher_forcing=p_teacher,
                                                                **spatial)
                    else:
                        outputs = model(inputs, **spatial)
                    targets_last_step = targets[:, -1, :]  # Sequence-to-one prediction

                    # Denormalize for hydrological losses/metrics
                    denorm_outputs, denorm_targets = self._denormalize_targets(outputs, targets_last_step)

                    # Extract grid indices for grid-aware losses
                    grid_indices = None
                    if self.config.training.loss_function.upper() in ['GRID_NSE', 'GRID_KGE']:
                        if isinstance(metadata, dict) and 'grid_idx' in metadata:
                            grid_indices = metadata['grid_idx'].to(inputs.device)
                        elif isinstance(metadata, (list, tuple)):
                            grid_idx_list = [int(m.get('grid_idx', -1).item() if hasattr(m.get('grid_idx', -1), 'item') else m.get('grid_idx', -1)) for m in metadata if isinstance(m, dict)]
                            grid_indices = torch.tensor(grid_idx_list, device=inputs.device) if grid_idx_list else None

                    # Compute data loss (hydrological losses on denormalized scale)
                    # For quantile_gaussian, use normalized outputs to preserve gradients
                    finetune_head = self.config.training.finetune_head
                    normalization_method = getattr(self.train_dataset, 'normalization_method', 'zscore')
                    
                    if normalization_method == 'quantile_gaussian':
                        # Use normalized data for loss (preserves gradients)
                        # KGE/NSE are relative metrics, work on normalized data too
                        data_loss = self.compute_head_specific_loss(
                            outputs, targets_last_step, finetune_head, criterion
                        )
                    elif self.config.training.loss_function.upper() in ['NSE', 'KGE', 'BIAS_NSE', 'BIAS_CORRECTED_NSE', 'GRID_NSE', 'GRID_KGE']:
                        data_loss = self.compute_head_specific_loss(
                            denorm_outputs, denorm_targets, finetune_head, criterion, grid_indices
                        )
                    else:
                        data_loss = self.compute_head_specific_loss(
                            outputs, targets_last_step, finetune_head, criterion
                        )

                    loss = data_loss

                    # Add physical constraints if enabled
                    if self.physical_constraint_loss is not None:
                        # Create task_outputs dict for per-variable constraints
                        task_outputs = {}
                        target_vars = self.config.data.variables.get('target', [])
                        for var_idx, var_name in enumerate(target_vars):
                            if denorm_outputs.shape[-1] > var_idx:
                                task_outputs[var_name] = denorm_outputs[:, var_idx:var_idx+1]
                        
                        constraint_losses = self.physical_constraint_loss(
                            denorm_outputs, denorm_targets, inputs, task_outputs=task_outputs
                        )
                        loss = loss + constraint_losses['total_constraint']
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                
                # Gradient clipping for stability
                grad_clip = self.config.training.optimizations.get('gradient_clipping', 1.0)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                # Step optimizer with scaled gradients
                scaler.step(optimizer)
                scaler.update()
            else:
                # Check for scheduled sampling (non-AMP path)
                ss_config = getattr(self.config.training, 'scheduled_sampling', None)
                use_scheduled_sampling = ss_config and getattr(ss_config, 'enabled', False)
                
                # lat/lon/graph key for the spatial GAT (empty when GAT is disabled)
                spatial = self._spatial_kwargs(model, metadata)

                if use_scheduled_sampling and hasattr(model, 'forward_autoregressive'):
                    p_teacher = self._get_scheduled_sampling_prob(self.current_epoch, is_training=True)
                    n_target_lags = self._get_n_target_lags()
                    outputs = model.forward_autoregressive(inputs, n_target_lags=n_target_lags, 
                                                            p_teacher_forcing=p_teacher,
                                                            **spatial)
                else:
                    outputs = model(inputs, **spatial)
                targets_last_step = targets[:, -1, :]

                # Denormalize for hydrological losses/metrics
                denorm_outputs, denorm_targets = self._denormalize_targets(outputs, targets_last_step)

                # Extract grid indices for grid-aware losses
                grid_indices = None
                if self.config.training.loss_function.upper() in ['GRID_NSE', 'GRID_KGE']:
                    if isinstance(metadata, dict) and 'grid_idx' in metadata:
                        grid_indices = metadata['grid_idx'].to(inputs.device)
                    elif isinstance(metadata, (list, tuple)):
                        grid_idx_list = [int(m.get('grid_idx', -1).item() if hasattr(m.get('grid_idx', -1), 'item') else m.get('grid_idx', -1)) for m in metadata if isinstance(m, dict)]
                        grid_indices = torch.tensor(grid_idx_list, device=inputs.device) if grid_idx_list else None

                # Compute data loss (hydrological losses on denormalized scale)
                # For quantile_gaussian, use normalized outputs for consistency
                finetune_head = self.config.training.finetune_head
                normalization_method = getattr(self.train_dataset, 'normalization_method', 'zscore')
                
                if normalization_method == 'quantile_gaussian':
                    # Use normalized data for loss (preserves gradients)
                    data_loss = self.compute_head_specific_loss(
                        outputs, targets_last_step, finetune_head, criterion
                    )
                elif self.config.training.loss_function.upper() in ['NSE', 'KGE', 'BIAS_NSE', 'BIAS_CORRECTED_NSE', 'GRID_NSE', 'GRID_KGE']:
                    data_loss = self.compute_head_specific_loss(
                        denorm_outputs, denorm_targets, finetune_head, criterion, grid_indices
                    )
                else:
                    data_loss = self.compute_head_specific_loss(
                        outputs, targets_last_step, finetune_head, criterion
                    )

                loss = data_loss

                # Add physical constraints if enabled
                if self.physical_constraint_loss is not None:
                    # Create task_outputs dict for per-variable constraints
                    task_outputs = {}
                    target_vars = self.config.data.variables.get('target', [])
                    for var_idx, var_name in enumerate(target_vars):
                        if denorm_outputs.shape[-1] > var_idx:
                            task_outputs[var_name] = denorm_outputs[:, var_idx:var_idx+1]
                    
                    constraint_losses = self.physical_constraint_loss(
                        denorm_outputs, denorm_targets, inputs, task_outputs=task_outputs
                    )
                    loss = loss + constraint_losses['total_constraint']
                
                loss.backward()
                
                # Gradient clipping for stability
                grad_clip = self.config.training.optimizations.get('gradient_clipping', 1.0)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
            
            # Optional per-batch memory logging
            if self.config.logging.get('enable_memory_logging', True):
                every = int(self.config.logging.get('memory_log_every_n_batches', 50) or 0)
                if every > 0 and (batch_idx % every == 0):
                    self.memory_monitor.log_memory_usage(f"train_epoch{self.current_epoch}_batch{batch_idx}")
            
            # Update metrics
            batch_size = inputs.size(0)
            total_loss += loss.detach().item() * batch_size
            total_data_loss += data_loss.detach().item() * batch_size
            total_samples += batch_size
            
            # Update metrics tracker with denormalized values and grid indices
            # Also pass normalized values for normalized space metrics
            self.metrics_tracker.update(
                denorm_outputs.detach(), denorm_targets.detach(), grid_indices,
                y_pred_normalized=outputs.detach(), y_true_normalized=targets_last_step.detach()
            )
            
            # Update progress bar
            current_total_loss = loss.item()
            if self.config.training.loss_function.upper() in ['NSE', 'KGE']:
                current_metric = 1 - data_loss.item()
                pbar.set_postfix(loss=f"{current_total_loss:.4f}",
                                 metric=f"{current_metric:.4f}")
            else:
                pbar.set_postfix(loss=f"{current_total_loss:.4f}")
            
            # Free intermediate GPU tensors to prevent fragmentation
            del loss, data_loss, outputs, targets_last_step, denorm_outputs, denorm_targets
            if grid_indices is not None:
                del grid_indices
        
        avg_total_loss = total_loss / total_samples
        avg_data_loss = total_data_loss / total_samples if total_samples > 0 else 0.0
        avg_metric = 1 - avg_data_loss if self.config.training.loss_function.upper() in ['NSE', 'KGE'] else None
        
        pbar.close()
        
        # Calculate hydrological metrics
        hydro_metrics = self.metrics_tracker.compute()
        self.metrics_tracker.reset()
        
        # Trim memory monitor history to prevent unbounded growth
        if len(self.memory_monitor.memory_history) > 200:
            self.memory_monitor.memory_history = self.memory_monitor.memory_history[-100:]
        
        # Log scenarios used (commented out to reduce verbose output)
        # if scenarios_used:
        #     scenarios_list = self._convert_tensors_to_scalars(scenarios_used)
        #     self.logger.info(f"  Training scenarios: {scenarios_list}")
        
        # Combine loss and metrics
        metrics = {'loss': avg_total_loss, 'scenarios_used': self._convert_tensors_to_scalars(scenarios_used)}
        if avg_metric is not None:
            metrics['metric'] = avg_metric
        metrics.update(hydro_metrics)
        
        return metrics
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader,
                      criterion: nn.Module) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Args:
            model: Model to validate
            val_loader: Validation data loader
            criterion: Loss criterion
            
        Returns:
            Dictionary with validation metrics
        """
        model.eval()
        total_loss = 0.0
        total_data_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(val_loader,
                   desc=f"Epoch {self.current_epoch}/{self.config.training.epochs} [Val]",
                   leave=False)
        
        # Track scenarios used in validation
        scenarios_used = set()
        
        with torch.no_grad():
          for batch_idx, batch_data in enumerate(pbar):
                # Unpack batch data - handle multiple return formats
                if len(batch_data) == 3:  # Standard format: input, target, metadata
                    inputs, targets, metadata = batch_data
                    # Track scenarios
                    if isinstance(metadata, (list, tuple)):
                        for meta in metadata:
                            if isinstance(meta, dict) and 'scenario_target' in meta:
                                scenarios_used.add(meta['scenario_target'])
                    elif isinstance(metadata, dict) and 'scenario_target' in metadata:
                        scenarios_used.add(metadata['scenario_target'])
                else:  # Legacy format without metadata
                    inputs, targets = batch_data[:2]
                    metadata = None
                del batch_data  # Free reference immediately
                
                # Move to device
                non_blocking = self.config.training.memory.get('non_blocking', True)
                inputs = inputs.to(self.config.device, non_blocking=non_blocking)
                targets = targets.to(self.config.device, non_blocking=non_blocking)
                
                # Validation uses autoregressive mode (p_teacher=0) to match inference
                ss_config = getattr(self.config.training, 'scheduled_sampling', None)
                use_scheduled_sampling = ss_config and getattr(ss_config, 'enabled', False)
                
                if use_scheduled_sampling and hasattr(model, 'forward_autoregressive'):
                    n_target_lags = self._get_n_target_lags()
                    # Validation: always use p_teacher=0 (fully autoregressive)
                    outputs = model.forward_autoregressive(inputs, n_target_lags=n_target_lags, 
                                                            p_teacher_forcing=0.0,
                                                            **self._spatial_kwargs(model, metadata))
                else:
                    outputs = model(inputs, **self._spatial_kwargs(model, metadata))
                targets_last_step = targets[:, -1, :]

                # Denormalize for hydrological losses/metrics
                denorm_outputs, denorm_targets = self._denormalize_targets(outputs, targets_last_step)

                # Compute data loss (hydrological losses on denormalized scale)
                finetune_head = self.config.training.finetune_head
                if self.config.training.loss_function.upper() in ['NSE', 'KGE', 'BIAS_NSE', 'BIAS_CORRECTED_NSE']:
                    data_loss = self.compute_head_specific_loss(
                        denorm_outputs, denorm_targets, finetune_head, criterion
                    )
                else:
                    data_loss = self.compute_head_specific_loss(
                        outputs, targets_last_step, finetune_head, criterion
                    )

                loss = data_loss

                # Add physical constraints if enabled (for consistent validation)
                if self.physical_constraint_loss is not None:
                    # Create task_outputs dict for per-variable constraints
                    task_outputs = {}
                    target_vars = self.config.data.variables.get('target', [])
                    for var_idx, var_name in enumerate(target_vars):
                        if denorm_outputs.shape[-1] > var_idx:
                            task_outputs[var_name] = denorm_outputs[:, var_idx:var_idx+1]
                    
                    constraint_losses = self.physical_constraint_loss(
                        denorm_outputs, denorm_targets, inputs, task_outputs=task_outputs
                    )
                    loss = loss + constraint_losses['total_constraint']
                
                # Update metrics
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_data_loss += data_loss.item() * batch_size
                total_samples += batch_size
                
                # Extract grid indices from metadata for per-grid metrics
                grid_indices = None
                if metadata is not None:
                    if isinstance(metadata, (list, tuple)):
                        grid_idx_list = []
                        for m in metadata:
                            if isinstance(m, dict) and 'grid_idx' in m:
                                idx = m['grid_idx']
                                # Handle if it's a tensor
                                if hasattr(idx, 'item'):
                                    idx = idx.item()
                                grid_idx_list.append(int(idx))
                            else:
                                grid_idx_list.append(-1)
                        grid_indices = torch.tensor(grid_idx_list)
                    elif isinstance(metadata, dict) and 'grid_idx' in metadata:
                        idx = metadata['grid_idx']
                        # Handle if it's a tensor or list
                        if hasattr(idx, 'tolist'):
                            grid_indices = torch.tensor(idx.tolist() if hasattr(idx, 'tolist') else [idx.item()] * batch_size)
                        elif isinstance(idx, (list, tuple)):
                            grid_indices = torch.tensor(idx)
                        else:
                            grid_indices = torch.tensor([int(idx)] * batch_size)
                
                # Update metrics tracker with denormalized values and grid indices
                # Also pass normalized values for normalized space metrics
                self.metrics_tracker.update(
                    denorm_outputs.detach(), denorm_targets.detach(), grid_indices,
                    y_pred_normalized=outputs.detach(), y_true_normalized=targets_last_step.detach()
                )
                
                # Update progress bar
                if self.config.training.loss_function.upper() in ['NSE', 'KGE']:
                    current_metric = 1 - data_loss.item()
                    pbar.set_postfix(loss=f"{loss.item():.4f}",
                                   metric=f"{current_metric:.4f}")
                else:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                
                # Free intermediate GPU tensors
                del loss, data_loss, outputs, targets_last_step, denorm_outputs, denorm_targets
                if grid_indices is not None:
                    del grid_indices
                
                # Optional per-batch memory logging for validation
                if self.config.logging.get('enable_memory_logging', True):
                    every = int(self.config.logging.get('memory_log_every_n_batches', 50) or 0)
                    if every > 0 and (len(val_loader) > 0):
                        # approximate batch index via total_samples // batch_size
                        batch_idx = total_samples // max(1, batch_size)
                        if batch_idx % every == 0:
                            self.memory_monitor.log_memory_usage(f"val_epoch{self.current_epoch}_batch{batch_idx}")
        
          pbar.close()
        
        # Averages
        avg_total_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_data_loss = total_data_loss / total_samples if total_samples > 0 else 0.0
        avg_metric = 1 - avg_data_loss if self.config.training.loss_function.upper() in ['NSE', 'KGE'] else None
        
        # Hydrological metrics on denormalized values
        hydro_metrics = self.metrics_tracker.compute()
        
        # Generate and log detailed grid report (BEFORE reset)
        detailed_report = self.metrics_tracker.generate_detailed_report()
        self.logger.info(detailed_report)
        
        self.metrics_tracker.reset()
        
        # Trim memory monitor history to prevent unbounded growth
        if len(self.memory_monitor.memory_history) > 200:
            self.memory_monitor.memory_history = self.memory_monitor.memory_history[-100:]
        
        # Combine loss and metrics
        metrics = {'loss': avg_total_loss, 'scenarios_used': self._convert_tensors_to_scalars(scenarios_used)}
        if avg_metric is not None:
            metrics['metric'] = avg_metric
        metrics.update(hydro_metrics)
        
        return metrics
    
    def _run_epoch_diagnostics(self, model: nn.Module, val_loader: DataLoader, 
                               val_dataset: UnifiedSurrogateDataset, epoch: int):
        """
        Run epoch diagnostics to generate best/worst grid time series and spatial metric maps.
        
        Args:
            model: Model to evaluate
            val_loader: Validation data loader
            val_dataset: Validation dataset (for grid coordinates)
            epoch: Current epoch number
        """
        # Only run every N epochs to save time (configurable)
        diag_interval = self.config.logging.get('diagnostics_interval', 5)
        if epoch % diag_interval != 0 and epoch != self.config.training.epochs - 1:
            return
        
        self.logger.info(f"Running epoch diagnostics for epoch {epoch}...")
        
        # Get grid coordinates from validation dataset
        try:
            lats = np.array(val_dataset.selected_lats_coords)
            lons = np.array(val_dataset.selected_lons_coords)
        except AttributeError:
            self.logger.warning("Could not get grid coordinates from dataset, skipping diagnostics")
            return
        
        # Get predictions and targets from the metrics tracker
        # The tracker stores all predictions/targets from the last validation
        try:
            if hasattr(self.metrics_tracker, 'last_predictions') and hasattr(self.metrics_tracker, 'last_targets'):
                predictions = self.metrics_tracker.last_predictions
                targets = self.metrics_tracker.last_targets
                grid_indices = getattr(self.metrics_tracker, 'last_grid_indices', None)
            else:
                self.logger.warning("Metrics tracker does not have stored predictions, skipping diagnostics")
                return
        except Exception as e:
            self.logger.warning(f"Could not get predictions from metrics tracker: {e}")
            return
        
        if predictions is None or targets is None or len(predictions) == 0:
            self.logger.warning("No predictions available for diagnostics")
            return
        
        # Convert to numpy if needed
        if hasattr(predictions, 'cpu'):
            predictions = predictions.cpu().numpy()
        if hasattr(targets, 'cpu'):
            targets = targets.cpu().numpy()
        if grid_indices is not None and hasattr(grid_indices, 'cpu'):
            grid_indices = grid_indices.cpu().numpy()
        
        # Get unique grid indices and their lat/lon
        n_vars = predictions.shape[-1] if len(predictions.shape) > 1 else 1
        
        if grid_indices is not None and len(grid_indices) == len(predictions):
            # Group predictions by grid index
            unique_grids = np.unique(grid_indices)
            n_grids = len(unique_grids)
            
            self.logger.info(f"Diagnostics: {len(predictions)} samples, {n_grids} unique grids")
            
            # Create per-grid grouped data
            predictions_per_grid = []
            targets_per_grid = []
            grid_lats = []
            grid_lons = []
            
            for grid_idx in unique_grids:
                mask = grid_indices == grid_idx
                pred_grid = predictions[mask]
                targ_grid = targets[mask]
                predictions_per_grid.append(pred_grid)
                targets_per_grid.append(targ_grid)
                
                # Map grid_idx to lat/lon
                if grid_idx < len(lats):
                    grid_lats.append(lats[grid_idx])
                    grid_lons.append(lons[grid_idx])
                else:
                    grid_lats.append(np.nan)
                    grid_lons.append(np.nan)
            
            # Pad to same length for stacking
            max_time = max(p.shape[0] for p in predictions_per_grid)
            predictions_padded = np.full((n_grids, max_time, n_vars), np.nan)
            targets_padded = np.full((n_grids, max_time, n_vars), np.nan)
            
            for i, (pred, targ) in enumerate(zip(predictions_per_grid, targets_per_grid)):
                predictions_padded[i, :pred.shape[0], :] = pred
                targets_padded[i, :targ.shape[0], :] = targ
            
            predictions_reshaped = predictions_padded
            targets_reshaped = targets_padded
            lats = np.array(grid_lats)
            lons = np.array(grid_lons)
        else:
            # Fallback: no grid indices available, use sample-based approach
            n_grids = len(lats)
            n_samples = predictions.shape[0]
            self.logger.warning(f"No grid indices available. Using sample-based fallback: {n_samples} samples, {n_grids} grids")
            
            # Treat each sample as a single time point
            predictions_reshaped = predictions[:n_grids, np.newaxis, :]
            targets_reshaped = targets[:n_grids, np.newaxis, :]
            lats = lats[:n_grids]
            lons = lons[:n_grids]
        
        variable_names = self.config.data.variables.get('target', [])
        
        save_epoch_diagnostics(
            predictions_reshaped, targets_reshaped,
            lats, lons, variable_names, epoch, self.config.out_dir
        )
        
        self.logger.info(f"Epoch diagnostics saved to {self.config.out_dir}/epoch_plots/epoch_{epoch:03d}/")

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       is_best: bool = False, epoch: int = None):
        """
        Save model checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Optional scheduler state
            is_best: Whether this is the best model so far
            epoch: Current epoch (for periodic saves)
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
            'config': self.config,
            'training_history': self.training_history
        }
        
        # Add scheduler state if provided
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if is_best:
            save_path = os.path.join(self.config.out_model_dir, 'best_model.pth')
            torch.save(checkpoint, save_path)
            self.logger.info(f"Saved best model to {save_path}")
        
        if epoch is not None and epoch % self.config.training.epoch_save == 0:
            save_path = os.path.join(self.config.out_model_dir, f'model_epoch_{epoch}.pth')
            torch.save(checkpoint, save_path)
            self.logger.info(f"Saved periodic checkpoint to {save_path}")
    
    def load_pretrained_model(self, checkpoint_path: str, model: nn.Module):
        """
        Load pretrained model weights for fine-tuning (without loading optimizer or training state)

        Args:
            checkpoint_path: Path to the pretrained model checkpoint file
            model: Model to load weights into

        Returns:
            Loaded checkpoint dictionary
        """
        self.logger.info("=" * 60)
        self.logger.info("LOADING PRETRAINED MODEL FOR FINE-TUNING")
        self.logger.info(f"Checkpoint path: {checkpoint_path}")
        self.logger.info("=" * 60)

        # Store initial random weights for verification (sample from first layer)
        initial_weights = {}
        for name, param in list(model.named_parameters())[:3]:  # Sample first 3 parameters
            initial_weights[name] = param.data.clone().cpu()

        # Use weights_only=False for compatibility with PyTorch 2.6+
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
        except Exception as e:
            self.logger.error(f"Failed to load pretrained model: {e}")
            raise

        # Handle different checkpoint formats
        # Format 1: {'model_state_dict': OrderedDict, 'optimizer_state_dict': ..., 'epoch': ...}
        # Format 2: OrderedDict (raw state_dict saved directly)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Standard checkpoint format with metadata
            state_dict = checkpoint['model_state_dict']
            checkpoint_metadata = checkpoint
        else:
            # Raw state_dict format (no metadata)
            self.logger.info("Checkpoint is in raw state_dict format (no metadata)")
            state_dict = checkpoint
            checkpoint_metadata = {'epoch': 'unknown'}

        # Robust fix for torch.compile prefix (_orig_mod.)
        # Handle both directions: 
        # 1. Checkpoint has _orig_mod. but model doesn't (loading compiled ckpt to normal model)
        # 2. Model has _orig_mod. but checkpoint doesn't (loading normal ckpt to compiled model)
        model_keys = set(model.state_dict().keys())
        new_state_dict = {}
        
        for k, v in state_dict.items():
            if k in model_keys:
                # Exact match
                new_state_dict[k] = v
            elif f"_orig_mod.{k}" in model_keys:
                # Checkpoint missing prefix (Model is compiled)
                new_state_dict[f"_orig_mod.{k}"] = v
            elif k.startswith("_orig_mod.") and k[10:] in model_keys:
                # Checkpoint has extra prefix (Model is normal)
                new_state_dict[k[10:]] = v
            else:
                # No match found, keep original for reporting
                new_state_dict[k] = v
        
        state_dict = new_state_dict

        # Check if checkpoint is from multi-task model
        has_task_heads = any('task_heads' in key for key in state_dict.keys())

        if not has_task_heads:
            self.logger.warning(" Checkpoint appears to be from single-task model (no task_heads found)")
            self.logger.warning("   Fine-tuning may not work as expected")

        # Load model state
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            self.logger.info("PRE-TRAINED MODEL WEIGHTS LOADED SUCCESSFULLY")
            self.logger.info(f"  Original training epoch: {checkpoint_metadata.get('epoch', 'unknown')}")

            # Display pretrained model performance
            if 'best_val_loss' in checkpoint_metadata:
                self.logger.info(f"  Best validation loss: {checkpoint_metadata.get('best_val_loss'):.6f}")
            if 'best_val_metric' in checkpoint_metadata:
                metric_name = checkpoint_metadata.get('config', {}).get('training', {}).get('loss_function', 'metric')
                self.logger.info(f"  Best validation {metric_name}: {checkpoint_metadata.get('best_val_metric'):.6f}")

            # Verify weights actually changed (not random initialization)
            weights_changed = False
            for name, param in list(model.named_parameters())[:3]:
                if name in initial_weights:
                    diff = (param.data.cpu() - initial_weights[name]).abs().sum().item()
                    if diff > 1e-6:  # Tolerance for numerical precision
                        weights_changed = True
                        break

            if weights_changed:
                self.logger.info("VERIFICATION: Weights successfully loaded (different from random initialization)")
            else:
                self.logger.warning(" WARNING: Weights may not have been loaded properly!")
                self.logger.warning("   Model weights appear identical to random initialization")

            # Report any issues with key matching
            if missing_keys:
                self.logger.warning(f" Missing keys in checkpoint: {len(missing_keys)} parameters")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        self.logger.warning(f"     - {key}")
                else:
                    self.logger.warning(f"     First 5: {missing_keys[:5]}")

            if unexpected_keys:
                self.logger.warning(f" Unexpected keys in checkpoint: {len(unexpected_keys)} parameters")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        self.logger.warning(f"     - {key}")
                else:
                    self.logger.warning(f"     First 5: {unexpected_keys[:5]}")

            # Count loaded parameters
            total_params = sum(p.numel() for p in model.parameters())
            loaded_params = total_params - sum(p.numel() for name, p in model.named_parameters() if name in missing_keys)

            self.logger.info(f"  Parameters loaded: {loaded_params:,} / {total_params:,} ({100*loaded_params/total_params:.1f}%)")

            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"Failed to load model state: {e}")
            raise

        # Do NOT load optimizer, scheduler, or training state for fine-tuning
        # Training will start from epoch 1 with fresh optimizer state

        return checkpoint_metadata

    def load_checkpoint(self, checkpoint_path: str, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """
        Load checkpoint and resume training state

        Args:
            checkpoint_path: Path to the checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Optional scheduler to load state into

        Returns:
            Loaded checkpoint dictionary
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Store initial random weights for verification (sample from first layer)
        initial_weights = {}
        for name, param in list(model.named_parameters())[:3]:
            initial_weights[name] = param.data.clone().cpu()

        # Use weights_only=False for compatibility with PyTorch 2.6+
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Raw state_dict format
            state_dict = checkpoint
            checkpoint = {'model_state_dict': state_dict}

        # Robust fix for torch.compile prefix (_orig_mod.)
        model_keys = set(model.state_dict().keys())
        new_state_dict = {}
        
        for k, v in state_dict.items():
            if k in model_keys:
                # Exact match
                new_state_dict[k] = v
            elif f"_orig_mod.{k}" in model_keys:
                # Checkpoint missing prefix (Model is compiled)
                new_state_dict[f"_orig_mod.{k}"] = v
            elif k.startswith("_orig_mod.") and k[10:] in model_keys:
                # Checkpoint has extra prefix (Model is normal)
                new_state_dict[k[10:]] = v
            else:
                # No match found, keep original for reporting
                new_state_dict[k] = v
        
        state_dict = new_state_dict

        # Load model state (non-strict to allow partial loading)
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            self.logger.info("=" * 60)
            self.logger.info("PRE-TRAINED MODEL WEIGHTS LOADED SUCCESSFULLY")
            self.logger.info(f"  Checkpoint: {checkpoint_path}")
            self.logger.info(f"  Original training epoch: {checkpoint.get('epoch', 'unknown')}")
            best_val_loss = checkpoint.get('best_val_loss', 'N/A')
            if isinstance(best_val_loss, (int, float)):
                self.logger.info(f"  Best validation loss: {best_val_loss:.6f}")
            else:
                self.logger.info(f"  Best validation loss: {best_val_loss}")
            
            # Verify weights actually changed
            weights_changed = False
            for name, param in list(model.named_parameters())[:3]:
                if name in initial_weights:
                    diff = (param.data.cpu() - initial_weights[name]).abs().sum().item()
                    if diff > 1e-6:
                        weights_changed = True
                        break
            
            if weights_changed:
                self.logger.info("VERIFICATION: Weights successfully loaded (different from random initialization)")
            else:
                self.logger.warning(" WARNING: Weights may not have been loaded properly!")
                self.logger.warning("   Model weights appear identical to random initialization")
            
            # Report key matching issues
            if missing_keys:
                self.logger.warning(f" Missing keys in checkpoint: {len(missing_keys)} parameters")
            if unexpected_keys:
                self.logger.warning(f" Unexpected keys in checkpoint: {len(unexpected_keys)} parameters")
            
            self.logger.info("=" * 60)
        except Exception as e:
            self.logger.error(f"Failed to load model state: {e}")
            raise

        # Load optimizer state
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info("Optimizer state loaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load optimizer state: {e}")

        # Load scheduler state if available and provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.logger.info("Scheduler state loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load scheduler state: {e}")

        # Restore training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_metric = checkpoint.get('best_val_metric', float('-inf'))

        # Restore training history if available (merge with defaults to ensure all keys exist)
        if 'training_history' in checkpoint:
            saved_history = checkpoint['training_history']
            # Merge with default structure to ensure all keys exist
            default_history = {
                'train_loss': [],
                'val_loss': [],
                'train_metric': [],
                'val_metric': [],
                'epoch_times': [],
                'learning_rates': []
            }
            for key in default_history:
                if key in saved_history:
                    self.training_history[key] = saved_history[key]
            self.logger.info("Training history restored")

        self.logger.info(f"Resumed from epoch {self.current_epoch}")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        if self.config.training.loss_function.upper() in ['NSE', 'KGE']:
            self.logger.info(f"Best validation {self.config.training.loss_function}: {self.best_val_metric:.4f}")

        return checkpoint
    
    def log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch_time: float):
        """Log metrics to various outputs"""
        
        # Console logging
        log_msg = f"Epoch [{self.current_epoch}/{self.config.training.epochs}] - "
        log_msg += f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}"
        
        if 'metric' in train_metrics:
            log_msg += f", Train {self.config.training.loss_function}: {train_metrics['metric']:.4f}"
            log_msg += f", Val {self.config.training.loss_function}: {val_metrics['metric']:.4f}"
        
        # Check if using grid-aware loss
        use_grid_metrics = self.config.training.loss_function.upper() in ['GRID_NSE', 'GRID_KGE']
        
        if use_grid_metrics:
            # Show grid-wise metrics only
            if 'grid_nse_mean_of_vars' in val_metrics:
                log_msg += f" | Grid NSE: T={train_metrics.get('grid_nse_mean_of_vars', 0):.3f}/V={val_metrics.get('grid_nse_mean_of_vars', 0):.3f}"
            if 'grid_kge_mean_of_vars' in val_metrics:
                log_msg += f" | Grid KGE: T={train_metrics.get('grid_kge_mean_of_vars', 0):.3f}/V={val_metrics.get('grid_kge_mean_of_vars', 0):.3f}"
            
            # Show per-variable grid metrics
            target_names = self.config.data.variables.get('target', [])
            per_var_msgs = []
            for target in target_names:
                grid_nse_key = f"{target}_grid_nse"
                grid_kge_key = f"{target}_grid_kge"
                has_nse = grid_nse_key in train_metrics and grid_nse_key in val_metrics
                has_kge = grid_kge_key in train_metrics and grid_kge_key in val_metrics
                if has_nse or has_kge:
                    parts = []
                    if has_nse:
                        parts.append(f"NSE T={train_metrics[grid_nse_key]:.3f}/V={val_metrics[grid_nse_key]:.3f}")
                    if has_kge:
                        parts.append(f"KGE T={train_metrics[grid_kge_key]:.3f}/V={val_metrics[grid_kge_key]:.3f}")
                    per_var_msgs.append(f"{target}: " + ", ".join(parts))
            if per_var_msgs:
                log_msg += "\n     " + " | ".join(per_var_msgs)
        else:
            # Show overall metrics (traditional approach)
            if 'overall_nse_median' in train_metrics:
                log_msg += f" | NSE(median): T={train_metrics['overall_nse_median']:.3f}/V={val_metrics.get('overall_nse_median', 0):.3f}"
            elif 'overall_nse_mean' in train_metrics:
                log_msg += f" | NSE(mean): T={train_metrics['overall_nse_mean']:.3f}/V={val_metrics.get('overall_nse_mean', 0):.3f}"
            elif 'overall_nse' in train_metrics:
                log_msg += f" | NSE: T={train_metrics['overall_nse']:.3f}/V={val_metrics.get('overall_nse', 0):.3f}"
            elif 'nse' in train_metrics:
                log_msg += f" | NSE: T={train_metrics['nse']:.3f}/V={val_metrics.get('nse', 0):.3f}"
            
            if 'overall_kge_median' in train_metrics:
                log_msg += f" | KGE(median): T={train_metrics['overall_kge_median']:.3f}/V={val_metrics.get('overall_kge_median', 0):.3f}"
            elif 'overall_kge_mean' in train_metrics:
                log_msg += f" | KGE(mean): T={train_metrics['overall_kge_mean']:.3f}/V={val_metrics.get('overall_kge_mean', 0):.3f}"
            elif 'overall_kge' in train_metrics:
                log_msg += f" | KGE: T={train_metrics['overall_kge']:.3f}/V={val_metrics.get('overall_kge', 0):.3f}"
            elif 'kge' in train_metrics:
                log_msg += f" | KGE: T={train_metrics['kge']:.3f}/V={val_metrics.get('kge', 0):.3f}"
            
            # Add per-output NSE and KGE if multi-output is enabled OR if fine-tuning multi-task model
            show_individual_metrics = False
            if self.is_multi_output:
                show_individual_metrics = self.config.training.multi_output.get('individual_metrics', True)
            elif hasattr(self.config.training, 'multi_output') and self.config.training.multi_output:
                show_individual_metrics = self.config.training.multi_output.get('individual_metrics', True)

            # Also show during fine-tuning
            if hasattr(self.config.training, 'finetune_head'):
                finetune_head = getattr(self.config.training, 'finetune_head', 'none')
                if finetune_head and finetune_head.lower() != 'none':
                    show_individual_metrics = True

            if show_individual_metrics:
                target_names = self.config.data.variables['target']
                per_var_msgs = []
                for target in target_names:
                    nse_t_key, nse_v_key = f"{target}_nse", f"{target}_nse"
                    kge_t_key, kge_v_key = f"{target}_kge", f"{target}_kge"
                    has_nse = (nse_t_key in train_metrics) and (nse_v_key in val_metrics)
                    has_kge = (kge_t_key in train_metrics) and (kge_v_key in val_metrics)
                    if has_nse or has_kge:
                        parts = []
                        if has_nse:
                            parts.append(f"NSE T={train_metrics[nse_t_key]:.3f}/V={val_metrics[nse_v_key]:.3f}")
                        if has_kge:
                            parts.append(f"KGE T={train_metrics[kge_t_key]:.3f}/V={val_metrics[kge_v_key]:.3f}")
                        per_var_msgs.append(f"{target}: " + ", ".join(parts))
                if per_var_msgs:
                    log_msg += "\n     " + " | ".join(per_var_msgs)
        
        log_msg += f" - Time: {epoch_time:.2f}s"
        
        # Add scenario information
        train_scenarios = train_metrics.get('scenarios_used', [])
        val_scenarios = val_metrics.get('scenarios_used', [])
        if train_scenarios:
            log_msg += f" - Train Scenarios: {len(train_scenarios)}"
        if val_scenarios:
            log_msg += f", Val Scenarios: {len(val_scenarios)}"
        
        self.logger.info(log_msg)
        
        # Detailed scenario logging (commented out to reduce verbose output)
        # if train_scenarios:
        #     self.logger.debug(f"Detailed train scenarios used: {train_scenarios}")
        # if val_scenarios:
        #     self.logger.debug(f"Detailed validation scenarios used: {val_scenarios}")
        
        # Tensorboard logging
        if self.writer:
            self.writer.add_scalar('Loss/train', train_metrics['loss'], self.current_epoch)
            self.writer.add_scalar('Loss/validation', val_metrics['loss'], self.current_epoch)
            
            if 'metric' in train_metrics:
                self.writer.add_scalar(f'Metric/train_{self.config.training.loss_function}', 
                                     train_metrics['metric'], self.current_epoch)
                self.writer.add_scalar(f'Metric/val_{self.config.training.loss_function}', 
                                     val_metrics['metric'], self.current_epoch)
            
            self.writer.add_scalar('Time/epoch_duration', epoch_time, self.current_epoch)
            # Memory scalars
            usage = self.memory_monitor.get_memory_usage()
            self.writer.add_scalar('Memory/RSS_MB', usage.get('rss', 0.0), self.current_epoch)
            self.writer.add_scalar('Memory/RAM_Used_Percent', usage.get('percent', 0.0), self.current_epoch)
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    self.writer.add_scalar(f'Memory/GPU{i}_Allocated_MB', usage.get(f'gpu_{i}_allocated', 0.0), self.current_epoch)
                    self.writer.add_scalar(f'Memory/GPU{i}_Reserved_MB', usage.get(f'gpu_{i}_reserved', 0.0), self.current_epoch)
        
        # Update training history
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['val_loss'].append(val_metrics['loss'])
        self.training_history['epoch_times'].append(epoch_time)
        
        if 'metric' in train_metrics:
            self.training_history['train_metric'].append(train_metrics['metric'])
            self.training_history['val_metric'].append(val_metrics['metric'])
    
    def train(self, resume_checkpoint: str = None):
        """
        Main training loop
        
        Args:
            resume_checkpoint: Path to checkpoint file to resume from (optional)
        """
        self.logger.info("=" * 60)
        self.logger.info("UNIFIED VIC SURROGATE MODEL TRAINING")
        self.logger.info(f"Author: Siddik Barbhuiya")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # Create datasets and loaders
        with self.memory_monitor.monitor_block('create_datasets_and_loaders'):
            train_loader, val_loader, test_loader = self.create_datasets_and_loaders()
        self.memory_monitor.log_memory_usage('post_dataloader_creation')
        
        # Create model
        with self.memory_monitor.monitor_block('create_model'):
            model = self.create_model()
        self.memory_monitor.log_memory_usage('post_model_creation')
        
        # Load pretrained model if specified (for fine-tuning)
        pretrained_model_path = getattr(self.config.training, 'pretrained_model_path', None)

        # Check if pretrained model path is provided and valid
        # Handle both None (not set) and string 'null' (YAML null value)
        if pretrained_model_path and str(pretrained_model_path).lower() not in ['none', 'null', '']:
            self.logger.info(f"Pretrained model path detected: {pretrained_model_path}")
            if os.path.exists(pretrained_model_path):
                self.load_pretrained_model(pretrained_model_path, model)
                self.logger.info("Pretrained model successfully loaded for fine-tuning")
            else:
                self.logger.error(f"Pretrained model file not found: {pretrained_model_path}")
                self.logger.error(f"Please verify the path exists and is accessible")
                raise FileNotFoundError(f"Pretrained model file not found: {pretrained_model_path}")
        else:
            self.logger.info("No pretrained model specified - training from random initialization")

        # Create optimizer, criterion, and scheduler AFTER loading pretrained model
        # This ensures optimizer is created with the correct frozen/unfrozen parameters
        optimizer, criterion, scheduler = self.create_optimizer_and_criterion(model)
        self.memory_monitor.log_memory_usage('post_optimizer_creation')

        # Load checkpoint if resuming training (different from pretrained model loading)
        if resume_checkpoint is not None:
            if os.path.exists(resume_checkpoint):
                self.load_checkpoint(resume_checkpoint, model, optimizer, scheduler)
                self.logger.info(f"Resuming training from epoch {self.current_epoch + 1}")
            else:
                self.logger.error(f"Checkpoint file not found: {resume_checkpoint}")
                raise FileNotFoundError(f"Checkpoint file not found: {resume_checkpoint}")
        
        # Training loop
        # Memory optimization recommendations
        memory_recommendations = optimize_for_memory(self.config, model, len(train_loader.dataset))
        self.logger.info(f"Memory recommendations: {memory_recommendations}")

        # Log training mode
        finetune_head = getattr(self.config.training, 'finetune_head', 'none')
        if pretrained_model_path and finetune_head and finetune_head.lower() != 'none':
            self.logger.info("=" * 60)
            self.logger.info("FINE-TUNING MODE ACTIVE")
            self.logger.info(f"  Pretrained model: {pretrained_model_path}")
            self.logger.info(f"  Fine-tuning head: {finetune_head.upper()}")
            self.logger.info(f"  Starting from epoch: 1 (fresh training with frozen layers)")
            self.logger.info("=" * 60)
        elif resume_checkpoint:
            self.logger.info("=" * 60)
            self.logger.info("RESUMING TRAINING FROM CHECKPOINT")
            self.logger.info(f"  Starting from epoch: {self.current_epoch + 1}")
            self.logger.info(f"  Checkpoint loaded: {resume_checkpoint}")
            self.logger.info("=" * 60)

        self.logger.info("Starting training loop...")
        self.memory_monitor.log_memory_usage('before_training_loop')

        # Adjust epoch range if resuming from checkpoint
        start_epoch = self.current_epoch + 1 if resume_checkpoint else 1
        
        for epoch in range(start_epoch, self.config.training.epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            self.memory_monitor.log_memory_usage(f'epoch_{epoch}_start')
            
            # Log scheduled sampling probability if enabled
            ss_config = getattr(self.config.training, 'scheduled_sampling', None)
            if ss_config and getattr(ss_config, 'enabled', False):
                p_real = self._get_scheduled_sampling_prob(epoch, is_training=True)
                self.logger.info(f"Scheduled Sampling: p_real={p_real:.3f} (epoch {epoch})")
            
            # Train epoch
            with self.memory_monitor.monitor_block(f'epoch_{epoch}_train'):
                train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            self._log_graph_stats(model, 'train')
            
            # Aggressively free GPU memory before validation
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Validate epoch
            with self.memory_monitor.monitor_block(f'epoch_{epoch}_validate'):
                val_metrics = self.validate_epoch(model, val_loader, criterion)
            self._log_graph_stats(model, 'validation')
            
            # Update learning rate scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(f"  Learning rate: {current_lr:.2e}")
            
            epoch_time = time.time() - epoch_start_time
            
            # Log metrics
            self.log_metrics(train_metrics, val_metrics, epoch_time)
            
            # Run epoch diagnostics (best/worst grid plots, spatial maps)
            try:
                self._run_epoch_diagnostics(model, val_loader, self.val_dataset, epoch)
            except Exception as e:
                self.logger.warning(f"Epoch diagnostics failed: {e}")
            
            # Progressive head freezing: freeze task heads that have reached NSE threshold
            try:
                self._check_progressive_freezing(model, val_metrics, epoch)
            except Exception as e:
                self.logger.warning(f"Progressive freezing check failed: {e}")
            
            # Post-epoch memory and cleanup
            self.memory_monitor.log_memory_usage(f'epoch_{epoch}_end_pre_cleanup')
            cleanup_memory()
            self.memory_monitor.log_memory_usage(f'epoch_{epoch}_post_cleanup')
            
            # Check for improvement
            current_val_loss = val_metrics['loss']
            lf = self.config.training.loss_function.upper()
            if lf == 'NSE':
                # Prefer averaged per-output NSE when available (higher is better)
                if 'overall_nse_mean' in val_metrics:
                    current_val_metric = val_metrics['overall_nse_mean']
                elif 'overall_nse' in val_metrics:
                    current_val_metric = val_metrics['overall_nse']
                elif 'nse' in val_metrics:
                    current_val_metric = val_metrics['nse']
                else:
                    current_val_metric = val_metrics.get('metric', 1 - current_val_loss)
            elif lf == 'KGE':
                # Prefer MEDIAN per-output KGE (robust to outliers like Grid 55)
                if 'overall_kge_median' in val_metrics:
                    current_val_metric = val_metrics['overall_kge_median']
                elif 'kge_median' in val_metrics:
                    current_val_metric = val_metrics['kge_median']
                elif 'overall_kge_mean' in val_metrics:
                    current_val_metric = val_metrics['overall_kge_mean']
                elif 'overall_kge' in val_metrics:
                    current_val_metric = val_metrics['overall_kge']
                elif 'kge' in val_metrics:
                    current_val_metric = val_metrics['kge']
                else:
                    current_val_metric = val_metrics.get('metric', 1 - current_val_loss)
            elif lf == 'GRID_NSE':
                # Use MEDIAN grid-wise NSE (robust to outliers)
                if 'grid_nse_median' in val_metrics:
                    current_val_metric = val_metrics['grid_nse_median']
                elif 'grid_nse_mean_of_vars' in val_metrics:
                    current_val_metric = val_metrics['grid_nse_mean_of_vars']
                elif 'grid_nse_mean' in val_metrics:
                    current_val_metric = val_metrics['grid_nse_mean']
                else:
                    current_val_metric = val_metrics.get('metric', 1 - current_val_loss)
            elif lf == 'GRID_KGE':
                # Use MEDIAN grid-wise KGE (robust to outliers)
                if 'grid_kge_median' in val_metrics:
                    current_val_metric = val_metrics['grid_kge_median']
                elif 'grid_kge_mean_of_vars' in val_metrics:
                    current_val_metric = val_metrics['grid_kge_mean_of_vars']
                elif 'grid_kge_mean' in val_metrics:
                    current_val_metric = val_metrics['grid_kge_mean']
                else:
                    current_val_metric = val_metrics.get('metric', 1 - current_val_loss)
            else:
                # For loss functions where lower is better (e.g., MSE/MAE), use negative loss for comparison
                current_val_metric = -current_val_loss

            
            improved = False
            if self.config.training.loss_function.upper() in ['NSE', 'KGE', 'GRID_NSE', 'GRID_KGE']:
                # Higher is better for NSE/KGE and grid-aware variants
                if current_val_metric > self.best_val_metric:
                    self.best_val_metric = current_val_metric
                    self.best_val_loss = current_val_loss
                    improved = True
            else:
                # Lower is better for MSE/MAE
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    improved = True
            
            # Save checkpoints
            if improved:
                self.patience_counter = 0
                self.save_checkpoint(model, optimizer, scheduler, is_best=True)
                self.logger.info(f"  -> Validation performance improved!")
            else:
                self.patience_counter += 1
                self.logger.info(f"  -> No improvement. Patience: {self.patience_counter}/{self.config.training.patience}")
            
            # Periodic save
            self.save_checkpoint(model, optimizer, scheduler, epoch=epoch)
            
            # Early stopping
            if self.patience_counter >= self.config.training.patience:
                self.logger.info("Early stopping triggered.")
                break
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info(f"Total training time: {total_time:.2f} seconds")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        if self.config.training.loss_function.upper() in ['NSE', 'KGE']:
            self.logger.info(f"Best validation {self.config.training.loss_function}: {self.best_val_metric:.4f}")
        # Final memory summary
        self.logger.info(self.memory_monitor.get_memory_summary())
        
        # Save training history
        history_file = os.path.join(self.config.out_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        self.logger.info(f"Training history saved to {history_file}")
        
        # Test evaluation if test data is available
        if test_loader:
            self.logger.info("Evaluating on test set...")
            self.evaluate_test(model, test_loader, criterion)
        
        # Create comprehensive grid analysis report
        try:
            self.logger.info("Creating comprehensive grid analysis report...")
            grid_report = analyze_grid_locations(
                self.config.out_dir, 
                create_plots=self.config.output.create_plots
            )
            self.logger.info("Grid analysis report completed successfully")
            
            # Print summary to console
            print("\n" + "=" * 60)
            print("GRID LOCATION SUMMARY")
            print("=" * 60)
            print(f"Grid analysis files saved to: {self.config.out_dir}")
            print("Files created:")
            print("  - grid_locations_train.csv")
            print("  - grid_locations_valid.csv") 
            print("  - grid_locations_test.csv (if available)")
            print("  - grid_locations_train.nc")
            print("  - grid_locations_valid.nc")
            print("  - grid_locations_test.nc (if available)")
            print("  - grid_analysis_report.txt")
            print("  - grid_summary_table.csv")
            if self.config.output.create_plots:
                print("  - all_modes_grid_locations.png")
                print("  - train_grid_locations.png")
                print("  - valid_grid_locations.png")
                print("  - spatial_histograms.png")
                print("  - grid_indices_analysis.png")
            print("=" * 60)
            
        except Exception as e:
            self.logger.warning(f"Could not create grid analysis report: {e}")
        
        # Cleanup
        if self.writer:
            self.writer.close()
        
        self.logger.info("=" * 60)
    
    def evaluate_test(self, model: nn.Module, test_loader: DataLoader, criterion: nn.Module):
        """Evaluate model on test set"""
        model.eval()
        total_loss = 0.0
        total_samples = 0
        
        # Initialize metrics tracker for test set
        test_metrics_tracker = HydrologicalMetricsTracker(variable_names=self.config.data.variables.get('target', []))

        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Testing", leave=False):
                # Unpack batch data - handle multiple return formats
                if len(batch_data) == 3:  # Standard format: input, target, metadata
                    inputs, targets, metadata = batch_data
                else:  # Legacy format without metadata
                    inputs, targets = batch_data[:2]
                    metadata = None
                del batch_data  # Free reference immediately
                
                non_blocking = self.config.training.memory.get('non_blocking', True)
                inputs = inputs.to(self.config.device, non_blocking=non_blocking)
                targets = targets.to(self.config.device, non_blocking=non_blocking)
                
                outputs = model(inputs, **self._spatial_kwargs(model, metadata))
                targets_last_step = targets[:, -1, :]

                # Denormalize for hydrological losses/metrics
                denorm_outputs, denorm_targets = self._denormalize_targets(outputs, targets_last_step)

                # Compute data loss (hydrological losses on denormalized scale)
                if self.config.training.loss_function.upper() in ['NSE', 'KGE']:
                    loss = criterion(denorm_outputs, denorm_targets)
                else:
                    loss = criterion(outputs, targets_last_step)                
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Update metrics tracker with denormalized values
                test_metrics_tracker.update(denorm_outputs.detach(), denorm_targets.detach())
        
        avg_test_loss = total_loss / total_samples
        avg_test_metric = 1 - avg_test_loss if self.config.training.loss_function.upper() in ['NSE', 'KGE'] else None
        
        # Compute final hydrological metrics
        hydro_metrics = test_metrics_tracker.compute()

        self.logger.info("=" * 60)
        self.logger.info("TEST SET EVALUATION")
        self.logger.info(f"Test Loss: {avg_test_loss:.4f}")
        if avg_test_metric is not None:
            self.logger.info(f"Test {self.config.training.loss_function}: {avg_test_metric:.4f}")

        for metric, value in hydro_metrics.items():
            self.logger.info(f"Test {metric}: {value:.4f}")
        self.logger.info("=" * 60)


def main():
    """Main training function"""
    # This will be called from the main script
    pass


if __name__ == '__main__':
    main()