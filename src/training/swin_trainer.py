"""
Unified Training Pipeline for Hydrological Surrogate Models
Author: Siddik
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
from ..models.lstm import UnifiedLSTM
from ..models.losses import LossFunction
from ..models.Multi_head_lstm import MultiTaskLSTM
from ..models.multi_task_lstm_final import MultiTaskVICLSTM
from ..models.swin_transformer import SwinTransformerHydro, MultiTaskSwinTransformer
from ..models.tcn import UnifiedTCN, MultiTaskTCN, MultiTaskTCNSeparate
from ..models.physics_constraints import PhysicalConstraintLoss
from ..data.dataset import UnifiedSurrogateDataset
from ..data.grid_splitter import create_grid_aware_datasets
from ..utils.memory import MemoryMonitor, DataLoaderOptimizer, optimize_for_memory, cleanup_memory
from ..utils.hydro_metrics import HydrologicalMetricsTracker, calculate_all_metrics
from ..analysis.grid_analyzer import analyze_grid_locations


class UnifiedTrainer:
    """
    Unified trainer for hydrological surrogate models with physics constraints
    Author: Siddik
    
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
        else:
            # Default to zscore
            denorm_outputs = outputs * std_or_range_t + mean_or_min_t
            denorm_targets = targets_last_step * std_or_range_t + mean_or_min_t
        
        return denorm_outputs, denorm_targets
    
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
            train_dataset = UnifiedSurrogateDataset(
                self.config, mode='train', 
                grid_indices=splits.get('train')
            )
            val_dataset = UnifiedSurrogateDataset(
                self.config, mode='valid',
                grid_indices=splits.get('val')
            )
            
            # Create test dataset if test files are configured
            test_dataset = None
            if (self.config.data.file_indices.get('test_files') and 
                self.config.data.temporal.get('test_range')):
                test_dataset = UnifiedSurrogateDataset(
                    self.config, mode='test',
                    grid_indices=splits.get('test')
                )
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
        if model_type in ['unified', 'unified_lstm']:
            self.logger.info("Creating UnifiedLSTM model...")
            model = UnifiedLSTM(
                n_exo=n_exo,
                ny=ny,
                lags=getattr(self.config.model, 'lags', [1]),  # Legacy compatibility
                n_lag_features=n_lag_features,
                hidden_size=hidden_size,
                dropout=dropout,
                num_layers=num_layers,
                lag_config=self.config.get_lag_configuration() if hasattr(self.config, 'get_lag_configuration') else None
            )
            
        elif model_type in ['multi_task_lstm', 'multitask_lstm']:
            self.logger.info("Creating MultiTaskLSTM model (separate encoders)...")
            # Calculate separate input dimensions
            n_forcing = len(self.config.data.variables.get('forcing', []))
            n_params = len(self.config.data.variables.get('parameters', []))
            model = MultiTaskLSTM(
                n_forcing=n_forcing,
                n_params=n_params,
                n_lag_features=n_lag_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                task_names=task_names
            )
            
        elif model_type in ['multi_task', 'multi_task_vic_lstm']:
            self.logger.info("Creating MultiTaskVICLSTM model...")
            # Get SM range for context-aware expert
            sm_range = {'mean': 0.0}  # Default, should be computed from data
            if 'sm' in task_names:
                try:
                    sm_idx = task_names.index('sm')
                    sm_mean, _ = self.train_dataset.get_normalization_stats(f'target_{sm_idx}')
                    sm_range = {'mean': float(sm_mean)}
                except:
                    self.logger.warning("Could not get SM statistics, using default mean=0.0")
            
            model = MultiTaskVICLSTM(
                n_exo=n_exo,
                n_lag_features=n_lag_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                task_names=task_names,
                sm_range=sm_range
            )
            
        elif model_type in ['unified_tcn', 'tcn']:
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
                task_names=task_names
            )
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: unified, unified_lstm, multi_task_lstm, multi_task, "
                           f"multi_task_vic_lstm, unified_tcn, tcn, multi_task_tcn, multi_task_tcn_separate, "
                           f"swin_transformer, swin, multi_task_swin, multi_task_swin_transformer")
        
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
    
    def create_optimizer_and_criterion(self, model: nn.Module) -> Tuple[torch.optim.Optimizer, nn.Module, Optional[torch.optim.lr_scheduler._LRScheduler]]:
        """
        Create optimizer, loss criterion, and learning rate scheduler
        
        Args:
            model: The model to optimize
            
        Returns:
            Tuple of (optimizer, criterion, scheduler)
        """
        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
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
        
        self.logger.info(f"Optimizer: Adam with lr={self.config.training.learning_rate}")
        self.logger.info(f"Loss function: {self.config.training.loss_function}")
        
        return optimizer, criterion, scheduler
    
    def train_epoch(self, model: UnifiedLSTM, train_loader: DataLoader,
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
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Track scenarios used in training
        scenarios_used = set()
        
        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch data - handle multiple return formats
            if len(batch_data) == 6:  # Full format: forcing, params, lags, combined_input, targets, metadata
                forcing_seq, param_vec, lag_features, inputs, targets, metadata = batch_data
                # Track scenarios
                if isinstance(metadata, (list, tuple)):
                    for meta in metadata:
                        if isinstance(meta, dict) and 'scenario_target' in meta:
                            scenarios_used.add(meta['scenario_target'])
                elif isinstance(metadata, dict) and 'scenario_target' in metadata:
                    scenarios_used.add(metadata['scenario_target'])
            elif len(batch_data) == 3:  # Legacy format with metadata
                inputs, targets, metadata = batch_data
                # Track scenarios
                if isinstance(metadata, (list, tuple)):
                    for meta in metadata:
                        if isinstance(meta, dict) and 'scenario_target' in meta:
                            scenarios_used.add(meta['scenario_target'])
                elif isinstance(metadata, dict) and 'scenario_target' in metadata:
                    scenarios_used.add(metadata['scenario_target'])
            else:  # Very legacy format without metadata
                inputs, targets = batch_data
            
            # Move to device
            non_blocking = self.config.training.memory.get('non_blocking', True)
            inputs = inputs.to(self.config.device, non_blocking=non_blocking)
            targets = targets.to(self.config.device, non_blocking=non_blocking)
            
            optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    targets_last_step = targets[:, -1, :]  # Sequence-to-one prediction

                    # Denormalize for hydrological losses/metrics
                    denorm_outputs, denorm_targets = self._denormalize_targets(outputs, targets_last_step)

                    # Compute data loss (hydrological losses on denormalized scale)
                    if self.config.training.loss_function.upper() in ['NSE', 'KGE']:
                        data_loss = criterion(denorm_outputs, denorm_targets)
                    else:
                        data_loss = criterion(outputs, targets_last_step)

                    loss = data_loss

                    # Add physical constraints if enabled
                    if self.physical_constraint_loss is not None:
                        constraint_losses = self.physical_constraint_loss(
                            outputs, targets_last_step, inputs
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
                outputs = model(inputs)
                targets_last_step = targets[:, -1, :]

                # Denormalize for hydrological losses/metrics
                denorm_outputs, denorm_targets = self._denormalize_targets(outputs, targets_last_step)

                # Compute data loss (hydrological losses on denormalized scale)
                if self.config.training.loss_function.upper() in ['NSE', 'KGE']:
                    data_loss = criterion(denorm_outputs, denorm_targets)
                else:
                    data_loss = criterion(outputs, targets_last_step)

                loss = data_loss
                
                # Add physical constraints if enabled
                if self.physical_constraint_loss is not None:
                    constraint_losses = self.physical_constraint_loss(
                        outputs, targets_last_step, inputs
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
            
            # Update metrics tracker with denormalized values
            self.metrics_tracker.update(denorm_outputs.detach(), denorm_targets.detach())
            
            # Update progress bar
            current_total_loss = loss.item()
            if self.config.training.loss_function.upper() in ['NSE', 'KGE']:
                current_metric = 1 - data_loss.item()
                pbar.set_postfix(loss=f"{current_total_loss:.4f}",
                                 metric=f"{current_metric:.4f}")
            else:
                pbar.set_postfix(loss=f"{current_total_loss:.4f}")
        
        avg_total_loss = total_loss / total_samples
        avg_data_loss = total_data_loss / total_samples if total_samples > 0 else 0.0
        avg_metric = 1 - avg_data_loss if self.config.training.loss_function.upper() in ['NSE', 'KGE'] else None
        
        pbar.close()
        
        # Calculate hydrological metrics
        hydro_metrics = self.metrics_tracker.compute()
        self.metrics_tracker.reset()
        
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
    
    def validate_epoch(self, model: UnifiedLSTM, val_loader: DataLoader,
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
        
        for batch_idx, batch_data in enumerate(pbar):
                # Unpack batch data - handle multiple return formats
                if len(batch_data) == 6:  # Full format: forcing, params, lags, combined_input, targets, metadata
                    forcing_seq, param_vec, lag_features, inputs, targets, metadata = batch_data
                    # Track scenarios
                    if isinstance(metadata, (list, tuple)):
                        for meta in metadata:
                            if isinstance(meta, dict) and 'scenario_target' in meta:
                                scenarios_used.add(meta['scenario_target'])
                    elif isinstance(metadata, dict) and 'scenario_target' in metadata:
                        scenarios_used.add(metadata['scenario_target'])
                elif len(batch_data) == 3:  # Legacy format with metadata
                    inputs, targets, metadata = batch_data
                    # Track scenarios
                    if isinstance(metadata, (list, tuple)):
                        for meta in metadata:
                            if isinstance(meta, dict) and 'scenario_target' in meta:
                                scenarios_used.add(meta['scenario_target'])
                    elif isinstance(metadata, dict) and 'scenario_target' in metadata:
                        scenarios_used.add(metadata['scenario_target'])
                else:  # Very legacy format without metadata
                    inputs, targets = batch_data
                
                # Move to device
                non_blocking = self.config.training.memory.get('non_blocking', True)
                inputs = inputs.to(self.config.device, non_blocking=non_blocking)
                targets = targets.to(self.config.device, non_blocking=non_blocking)
                
                outputs = model(inputs)
                targets_last_step = targets[:, -1, :]

                # Denormalize for hydrological losses/metrics
                denorm_outputs, denorm_targets = self._denormalize_targets(outputs, targets_last_step)

                # Compute data loss (hydrological losses on denormalized scale)
                if self.config.training.loss_function.upper() in ['NSE', 'KGE']:
                    data_loss = criterion(denorm_outputs, denorm_targets)
                else:
                    data_loss = criterion(outputs, targets_last_step)

                loss = data_loss
                
                # Add physical constraints if enabled (for consistent validation)
                if self.physical_constraint_loss is not None:
                    constraint_losses = self.physical_constraint_loss(
                        outputs, targets_last_step, inputs
                    )
                    loss = loss + constraint_losses['total_constraint']
                
                # Update metrics
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_data_loss += data_loss.item() * batch_size
                total_samples += batch_size
                
                # Update metrics tracker with denormalized values
                self.metrics_tracker.update(denorm_outputs.detach(), denorm_targets.detach())
                
                # Update progress bar
                if self.config.training.loss_function.upper() in ['NSE', 'KGE']:
                    current_metric = 1 - data_loss.item()
                    pbar.set_postfix(loss=f"{loss.item():.4f}",
                                   metric=f"{current_metric:.4f}")
                else:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                
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
        self.metrics_tracker.reset()
        
        # Combine loss and metrics
        metrics = {'loss': avg_total_loss, 'scenarios_used': self._convert_tensors_to_scalars(scenarios_used)}
        if avg_metric is not None:
            metrics['metric'] = avg_metric
        metrics.update(hydro_metrics)
        
        return metrics
    def save_checkpoint(self, model: UnifiedLSTM, optimizer: torch.optim.Optimizer,
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
        
        # Use weights_only=False for compatibility with PyTorch 2.6+
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        # Load model state
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Model state loaded successfully")
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
        
        # Restore training history if available
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
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
        
        # Add NSE and KGE metrics to console output (prefer averaged NSE across outputs)
        if 'overall_nse_mean' in train_metrics:
            log_msg += f" | NSE(mean): T={train_metrics['overall_nse_mean']:.3f}/V={val_metrics.get('overall_nse_mean', 0):.3f}"
        elif 'overall_nse' in train_metrics:
            log_msg += f" | NSE: T={train_metrics['overall_nse']:.3f}/V={val_metrics.get('overall_nse', 0):.3f}"
        elif 'nse' in train_metrics:
            log_msg += f" | NSE: T={train_metrics['nse']:.3f}/V={val_metrics.get('nse', 0):.3f}"
        
        if 'overall_kge_mean' in train_metrics:
            log_msg += f" | KGE(mean): T={train_metrics['overall_kge_mean']:.3f}/V={val_metrics.get('overall_kge_mean', 0):.3f}"
        elif 'overall_kge' in train_metrics:
            log_msg += f" | KGE: T={train_metrics['overall_kge']:.3f}/V={val_metrics.get('overall_kge', 0):.3f}"
        elif 'kge' in train_metrics:
            log_msg += f" | KGE: T={train_metrics['kge']:.3f}/V={val_metrics.get('kge', 0):.3f}"
        
        # Add per-output NSE and KGE if multi-output is enabled
        if self.is_multi_output and self.config.training.multi_output.get('individual_metrics', True):
            target_names = self.config.data.variables['target']
            # Compact per-variable summary: {var}: NSE T/V, KGE T/V
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
                log_msg += " | " + " | ".join(per_var_msgs)
        
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
        self.logger.info(f"Author: Siddik")
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
        
        # Create optimizer, criterion, and scheduler
        optimizer, criterion, scheduler = self.create_optimizer_and_criterion(model)
        self.memory_monitor.log_memory_usage('post_optimizer_creation')
        
        # Load checkpoint if resuming training
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
        
        self.logger.info("Starting training loop...")
        self.memory_monitor.log_memory_usage('before_training_loop')
        
        # Adjust epoch range if resuming from checkpoint
        start_epoch = self.current_epoch + 1 if resume_checkpoint else 1
        
        for epoch in range(start_epoch, self.config.training.epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            self.memory_monitor.log_memory_usage(f'epoch_{epoch}_start')
            
            # Train epoch
            with self.memory_monitor.monitor_block(f'epoch_{epoch}_train'):
                train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # Validate epoch
            with self.memory_monitor.monitor_block(f'epoch_{epoch}_validate'):
                val_metrics = self.validate_epoch(model, val_loader, criterion)
            
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
                # Prefer averaged per-output KGE when available (higher is better)
                if 'overall_kge_mean' in val_metrics:
                    current_val_metric = val_metrics['overall_kge_mean']
                elif 'overall_kge' in val_metrics:
                    current_val_metric = val_metrics['overall_kge']
                elif 'kge' in val_metrics:
                    current_val_metric = val_metrics['kge']
                else:
                    current_val_metric = val_metrics.get('metric', 1 - current_val_loss)
            else:
                # For loss functions where lower is better (e.g., MSE/MAE), use negative loss for comparison
                current_val_metric = -current_val_loss
            
            improved = False
            if self.config.training.loss_function.upper() in ['NSE', 'KGE']:
                # Higher is better for NSE/KGE
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
    
    def evaluate_test(self, model: UnifiedLSTM, test_loader: DataLoader, criterion: nn.Module):
        """Evaluate model on test set"""
        model.eval()
        total_loss = 0.0
        total_samples = 0
        
        # Initialize metrics tracker for test set
        test_metrics_tracker = HydrologicalMetricsTracker(variable_names=self.config.data.variables.get('target', []))

        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Testing", leave=False):
                # Unpack batch data - handle multiple return formats
                if len(batch_data) == 6:  # Full format: forcing, params, lags, combined_input, targets, metadata
                    forcing_seq, param_vec, lag_features, inputs, targets, metadata = batch_data
                elif len(batch_data) == 3:  # Legacy format with metadata
                    inputs, targets, metadata = batch_data
                else:  # Very legacy format without metadata
                    inputs, targets = batch_data
                
                non_blocking = self.config.training.memory.get('non_blocking', True)
                inputs = inputs.to(self.config.device, non_blocking=non_blocking)
                targets = targets.to(self.config.device, non_blocking=non_blocking)
                
                outputs = model(inputs)
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