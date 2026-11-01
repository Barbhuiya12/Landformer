"""
Unified Configuration Loader for VIC Surrogate Model
Author: Siddik Barbhuiya
Description: Loads and manages configuration from YAML file

This module implements a unified configuration system for the VIC surrogate model
that provides type-safe access to all model parameters, data paths, training settings,
and hardware configurations.

The configuration system uses YAML files as the primary source of truth and provides:
1. Type-safe configuration objects using dataclasses
2. Automatic directory structure setup
3. Device configuration with automatic CUDA detection
4. Backward compatibility with legacy config formats
5. Configuration saving and loading capabilities

The system is organized into modular configuration sections:
- ModelConfig: Neural network architecture parameters
- DataConfig: Data paths, variables, and processing settings
- TrainingConfig: Training hyperparameters and optimization settings
- OutputConfig: Experiment output and logging settings
- Hardware and logging configurations

This approach ensures consistent configuration management across all components
of the VIC surrogate modeling pipeline.
"""

import yaml
import os
import torch
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model architecture configuration
    
    Configuration for all supported model architectures (LSTM, Swin Transformer, TCN).
    
    Attributes:
        # Common parameters
        model_type (str): Model architecture type ('unified', 'multi_task_swin', etc.)
        dropout (float): Dropout rate for regularization (0.0 to 1.0)
        sequence_length (int): Length of input sequences
        
        # LSTM-specific parameters
        hidden_size (int): Number of hidden units in LSTM layers
        num_layers (int): Number of LSTM layers
        enable_lags (bool): Whether to use lag features at all
        lag_config (Dict[str, Dict[str, List[int]]]): Advanced lag configuration per variable
        lags (List[int], optional): Legacy lag values for backward compatibility
        multi_task (Dict[str, Any]): Multi-task specific configuration
        
        # Swin Transformer parameters
        embed_dim (int): Embedding dimension for Swin Transformer
        depths (List[int]): Depths for each Swin stage
        num_heads (List[int]): Number of attention heads for each stage
        window_size (int): Window size for Swin attention
        patch_size (int): Patch size for input tokenization
        drop_path_rate (float): Drop path rate for regularization
        
        # TCN parameters
        hidden_channels (List[int]): Hidden channels for TCN layers
        kernel_size (int): Kernel size for TCN convolutions
    """
    # Common parameters (required)
    model_type: str
    dropout: float
    sequence_length: int
    
    # LSTM parameters (with defaults)
    hidden_size: int = 128
    num_layers: int = 1
    enable_lags: bool = True
    lag_config: Dict[str, Dict[str, List[int]]] = None
    lags: List[int] = None  # Legacy support
    multi_task: Dict[str, Any] = None  # Multi-task configuration
    
    # Swin Transformer parameters (with defaults)
    embed_dim: int = 96
    depths: List[int] = None  # Will default to [2, 2, 6, 2]
    num_heads: List[int] = None  # Will default to [3, 6, 12, 24]
    window_size: int = 7
    patch_size: int = 10
    drop_path_rate: float = 0.1
    
    # TCN parameters (with defaults)
    hidden_channels: List[int] = None  # Will default to [64, 128, 128, 64]
    kernel_size: int = 3
    
    # Focal Loss configuration (Author: Siddik Barbhuiya)
    focal_loss: Dict[str, Any] = None

    # Spatial graph attention (GAT) after the Swin encoder; multi_task_swin only.
    # Keys: enabled (bool), max_distance (deg), num_heads (int), dropout (float)
    spatial_attention: Dict[str, Any] = None


@dataclass
class DataConfig:
    """Data configuration

    Configuration for data loading, processing, and sampling.

    Attributes:
        spatial (Dict[str, Any]): Spatial domain boundaries and sampling
        temporal (Dict[str, List[str]]): Time ranges for train/val/test
        file_indices (Dict[str, List[int]]): File ranges for different scenarios
        grid_sampling (Dict[str, Any]): Grid cell sampling strategy and parameters
            - exclude_previous_grids (Dict[str, Any]): Configuration for excluding grids from previous runs
                - enabled (bool): Whether to enable grid exclusion
                - previous_runs (List[str]): List of paths to grid_locations_*.pkl files from previous runs
        variables (Dict[str, List[str]]): Input/output variable names
        paths (Dict[str, str]): File paths for data sources
        processing (Dict[str, Any]): Data processing options
        cache_size_limit (int): Maximum number of items in the data cache
    """
    spatial: Dict[str, Any]
    temporal: Dict[str, List[str]]
    file_indices: Dict[str, List[int]]
    grid_sampling: Dict[str, Any]
    variables: Dict[str, List[str]]
    paths: Dict[str, str]
    processing: Dict[str, Any] = None
    cache_size_limit: int = 5


@dataclass
class TrainingConfig:
    """Training configuration

    Configuration for model training hyperparameters and optimization.

    Attributes:
        epochs (int): Maximum number of training epochs
        batch_size (int): Training batch size
        learning_rate (float): Initial learning rate
        loss_function (str): Primary loss function ('MSE', 'NSE', 'KGE', etc.)
        memory (Dict[str, Any]): Memory optimization settings
        batch_size_valid (int): Validation batch size
        patience (int): Early stopping patience
        epoch_save (int): Save model every N epochs
        optimizations (Dict[str, Any]): Data processing optimizations
        multi_output (Dict[str, Any]): Multi-output training settings
        physical_constraints (Dict[str, Any]): Physics-informed constraints
        scheduler (Dict[str, Any]): Learning rate scheduler settings
        multi_task (Dict[str, Any]): Multi-task specific training settings
        finetune_head (str): Head to fine-tune for multi-task models. Allowed values:
            - "none" (default): Train all heads normally
            - "SM": Fine-tune only SM head, freeze Q and ET heads
            - "Q": Fine-tune only Q head, freeze SM and ET heads
            - "ET": Fine-tune only ET head, freeze SM and Q heads
            - "evap": Fine-tune only evap head, freeze sm and Q heads
        pretrained_model_path (str): Path to pretrained model checkpoint for fine-tuning.
            If provided, loads model weights (but not optimizer state) before training.
            Use with finetune_head to train specific heads on new data.
    """
    epochs: int
    batch_size: int
    learning_rate: float
    loss_function: str
    memory: Dict[str, Any]
    batch_size_valid: int = 256
    patience: int = 10
    epoch_save: int = 10
    optimizations: Dict[str, Any] = None
    multi_output: Dict[str, Any] = None
    physical_constraints: Dict[str, Any] = None
    scheduler: Dict[str, Any] = None
    multi_task: Dict[str, Any] = None  # Multi-task configuration
    finetune_head: str = "none"  # Fine-tune specific head: "none", "SM", "Q", "ET", "evap"
    pretrained_model_path: str = None  # Path to pretrained model for fine-tuning
    resume_training: bool = False  # Whether to resume training from a checkpoint
    restart_path: str = None  # Specific checkpoint path to resume from

@dataclass
class PhysicalConstraintsConfig:
    enabled: bool
    mass_conservation: Dict[str, Any]
    water_balance: Dict[str, Any]
    bounds: Dict[str, Dict[str, float]]


@dataclass
class OutputConfig:
    """Output configuration
    
    Configuration for experiment output management.
    
    Attributes:
        base_dir (str): Base directory for all experiment outputs
        experiment_name (str): Name of the current experiment
        save_predictions (bool): Whether to save model predictions (optional)
        save_metrics (bool): Whether to save evaluation metrics (optional)
        create_plots (bool): Whether to generate visualization plots (optional)
        multi_task (Dict[str, Any]): Multi-task specific output settings (optional)
    """
    base_dir: str
    experiment_name: str
    save_predictions: bool = True
    save_metrics: bool = True
    create_plots: bool = True
    multi_task: Dict[str, Any] = None  # Multi-task configuration


@dataclass
class UnifiedConfig:
    """Unified configuration container
    
    Main configuration container that combines all configuration sections.
    Automatically computes derived properties like output directories and device settings.
    
    Attributes:
        model (ModelConfig): Model architecture configuration
        data (DataConfig): Data loading and processing configuration
        training (TrainingConfig): Training hyperparameters and settings
        output (OutputConfig): Experiment output configuration
        hardware (Dict[str, Any]): Hardware and device settings
        logging (Dict[str, Any]): Logging configuration
        evaluation (Dict[str, Any]): Evaluation and testing settings
        device (str): Computed device string ('cuda:0', 'cpu', etc.)
        out_dir (str): Computed experiment output directory
        out_model_dir (str): Computed model save directory
        tensorb_dir (str): Computed TensorBoard log directory
        figure_dir (str): Computed figure output directory
    """
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    output: OutputConfig
    hardware: Dict[str, Any]
    logging: Dict[str, Any]
    evaluation: Dict[str, Any]
    
    # Computed properties
    device: str = None
    input_dir: str = None
    out_dir: str = None
    out_model_dir: str = None
    tensorb_dir: str = None
    figure_dir: str = None

    def __post_init__(self):
        """Setup computed properties after initialization"""
        self._setup_device()
        self._setup_directories()
    
    def _setup_device(self):
        """Setup device configuration"""
        device_config = self.hardware.get('device', 'auto')
        if device_config == 'auto':
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device_config
    
    def _setup_directories(self):
        """Setup output directories"""
        base_dir = self.output.base_dir
        exp_name = self.output.experiment_name
        
        self.out_dir = os.path.join(base_dir, exp_name)
        self.out_model_dir = os.path.join(self.out_dir, 'models')
        self.tensorb_dir = os.path.join(self.out_dir, 'tensorboard')
        self.figure_dir = os.path.join(self.out_dir, 'figures')
        
        # Create directories
        for directory in [self.out_dir, self.out_model_dir, self.tensorb_dir, self.figure_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def get_lag_configuration(self) -> Dict[str, Dict[str, List[int]]]:
        """
        Get processed lag configuration with backward compatibility
        
        Returns:
            Dictionary with lag configuration for each variable type:
            {
                'forcing': {'var_name': [lag1, lag2, ...]},
                'parameters': {'var_name': [lag1, lag2, ...]},  
                'target': {'var_name': [lag1, lag2, ...]}
            }
        """
        if not self.model.enable_lags:
            # Return empty lag configuration
            return {
                'forcing': {},
                'parameters': {},
                'target': {}
            }
        
        # Use new lag_config if available
        if self.model.lag_config is not None:
            # Ensure all categories exist
            lag_config = {
                'forcing': self.model.lag_config.get('forcing', {}),
                'parameters': self.model.lag_config.get('parameters', {}),
                'target': self.model.lag_config.get('target', {})
            }
            return lag_config
        
        # Backward compatibility: use legacy lags for target variables only
        elif self.model.lags is not None:
            target_lags = {}
            for var in self.data.variables['target']:
                target_lags[var] = self.model.lags
            
            return {
                'forcing': {},
                'parameters': {},
                'target': target_lags
            }
        
        # No lags configured
        return {
            'forcing': {},
            'parameters': {},
            'target': {}
        }
    
    def get_total_lag_features(self) -> int:
        """
        Calculate total number of lag features
        
        Returns:
            Total number of lag feature dimensions
        """
        lag_config = self.get_lag_configuration()
        total_lag_features = 0
        
        for var_type in ['forcing', 'parameters', 'target']:
            var_list = self.data.variables.get(var_type, [])
            var_lags = lag_config[var_type]
            
            for var_name in var_list:
                if var_name in var_lags:
                    # Count number of lags for this variable
                    n_lags = len(var_lags[var_name])
                    total_lag_features += n_lags
        
        return total_lag_features
    
    def get_lag_summary(self) -> Dict[str, Any]:
        """
        Get summary of lag configuration for logging/debugging
        
        Returns:
            Summary dictionary with lag information
        """
        lag_config = self.get_lag_configuration()
        
        summary = {
            'enabled': self.model.enable_lags,
            'total_lag_features': self.get_total_lag_features(),
            'by_variable_type': {}
        }
        
        for var_type in ['forcing', 'parameters', 'target']:
            var_lags = lag_config[var_type]
            var_summary = {
                'variables_with_lags': list(var_lags.keys()),
                'total_features': sum(len(lags) for lags in var_lags.values())
            }
            summary['by_variable_type'][var_type] = var_summary
        
        return summary


def load_config(config_path: str) -> UnifiedConfig:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        UnifiedConfig object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    # Create dataclass instances with safe field extraction
    model_dict = config_dict['model']
    model_config = ModelConfig(
        # Common parameters
        model_type=model_dict.get('model_type', 'unified'),
        dropout=model_dict['dropout'],
        sequence_length=model_dict['sequence_length'],
        
        # LSTM parameters
        hidden_size=model_dict.get('hidden_size', 128),
        num_layers=model_dict.get('num_layers', 1),
        enable_lags=model_dict.get('enable_lags', True),
        lag_config=model_dict.get('lag_config', None),
        lags=model_dict.get('lags', None),
        multi_task=model_dict.get('multi_task', None),
        
        # Swin Transformer parameters  
        embed_dim=model_dict.get('embed_dim', 96),
        depths=model_dict.get('depths', [2, 2, 6, 2]),
        num_heads=model_dict.get('num_heads', [3, 6, 12, 24]),
        window_size=model_dict.get('window_size', 7),
        patch_size=model_dict.get('patch_size', 10),
        drop_path_rate=model_dict.get('drop_path_rate', 0.1),
        
        # TCN parameters
        hidden_channels=model_dict.get('hidden_channels', [64, 128, 128, 64]),
        kernel_size=model_dict.get('kernel_size', 3),
        
        # Focal Loss configuration
        focal_loss=model_dict.get('focal_loss', None),

        # Spatial graph attention (previously dropped here, so it was never applied)
        spatial_attention=model_dict.get('spatial_attention', None)
    )
    
    data_config = DataConfig(**config_dict['data'])
    
    # Handle training config with optional multi-task fields
    training_dict = config_dict['training']
    training_config = TrainingConfig(
        epochs=training_dict['epochs'],
        batch_size=training_dict['batch_size'],
        learning_rate=training_dict['learning_rate'],
        loss_function=training_dict['loss_function'],
        memory=training_dict['memory'],
        batch_size_valid=training_dict.get('batch_size_valid', 256),
        patience=training_dict.get('patience', 10),
        epoch_save=training_dict.get('epoch_save', 10),
        optimizations=training_dict.get('optimizations', None),
        multi_output=training_dict.get('multi_output', None),
        physical_constraints=training_dict.get('physical_constraints', None),
        scheduler=training_dict.get('scheduler', None),
        multi_task=training_dict.get('multi_task', None),
        finetune_head=training_dict.get('finetune_head', 'none'),
        pretrained_model_path=training_dict.get('pretrained_model_path', None),
        resume_training=training_dict.get('resume_training', False),
        restart_path=training_dict.get('restart_path', None)
    )
    
    # Handle optional output config fields
    output_dict = config_dict['output']
    output_config = OutputConfig(
        base_dir=output_dict['base_dir'],
        experiment_name=output_dict['experiment_name'],
        save_predictions=output_dict.get('save_predictions', True),
        save_metrics=output_dict.get('save_metrics', True),
        create_plots=output_dict.get('create_plots', True),
        multi_task=output_dict.get('multi_task', None)
    )
    
    # Create unified config
    unified_config = UnifiedConfig(
        model=model_config,
        data=data_config,
        training=training_config,
        output=output_config,
        hardware=config_dict['hardware'],
        logging=config_dict['logging'],
        evaluation=config_dict['evaluation']
    )
    
    return unified_config


def save_config(config: UnifiedConfig, save_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: UnifiedConfig object
        save_path: Path to save the configuration
    """
    config_dict = {
        'model': {
            'hidden_size': config.model.hidden_size,
            'dropout': config.model.dropout,
            'sequence_length': config.model.sequence_length,
            'lags': config.model.lags
        },
        'data': {
            'spatial': config.data.spatial,
            'temporal': config.data.temporal,
            'file_indices': config.data.file_indices,
            'grid_sampling': config.data.grid_sampling,
            'variables': config.data.variables,
            'paths': config.data.paths
        },
        'training': {
            'epochs': config.training.epochs,
            'batch_size': config.training.batch_size,
            'batch_size_valid': config.training.batch_size_valid,
            'learning_rate': config.training.learning_rate,
            'patience': config.training.patience,
            'epoch_save': config.training.epoch_save,
            'loss_function': config.training.loss_function,
            'memory': config.training.memory,
            'multi_output': config.training.multi_output,
            'physical_constraints': config.training.physical_constraints,
            'finetune_head': config.training.finetune_head,
            'pretrained_model_path': config.training.pretrained_model_path,
            'resume_training': config.training.resume_training,
            'restart_path': config.training.restart_path
        },
        'output': {
            'base_dir': config.output.base_dir,
            'experiment_name': config.output.experiment_name,
            'save_predictions': config.output.save_predictions,
            'save_metrics': config.output.save_metrics,
            'create_plots': config.output.create_plots
        },
        'hardware': config.hardware,
        'logging': config.logging,
        'evaluation': config.evaluation
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False, indent=2)


# Legacy compatibility function
def create_legacy_config(config: UnifiedConfig):
    """
    Create legacy-style config object for backward compatibility
    
    Args:
        config: UnifiedConfig object
        
    Returns:
        Legacy config object with dot notation access
    """
    class LegacyConfig:
        def __init__(self, unified_config: UnifiedConfig):
            # Core training parameters
            self.epochs = unified_config.training.epochs
            self.lr = unified_config.training.learning_rate
            self.hiddensize = unified_config.model.hidden_size
            self.batch_size = unified_config.training.batch_size
            self.batch_size_valid = unified_config.training.batch_size_valid
            self.seq_length = unified_config.model.sequence_length
            self.patience = unified_config.training.patience
            self.epoch_save = unified_config.training.epoch_save
            
            # Device and directories
            self.device = unified_config.device
            self.out_dir = unified_config.out_dir
            self.out_model_dir = unified_config.out_model_dir
            self.tensorb_dir = unified_config.tensorb_dir
            self.figure_dir = unified_config.figure_dir
            
            # Additional attributes that might be used
            self.local_rank = 0
            self.nbatch_valid = 50
    
    return LegacyConfig(config)