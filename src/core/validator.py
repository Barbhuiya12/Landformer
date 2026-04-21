"""
Configuration Validator for Unified VIC Surrogate Model
Author: Siddik
Description: Comprehensive validation of configuration parameters
"""

import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import torch

from .config import UnifiedConfig


class ConfigValidator:
    """
    Comprehensive configuration validator
    Author: Siddik
    """
    
    def __init__(self, config: UnifiedConfig):
        """
        Initialize validator with configuration
        
        Args:
            config: UnifiedConfig object to validate
        """
        self.config = config
        self.errors = []
        self.warnings = []
        self.logger = logging.getLogger(__name__)
    
    def validate_all(self) -> bool:
        """
        Run all validation checks
        
        Returns:
            True if all validations pass, False otherwise
        """
        self.errors.clear()
        self.warnings.clear()
        
        # Run all validation methods
        self.validate_model_config()
        self.validate_data_config()
        self.validate_training_config()
        self.validate_hardware_config()
        self.validate_output_config()
        self.validate_paths()
        
        # Log results
        if self.errors:
            self.logger.error(f"Configuration validation failed with {len(self.errors)} errors")
            for error in self.errors:
                self.logger.error(f"  - {error}")
        
        if self.warnings:
            self.logger.warning(f"Configuration validation completed with {len(self.warnings)} warnings")
            for warning in self.warnings:
                self.logger.warning(f"  - {warning}")
        
        return len(self.errors) == 0
    
    def validate_model_config(self):
        """Validate model configuration parameters"""
        model = self.config.model
        
        # Hidden size validation
        if not isinstance(model.hidden_size, int) or model.hidden_size <= 0:
            self.errors.append("Model hidden_size must be a positive integer")
        elif model.hidden_size < 32:
            self.warnings.append("Model hidden_size is very small (< 32), may lead to underfitting")
        elif model.hidden_size > 1024:
            self.warnings.append("Model hidden_size is very large (> 1024), may cause memory issues")
        
        # Dropout validation
        if not isinstance(model.dropout, (int, float)) or not (0 <= model.dropout < 1):
            self.errors.append("Model dropout must be a float between 0 and 1")
        elif model.dropout > 0.8:
            self.warnings.append("Model dropout is very high (> 0.8), may cause underfitting")
        
        # Sequence length validation
        if not isinstance(model.sequence_length, int) or model.sequence_length <= 0:
            self.errors.append("Model sequence_length must be a positive integer")
        elif model.sequence_length < 5:
            self.warnings.append("Model sequence_length is very short (< 5)")
        elif model.sequence_length > 365:
            self.warnings.append("Model sequence_length is very long (> 365), may cause memory issues")
        
        # Flexible lag validation
        if model.enable_lags:
            lag_config = self.config.get_lag_configuration()
            has_any_lags = False
            max_lag = 0
            
            # Check if any lags are configured
            for var_type in ['forcing', 'parameters', 'target']:
                var_lags = lag_config.get(var_type, {}) or {}
                for var_name, lags in var_lags.items():
                    if lags:
                        has_any_lags = True
                        for lag in lags:
                            if not isinstance(lag, int) or lag <= 0:
                                self.errors.append(f"All lag values must be positive integers (found {lag} for {var_type}.{var_name})")
                            else:
                                max_lag = max(max_lag, lag)
            
            # Check if lags are enabled but none configured
            if not has_any_lags:
                self.warnings.append("Lags are enabled (enable_lags: true) but no lag features are configured")
            
            # Check maximum lag vs sequence length
            if max_lag > 0 and max_lag >= model.sequence_length:
                self.errors.append(f"Maximum lag value ({max_lag}) must be less than sequence_length ({model.sequence_length})")
        
        # Legacy lags validation (backward compatibility)
        if hasattr(model, 'lags') and model.lags is not None:
            if not isinstance(model.lags, list) or not model.lags:
                self.errors.append("Legacy model.lags must be a non-empty list if specified")
            else:
                for lag in model.lags:
                    if not isinstance(lag, int) or lag <= 0:
                        self.errors.append("All legacy lag values must be positive integers")
                        break
                
                if max(model.lags) >= model.sequence_length:
                    self.errors.append("Maximum legacy lag value must be less than sequence_length")
    
    def validate_data_config(self):
        """Validate data configuration parameters"""
        data = self.config.data
        
        # Spatial bounds validation
        spatial = data.spatial
        if 'lat_min' in spatial and 'lat_max' in spatial:
            if spatial['lat_min'] >= spatial['lat_max']:
                self.errors.append("lat_min must be less than lat_max")
            if not (-90 <= spatial['lat_min'] <= 90) or not (-90 <= spatial['lat_max'] <= 90):
                self.errors.append("Latitude values must be between -90 and 90")
        
        if 'lon_min' in spatial and 'lon_max' in spatial:
            if spatial['lon_min'] >= spatial['lon_max']:
                self.errors.append("lon_min must be less than lon_max")
            if not (-180 <= spatial['lon_min'] <= 180) or not (-180 <= spatial['lon_max'] <= 180):
                self.errors.append("Longitude values must be between -180 and 180")
        
        # Grid sampling validation
        grid_sampling = data.grid_sampling
        if 'use_exact_counts' in grid_sampling:
            if grid_sampling['use_exact_counts']:
                # Validate exact counts
                required_counts = ['train_grids', 'validation_grids', 'test_grids']
                for count_key in required_counts:
                    if count_key in grid_sampling:
                        count_val = grid_sampling[count_key]
                        if not isinstance(count_val, int) or count_val <= 0:
                            self.errors.append(f"{count_key} must be a positive integer")
            else:
                # Validate percentages
                required_percentages = ['train_percentage', 'validation_percentage', 'test_percentage']
                total_percentage = 0
                for pct_key in required_percentages:
                    if pct_key in grid_sampling:
                        pct_val = grid_sampling[pct_key]
                        if not isinstance(pct_val, (int, float)) or not (0 < pct_val <= 1):
                            self.errors.append(f"{pct_key} must be a float between 0 and 1")
                        else:
                            total_percentage += pct_val
                
                if abs(total_percentage - 1.0) > 1e-6:
                    self.warnings.append(f"Grid sampling percentages sum to {total_percentage:.3f}, not 1.0")
        
        # Variables validation
        variables = data.variables
        required_var_types = ['forcing', 'parameters', 'target']
        for var_type in required_var_types:
            if var_type not in variables or not isinstance(variables[var_type], list):
                self.errors.append(f"Data variables must include '{var_type}' as a list")
            elif not variables[var_type]:
                self.errors.append(f"Data variables '{var_type}' cannot be empty")
    
    def validate_training_config(self):
        """Validate training configuration parameters"""
        training = self.config.training
        
        # Basic parameter validation
        if not isinstance(training.epochs, int) or training.epochs <= 0:
            self.errors.append("Training epochs must be a positive integer")
        elif training.epochs > 1000:
            self.warnings.append("Training epochs is very high (> 1000)")
        
        if not isinstance(training.batch_size, int) or training.batch_size <= 0:
            self.errors.append("Training batch_size must be a positive integer")
        elif training.batch_size > 5000:
            self.warnings.append("Training batch_size is very large (> 5000), may cause memory issues")
        
        if not isinstance(training.batch_size_valid, int) or training.batch_size_valid <= 0:
            self.errors.append("Training batch_size_valid must be a positive integer")
        
        if not isinstance(training.learning_rate, (int, float)) or training.learning_rate <= 0:
            self.errors.append("Training learning_rate must be a positive number")
        elif training.learning_rate > 0.01:
            self.warnings.append("Training learning_rate is very high (> 0.01)")
        elif training.learning_rate < 1e-6:
            self.warnings.append("Training learning_rate is very low (< 1e-6)")
        
        if not isinstance(training.patience, int) or training.patience <= 0:
            self.errors.append("Training patience must be a positive integer")
        elif training.patience < 5:
            self.warnings.append("Training patience is very low (< 5)")
        
        # Loss function validation
        valid_loss_functions = ['MSE', 'NSE', 'KGE', 'MAE', 'MULTI_NSE', 'MULTI_KGE']
        if training.loss_function.upper() not in valid_loss_functions:
            self.errors.append(f"Training loss_function must be one of {valid_loss_functions}")
        
        # Memory configuration validation
        memory = training.memory
        if 'num_workers' in memory:
            if not isinstance(memory['num_workers'], int) or memory['num_workers'] < 0:
                self.errors.append("Memory num_workers must be a non-negative integer")
            elif memory['num_workers'] > 16:
                self.warnings.append("Memory num_workers is very high (> 16), may cause overhead")

        # Mixed precision validation
        if self.config.training.optimizations.get('use_amp', False):
            if not torch.cuda.is_available():
                self.warnings.append("Mixed precision (use_amp) enabled but CUDA not available")
    
    def validate_hardware_config(self):
        """Validate hardware configuration"""
        hardware = self.config.hardware
        
        # Device validation
        if 'device' in hardware:
            device = hardware['device']
            if device != 'auto' and device != 'cpu':
                if device.startswith('cuda'):
                    if not torch.cuda.is_available():
                        self.errors.append("CUDA device specified but CUDA is not available")
                    else:
                        # Check specific GPU index
                        if ':' in device:
                            try:
                                gpu_id = int(device.split(':')[1])
                                if gpu_id >= torch.cuda.device_count():
                                    self.errors.append(f"GPU {gpu_id} not available (only {torch.cuda.device_count()} GPUs found)")
                            except ValueError:
                                self.errors.append("Invalid CUDA device format (use 'cuda:0', 'cuda:1', etc.)")
                else:
                    self.warnings.append(f"Unknown device '{device}', will use auto-detection")
    
    def validate_output_config(self):
        """Validate output configuration"""
        output = self.config.output
        
        # Base directory validation
        base_dir = output.base_dir
        if not isinstance(base_dir, str) or not base_dir:
            self.errors.append("Output base_dir must be a non-empty string")
        else:
            # Check if parent directory exists and is writable
            parent_dir = Path(base_dir).parent
            if not parent_dir.exists():
                self.errors.append(f"Output base directory parent '{parent_dir}' does not exist")
            elif not os.access(parent_dir, os.W_OK):
                self.errors.append(f"Output base directory parent '{parent_dir}' is not writable")
        
        # Experiment name validation
        exp_name = output.experiment_name
        if not isinstance(exp_name, str) or not exp_name:
            self.errors.append("Output experiment_name must be a non-empty string")
        elif any(char in exp_name for char in '<>:"/\\|?*'):
            self.errors.append("Output experiment_name contains invalid characters")
    
    def validate_paths(self):
        """Validate file paths in configuration"""
        paths = self.config.data.paths
        
        # Check required paths
        required_paths = ['forcing', 'parameters']
        for path_type in required_paths:
            if path_type not in paths:
                self.errors.append(f"Data paths must include '{path_type}'")
            else:
                path = paths[path_type]
                if not isinstance(path, str) or not path:
                    self.errors.append(f"Data path '{path_type}' must be a non-empty string")
                elif not os.path.exists(path):
                    self.errors.append(f"Data path '{path_type}': '{path}' does not exist")
                elif not os.access(path, os.R_OK):
                    self.errors.append(f"Data path '{path_type}': '{path}' is not readable")
        
        # Check target paths - either single 'target' or multiple 'target_files'
        if 'target' not in paths and 'target_files' not in paths:
            self.errors.append("Data paths must include either 'target' (single file) or 'target_files' (multiple files)")
        elif 'target' in paths:
            # Single target file approach
            path = paths['target']
            if not isinstance(path, str) or not path:
                self.errors.append("Data path 'target' must be a non-empty string")
            elif not os.path.exists(path):
                self.errors.append(f"Data path 'target': '{path}' does not exist")
            elif not os.access(path, os.R_OK):
                self.errors.append(f"Data path 'target': '{path}' is not readable")
        elif 'target_files' in paths:
            # Multiple target files approach
            target_files = paths['target_files']
            if not isinstance(target_files, dict) or not target_files:
                self.errors.append("Data path 'target_files' must be a non-empty dictionary")
            else:
                for var_name, path in target_files.items():
                    if not isinstance(path, str) or not path:
                        self.errors.append(f"Target file path for '{var_name}' must be a non-empty string")
                    elif not os.path.exists(path):
                        self.errors.append(f"Target file path for '{var_name}': '{path}' does not exist")
                    elif not os.access(path, os.R_OK):
                        self.errors.append(f"Target file path for '{var_name}': '{path}' is not readable")
    
    def get_validation_summary(self) -> str:
        """
        Get a formatted summary of validation results
        
        Returns:
            Formatted validation summary string
        """
        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("CONFIGURATION VALIDATION SUMMARY")
        summary_lines.append("=" * 60)
        
        if not self.errors and not self.warnings:
            summary_lines.append("✅ Configuration is valid!")
        else:
            if self.errors:
                summary_lines.append(f"❌ {len(self.errors)} ERRORS found:")
                for i, error in enumerate(self.errors, 1):
                    summary_lines.append(f"   {i}. {error}")
                summary_lines.append("")
            
            if self.warnings:
                summary_lines.append(f"⚠️  {len(self.warnings)} WARNINGS:")
                for i, warning in enumerate(self.warnings, 1):
                    summary_lines.append(f"   {i}. {warning}")
        
        summary_lines.append("=" * 60)
        return "\n".join(summary_lines)