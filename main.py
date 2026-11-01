#!/usr/bin/env python3
"""
Unified VIC Surrogate Model - Main Entry Point
Author: Siddik Barbhuiya
Description: Main script to run the unified surrogate model with YAML configuration

This is the primary entry point for the VIC surrogate model system. It provides a unified
interface for training, configuration management, grid splitting, and analysis.

The script supports multiple modes of operation:
1. Training mode: Run the full surrogate model training pipeline
2. Configuration creation: Generate default configuration templates
3. Grid splitting: Create reproducible train/validation/test grid splits
4. Grid analysis: Analyze existing grid location files
5. Configuration validation: Validate configuration file integrity

The workflow follows these steps:
1. Load and validate configuration from YAML file
2. Set random seeds for reproducibility
3. Create grid-aware datasets with spatial splitting
4. Initialize the UnifiedTrainer with configuration
5. Execute the training pipeline with monitoring
6. Save results, metrics, and model checkpoints

Key features:
- Reproducible experiments with seed management
- Comprehensive configuration validation
- System information reporting
- Error handling and logging
- Multiple operational modes via command-line arguments
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import random
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.core.config import load_config, save_config
from src.training.trainer import UnifiedTrainer
from src.data.grid_splitter import create_grid_aware_datasets
from src.core.validator import ConfigValidator
from src.analysis.grid_analyzer import analyze_grid_locations


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_system_info():
    """Print system and environment information"""
    print("=" * 80)
    print("UNIFIED VIC SURROGATE MODEL")
    print("Author: Siddik Barbhuiya")
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory // 1024**3} GB")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 80)


def create_default_config():
    """Create a default configuration file"""
    config_path = "config.yaml"
    
    if os.path.exists(config_path):
        response = input(f"Configuration file '{config_path}' already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Configuration file not created.")
            return
    
    # Copy the template config
    import shutil
    current_dir = Path(__file__).parent
    source_config = current_dir / "config" / "templates" / "default_config.yaml"
    
    if source_config.exists():
        shutil.copy2(source_config, config_path)
        print(f"Default configuration created at '{config_path}'")
        print("Please edit the configuration file before training.")
    else:
        print("Error: Default configuration template not found!")


def validate_config(config):
    """Validate configuration parameters - Enhanced by Siddik"""
    validator = ConfigValidator(config)
    is_valid = validator.validate_all()

    if not is_valid:
        print(validator.get_validation_summary())

    return is_valid


def normalize_finetune_head(finetune_head: str):
    """Normalize finetune_head argument to match actual head names.

    Maps common variations to the correct head names used in the model.
    Valid head names: 'none', 'evap', 'sm', 'Q'
    """
    if finetune_head is None:
        return None

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

    normalized = head_mapping.get(finetune_head)
    if normalized is None:
        raise ValueError(
            f"Invalid finetune_head '{finetune_head}'. "
            f"Valid values are: 'none', 'evap', 'sm', 'Q' "
            f"(also accepts case-insensitive variants like 'q', 'et', 'ET', 'SM')"
        )

    return normalized


def run_training(config_path: str, resume: bool = False, checkpoint_path: str = None, finetune_head: str = None):
    """Run the training process"""

    # Load configuration
    try:
        config = load_config(config_path)
        print(f"Configuration loaded from: {config_path}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Override finetune_head from command-line argument if provided
    if finetune_head is not None:
        normalized_head = normalize_finetune_head(finetune_head)
        config.training.finetune_head = normalized_head
        print(f"Fine-tune head overridden from CLI: {finetune_head} -> {normalized_head}")

    # Handle resume logic from config if not specified in CLI
    if not resume and config.training.resume_training:
        resume = True
        print("Resuming training enabled via configuration file")
    
    if not checkpoint_path and config.training.restart_path:
        checkpoint_path = config.training.restart_path
        print(f"Checkpoint path set via configuration file: {checkpoint_path}")

    # Validate configuration
    if not validate_config(config):
        print("Configuration validation failed. Please fix the errors and try again.")
        return
    
    # Set random seeds
    evaluation_config = config.evaluation
    seed = evaluation_config.get('grid_split_seed', 42)
    set_seeds(seed)
    print(f"Random seed set to: {seed}")
    
    # Save configuration to output directory
    config_backup_path = os.path.join(config.out_dir, 'config_backup.yaml')
    save_config(config, config_backup_path)
    
    # Print configuration summary
    print("\nConfiguration Summary:")

    # Get model type
    model_type = getattr(config.model, 'model_type', 'unknown')

    # Build model info based on type
    if 'swin' in model_type.lower():
        embed_dim = getattr(config.model, 'embed_dim', 96)
        depths = getattr(config.model, 'depths', [2, 2, 6, 2])
        model_info = f"{model_type} (embed_dim={embed_dim}, depths={depths})"
    elif 'tcn' in model_type.lower():
        hidden_channels = getattr(config.model, 'hidden_channels', [64, 128, 128, 64])
        kernel_size = getattr(config.model, 'kernel_size', 3)
        model_info = f"{model_type} (channels={hidden_channels}, kernel={kernel_size})"
    else:
        hidden_size = getattr(config.model, 'hidden_size', 128)
        num_layers = getattr(config.model, 'num_layers', 2)
        model_info = f"{model_type} (hidden_size={hidden_size}, layers={num_layers})"

    print(f"  Model: {model_info}")
    print(f"  Training: {config.training.epochs} epochs, lr={config.training.learning_rate}")
    print(f"  Loss function: {config.training.loss_function}")
    print(f"  Device: {config.device}")
    print(f"  Output directory: {config.out_dir}")
    
    # Handle resume functionality
    resume_checkpoint_path = None
    if resume:
        if checkpoint_path:
            resume_checkpoint_path = checkpoint_path
        else:
            # Try to find the latest checkpoint automatically
            models_dir = os.path.join(config.out_dir, 'models')
            if os.path.exists(models_dir):
                # Look for best_model.pth first, then latest epoch
                best_model = os.path.join(models_dir, 'best_model.pth')
                if os.path.exists(best_model):
                    resume_checkpoint_path = best_model
                    print(f"Resuming from best model: {resume_checkpoint_path}")
                else:
                    # Find latest epoch checkpoint
                    epoch_files = [f for f in os.listdir(models_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
                    if epoch_files:
                        # Sort by epoch number
                        epoch_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
                        resume_checkpoint_path = os.path.join(models_dir, epoch_files[-1])
                        print(f"Resuming from latest checkpoint: {resume_checkpoint_path}")
        
        if resume_checkpoint_path and not os.path.exists(resume_checkpoint_path):
            print(f"Warning: Checkpoint file not found: {resume_checkpoint_path}")
            print("Starting fresh training instead.")
            resume_checkpoint_path = None
    
    try:
        # Create trainer and run training
        trainer = UnifiedTrainer(config)
        trainer.train(resume_checkpoint=resume_checkpoint_path)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        print(f"\nTraining failed: {e}")
        raise


def create_grid_splits(config_path: str):
    """Create and save grid splits for reproducible experiments"""
    
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    print("Creating grid splits...")
    
    splits = create_grid_aware_datasets(
        config, 
        use_existing_splits=False,  # Force creation of new splits
        create_stratified=True      # Use stratified splits
    )
    
    print("Grid splits created successfully!")
    print(f"Training grids: {len(splits['train'])}")
    print(f"Validation grids: {len(splits['val'])}")
    print(f"Test grids: {len(splits['test'])}")


def analyze_grids(output_dir: str):
    """Analyze existing grid location files"""
    
    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' not found!")
        return
    
    print("Analyzing grid locations...")
    
    try:
        report = analyze_grid_locations(output_dir, create_plots=True)
        
        print("\nGrid analysis completed successfully!")
        print(f"Analysis files saved to: {output_dir}")
        print("\nKey files created:")
        print("  - grid_analysis_report.txt")
        print("  - grid_summary_table.csv")
        print("  - Multiple visualization plots")
        
        # Print a summary of the report
        lines = report.split('\n')
        overview_started = False
        for line in lines:
            if "OVERVIEW:" in line:
                overview_started = True
            elif overview_started and line.strip() == "":
                break
            elif overview_started:
                print(line)
                
    except Exception as e:
        print(f"Error analyzing grids: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Unified VIC Surrogate Model by Siddik",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config config.yaml                      # Train with configuration file
  %(prog)s --config config.yaml --resume             # Resume training from latest checkpoint
  %(prog)s --config config.yaml --checkpoint path/to/model.pth  # Resume from specific checkpoint
  %(prog)s --create-config                           # Create default configuration
  %(prog)s --create-splits --config config.yaml     # Create grid splits
  %(prog)s --analyze-grids /path/to/output/dir       # Analyze existing grid files
  %(prog)s --info                                    # Show system information
        """
    )
    
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--create-config', action='store_true',
                       help='Create a default configuration file')
    parser.add_argument('--create-splits', action='store_true',
                       help='Create grid splits for training/testing')
    parser.add_argument('--analyze-grids', type=str, metavar='OUTPUT_DIR',
                       help='Analyze existing grid location files in specified directory')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, metavar='PATH',
                       help='Path to specific checkpoint file to resume from')
    parser.add_argument('--info', action='store_true',
                       help='Show system information and exit')
    parser.add_argument('--validate', action='store_true',
                       help='Validate configuration file and exit')
    parser.add_argument('--finetune-head', type=str, default=None,
                       help='Fine-tune only specified head for multi-task models (overrides config value). '
                            'Valid values: none, evap, sm, Q')

    args = parser.parse_args()
    
    # Show system info if requested or when starting training
    if args.info:
        print_system_info()
        return
    
    # Create default configuration
    if args.create_config:
        create_default_config()
        return
    
    # Validate configuration
    if args.validate:
        try:
            config = load_config(args.config)
            if validate_config(config):
                print("Configuration is valid!")
            else:
                print("Configuration validation failed.")
                sys.exit(1)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
        return
    
    # Create grid splits
    if args.create_splits:
        create_grid_splits(args.config)
        return
    
    # Analyze existing grid files
    if args.analyze_grids:
        analyze_grids(args.analyze_grids)
        return
    
    # Default: run training
    print_system_info()
    
    # Check if configuration file exists
    if not os.path.exists(args.config):
        print(f"Configuration file '{args.config}' not found!")
        print("Use --create-config to create a default configuration file.")
        sys.exit(1)
    
    # Run training
    run_training(args.config, args.resume, args.checkpoint, args.finetune_head)


if __name__ == '__main__':
    main()