#!/usr/bin/env python3
"""
Swin Transformer Surrogate Model - Main Entry Point
Author: Siddik
Description: Main script to run the Swin Transformer surrogate model with YAML configuration
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
from src.training.swin_trainer import UnifiedTrainer
from src.data.grid_splitter import create_grid_aware_datasets
from src.core.validator import ConfigValidator


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
    print("SWIN TRANSFORMER SURROGATE MODEL")
    print("Author: Siddik")
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


def validate_config(config):
    """Validate configuration parameters"""
    validator = ConfigValidator(config)
    is_valid = validator.validate_all()

    if not is_valid:
        print(validator.get_validation_summary())

    return is_valid


def run_training(config_path: str, resume: bool = False, checkpoint_path: str = None):
    """Run the training process"""

    # Load configuration
    try:
        config = load_config(config_path)
        print(f"Configuration loaded from: {config_path}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

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
    print(f"  Model: {config.model.model_type}")
    print(f"  Embed dim: {config.model.embed_dim}, Patch size: {config.model.patch_size}")
    print(f"  Window size: {config.model.window_size}, Depths: {config.model.depths}")
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
            models_dir = os.path.join(config.out_dir, 'models')
            if os.path.exists(models_dir):
                best_model = os.path.join(models_dir, 'best_model.pth')
                if os.path.exists(best_model):
                    resume_checkpoint_path = best_model
                    print(f"Resuming from best model: {resume_checkpoint_path}")
                else:
                    epoch_files = [f for f in os.listdir(models_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
                    if epoch_files:
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


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Swin Transformer Surrogate Model by Siddik",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config config_swin.yaml                      # Train with configuration file
  %(prog)s --config config_swin.yaml --resume             # Resume training from latest checkpoint
  %(prog)s --config config_swin.yaml --checkpoint path/to/model.pth  # Resume from specific checkpoint
  %(prog)s --info                                          # Show system information
        """
    )

    parser.add_argument('--config', '-c', type=str, default='config_swin.yaml',
                       help='Path to configuration file (default: config_swin.yaml)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, metavar='PATH',
                       help='Path to specific checkpoint file to resume from')
    parser.add_argument('--info', action='store_true',
                       help='Show system information and exit')
    parser.add_argument('--validate', action='store_true',
                       help='Validate configuration file and exit')

    args = parser.parse_args()

    # Show system info if requested or when starting training
    if args.info:
        print_system_info()
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

    # Default: run training
    print_system_info()

    # Check if configuration file exists
    if not os.path.exists(args.config):
        print(f"Configuration file '{args.config}' not found!")
        sys.exit(1)

    # Run training
    run_training(args.config, args.resume, args.checkpoint)


if __name__ == '__main__':
    main()
