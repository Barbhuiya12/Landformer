"""
Simple Inference Module for VIC Surrogate Models
Author: Siddik Barbhuiya
Description: Easy-to-use inference interface for Jupyter notebooks

This module provides a simple way to:
1. Load trained models (LSTM, TCN, Swin Transformer)
2. Make predictions on new data
3. Visualize results
4. Export predictions

Usage in Jupyter:
    from src.inference import VICSurrogateInference

    # Initialize
    inferencer = VICSurrogateInference(
        model_path='path/to/best_model.pth',
        config_path='path/to/config.yaml'
    )

    # Make predictions
    predictions = inferencer.predict(forcing_data, parameter_data)

    # Visualize
    inferencer.plot_predictions(predictions, observations)
"""

import torch
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import logging
from pathlib import Path
from typing import Dict, Union, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Import model architectures
from .models.swin_transformer import MultiTaskSwinTransformer
from .models.tcn import MultiTaskTCN
from .models.lstm import MultiTaskLSTM
from .core.config import UnifiedConfig
from .utils.hydro_metrics import calculate_all_metrics


class VICSurrogateInference:
    """
    Simple inference class for VIC surrogate models

    Easy-to-use interface for loading trained models and making predictions
    """

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = 'cuda:0'
    ):
        """
        Initialize the inference module

        Args:
            model_path: Path to trained model checkpoint (.pth file)
            config_path: Path to training config (optional, will try to auto-detect)
            device: Device to run inference on ('cuda:0', 'cuda:1', or 'cpu')
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger = self._setup_logger()

        # Load model and config
        self.logger.info(f"Loading model from: {model_path}")
        # PyTorch 2.6+ requires weights_only=False for custom objects
        self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Try to load config
        if config_path is None:
            config_path = self._find_config()

        if config_path:
            self.logger.info(f"Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                self.config_dict = yaml.safe_load(f)
            self.config = UnifiedConfig(self.config_dict)
        else:
            self.logger.warning("No config file found. Using checkpoint metadata.")
            self.config = None

        # Initialize model
        self.model = self._load_model()
        self.model.eval()

        # Get model info
        self._print_model_info()

        # Store normalizers if available
        self.scaler = self.checkpoint.get('scaler', None)

    def _setup_logger(self):
        """Setup logging"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _find_config(self) -> Optional[str]:
        """Try to find config file automatically"""
        # Look in common locations
        model_dir = self.model_path.parent.parent

        possible_configs = [
            model_dir / 'config.yaml',
            model_dir / 'training_config.yaml',
            model_dir.parent / 'config.yaml',
        ]

        for config_path in possible_configs:
            if config_path.exists():
                return str(config_path)

        return None

    def _load_model(self):
        """Load model architecture and weights"""
        # Get model config from checkpoint or config file
        if 'model_config' in self.checkpoint:
            model_config = self.checkpoint['model_config']
        elif 'config' in self.checkpoint:
            # Try to get config from checkpoint
            model_config = self.checkpoint['config'].get('model', {}) if isinstance(self.checkpoint['config'], dict) else {}
        elif self.config:
            model_config = self.config.model
        else:
            # Try to infer from model state dict
            self.logger.warning("No model config found. Attempting to infer from checkpoint...")
            model_config = self._infer_model_config_from_checkpoint()

        # Determine model type
        model_type = model_config.get('model_type', 'multi_task_swin')

        # Create model
        if 'swin' in model_type.lower():
            # Backward compatibility: use_hierarchical defaults to True (current behavior)
            # Set to False in config/checkpoint for legacy models with constant dimensions
            use_hierarchical = model_config.get('use_hierarchical', True)
            model = MultiTaskSwinTransformer(
                n_exo=model_config.get('input_dims', {}).get('n_forcing', 8) +
                      model_config.get('input_dims', {}).get('n_params', 13),
                n_lag_features=model_config.get('input_dims', {}).get('n_lag_features', 0),
                seq_len=model_config.get('sequence_length', 270),
                patch_size=model_config.get('patch_size', 10),
                embed_dim=model_config.get('embed_dim', 96),
                depths=model_config.get('depths', [2, 2, 6, 2]),
                num_heads=model_config.get('num_heads', [4, 8, 8, 8]),
                window_size=model_config.get('window_size', 12),
                dropout=0.0,  # No dropout during inference
                drop_path_rate=0.0,  # No drop path during inference
                task_names=['evap', 'soil_moisture_total', 'Q'],
                use_hierarchical=use_hierarchical
            )
        elif 'tcn' in model_type.lower():
            model = MultiTaskTCN(
                n_exo=model_config.get('input_dims', {}).get('n_forcing', 8) +
                      model_config.get('input_dims', {}).get('n_params', 13),
                n_lag_features=model_config.get('input_dims', {}).get('n_lag_features', 0),
                hidden_units=model_config.get('hidden_units', 64),
                num_layers=model_config.get('num_layers', 4),
                kernel_size=model_config.get('kernel_size', 3),
                dropout=0.0,
                task_names=['evap', 'soil_moisture_total', 'Q']
            )
        elif 'lstm' in model_type.lower():
            from .models.lstm import MultiTaskLSTM
            model = MultiTaskLSTM(
                n_exo=model_config.get('input_dims', {}).get('n_forcing', 8) +
                      model_config.get('input_dims', {}).get('n_params', 13),
                n_lag_features=model_config.get('input_dims', {}).get('n_lag_features', 0),
                seq_len=model_config.get('sequence_length', 270),
                hidden_size=model_config.get('hidden_size', 256),
                num_layers=model_config.get('num_layers', 2),
                dropout=0.0,
                task_names=['evap', 'soil_moisture_total', 'Q']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported: swin, tcn, lstm")

        # Load weights
        if 'model_state_dict' in self.checkpoint:
            model.load_state_dict(self.checkpoint['model_state_dict'])
        elif 'state_dict' in self.checkpoint:
            model.load_state_dict(self.checkpoint['state_dict'])
        else:
            model.load_state_dict(self.checkpoint)

        model.to(self.device)
        return model

    def _print_model_info(self):
        """Print model information"""
        self.logger.info("=" * 60)
        self.logger.info("MODEL INFORMATION")
        self.logger.info("=" * 60)

        if 'epoch' in self.checkpoint:
            self.logger.info(f"Training epoch: {self.checkpoint['epoch']}")

        if 'best_val_loss' in self.checkpoint:
            self.logger.info(f"Best validation loss: {self.checkpoint['best_val_loss']:.4f}")

        if 'best_val_metric' in self.checkpoint:
            self.logger.info(f"Best validation KGE: {self.checkpoint['best_val_metric']:.4f}")

        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info("=" * 60)

    def predict(
        self,
        forcing: Union[np.ndarray, pd.DataFrame, xr.DataArray],
        parameters: Union[np.ndarray, pd.DataFrame, xr.DataArray],
        normalize: bool = True,
        return_dict: bool = True
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Make predictions on new data

        Args:
            forcing: Forcing data (batch, time, n_forcing)
                     Shape: (n_samples, 270, 8) for 270 days, 8 variables
            parameters: Parameter data (batch, n_params)
                       Shape: (n_samples, 13) for 13 soil parameters
            normalize: Whether to normalize inputs (if scaler available)
            return_dict: Return as dict or array

        Returns:
            predictions: Dictionary with keys ['evap', 'soil_moisture_total', 'Q']
                        or array of shape (n_samples, 3)

        Example:
            >>> predictions = inferencer.predict(forcing_data, param_data)
            >>> print(predictions['Q'])  # Discharge predictions
        """
        self.model.eval()

        # Convert to numpy if needed
        forcing = self._to_numpy(forcing)
        parameters = self._to_numpy(parameters)

        # Ensure correct shapes
        if forcing.ndim == 2:
            forcing = forcing[np.newaxis, :, :]  # Add batch dimension

        if parameters.ndim == 1:
            parameters = parameters[np.newaxis, :]  # Add batch dimension

        # Expand parameters to match sequence length
        n_samples = forcing.shape[0]
        seq_len = forcing.shape[1]
        params_expanded = np.tile(parameters[:, np.newaxis, :], (1, seq_len, 1))

        # Combine forcing and parameters
        inputs = np.concatenate([forcing, params_expanded], axis=2)

        # Normalize if scaler available
        if normalize and self.scaler is not None:
            # Note: This is simplified - may need adjustment based on your scaler
            self.logger.warning("Normalization not fully implemented. Using raw data.")

        # Convert to tensor
        inputs_tensor = torch.FloatTensor(inputs).to(self.device)

        # Make predictions
        with torch.no_grad():
            outputs = self.model(inputs_tensor)

        # Convert to numpy
        predictions = outputs.cpu().numpy()

        # Return as dict or array
        if return_dict:
            return {
                'evap': predictions[:, 0],
                'soil_moisture_total': predictions[:, 1],
                'Q': predictions[:, 2]
            }
        else:
            return predictions

    def predict_batch(
        self,
        forcing_batch: np.ndarray,
        parameters_batch: np.ndarray,
        batch_size: int = 32
    ) -> Dict[str, np.ndarray]:
        """
        Predict on large batches with batching

        Args:
            forcing_batch: Large forcing array
            parameters_batch: Large parameter array
            batch_size: Batch size for processing

        Returns:
            predictions: Dictionary with predictions for all samples
        """
        n_samples = forcing_batch.shape[0]
        all_predictions = {'evap': [], 'soil_moisture_total': [], 'Q': []}

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            forcing = forcing_batch[i:end_idx]
            params = parameters_batch[i:end_idx]

            preds = self.predict(forcing, params, return_dict=True)

            for key in all_predictions:
                all_predictions[key].append(preds[key])

        # Concatenate all batches
        for key in all_predictions:
            all_predictions[key] = np.concatenate(all_predictions[key], axis=0)

        return all_predictions

    def evaluate(
        self,
        forcing: np.ndarray,
        parameters: np.ndarray,
        observations: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate predictions against observations

        Args:
            forcing: Forcing data
            parameters: Parameter data
            observations: Dictionary with true values {'evap': ..., 'soil_moisture_total': ..., 'Q': ...}

        Returns:
            metrics: Dictionary with NSE, KGE, correlation, bias for each variable

        Example:
            >>> obs = {'Q': true_discharge, 'evap': true_evap, 'soil_moisture_total': true_sm}
            >>> metrics = inferencer.evaluate(forcing, params, obs)
            >>> print(f"Q NSE: {metrics['Q']['nse']:.3f}")
        """
        # Make predictions
        predictions = self.predict(forcing, parameters, return_dict=True)

        # Calculate metrics for each variable
        results = {}

        for var in ['evap', 'soil_moisture_total', 'Q']:
            if var in observations:
                pred = predictions[var]
                obs = observations[var]

                # Calculate metrics
                metrics = calculate_all_metrics(obs, pred)
                results[var] = metrics

                self.logger.info(f"{var}: NSE={metrics['nse']:.3f}, KGE={metrics['kge']:.3f}, "
                               f"R={metrics['correlation']:.3f}, Bias={metrics['bias']:.3f}")

        return results

    def plot_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        observations: Optional[Dict[str, np.ndarray]] = None,
        sample_idx: int = 0,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ):
        """
        Plot predictions vs observations

        Args:
            predictions: Dictionary with predictions
            observations: Dictionary with true values (optional)
            sample_idx: Which sample to plot (if multiple)
            figsize: Figure size
            save_path: Path to save figure (optional)

        Example:
            >>> predictions = inferencer.predict(forcing, params)
            >>> inferencer.plot_predictions(predictions, observations, sample_idx=0)
        """
        fig, axes = plt.subplots(3, 2, figsize=figsize)

        variables = ['evap', 'soil_moisture_total', 'Q']
        titles = ['Evapotranspiration', 'Soil Moisture', 'Discharge']

        for i, (var, title) in enumerate(zip(variables, titles)):
            # Time series plot
            ax1 = axes[i, 0]
            pred = predictions[var][sample_idx] if predictions[var].ndim > 1 else predictions[var]

            ax1.plot(pred, label='Prediction', linewidth=2)

            if observations and var in observations:
                obs = observations[var][sample_idx] if observations[var].ndim > 1 else observations[var]
                ax1.plot(obs, label='Observation', linewidth=2, alpha=0.7)

                # Calculate metrics
                metrics = calculate_all_metrics(obs, pred)
                ax1.text(0.02, 0.98, f"NSE: {metrics['nse']:.3f}\nKGE: {metrics['kge']:.3f}",
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax1.set_xlabel('Time Step')
            ax1.set_ylabel(title)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f'{title} - Time Series')

            # Scatter plot
            if observations and var in observations:
                ax2 = axes[i, 1]
                obs = observations[var][sample_idx] if observations[var].ndim > 1 else observations[var]

                ax2.scatter(obs, pred, alpha=0.5, s=20)

                # 1:1 line
                min_val = min(obs.min(), pred.min())
                max_val = max(obs.max(), pred.max())
                ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')

                ax2.set_xlabel(f'Observed {title}')
                ax2.set_ylabel(f'Predicted {title}')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_title(f'{title} - Scatter Plot')
            else:
                # Just plot histogram if no observations
                ax2 = axes[i, 1]
                ax2.hist(pred, bins=30, alpha=0.7)
                ax2.set_xlabel(title)
                ax2.set_ylabel('Frequency')
                ax2.set_title(f'{title} - Distribution')
                ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Figure saved to: {save_path}")

        plt.show()

    def export_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        output_path: str,
        format: str = 'csv'
    ):
        """
        Export predictions to file

        Args:
            predictions: Dictionary with predictions
            output_path: Path to save predictions
            format: Format to save ('csv', 'netcdf', 'pickle')

        Example:
            >>> predictions = inferencer.predict(forcing, params)
            >>> inferencer.export_predictions(predictions, 'results.csv')
        """
        if format == 'csv':
            df = pd.DataFrame(predictions)
            df.to_csv(output_path, index=False)

        elif format == 'netcdf':
            ds = xr.Dataset({
                key: xr.DataArray(value, dims=['sample'])
                for key, value in predictions.items()
            })
            ds.to_netcdf(output_path)

        elif format == 'pickle':
            pd.to_pickle(predictions, output_path)

        else:
            raise ValueError(f"Unknown format: {format}")

        self.logger.info(f"Predictions exported to: {output_path}")

    def _to_numpy(self, data) -> np.ndarray:
        """Convert various data types to numpy array"""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, xr.DataArray):
            return data.values
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        else:
            return np.array(data)


# Convenience function for quick inference
def quick_inference(
    model_path: str,
    forcing: np.ndarray,
    parameters: np.ndarray,
    config_path: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Quick inference without creating inference object

    Args:
        model_path: Path to model checkpoint
        forcing: Forcing data
        parameters: Parameter data
        config_path: Path to config (optional)

    Returns:
        predictions: Dictionary with predictions

    Example:
        >>> preds = quick_inference('best_model.pth', forcing, params)
        >>> print(preds['Q'])
    """
    inferencer = VICSurrogateInference(model_path, config_path)
    return inferencer.predict(forcing, parameters)
