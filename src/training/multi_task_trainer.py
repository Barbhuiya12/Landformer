"""
Multi-Task VIC Surrogate Trainer - FIXED VERSION
Author: Siddik Barbhuiya
Description: Fixed training pipeline for multi-task VIC LSTM model

Key fixes:
- Proper tensor shape handling for predictions and targets
- Added tqdm progress bars
- Fixed NSE/KGE calculation with correct shape alignment
- Better error handling and debugging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from tqdm import tqdm
from ..core.config import UnifiedConfig
from ..models.multi_task_lstm import MultiTaskVICLSTM, MultiTaskLoss
from ..data.dataset import UnifiedSurrogateDataset
from ..utils.hydro_metrics import nash_sutcliffe_efficiency, kling_gupta_efficiency, pearson_correlation


class MultiTaskTrainer:
    """
    Multi-Task VIC Surrogate Model Trainer - FIXED VERSION
    
    Features:
    - Multi-task learning with shared and task-specific layers
    - Task balancing and uncertainty weighting
    - Gradient analysis for task interference
    - Individual and joint task evaluation
    - FIXED: Proper tensor shape handling
    - ADDED: tqdm progress bars
    """
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)
        
        # Task configuration
        self.task_names = config.data.variables['target']  # Should match config order!
        self.num_tasks = len(self.task_names)
        
        # Task order confirmed
        self.logger.info(f"Multi-task training order: {self.task_names}")
        
        # Store training dataset reference for denormalization
        self.train_dataset = None
        
        # Model components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'task_losses': {task: [] for task in self.task_names},
            'task_metrics': {task: {'nse': [], 'kge': [], 'corr': []} for task in self.task_names}
        }
        
        # Gradient analysis
        self.gradient_similarities = []
        self.task_weights_history = []
    
    def build_model(self, train_dataset: UnifiedSurrogateDataset = None) -> MultiTaskVICLSTM:
        """Build simplified multi-task VIC LSTM model"""
        
        # Calculate input dimensions
        n_forcing = len(self.config.data.variables['forcing'])
        n_parameters = len(self.config.data.variables['parameters'])
        n_lag_features = self.config.get_total_lag_features()
        
        n_exo = n_forcing + n_parameters
        
        # Extract SM normalized range from training dataset (optional for simplified model)
        sm_range = None
        if train_dataset is not None and 'sm' in self.task_names:
            try:
                sm_data_sample = []
                
                # Sample some SM data to find range
                sample_size = min(1000, len(train_dataset))
                for i in range(0, sample_size, len(train_dataset) // sample_size):
                    try:
                        _, targets, _ = train_dataset[i]
                        sm_idx = self.task_names.index('sm')
                        if targets.shape[-1] > sm_idx:
                            sm_data_sample.extend(targets[:, sm_idx].flatten().tolist())
                    except:
                        continue
                
                if sm_data_sample:
                    sm_min = min(sm_data_sample)
                    sm_max = max(sm_data_sample)
                    sm_mean = sum(sm_data_sample) / len(sm_data_sample)
                    sm_range = {'min': sm_min, 'max': sm_max, 'mean': sm_mean}
                    self.logger.info(f"SM range stats: mean={sm_mean:.4f}, range=[{sm_min:.4f}, {sm_max:.4f}]")
                    
            except Exception as e:
                self.logger.warning(f"Could not extract SM range: {e}")
        
        # Create simplified multi-task model
        model = MultiTaskVICLSTM(
            n_exo=n_exo,
            n_lag_features=n_lag_features,
            hidden_size=self.config.model.hidden_size,
            dropout=self.config.model.dropout,
            task_names=self.task_names,
            sm_range=sm_range
        )
        
        return model.to(self.device)
    
    def denormalize_predictions(self, predictions: np.ndarray, targets: np.ndarray) -> tuple:
        """
        Denormalize predictions and targets to original scale using training dataset statistics
        
        CRITICAL: This fixes the root cause - NSE/KGE must be calculated on denormalized values!
        """
        if self.train_dataset is None:
            self.logger.warning("No training dataset available for denormalization. Using normalized values.")
            return predictions, targets
        
        try:
            denorm_preds = predictions.copy()
            denorm_targets = targets.copy()
            
            # Get normalization method
            norm_method = getattr(self.train_dataset, 'normalization_method', 'zscore')
            
            # Denormalize each task
            for i, task_name in enumerate(self.task_names):
                if i >= predictions.shape[1] or i >= targets.shape[1]:
                    continue
                    
                # Get normalization statistics for this task
                mean, std = self.train_dataset.get_normalization_stats(task_name)
                
                if norm_method == 'zscore':
                    # Reverse z-score: x_original = x_norm * std + mean
                    denorm_preds[:, i] = predictions[:, i] * std + mean
                    denorm_targets[:, i] = targets[:, i] * std + mean
                elif norm_method == 'minmax':
                    # Reverse min-max: x_original = x_norm * (max-min) + min
                    # Note: 'mean' stores min, 'std' stores (max-min)
                    denorm_preds[:, i] = predictions[:, i] * std + mean
                    denorm_targets[:, i] = targets[:, i] * std + mean
                
                elif norm_method == 'log_zscore':
                    # Reverse z-score
                    log_preds = predictions[:, i] * std + mean
                    log_targets = targets[:, i] * std + mean
                    
                    # Reverse log1p (expm1)
                    denorm_preds[:, i] = np.expm1(log_preds)
                    denorm_targets[:, i] = np.expm1(log_targets)
                    
                    # Ensure non-negative
                    denorm_preds[:, i] = np.maximum(denorm_preds[:, i], 0)
                    denorm_targets[:, i] = np.maximum(denorm_targets[:, i], 0)
                
                # Log denormalization stats occasionally
                if task_name == 'sm' and np.random.rand() < 0.01:  # 1% of the time
                    self.logger.debug(f"SM denormalization: mean={mean:.4f}, std={std:.4f}")
            
            return denorm_preds, denorm_targets
            
        except Exception as e:
            self.logger.error(f"Error during denormalization: {e}")
            self.logger.warning("Using normalized values for metrics calculation.")
            return predictions, targets
    
    def build_criterion(self) -> MultiTaskLoss:
        """Build multi-task loss function"""
        
        # Get task balancing strategy from config
        multi_task_config = getattr(self.config.training, 'multi_task', {})
        task_balancing = multi_task_config.get('task_balancing', 'uncertainty_weighting')
        
        # Initialize task weights and uncertainty weighting based on config
        task_weights = None
        use_uncertainty_weighting = True  # Default
        
        if task_balancing == 'manual':
            # Use manual weights from config
            task_weights = multi_task_config.get('task_weights', {})
            use_uncertainty_weighting = False
            self.logger.info(f"Using manual task weights from config: {task_weights}")
            
        elif task_balancing == 'equal':
            # Equal weights for all tasks
            task_weights = {task: 1.0 for task in self.task_names}
            use_uncertainty_weighting = False
            self.logger.info("Using equal task weights")
            
        elif task_balancing == 'uncertainty_weighting':
            # Use uncertainty weighting (homoscedastic)
            use_uncertainty_weighting = True
            task_weights = None
            self.logger.info("Using uncertainty-based task weighting")
        
        # Fallback to legacy multi_output weights if available
        if task_weights is None and hasattr(self.config.training, 'multi_output') and self.config.training.multi_output:
            if 'loss_weights' in self.config.training.multi_output:
                weights = self.config.training.multi_output['loss_weights']
                task_weights = {task: weights[i] for i, task in enumerate(self.task_names)}
                use_uncertainty_weighting = False
                self.logger.info(f"Using legacy multi_output weights: {task_weights}")
        
        # Build criterion with configurable weights
        criterion = MultiTaskLoss(
            task_weights=task_weights,
            loss_type='NSE',  # Use Nash-Sutcliffe Efficiency for hydrological modeling
            use_uncertainty_weighting=use_uncertainty_weighting
        )
        
        return criterion.to(self.device)
    
    def setup_training(self, train_dataset: UnifiedSurrogateDataset = None):
        """Setup model, optimizer, and loss function"""
        
        # Store training dataset for denormalization
        self.train_dataset = train_dataset
        
        # Build model with SM statistics from training data
        self.model = self.build_model(train_dataset)
        self.logger.info(f"Built multi-task model with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        # Build loss function
        self.criterion = self.build_criterion()
        
        # Setup optimizer for simplified architecture
        # Get shared layer and task-specific head parameters
        shared_params = list(self.model.shared_layer.parameters())
        
        # Task-specific head parameters (separate optimizer for SM if desired)
        if 'sm' in self.task_names:
            sm_head_params = list(self.model.task_heads['sm'].parameters())
            other_head_params = []
            for task in self.task_names:
                if task != 'sm':
                    other_head_params.extend(list(self.model.task_heads[task].parameters()))
            
            # Main optimizer for shared layers and non-SM heads
            self.optimizer = optim.Adam(
                shared_params + other_head_params,
                lr=self.config.training.learning_rate,
                weight_decay=1e-5
            )
            
            # Special optimizer for soil moisture head with higher learning rate
            self.sm_optimizer = optim.Adam(
                sm_head_params,
                lr=self.config.training.learning_rate * 2.0,  # 2x higher learning rate
                weight_decay=1e-6  # Lower weight decay
            )
            
            self.logger.info(f"Created separate SM optimizer with {self.config.training.learning_rate * 2.0:.6f} learning rate")
        else:
            # Single optimizer for all parameters
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=1e-5
            )
            self.sm_optimizer = None
            
        self.logger.info(f"Optimizer setup complete for simplified architecture")
        
        # Setup scheduler (if configured)
        if hasattr(self.config.training, 'scheduler') and self.config.training.scheduler.get('enabled', False):
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=self.config.training.scheduler.get('min_lr', 1e-6)
            )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with progress bar"""
        
        self.model.train()
        epoch_losses = {f'{task}_loss': 0.0 for task in self.task_names}
        epoch_losses['total_loss'] = 0.0
        num_batches = 0
        
        # Track gradient similarities
        batch_grad_similarities = []
        
        # Progress bar for training
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                         desc=f"Epoch {self.current_epoch:3d} - Train", leave=False)
        
        for batch_idx, batch in train_pbar:
            # Unpack batch - dataset returns (inputs, targets, metadata)
            inputs, targets, metadata = batch
            
            # Move data to device
            inputs = inputs.to(self.device)  # (batch_size, seq_len, n_features)
            targets = targets.to(self.device)  # (batch_size, seq_len, n_targets) or (batch_size, n_targets)
            
            # Handle 3D target tensor - take last timestep for multi-task prediction
            if len(targets.shape) == 3:  # (batch_size, seq_len, n_targets)
                if batch_idx == 0:  # Only log once per epoch
                    self.logger.warning(f"Unusual target shape: {targets.shape}")
                targets = targets[:, -1, :]  # Take last timestep: (batch_size, n_targets)
            
            # Forward pass
            self.optimizer.zero_grad()
            self.sm_optimizer.zero_grad()
            outputs = self.model(inputs, return_task_outputs=True)
            
            # Calculate losses
            losses = self.criterion(
                predictions=outputs['final_prediction'],
                targets=targets,
                task_names=self.task_names
            )
            
            # Get the base total loss
            total_loss = losses['total_loss']
            
            # Enhanced loss + SM boosting (weight preservation disabled)
            ewc_penalty = 0.0  # No EWC penalty since weight preservation is disabled
            
            if losses['sm_loss'] > 1.0:
                # Boost the total loss with extra SM component + EWC penalty
                enhanced_total_loss = total_loss + losses['sm_loss'] * 1.0 + ewc_penalty
            else:
                # Regular loss + EWC penalty to preserve good patterns
                enhanced_total_loss = total_loss + ewc_penalty
            
            # Log EWC penalty if significant
            if ewc_penalty > 0.01:
                epoch_losses['ewc_penalty'] = ewc_penalty.item()
            
            # Single backward pass with enhanced loss
            enhanced_total_loss.backward()
            
            # Gradient analysis (every 10 batches to save compute)
            if batch_idx % 10 == 0:
                grad_sim = self.analyze_gradient_similarity()
                if grad_sim is not None:
                    batch_grad_similarities.append(grad_sim)
            
            # Gradient clipping (if configured)
            if hasattr(self.config.training, 'optimizations'):
                clip_val = self.config.training.optimizations.get('gradient_clipping', None)
                if clip_val:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
            
            # Step both optimizers
            self.optimizer.step()
            self.sm_optimizer.step()
            
            # Accumulate losses
            for loss_name, loss_value in losses.items():
                epoch_losses[loss_name] += loss_value.item()
            
            # Update progress bar
            task_losses_str = " | ".join([f"{task}: {losses.get(f'{task}_loss', torch.tensor(0)).item():.4f}" 
                                        for task in self.task_names])
            train_pbar.set_postfix_str(f"Loss: {total_loss.item():.4f} | {task_losses_str}")
            
            num_batches += 1
        
        # Average losses
        for loss_name in epoch_losses:
            epoch_losses[loss_name] /= num_batches
        
        # Store gradient similarities
        if batch_grad_similarities:
            avg_grad_sim = np.mean(batch_grad_similarities, axis=0)
            self.gradient_similarities.append(avg_grad_sim)
        
        return epoch_losses
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch with FIXED shape handling"""
        
        self.model.eval()
        epoch_losses = {f'{task}_loss': 0.0 for task in self.task_names}
        epoch_losses['total_loss'] = 0.0
        
        # Metrics for each task - use lists to collect individual values
        task_metrics = {
            task: {'predictions': [], 'targets': []} 
            for task in self.task_names
        }
        
        num_batches = 0
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch:3d} - Valid", leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                # Unpack batch - dataset returns (inputs, targets, metadata)
                inputs, targets, metadata = batch
                
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Handle 3D target tensor - take last timestep for multi-task prediction
                if len(targets.shape) == 3:  # (batch_size, seq_len, n_targets)
                    if num_batches == 0:  # Only log once per epoch
                        self.logger.warning(f"Unusual target shape: {targets.shape}")
                    targets = targets[:, -1, :]  # Take last timestep: (batch_size, n_targets)
                
                # Forward pass
                outputs = self.model(inputs, return_task_outputs=True)
                
                # Calculate losses
                losses = self.criterion(
                    predictions=outputs['final_prediction'],
                    targets=targets,
                    task_names=self.task_names
                )
                
                # Accumulate losses
                for loss_name, loss_value in losses.items():
                    epoch_losses[loss_name] += loss_value.item()
                
                # Store predictions for metrics with proper shape handling
                final_pred = outputs['final_prediction'].cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                # CRITICAL FIX: Denormalize for proper NSE/KGE calculation
                denorm_pred, denorm_targets = self.denormalize_predictions(final_pred, targets_np)
                
                # Log basic validation info on first batch
                if num_batches == 0:
                    self.logger.debug(f"Validation batch shapes - Predictions: {final_pred.shape}, Targets: {targets_np.shape}")
                    for i, task_name in enumerate(self.task_names):
                        if i < final_pred.shape[1] and i < targets_np.shape[1]:
                            # Check for NaN values
                            if np.isnan(final_pred[:, i]).any():
                                self.logger.error(f"{task_name} predictions contain NaN values!")
                            if np.isnan(denorm_pred[:, i]).any():
                                self.logger.error(f"{task_name} denormalized predictions contain NaN values!")
                
                for i, task in enumerate(self.task_names):
                    # Use DENORMALIZED predictions/targets for metrics
                    task_pred = denorm_pred[:, i].flatten()
                    
                    # Handle target shape properly
                    if len(denorm_targets.shape) == 2:
                        if denorm_targets.shape[1] == len(self.task_names):
                            # Targets are [batch_size, n_tasks] - correct format
                            task_target = denorm_targets[:, i].flatten()
                        else:
                            # Targets might be [batch_size, 1] - need to handle differently
                            self.logger.warning(f"Unexpected target shape: {denorm_targets.shape}")
                            task_target = denorm_targets.flatten()[:len(task_pred)]  # Truncate to match pred size
                    else:
                        # Handle other shapes
                        self.logger.warning(f"Unusual target shape: {denorm_targets.shape}")
                        task_target = denorm_targets.flatten()[:len(task_pred)]  # Truncate to match pred size
                    
                    # Ensure same length
                    min_len = min(len(task_pred), len(task_target))
                    task_pred = task_pred[:min_len]
                    task_target = task_target[:min_len]
                    
                    task_metrics[task]['predictions'].extend(task_pred.tolist())
                    task_metrics[task]['targets'].extend(task_target.tolist())
                
                # Update progress bar
                val_pbar.set_postfix_str(f"Loss: {losses['total_loss'].item():.4f}")
                
                num_batches += 1
        
        # Average losses
        for loss_name in epoch_losses:
            epoch_losses[loss_name] /= num_batches
        
        # Calculate detailed metrics for each task with debugging
        detailed_metrics = {}
        for task in self.task_names:
            # Convert lists to numpy arrays
            pred = np.array(task_metrics[task]['predictions'])
            target = np.array(task_metrics[task]['targets'])
            
            
            # Calculate metrics only if we have data and shapes match
            if len(pred) > 0 and len(target) > 0 and pred.shape == target.shape:
                try:
                    # Convert to torch tensors
                    pred_tensor = torch.tensor(pred).float()
                    target_tensor = torch.tensor(target).float()
                    
                    # Calculate NSE and KGE using our loss functions for consistency
                    nse_val = nash_sutcliffe_efficiency(pred_tensor, target_tensor).item()
                    kge_val = kling_gupta_efficiency(pred_tensor, target_tensor).item()
                    corr_val = pearson_correlation(pred_tensor, target_tensor).item()
                    
                    detailed_metrics[f'{task}_nse'] = nse_val
                    detailed_metrics[f'{task}_kge'] = kge_val
                    detailed_metrics[f'{task}_corr'] = corr_val
                    
                        
                except Exception as e:
                    self.logger.warning(f"Error calculating metrics for {task}: {e}")
                    detailed_metrics[f'{task}_nse'] = 0.0
                    detailed_metrics[f'{task}_kge'] = 0.0
                    detailed_metrics[f'{task}_corr'] = 0.0
            else:
                if len(pred) > 0 and len(target) > 0 and pred.shape != target.shape:
                    self.logger.warning(f"Task {task}: Shape mismatch - pred {pred.shape} vs target {target.shape}")
                elif len(pred) == 0:
                    self.logger.warning(f"Task {task}: Empty predictions")
                elif len(target) == 0:
                    self.logger.warning(f"Task {task}: Empty targets")
                detailed_metrics[f'{task}_nse'] = 0.0
                detailed_metrics[f'{task}_kge'] = 0.0
                detailed_metrics[f'{task}_corr'] = 0.0
        
        epoch_losses.update(detailed_metrics)
        
        # Performance-Aware Weight Preservation: Monitor and preserve good weights
        for task in self.task_names:
            if f'{task}_nse' in detailed_metrics:
                nse_score = detailed_metrics[f'{task}_nse']
                
                # Weight preservation disabled - skip this functionality
                if nse_score > 0.7:  # Just log good performance
                    self.logger.info(f"Task {task} achieving good performance (NSE: {nse_score:.4f})")
                    # Weight preservation would normally happen here, but it's disabled
        
        return epoch_losses
    
    def analyze_gradient_similarity(self) -> Optional[np.ndarray]:
        """
        Analyze gradient similarity between tasks
        Helps identify task interference/cooperation
        """
        task_grads = self.model.get_task_gradients()
        
        if len(task_grads) < 2:
            return None
        
        # Calculate cosine similarities between task gradients
        grad_similarities = np.zeros((self.num_tasks, self.num_tasks))
        task_list = list(task_grads.keys())
        
        for i, task1 in enumerate(task_list):
            for j, task2 in enumerate(task_list):
                if i != j:
                    grad1 = task_grads[task1]
                    grad2 = task_grads[task2]
                    
                    # Handle different gradient vector sizes by padding shorter one
                    if grad1.size(0) != grad2.size(0):
                        max_size = max(grad1.size(0), grad2.size(0))
                        if grad1.size(0) < max_size:
                            pad_size = max_size - grad1.size(0)
                            grad1 = torch.cat([grad1, torch.zeros(pad_size, device=grad1.device)])
                        if grad2.size(0) < max_size:
                            pad_size = max_size - grad2.size(0)
                            grad2 = torch.cat([grad2, torch.zeros(pad_size, device=grad2.device)])
                    
                    # Cosine similarity
                    cos_sim = torch.nn.functional.cosine_similarity(
                        grad1.unsqueeze(0), 
                        grad2.unsqueeze(0)
                    ).item()
                    grad_similarities[i, j] = cos_sim
        
        return grad_similarities
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Main training loop with progress bars and better logging
        """
        self.logger.info("Starting multi-task training...")
        
        # Main epoch progress bar
        epoch_pbar = tqdm(range(self.config.training.epochs), 
                         desc="Training Progress", 
                         position=0)
        
        for epoch in epoch_pbar:
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Update main progress bar
            epoch_pbar.set_postfix({
                'Train': f"{train_metrics['total_loss']:.4f}",
                'Valid': f"{val_metrics['total_loss']:.4f}",
                'Best': f"{self.best_val_loss:.4f}"
            })
            
            # Log progress
            self.log_epoch_progress(train_metrics, val_metrics)
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.best_epoch = epoch
                self.save_checkpoint(is_best=True)
            
            # Store history
            self.training_history['train_loss'].append(train_metrics['total_loss'])
            self.training_history['val_loss'].append(val_metrics['total_loss'])
            
            for task in self.task_names:
                self.training_history['task_losses'][task].append(val_metrics[f'{task}_loss'])
                self.training_history['task_metrics'][task]['nse'].append(val_metrics.get(f'{task}_nse', 0.0))
                self.training_history['task_metrics'][task]['kge'].append(val_metrics.get(f'{task}_kge', 0.0))
                self.training_history['task_metrics'][task]['corr'].append(val_metrics.get(f'{task}_corr', 0.0))
            
            # Early stopping check
            if self.should_stop_early(epoch):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Training completed
        self.logger.info(f"Training completed. Best epoch: {self.best_epoch}, Best loss: {self.best_val_loss:.6f}")
        
        return {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'gradient_similarities': self.gradient_similarities
        }
    
    def log_epoch_progress(self, train_metrics: Dict, val_metrics: Dict):
        """Log training progress"""
        
        # Main losses
        self.logger.info(
            f"Epoch {self.current_epoch:3d} | "
            f"Train Loss: {train_metrics['total_loss']:.6f} | "
            f"Val Loss: {val_metrics['total_loss']:.6f}"
        )
        
        # Task-specific losses
        task_loss_str = " | ".join([
            f"{task}: {val_metrics[f'{task}_loss']:.6f}"
            for task in self.task_names
        ])
        self.logger.info(f"  Task Losses: {task_loss_str}")
        
        # Task-specific metrics (NSE and KGE) - Always show them
        nse_values = [val_metrics.get(f'{task}_nse', 0.0) for task in self.task_names]
        kge_values = [val_metrics.get(f'{task}_kge', 0.0) for task in self.task_names]
        corr_values = [val_metrics.get(f'{task}_corr', 0.0) for task in self.task_names]
        
        # NSE metrics
        nse_str = " | ".join([
            f"{task}_NSE: {val_metrics.get(f'{task}_nse', 0.0):.4f}"
            for task in self.task_names
        ])
        self.logger.info(f"  Task NSE: {nse_str}")
        
        # KGE metrics  
        kge_str = " | ".join([
            f"{task}_KGE: {val_metrics.get(f'{task}_kge', 0.0):.4f}"
            for task in self.task_names
        ])
        self.logger.info(f"  Task KGE: {kge_str}")
        
        # Correlation metrics
        corr_str = " | ".join([
            f"{task}_CORR: {val_metrics.get(f'{task}_corr', 0.0):.4f}"
            for task in self.task_names
        ])
        self.logger.info(f"  Task CORR: {corr_str}")
        
        # Log task weights being used
        if hasattr(self.criterion, 'task_weights') and self.criterion.task_weights:
            task_weight_str = " | ".join([
                f"{task}_Weight: {self.criterion.task_weights.get(task, 1.0):.1f}x"
                for task in self.task_names
            ])
            self.logger.info(f"  Manual Task Weights: {task_weight_str}")
        elif hasattr(self.criterion, 'log_vars') and self.criterion.use_uncertainty_weighting:
            precision_weights = torch.exp(-self.criterion.log_vars).detach().cpu().numpy()
            log_vars = self.criterion.log_vars.detach().cpu().numpy()
            
            precision_str = " | ".join([
                f"{task}_Precision: {precision_weights[i]:.4f}"
                for i, task in enumerate(self.task_names)
            ])
            self.logger.info(f"  Uncertainty Precision: {precision_str}")
            
            logvar_str = " | ".join([
                f"{task}_LogVar: {log_vars[i]:.4f}"
                for i, task in enumerate(self.task_names)
            ])
            self.logger.info(f"  Uncertainty LogVars: {logvar_str}")
    
    def should_stop_early(self, current_epoch: int) -> bool:
        """Check if training should stop early"""
        patience = getattr(self.config.training, 'patience', 20)
        return (current_epoch - self.best_epoch) >= patience
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save path
        save_dir = Path(self.config.out_model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            save_path = save_dir / 'best_multi_task_model.pth'
            torch.save(checkpoint, save_path)
            self.logger.info(f"Saved best model to {save_path}")
        
        # Also save regular checkpoint
        if self.current_epoch % self.config.training.epoch_save == 0:
            save_path = save_dir / f'multi_task_model_epoch_{self.current_epoch}.pth'
            torch.save(checkpoint, save_path)