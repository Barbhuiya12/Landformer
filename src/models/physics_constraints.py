"""
Physics-informed constraints for hydrological models.

This module implements mass conservation constraints
for training hydrological surrogate models. These constraints help ensure
that the model predictions adhere to fundamental physical laws governing
hydrological processes.

The module contains two main classes:
1. MassConservationConstraint: Ensures non-negative fluxes and realistic bounds
2. PhysicalConstraintLoss: Combines enabled physical constraints into a single loss term

These constraints are added to the primary loss function during training to guide
the model toward physically consistent predictions.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Water balance constraint removed from the codebase (kept only mass conservation)


class TemporalSmoothnessConstraint(nn.Module):
    """
    Implements temporal smoothness constraint for Soil Moisture.
    Penalizes rapid changes in soil moisture: Loss += |SM_t - SM_{t-1}|
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        pcfg = getattr(config.training, 'physical_constraints', {}) or {}
        ts_config = pcfg.get('temporal_smoothness', {}) if isinstance(pcfg, dict) else {}
        
        self.weight = ts_config.get('weight', 0.1)
        self.target_var = ts_config.get('target', 'soil_moisture_total')
        logger.info(f"Initialized TemporalSmoothnessConstraint for {self.target_var} with weight={self.weight}")

    def forward(self, task_outputs, inputs, targets=None):
        """
        Args:
            task_outputs: Dict with keys like 'soil_moisture_total' - shape (B, 1) or (B, T)
        """
        sm = task_outputs.get(self.target_var, None)
        if sm is None:
            return 0.0
        
        # If single-step output (B, 1), we can't compute temporal difference
        if sm.dim() <= 1 or sm.shape[-1] <= 1:
            return 0.0
            
        # Calculate absolute difference between consecutive time steps
        diff = torch.abs(sm[:, 1:] - sm[:, :-1])
        loss = diff.mean()
        
        return self.weight * loss


class RunoffConstraint(nn.Module):
    """
    Implements constraints for Discharge/Runoff (Q):
    1. Sparsity (L1): Encourages Q=0 when no rain.
    2. Recession: Q should decrease when P=0.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        pcfg = getattr(config.training, 'physical_constraints', {}) or {}
        rc_config = pcfg.get('runoff_constraint', {}) if isinstance(pcfg, dict) else {}
        
        self.sparsity_weight = rc_config.get('sparsity_weight', 0.01)
        self.recession_weight = rc_config.get('recession_weight', 0.1)
        self.precip_col = rc_config.get('precip_col', 0) # Index of precipitation in inputs
        
        logger.info(f"Initialized RunoffConstraint with sparsity={self.sparsity_weight}, recession={self.recession_weight}")


    def forward(self, task_outputs, inputs, targets=None):
        """
        Args:
            task_outputs: Dict with key 'Q' - shape (B, 1) for single-step or (B, T) for multi-step
            inputs: (B, T, C) - C[0] is assumed to be Precip
        """
        q = task_outputs.get('Q', None)
        if q is None:
            return 0.0
        
        # Squeeze if needed: (B, 1) -> (B,) for sparsity
        q_flat = q.squeeze(-1) if q.dim() > 1 else q
        
        # 1. Sparsity (L1 Regularization) - always applicable
        sparsity_loss = torch.abs(q_flat).mean()
        
        # 2. Recession Constraint (Q_t < Q_{t-1} if P_t = 0)
        # Only applicable if we have temporal sequence in outputs (dim > 1 and time > 1)
        recession_loss = torch.tensor(0.0, device=q.device)
        
        if q.dim() > 1 and q.shape[1] > 1:
            # Multi-step output: can compute recession
            precip = inputs[..., self.precip_col]
            is_dry = (precip < 1e-5).float()
            
            q_diff = q[:, 1:] - q[:, :-1]
            recession_violation = torch.relu(q_diff) * is_dry[:, 1:]
            recession_loss = recession_violation.mean()
        
        total_loss = (self.sparsity_weight * sparsity_loss) + (self.recession_weight * recession_loss)
        return total_loss


class PhysicalConstraintLoss(nn.Module):
    """
    Combines multiple physical constraints into a single loss term.
    """
    
    def __init__(self, config):
        """Initialize physical constraint losses."""
        super().__init__()
        self.constraints = nn.ModuleDict()
        
        # Check if physical_constraints exists
        pcfg = getattr(config.training, 'physical_constraints', None)
        if pcfg is None:
            pcfg = getattr(config.training, 'physics_constraints', None)
        if pcfg is None:
            return
        
        # Add Temporal Smoothness (SM)
        ts_config = pcfg.get('temporal_smoothness', {}) if isinstance(pcfg, dict) else {}
        if isinstance(pcfg, dict) and pcfg.get('enabled', False) and ts_config.get('enabled', False):
            self.constraints['temporal_smoothness'] = TemporalSmoothnessConstraint(config)
            logger.info("Added TemporalSmoothness constraint for Soil Moisture")

        # Add Runoff Constraints (Sparsity + Recession)
        rc_config = pcfg.get('runoff_constraint', {}) if isinstance(pcfg, dict) else {}
        if isinstance(pcfg, dict) and pcfg.get('enabled', False) and rc_config.get('enabled', False):
            self.constraints['runoff_constraint'] = RunoffConstraint(config)
            logger.info("Added Runoff constraints (Sparsity + Recession)")
        
        if not self.constraints:
            logger.info("No physical constraints enabled")
    
    def forward(self, outputs, targets, inputs, task_outputs=None):
        """
        Calculate total physical constraint loss.
        """
        constraint_losses = {}
        total_loss = 0.0
        
        for name, constraint in self.constraints.items():
            if name in ['temporal_smoothness', 'runoff_constraint']:
                if task_outputs is not None:
                    loss = constraint(task_outputs, inputs, targets)
                    constraint_losses[name] = loss.item() if torch.is_tensor(loss) else loss
                    total_loss += loss
            
        constraint_losses['total_constraint'] = total_loss
        
        return constraint_losses
