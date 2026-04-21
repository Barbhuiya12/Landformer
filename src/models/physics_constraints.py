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

class MassConservationConstraint(nn.Module):
    """
    Implements mass conservation constraints for hydrological variables.
    Ensures all outputs are non-negative (user request).
    
    This constraint penalizes any negative values across the model outputs,
    effectively enforcing y >= 0 element-wise.
    - No upper bound penalty for soil moisture (SM) is applied
    
    These constraints are particularly important for preventing unphysical
    predictions that might occur during early training stages.
    
    Attributes:
        weight (float): Weight multiplier for the constraint loss
        check_negative (bool): Whether to check for negative fluxes
        min_bounds (dict): Minimum bounds for variables
        max_bounds (dict): Maximum bounds for variables
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Support dict-style config and legacy alt key
        pcfg = getattr(config.training, 'physical_constraints', None)
        if pcfg is None:
            pcfg = getattr(config.training, 'physics_constraints', None)
        pcfg = pcfg or {}
        mc_config = pcfg.get('mass_conservation', {}) if isinstance(pcfg, dict) else {}

        self.weight = mc_config.get('weight', 0.05)
        self.check_negative = mc_config.get('check_negative_fluxes', True)
        
        # Physical bounds for variables (in normalized space)
        bounds_config = pcfg.get('bounds', {})
        self.min_bounds = bounds_config.get('min', {})
        # self.max_bounds = bounds_config.get('max', {})
        
        logger.info(f"Initialized MassConservationConstraint with weight={self.weight}")
    
    def forward(self, outputs, targets, inputs):
        """
        Calculate mass conservation constraint loss.
        
        Args:
            outputs: Model predictions [batch, n_outputs]
            targets: Target values [batch, n_outputs]
            inputs: Input features [batch, seq_len, n_features]
        
        Returns:
            Mass conservation loss
        """
        loss = 0.0
        
        if self.check_negative:
            # Penalize any negative value across all outputs
            negative_vals = torch.relu(-outputs)
            loss = negative_vals.mean()
        
        return self.weight * loss

class WaterBalance(nn.Module):
    """
    Implements water balance constraint for hydrological models.
    Ensures long-term P ≈ ET + Q (ΔS ≈ 0).
    Works on annual (or long aggregation) scale.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Support dict-style config and legacy alt key
        pcfg = getattr(config.training, 'physical_constraints', None)
        if pcfg is None:
            pcfg = getattr(config.training, 'physics_constraints', None)
        pcfg = pcfg or {}
        mc_config = pcfg.get('water_balance', {}) if isinstance(pcfg, dict) else {}

        self.weight = mc_config.get('weight', 0.05)

    def forward(self, preds: Dict[str, torch.Tensor], 
                forcings: torch.Tensor, 
                targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute water balance constraint loss.

        Args:
            preds: dict of predictions { "evap": (B, T), "Q": (B, T), ... }
            forcings: input forcing tensor (B, T, n_features), must include precipitation as first/known channel
            targets: optional ground truth targets for checking balance (not required)

        Returns:
            dict: { "water_balance": scalar loss, "total_constraint": weighted loss }
        """
        # --- Extract forcing precipitation (assume forcing[...,0] = precip) ---
        P = forcings[..., 0]          # (B, T)
        ET = preds.get("evap", None)  # (B, T) or (B,)
        Q  = preds.get("Q", None)     # (B, T) or (B,)

        if ET is None or Q is None:
            return {"water_balance": torch.tensor(0.0, device=forcings.device),
                    "total_constraint": torch.tensor(0.0, device=forcings.device)}

        # Align dimensions (ensure seq_len dimension)
        if ET.dim() == 2 and P.dim() == 3:  # (B,T) vs (B,T,1)
            P = P.squeeze(-1)

        # --- Annual/long-term aggregation ---
        # Sum over time dimension (T)
        P_annual  = P.sum(dim=1)   # (B,)
        ET_annual = ET.sum(dim=1)  # (B,)
        Q_annual  = Q.sum(dim=1)   # (B,)

        # --- Compute residual ---
        residual = P_annual - (ET_annual + Q_annual)  # (B,)

        # --- Constraint loss (MSE of residuals) ---
        wb_loss = torch.mean(residual ** 2)

        return {
            "water_balance": wb_loss,
            "total_constraint": self.weight * wb_loss
        }


class PhysicalConstraintLoss(nn.Module):
    """
    Combines multiple physical constraints into a single loss term.
    
    This class serves as a container for all enabled physical constraints,
    managing their initialization based on configuration and computing
    their combined contribution to the total loss function.
    
    The class dynamically initializes constraint modules based on the
    configuration settings and only includes enabled constraints in
    the forward pass computation.
    
    Attributes:
        constraints (nn.ModuleDict): Dictionary of enabled constraint modules
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
        
        # Add mass conservation constraint if enabled
        mc_config = pcfg.get('mass_conservation', {}) if isinstance(pcfg, dict) else {}
        if isinstance(pcfg, dict) and pcfg.get('enabled', False) and mc_config.get('enabled', False):
            self.constraints['mass_conservation'] = MassConservationConstraint(config)
        
        # Add water balance constraint if enabled
        wb_config = pcfg.get('water_balance', {}) if isinstance(pcfg, dict) else {}
        if isinstance(pcfg, dict) and pcfg.get('enabled', False) and wb_config.get('enabled', False):
            self.constraints['water_balance'] = WaterBalance(config)
            logger.info("Added WaterBalance constraint for annual P ≈ ET + Q")
        
        if not self.constraints:
            logger.info("No physical constraints enabled")
    
    def forward(self, outputs, targets, inputs, task_outputs=None):
        """
        Calculate total physical constraint loss.
        
        Args:
            outputs: Model predictions (tensor or dict)
            targets: Target values
            inputs: Input features [batch, seq_len, features] 
            task_outputs: Dict of task-specific outputs {'evap': tensor, 'Q': tensor} for multi-task
        
        Returns:
            Dictionary with individual and total constraint losses
        """
        constraint_losses = {}
        total_loss = 0.0
        
        for name, constraint in self.constraints.items():
            if name == 'water_balance' and task_outputs is not None:
                # Water balance needs task-specific outputs and forcing data
                loss_dict = constraint(task_outputs, inputs, targets)
                loss = loss_dict['total_constraint']
                constraint_losses[name] = loss.item() if torch.is_tensor(loss) else loss
                # Also store individual water balance loss for logging
                constraint_losses['water_balance_raw'] = loss_dict['water_balance'].item()
            else:
                # Standard constraints (mass conservation)
                loss = constraint(outputs, targets, inputs)
                constraint_losses[name] = loss.item() if torch.is_tensor(loss) else loss
            
            total_loss += loss
        
        constraint_losses['total_constraint'] = total_loss
        
        return constraint_losses
