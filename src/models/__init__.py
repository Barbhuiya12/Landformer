"""
Models module for Swin Transformer surrogate model
"""

from .swin_transformer import (
    SwinTransformerHydro,
    MultiTaskSwinTransformer
)
from .losses import LossFunction
from .physics_constraints import PhysicalConstraintLoss

__all__ = [
    'SwinTransformerHydro',
    'MultiTaskSwinTransformer',
    'LossFunction',
    'PhysicalConstraintLoss'
]
