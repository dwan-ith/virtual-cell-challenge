"""Hybrid zero-shot perturbation model for Arc Virtual Cell Challenge 2026.

Public API (v2 stack):
    HybridPredictor      -- LFC-space ensemble (prior + residual [+ Stack])
    LFCTransferPrior     -- similarity-weighted cross-context transfer
    ResidualLFCModel     -- basal-conditioned residual corrector
    DECalibration        -- metric-aligned LFC post-processing
    PseudobulkStore      -- harmonized observed LFC store built from raw screens
    build_submission     -- rule-checked AnnData writer
"""

from .ensemble import HybridPredictor
from .prior import LFCTransferPrior
from .residual import ResidualConfig, ResidualLFCModel
from .calibrate import DECalibration
from .dataset import PseudobulkStore, build_pseudobulk_store
from .submission import build_submission, validate_submission

__all__ = [
    "HybridPredictor",
    "LFCTransferPrior",
    "ResidualLFCModel",
    "ResidualConfig",
    "DECalibration",
    "PseudobulkStore",
    "build_pseudobulk_store",
    "build_submission",
    "validate_submission",
]
__version__ = "1.0.0"
