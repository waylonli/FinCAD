"""
Context-aware decoding (CAD) utilities.
"""

from .decoder import CADConfig, ContextAwareDecoder
from .calibrator import CADCalibrator, AlphaCalibrationResult

__all__ = [
    "CADConfig",
    "ContextAwareDecoder",
    "CADCalibrator",
    "AlphaCalibrationResult",
]
