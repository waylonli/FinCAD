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

try:
    from .discovery import NegativePromptBuilder, OptimizedInstruction

    __all__ += ["NegativePromptBuilder", "OptimizedInstruction"]
except ImportError:
    pass
