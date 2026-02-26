"""Adversarial bias discovery module via DSPy.

Optimizes negative-prompt instructions for context-aware decoding (CAD)
using discrete prompt optimization (MIPROv2 / COPRO).
"""
from .config import (
    CalibrationDatasetConfig,
    CalibrationExample,
    DiscoveryConfig,
    OptimizedInstruction,
)
from .builder import NegativePromptBuilder
from .registry import load_instruction, save_instruction
from .calibration_data import build_calibration_dataset
from .optimizer import run_optimization

__all__ = [
    "CalibrationDatasetConfig",
    "CalibrationExample",
    "DiscoveryConfig",
    "NegativePromptBuilder",
    "OptimizedInstruction",
    "build_calibration_dataset",
    "load_instruction",
    "run_optimization",
    "save_instruction",
]
