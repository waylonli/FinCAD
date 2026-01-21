"""
Vector steering toolkit for removing look-ahead bias in financial LLMs.

This package provides:
- Adapters for different inference backends (Transformers today, vLLM-ready hooks).
- A steering controller that scans layers with probes, extracts bias vectors, and
  applies them during generation.
- Reusable dataset helpers for memory vs logic contrast pairs.
"""

from .adapters import AdapterInitConfig, BaseLMAdapter, TransformersAdapter, VLLMAdapter
from .datasets import (
    build_financial_contrast_pairs,
    build_entity_defocus_pairs,
    get_contrast_pairs,
)
from .steering import ScanSettings, SteeringController

__all__ = [
    "BaseLMAdapter",
    "TransformersAdapter",
    "VLLMAdapter",
    "AdapterInitConfig",
    "ScanSettings",
    "SteeringController",
    "build_financial_contrast_pairs",
    "build_entity_defocus_pairs",
    "get_contrast_pairs",
]
