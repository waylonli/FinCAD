"""Dataclasses for the adversarial bias discovery module."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CalibrationExample:
    """A single calibration data point derived from historical prices."""

    ticker: str
    date: str  # YYYY-MM-DD
    future_return: float
    direction: str  # "up" or "down"


@dataclass
class CalibrationDatasetConfig:
    """Parameters for building the calibration dataset from price data."""

    price_csv: str
    date_range: tuple[str, str] = ("2005-01-01", "2015-01-01")
    forward_days: int = 63
    max_examples: int = 200
    sample_freq: str = "QE"
    min_abs_return: float = 0.05


@dataclass
class DiscoveryConfig:
    """Top-level configuration for the discovery optimisation run."""

    model_name: str
    optimizer: str = "MIPROv2"
    num_candidates: int = 10
    num_trials: int = 30
    calibration: CalibrationDatasetConfig = field(default_factory=CalibrationDatasetConfig)
    output_dir: str = "results/discovery"
    server_url: Optional[str] = None


@dataclass
class OptimizedInstruction:
    """An optimized memory-activation instruction, JSON-serializable."""

    instruction: str
    model_name: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    logit_gap_profile: Optional[Dict[str, Any]] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> OptimizedInstruction:
        return cls(
            instruction=d["instruction"],
            model_name=d["model_name"],
            score=d["score"],
            metadata=d.get("metadata", {}),
            logit_gap_profile=d.get("logit_gap_profile"),
        )

    @classmethod
    def from_json(cls, s: str) -> OptimizedInstruction:
        return cls.from_dict(json.loads(s))
