"""DSPy Module wrapping the MemoryProbe signature."""
from __future__ import annotations

try:
    import dspy
except ImportError:
    dspy = None  # type: ignore[assignment]

if dspy is not None:
    from .signatures import MemoryProbe

    class MemoryProbeModule(dspy.Module):
        """Predict outcome from parametric memory alone."""

        def __init__(self) -> None:
            super().__init__()
            self.probe = dspy.Predict(MemoryProbe)

        def forward(self, task: str) -> dspy.Prediction:
            return self.probe(task=task)
