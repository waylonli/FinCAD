"""DSPy Module wrapping the MemoryProbe signature."""
from __future__ import annotations

try:
    import dspy
except ImportError:
    dspy = None  # type: ignore[assignment]

if dspy is not None:
    from .signatures import MemoryProbe

    from .signatures import CALIBRATION_TASK

    class MemoryProbeModule(dspy.Module):
        """Predict stock direction from parametric memory alone."""

        def __init__(self) -> None:
            super().__init__()
            self.probe = dspy.Predict(MemoryProbe)

        def forward(self, entity: str, date: str) -> dspy.Prediction:
            # MIPROv2 only sees/optimises the signature (entity, date → direction).
            # We append CALIBRATION_TASK at runtime so the LM gets F_task during
            # metric evaluation, but the proposer never sees it.
            entity_with_task = f"{entity}\n\n{CALIBRATION_TASK}"
            pred = self.probe(entity=entity_with_task, date=date)
            # Normalize direction to "up" or "down"
            raw = pred.direction.strip().lower()
            if "up" in raw:
                pred.direction = "up"
            elif "down" in raw:
                pred.direction = "down"
            else:
                pred.direction = raw
            return pred
