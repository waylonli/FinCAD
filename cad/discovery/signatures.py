"""DSPy Signature for memory probing."""
from __future__ import annotations

try:
    import dspy
except ImportError:
    dspy = None  # type: ignore[assignment]


if dspy is not None:

    class MemoryProbe(dspy.Signature):
        """Using only your internal knowledge and training data, predict
        whether the stock will go up or down after the given date.
        Think about everything you know about this entity's trajectory."""

        entity: str = dspy.InputField(desc="Stock ticker symbol")
        date: str = dspy.InputField(desc="Date in YYYY-MM-DD format")
        direction: str = dspy.OutputField(desc="Either 'up' or 'down'")
