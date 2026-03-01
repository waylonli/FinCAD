"""DSPy Signature for memory probing."""
from __future__ import annotations

try:
    import dspy
except ImportError:
    dspy = None  # type: ignore[assignment]


if dspy is not None:

    class MemoryProbe(dspy.Signature):
        """Recall what you know from your training data about this
        financial entity and predict the outcome."""

        task: str = dspy.InputField(desc="Task instruction")
        answer: str = dspy.OutputField(desc="one word")
