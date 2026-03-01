"""DSPy Signature for memory probing."""
from __future__ import annotations

try:
    import dspy
except ImportError:
    dspy = None  # type: ignore[assignment]


if dspy is not None:

    CALIBRATION_TASK = (
        "Predict whether the stock price went up or down after this date. "
        "Respond with a single word: up or down."
    )

    class MemoryProbe(dspy.Signature):
        """Using only your internal knowledge and training data, recall
        what you know about this entity's historical trajectory."""

        task: str = dspy.InputField(desc="Task instruction")
        direction: str = dspy.OutputField(desc="Your prediction")
