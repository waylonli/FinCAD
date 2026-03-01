"""DSPy metric for bias activation scoring."""
from __future__ import annotations


def _normalize(text: str) -> str:
    """Extract a canonical label from free-form model output."""
    t = text.strip().lower()
    # Check for exact single-word answer first
    if t in ("up", "down"):
        return t
    # Fallback: check for substring presence, preferring the first match
    # to handle verbose outputs. Check "down" before "up" to avoid
    # false positives from words containing "up" (e.g., "update").
    if "down" in t:
        return "down"
    if " up" in t or t.startswith("up"):
        return "up"
    return t


def bias_activation_score(example, prediction, trace=None) -> float:
    """Return 1.0 if the predicted answer matches ground truth, else 0.0.

    This metric rewards instructions that successfully activate the model's
    parametric memory about an entity's future trajectory — exactly the bias
    we want the negative prompt to elicit for CAD subtraction.
    """
    gold = _normalize(example.answer)
    pred = _normalize(prediction.answer)
    return 1.0 if pred == gold else 0.0
