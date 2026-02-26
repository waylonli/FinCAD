"""DSPy metric for bias activation scoring."""
from __future__ import annotations


def bias_activation_score(example, prediction, trace=None) -> float:
    """Return 1.0 if the predicted direction matches ground truth, else 0.0.

    This metric rewards instructions that successfully activate the model's
    parametric memory about an entity's future trajectory — exactly the bias
    we want the negative prompt to elicit for CAD subtraction.
    """
    gold = example.direction.strip().lower()
    pred = prediction.direction.strip().lower()
    return 1.0 if pred == gold else 0.0
