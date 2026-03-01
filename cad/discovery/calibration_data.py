"""Build a calibration dataset from historical price data.

The calibration set consists of (ticker, date) pairs with known future
direction labels.  These are used by DSPy to optimise the memory-activation
instruction so that it maximises parametric recall.
"""
from __future__ import annotations

from typing import List

import pandas as pd

from .config import CalibrationDatasetConfig, CalibrationExample

try:
    import dspy
except ImportError:
    dspy = None  # type: ignore[assignment]


def build_calibration_dataset(
    cfg: CalibrationDatasetConfig,
) -> List[CalibrationExample]:
    """Load prices and build labelled (ticker, date, direction) examples.

    Steps:
    1. Load ``price_data.csv`` (columns: date, symbol, adjusted_close).
    2. For each ticker, sample dates at ``sample_freq`` within ``date_range``.
    3. Compute forward returns over ``forward_days`` trading days.
    4. Filter out flat moves (abs return < ``min_abs_return``).
    5. Label direction as ``"up"`` or ``"down"``.
    6. Balance classes and cap at ``max_examples``.
    """
    df = pd.read_csv(cfg.price_csv, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

    pcol = "adjusted_close" if "adjusted_close" in df.columns else "close"
    start, end = pd.Timestamp(cfg.date_range[0]), pd.Timestamp(cfg.date_range[1])
    df = df[["date", "symbol", pcol]].dropna()
    df = df[(df["date"] >= start) & (df["date"] <= end)]

    examples: List[CalibrationExample] = []

    for symbol, grp in df.groupby("symbol"):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) < cfg.forward_days + 1:
            continue

        # Sample dates at the requested frequency
        grp = grp.set_index("date")
        sampled = grp.resample(cfg.sample_freq).first().dropna()

        for sample_date in sampled.index:
            # Find the position in the original sorted group
            mask = grp.index >= sample_date
            future = grp.loc[mask]
            if len(future) <= cfg.forward_days:
                continue

            price_now = future[pcol].iloc[0]
            price_future = future[pcol].iloc[cfg.forward_days]
            ret = (price_future - price_now) / price_now

            if abs(ret) < cfg.min_abs_return:
                continue

            direction = "up" if ret > 0 else "down"
            examples.append(
                CalibrationExample(
                    ticker=str(symbol),
                    date=sample_date.strftime("%Y-%m-%d"),
                    future_return=round(float(ret), 6),
                    direction=direction,
                )
            )

    # Balance classes
    ups = [e for e in examples if e.direction == "up"]
    downs = [e for e in examples if e.direction == "down"]
    n = min(len(ups), len(downs), cfg.max_examples // 2)
    balanced = ups[:n] + downs[:n]

    return balanced


def to_dspy_examples(
    examples: List[CalibrationExample],
) -> list:
    """Convert calibration examples to ``dspy.Example`` objects.

    Each example has a single input ``task`` (containing entity, date, and
    calibration instruction), with label ``answer`` (``"up"`` or ``"down"``).
    """
    if dspy is None:
        raise ImportError("dspy is required for to_dspy_examples(). Install with: pip install dspy")

    dspy_examples = []
    for ex in examples:
        # Only include entity, date, and output options — NOT a prediction
        # instruction.  This forces T* to be the critical signal that
        # activates memory.  "Options: up, down" constrains the model's
        # output format but is invisible to the proposer (data_aware=False).
        task_text = (
            f"Entity: {ex.ticker}\n"
            f"Date: {ex.date}\n"
            f"Options: up, down"
        )
        dspy_ex = dspy.Example(
            task=task_text,
            answer=ex.direction,
        ).with_inputs("task")
        dspy_examples.append(dspy_ex)
    return dspy_examples
