"""Save and load optimized instructions as JSON files."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .config import OptimizedInstruction


def _model_slug(model_name: str) -> str:
    """Convert a model name to a filesystem-safe slug."""
    short = model_name.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9._-]", "_", short)


def save_instruction(
    inst: OptimizedInstruction,
    *,
    output_dir: str = "results/discovery",
) -> Path:
    """Save an ``OptimizedInstruction`` to ``{output_dir}/{model_slug}.json``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{_model_slug(inst.model_name)}.json"
    path.write_text(inst.to_json())
    return path


def load_instruction(
    model_name: Optional[str] = None,
    *,
    path: Optional[str] = None,
    output_dir: str = "results/discovery",
) -> OptimizedInstruction:
    """Load an ``OptimizedInstruction`` from JSON.

    Either ``path`` (direct file path) or ``model_name`` (resolved via
    ``output_dir``) must be provided.
    """
    if path is not None:
        p = Path(path)
    elif model_name is not None:
        p = Path(output_dir) / f"{_model_slug(model_name)}.json"
    else:
        raise ValueError("Provide either path or model_name.")

    if not p.exists():
        raise FileNotFoundError(f"No optimized instruction found at {p}")
    return OptimizedInstruction.from_json(p.read_text())
