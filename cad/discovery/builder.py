"""NegativePromptBuilder: combine optimized instruction with task prompt."""
from __future__ import annotations

from pathlib import Path

from .config import OptimizedInstruction
from .registry import load_instruction


class NegativePromptBuilder:
    """Build a negative (prior) prompt from an optimized instruction.

    The negative prompt is: ``T*(s, t) ⊕ F_task``.

    *  ``T*`` is the optimised memory-activation instruction (model-specific,
       task-agnostic), combined with entity/date fields.
    *  ``F_task`` is the task-specific instruction copied from the context
       prompt (task framing + output format), ensuring logit alignment.
    """

    def __init__(self, instruction: OptimizedInstruction) -> None:
        self._instruction = instruction

    @classmethod
    def from_file(cls, path: str | Path) -> NegativePromptBuilder:
        """Load an ``OptimizedInstruction`` from a JSON file."""
        inst = load_instruction(path=str(path))
        return cls(inst)

    @classmethod
    def from_model(
        cls, model_name: str, output_dir: str = "results/discovery"
    ) -> NegativePromptBuilder:
        """Load the saved instruction for a given model name."""
        inst = load_instruction(model_name=model_name, output_dir=output_dir)
        return cls(inst)

    @property
    def instruction(self) -> OptimizedInstruction:
        return self._instruction

    def build(
        self,
        entity: str = "",
        date: str = "",
        task_prompt: str = "",
    ) -> str:
        """Build the full negative prompt.

        Structure::

            [Optimised memory-activation instruction T*]

            Entity: {entity}
            Date: {date}

            [Task-specific instruction F_task]

        The entity/date are appended as structured fields, mirroring how
        DSPy presented them as InputFields during optimisation.  ``F_task``
        is the task-specific instruction (task framing + output format)
        copied from the context prompt to ensure logit alignment
        (see methodology §3.1, Eq. 2).

        Parameters
        ----------
        entity:
            Ticker symbol or entity name. Empty string for non-entity tasks
            (e.g. MCQ, math).
        date:
            Date string. Empty string for non-temporal tasks.
        task_prompt:
            The task-specific instruction from the context prompt, including
            both the task framing and output format specification.  Ensures
            logit alignment between context and prior streams.
        """
        parts = [self._instruction.instruction.rstrip()]

        # Append entity/date as structured fields
        if entity or date:
            fields = []
            if entity:
                fields.append(f"Entity: {entity}")
            if date:
                fields.append(f"Date: {date}")
            parts.append("\n".join(fields))

        if task_prompt:
            parts.append(task_prompt.strip())

        return "\n\n".join(parts)
