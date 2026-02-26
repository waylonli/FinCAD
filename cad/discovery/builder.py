"""NegativePromptBuilder: combine optimized instruction with output format spec."""
from __future__ import annotations

from pathlib import Path

from .config import OptimizedInstruction
from .registry import load_instruction


class NegativePromptBuilder:
    """Build a negative (prior) prompt from an optimized instruction.

    The negative prompt is: ``[Memory Activation Instruction] + [Output Format Spec]``.
    The instruction is model-specific but task-agnostic; the output format is
    copied from the context prompt to preserve logit alignment.
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
        output_format_spec: str = "",
    ) -> str:
        """Build the full negative prompt.

        Substitutes ``{entity}`` and ``{date}`` placeholders in the optimized
        instruction, then appends the output format specification.

        Parameters
        ----------
        entity:
            Ticker symbol or entity name. Empty string for non-entity tasks
            (e.g. MCQ, math).
        date:
            Date string. Empty string for non-temporal tasks.
        output_format_spec:
            The output format portion of the context prompt, ensuring logit
            alignment between context and prior streams.
        """
        text = self._instruction.instruction
        text = text.replace("{entity}", entity)
        text = text.replace("{date}", date)

        parts = [text.rstrip()]
        if output_format_spec:
            parts.append(output_format_spec.strip())

        return "\n\n".join(parts)
