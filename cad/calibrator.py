from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import math

import torch

logger = logging.getLogger(__name__)

_COMPLETION_TEMPLATE = "After {date}, {entity} stock went"


@dataclass
class AlphaCalibrationResult:
    alpha: float
    entropy: float    # normalised binary entropy over {up, down}
    p_yes: float      # p_up (after normalisation)
    p_no: float       # p_down (after normalisation)
    prompt: str


class CADCalibrator:
    """Calibrate CAD alpha per entity-date pair via completion-based probing.

    Uses a completion prompt ``"[T*] After {date}, {entity} stock went"``
    and compares logits for "up" vs "down" tokens at the next-token position.
    This avoids:
    - Safety refusals (plain completion, not a question)
    - Signal dilution (100% of signal in the measured token)
    - Chat template artifacts (always uses raw tokenization)

    The logit gap between "up" and "down" is mapped to α via an
    exponential transform: α = exp(gap) − 1.  This is approximately
    linear for small gaps (out-of-sample entities stay near the raw
    gap) and superlinear for large gaps (in-sample entities are
    amplified to α ≈ 2–3).
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        use_chat_template: bool = False,
        optimized_instruction: str = "",
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.optimized_instruction = optimized_instruction
        # Pre-compute token IDs for "up" and "down" variants
        self._up_ids = self._token_ids([" up", " Up", "up", "Up"])
        self._down_ids = self._token_ids([" down", " Down", "down", "Down"])

    def calibrate_alpha(
        self,
        ticker: str,
        date: str = "",
        **kwargs,
    ) -> AlphaCalibrationResult:
        # Build completion prompt: T* ⊕ "After {date}, {entity} stock went"
        parts = []
        if self.optimized_instruction:
            parts.append(self.optimized_instruction.rstrip())
        parts.append(_COMPLETION_TEMPLATE.format(entity=ticker, date=date))
        prompt = "\n\n".join(parts)

        entropy, p_up, p_down, l_up, l_down = self._probe(prompt)

        # α = exp(gap) − 1: approximately linear for small gaps,
        # superlinear for large gaps.  No free parameters.
        gap = abs(l_up - l_down)
        alpha = math.exp(gap) - 1.0

        preferred = "up" if l_up > l_down else "down"
        logger.info(
            "Probe %s @ %s → α=%.3f (gap=%.3f) p_up=%.4f p_down=%.4f "
            "logit_up=%.2f logit_down=%.2f preferred=%s",
            ticker, date, alpha, gap, p_up, p_down, l_up, l_down, preferred,
        )
        return AlphaCalibrationResult(
            alpha=alpha,
            entropy=gap,       # raw logit gap stored here for logging
            p_yes=p_up,
            p_no=p_down,
            prompt=prompt,
        )

    def _probe(self, prompt: str) -> tuple[float, float, float, float, float]:
        """Run the completion probe and return (entropy, p_up, p_down, l_up, l_down)."""
        inputs = self._encode_plain(prompt)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]

        if not self._up_ids or not self._down_ids:
            return 1.0, 0.5, 0.5, 0.0, 0.0

        # Best logit for each class
        l_up = logits[0, self._up_ids].max().item()
        l_down = logits[0, self._down_ids].max().item()

        # 2-class softmax at τ=1 (model's raw logit calibration)
        l_max = max(l_up, l_down)
        e_up = math.exp(l_up - l_max)
        e_down = math.exp(l_down - l_max)
        total = e_up + e_down
        p_up = e_up / total
        p_down = e_down / total

        # Normalised binary entropy
        entropy = 0.0
        for p in (p_up, p_down):
            if p > 0:
                entropy -= p * math.log(p)
        entropy /= math.log(2.0)

        return entropy, p_up, p_down, l_up, l_down

    def _token_ids(self, tokens) -> list[int]:
        ids = []
        for t in tokens:
            token_ids = self.tokenizer.encode(t, add_special_tokens=False)
            if len(token_ids) == 1:
                ids.append(token_ids[0])
        return sorted(set(ids))

    def _encode_plain(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode as plain text (no chat template) for completion probing."""
        encoded = self.tokenizer(text, return_tensors="pt", padding=False)
        return {k: v.to(self.device) for k, v in encoded.items()}
