from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import math

import torch

logger = logging.getLogger(__name__)

_COMPLETION_TEMPLATE = "After {date}, {entity} stock went"

_ALPHA_CAP = 3.0


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

    Alpha is computed using **entropy-based scaling** (Eq. 6 in the paper):
        α = α_min + (α_max − α_min) * (1 − Ĥ)
    where Ĥ is the normalised binary entropy of the up/down distribution.

    When the model is confident (low entropy → strong memorisation), α is high.
    When the model is uncertain (high entropy → weak/no memorisation), α is low.
    This naturally assigns low penalties out-of-sample where there is nothing
    to subtract, and high penalties in-sample where look-ahead bias is strong.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        use_chat_template: bool = False,
        optimized_instruction: str = "",
        logit_gap_profile: Optional[Dict[str, float]] = None,
        alpha_min: float = 0.0,
        alpha_max: float = 12.0,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.optimized_instruction = optimized_instruction
        # Pre-compute token IDs for "up" and "down" variants
        self._up_ids = self._token_ids([" up", " Up", "up", "Up"])
        self._down_ids = self._token_ids([" down", " Down", "down", "Down"])

        # Entropy-based alpha bounds (model-variant via logit_gap_profile)
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._entropy_threshold = 0.85  # fallback when no profile
        if logit_gap_profile is not None:
            self._alpha_min = logit_gap_profile.get("alpha_min", alpha_min)
            self._alpha_max = logit_gap_profile.get("alpha_max", alpha_max)
            # Auto-compute threshold: OOS_mean - IS_std
            # This places the threshold one in-sample std below the OOS mean,
            # ensuring OOS probes (near OOS_mean) get α=0 while in-sample
            # probes that are anomalously confident get penalised.
            oos_mean = logit_gap_profile.get("entropy_oos_mean")
            is_std = logit_gap_profile.get("entropy_in_sample_std")
            if oos_mean is not None and is_std is not None:
                self._entropy_threshold = oos_mean - is_std
            else:
                self._entropy_threshold = logit_gap_profile.get("entropy_threshold", 0.85)
        logger.info("Entropy-based alpha: α_min=%.2f  α_max=%.2f  H_thresh=%.3f",
                     self._alpha_min, self._alpha_max, self._entropy_threshold)

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

        # Entropy-based alpha: α = α_min + (α_max - α_min) * (1 - Ĥ)
        # Ĥ ∈ [0, 1]: 0 = fully confident (max penalty), 1 = uniform (min penalty)
        #
        # Entropy threshold: when Ĥ > threshold, the model is too uncertain
        # for the probe to distinguish look-ahead bias from brand priors.
        # In this regime we set α = 0 to avoid subtracting legitimate reasoning.
        entropy_threshold = self._entropy_threshold
        if entropy > entropy_threshold:
            alpha = 0.0
        else:
            # Scale within the confident regime [0, threshold]
            confidence = 1.0 - entropy / entropy_threshold  # 0 at threshold, 1 at entropy=0
            alpha = self._alpha_min + (self._alpha_max - self._alpha_min) * confidence
            alpha = max(0.0, min(alpha, _ALPHA_CAP))

        gap = abs(l_up - l_down)
        preferred = "up" if l_up > l_down else "down"
        logger.info(
            "Probe %s @ %s → α=%.3f (entropy=%.3f gap=%.3f) p_up=%.4f p_down=%.4f "
            "logit_up=%.2f logit_down=%.2f preferred=%s",
            ticker, date, alpha, entropy, gap, p_up, p_down, l_up, l_down, preferred,
        )
        return AlphaCalibrationResult(
            alpha=alpha,
            entropy=entropy,
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
