from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

_DATED_TEMPLATE = "After {date}, {entity} stock went"

_ALPHA_CAP = 4.0


@dataclass
class AlphaCalibrationResult:
    alpha: float
    entropy: float         # H_dated — entropy of the dated probe
    entity_date_var: float # entity's date-variance (from calibrate_entity)
    delta_temporal: float  # entity_mean_H - H_dated (positive = more confident than avg)
    p_yes: float           # p_up (after normalisation)
    p_no: float            # p_down (after normalisation)
    prompt: str


class CADCalibrator:
    """Calibrate CAD alpha per entity-date pair via date-variance probing.

    **Entity calibration** (once per entity, before the backtest loop):
    Probes the entity across 12 calibration dates (2005-2015) and computes:
      - ``date_variance``: std of entropy across dates.
        High → model's confidence swings with date → temporal memorisation.
        Low  → model's confidence is stable → brand prior.
      - ``mean_entropy``: average entropy across calibration dates.

    **Per-date alpha** (each backtest date):
    Probes the specific date and computes:
      - ``delta = mean_entropy - H_dated``: how much more confident the model
        is at this specific date compared to its average.

    Alpha combines both signals:
        ``α = α_max · (date_variance / DV_ref) · max(0, delta) / Δ_range``

    Entities with low date-variance (brand priors) get low α regardless of
    how confident the model is.  Entities with high date-variance get α
    proportional to how much this specific date exceeds their average.
    """

    # Calibration dates: quarterly from 2005-2015 (within any modern LLM's training data)
    _CALIBRATION_DATES = [
        "2006-03-31", "2007-06-30", "2008-09-30", "2009-03-31",
        "2010-06-30", "2011-09-30", "2012-03-31", "2013-06-30",
        "2014-03-31", "2014-09-30", "2015-03-31", "2015-09-30",
    ]

    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        use_chat_template: bool = False,
        optimized_instruction: str = "",
        logit_gap_profile: Optional[Dict[str, float]] = None,
        alpha_min: float = 0.0,
        alpha_max: float = 3.0,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.optimized_instruction = optimized_instruction
        # Pre-compute token IDs for "up" and "down" variants
        self._up_ids = self._token_ids([" up", " Up", "up", "Up"])
        self._down_ids = self._token_ids([" down", " Down", "down", "Down"])

        # Alpha bounds
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        # Date-variance reference: the DV level that maps to full α.
        # From profiling OOS mean — entities at this DV level get scale=1.0
        self._dv_ref = 0.08  # conservative default
        # Delta range: how much excess confidence (vs entity mean) maps to full α
        self._delta_range = 0.10  # default
        if logit_gap_profile is not None:
            self._alpha_min = logit_gap_profile.get("alpha_min", alpha_min)
            # alpha_max is capped at _ALPHA_CAP — the DV-scale formula doesn't
            # need alpha_max > cap (unlike the old entropy-threshold approach)
            self._alpha_max = min(
                logit_gap_profile.get("alpha_max", alpha_max), _ALPHA_CAP)
            # Use OOS entropy std as DV reference — OOS entities have this much
            # "natural" date-variance from non-memorisation sources
            dv = logit_gap_profile.get("entropy_oos_std")
            if dv is not None and dv > 0:
                self._dv_ref = dv
            # Delta range: IS entropy std — the spread of per-date confidence
            dr = logit_gap_profile.get("entropy_in_sample_std")
            if dr is not None and dr > 0:
                self._delta_range = dr
        logger.info("CADCalibrator: α_max=%.1f  DV_ref=%.4f  Δ_range=%.3f",
                     self._alpha_max, self._dv_ref, self._delta_range)

        # Per-entity state (populated by calibrate_entity)
        self._entity_date_variance: Dict[str, float] = {}
        self._entity_mean_entropy: Dict[str, float] = {}

    def calibrate_entity(self, ticker: str, dates: List[str] | None = None) -> float:
        """Probe *ticker* across calibration dates. Call once per entity before backtest.

        Returns the date-variance (std of entropy across dates).
        """
        if dates is None:
            dates = self._CALIBRATION_DATES
        entropies = []
        for date in dates:
            parts = []
            if self.optimized_instruction:
                parts.append(self.optimized_instruction.rstrip())
            parts.append(_DATED_TEMPLATE.format(entity=ticker, date=date))
            prompt = "\n\n".join(parts)
            h, _, _, _, _ = self._probe(prompt)
            entropies.append(h)

        import numpy as _np
        entity_std = float(_np.std(entropies)) if len(entropies) >= 2 else 0.0
        entity_mean = float(_np.mean(entropies))
        self._entity_date_variance[ticker] = entity_std
        self._entity_mean_entropy[ticker] = entity_mean
        logger.info("Entity calibration %s: DV=%.4f  mean_H=%.3f  range=[%.3f, %.3f]  (%d dates)",
                     ticker, entity_std, entity_mean, min(entropies), max(entropies), len(dates))
        return entity_std

    def calibrate_alpha(
        self,
        ticker: str,
        date: str = "",
        **kwargs,
    ) -> AlphaCalibrationResult:
        # Dated probe
        parts = []
        if self.optimized_instruction:
            parts.append(self.optimized_instruction.rstrip())
        parts.append(_DATED_TEMPLATE.format(entity=ticker, date=date))
        prompt = "\n\n".join(parts)

        h_dated, p_up, p_down, l_up, l_down = self._probe(prompt)

        # Entity state (from calibrate_entity)
        entity_dv = self._entity_date_variance.get(ticker, 0.0)
        entity_mean_h = self._entity_mean_entropy.get(ticker, h_dated)

        # Per-date signal: how much more confident than the entity's average
        # Positive = this date makes the model unusually confident → memorisation
        # Negative/zero = average or less confident → no memorisation at this date
        delta = entity_mean_h - h_dated

        # α = α_max × DV_scale × delta_scale
        #
        # DV_scale: entity date-variance relative to OOS reference.
        #   Low DV (brand prior): scale ≈ 0 → α ≈ 0
        #   High DV (date-specific): scale → 1 → α driven by per-date signal
        #
        # delta_scale: per-date excess confidence relative to entity mean.
        #   H_dated ≈ entity_mean → delta ≈ 0 → no penalty
        #   H_dated << entity_mean → large delta → strong penalty
        dv_scale = min(1.0, entity_dv / self._dv_ref) if self._dv_ref > 0 else 1.0
        delta_scale = max(0.0, delta / self._delta_range) if self._delta_range > 0 else 0.0

        alpha = self._alpha_max * dv_scale * delta_scale
        alpha = max(self._alpha_min, min(alpha, _ALPHA_CAP))

        preferred = "up" if l_up > l_down else "down"
        logger.info(
            "Probe %s @ %s → α=%.3f (H=%.3f mean_H=%.3f Δ=%+.3f DV=%.4f "
            "dv_s=%.2f Δ_s=%.2f) p_up=%.4f p_down=%.4f preferred=%s",
            ticker, date, alpha, h_dated, entity_mean_h, delta, entity_dv,
            dv_scale, delta_scale, p_up, p_down, preferred,
        )
        return AlphaCalibrationResult(
            alpha=alpha,
            entropy=h_dated,
            entity_date_var=entity_dv,
            delta_temporal=delta,
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

        l_up = logits[0, self._up_ids].max().item()
        l_down = logits[0, self._down_ids].max().item()

        l_max = max(l_up, l_down)
        e_up = math.exp(l_up - l_max)
        e_down = math.exp(l_down - l_max)
        total = e_up + e_down
        p_up = e_up / total
        p_down = e_down / total

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
