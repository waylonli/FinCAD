"""Local HuggingFace-native filing scorer with CAD support.

Replaces the abrdn LangChain/OpenAI scorer with one that runs locally on
any open-source model (Qwen2.5, Llama, Phi, etc.) via the project's
``ContextAwareDecoder``.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from cad import CADConfig, ContextAwareDecoder
from cad.calibrator import CADCalibrator

from .prompts import DEFAULT_CATEGORY_PROMPTS, build_prior_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config & result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ScoringConfig:
    decoding_mode: str = "baseline"          # "baseline" or "cad"
    cad_alpha: float = 1.0
    temperature: float = 0.0
    max_new_tokens: int = 512
    cad_top_p: float = 1.0
    cad_prior_mode: str = "no_context"       # "no_context" or "bias_amplified"
    use_calibrator: bool = False
    calibrator_alpha_min: float = 0.0
    calibrator_alpha_max: float = 5.0


@dataclass
class FilingScoringResult:
    symbol: str
    report_date: pd.Timestamp
    quality_score: float
    category_scores: Dict[str, float]
    category_explanations: Dict[str, str]
    alpha_used: float
    raw_generations: Dict[str, str]


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class HFFilingScorer:
    """Score filings using a local HF model with optional CAD decoding."""

    def __init__(
        self,
        decoder: ContextAwareDecoder,
        scoring_config: ScoringConfig,
        calibrator: Optional[CADCalibrator] = None,
        category_prompts: Optional[Dict[str, str]] = None,
    ) -> None:
        self.decoder = decoder
        self.config = scoring_config
        self.calibrator = calibrator
        self.category_prompts = category_prompts or DEFAULT_CATEGORY_PROMPTS

    def score_filing(
        self,
        text: str,
        symbol: str,
        report_date: pd.Timestamp,
    ) -> FilingScoringResult:
        """Score a single filing across all categories.

        For baseline mode, ``alpha=0`` is passed to the decoder so the
        combined-logit formula reduces to standard generation.
        """
        alpha = 0.0
        if self.config.decoding_mode == "cad":
            if self.config.use_calibrator and self.calibrator is not None:
                cal_result = self.calibrator.calibrate_alpha(
                    symbol,
                    alpha_min=self.config.calibrator_alpha_min,
                    alpha_max=self.config.calibrator_alpha_max,
                )
                alpha = cal_result.alpha
                logger.info(
                    "Calibrated alpha for %s: %.3f (entropy=%.3f)",
                    symbol, alpha, cal_result.entropy,
                )
            else:
                alpha = self.config.cad_alpha

        cad_config = CADConfig(
            alpha=alpha,
            top_p=self.config.cad_top_p,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
        )

        category_scores: Dict[str, float] = {}
        category_explanations: Dict[str, str] = {}
        raw_generations: Dict[str, str] = {}

        for category, prompt_template in self.category_prompts.items():
            context_prompt = prompt_template.format(input_annual_report=text)
            prior_prompt = build_prior_prompt(
                category, symbol, self.config.cad_prior_mode,
            )

            generation = self.decoder.generate(context_prompt, prior_prompt, cad_config)
            raw_generations[category] = generation

            score, explanation = _parse_score_json(generation)
            category_scores[category] = score
            category_explanations[category] = explanation

        valid_scores = [s for s in category_scores.values() if 0 <= s <= 100]
        quality_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        return FilingScoringResult(
            symbol=symbol,
            report_date=report_date,
            quality_score=quality_score,
            category_scores=category_scores,
            category_explanations=category_explanations,
            alpha_used=alpha,
            raw_generations=raw_generations,
        )

    def score_filings_to_dataframe(
        self,
        filings: List[Tuple[str, str, pd.Timestamp]],
        *,
        cache_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Score multiple filings and return a tidy DataFrame.

        Parameters
        ----------
        filings:
            List of ``(text, symbol, report_date)`` tuples.
        cache_path:
            If provided, append results to this CSV after each filing for
            resumability.
        """
        records: List[Dict] = []

        # Load existing cache to skip already-scored filings
        cached_keys: set = set()
        if cache_path is not None and cache_path.exists():
            try:
                existing = pd.read_csv(cache_path)
                for _, row in existing.iterrows():
                    cached_keys.add((str(row["symbol"]).upper(), str(row["report_date"])))
                records.extend(existing.to_dict("records"))
                logger.info("Loaded %d cached scores from %s", len(existing), cache_path)
            except Exception as exc:
                logger.warning("Could not read score cache %s: %s", cache_path, exc)

        for text, symbol, report_date in filings:
            key = (symbol.upper(), str(report_date))
            if key in cached_keys:
                logger.info("Skipping cached score for %s @ %s", symbol, report_date)
                continue

            logger.info("Scoring %s @ %s ...", symbol, report_date)
            result = self.score_filing(text, symbol, report_date)

            record = {
                "symbol": result.symbol,
                "report_date": str(result.report_date),
                "quality_score": result.quality_score,
                "alpha_used": result.alpha_used,
            }
            for cat, score in result.category_scores.items():
                record[f"category_{cat}_score"] = score
            records.append(record)

            if cache_path is not None:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(records).to_csv(cache_path, index=False)
                except Exception as exc:
                    logger.warning("Failed to write score cache: %s", exc)

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_score_json(text: str) -> Tuple[float, str]:
    """Extract ``(score, explanation)`` from a generation.

    Tries ``json.loads`` first, then falls back to regex extraction.
    Score is clamped to [0, 100].
    """
    # Try to find a JSON object in the text
    match = _JSON_RE.search(text)
    if match:
        try:
            obj = json.loads(match.group())
            score = float(obj.get("score", 0))
            explanation = str(obj.get("explanation", ""))
            return max(0.0, min(100.0, score)), explanation
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Fallback: try to find a bare number
    numbers = re.findall(r"\b(\d{1,3})\b", text)
    if numbers:
        score = float(numbers[0])
        return max(0.0, min(100.0, score)), text.strip()[:200]

    logger.warning("Could not parse score from generation: %s", text[:200])
    return 0.0, ""
