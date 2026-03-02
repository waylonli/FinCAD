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
from tqdm import tqdm

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
    calibrator_alpha_min: float = 0.5
    calibrator_alpha_max: float = 1.5
    chunk_size: int = 8192                   # max filing text tokens per chunk
    chunk_overlap: int = 256                 # overlap tokens between consecutive chunks


@dataclass
class FilingScoringResult:
    symbol: str
    report_date: pd.Timestamp
    quality_score: float
    category_scores: Dict[str, float]
    category_explanations: Dict[str, str]
    alpha_used: float
    raw_generations: Dict[str, str]
    num_chunks: int = 1


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
        neg_prompt_builder=None,
    ) -> None:
        self.decoder = decoder
        self.config = scoring_config
        self.calibrator = calibrator
        self.category_prompts = category_prompts or DEFAULT_CATEGORY_PROMPTS
        self.neg_prompt_builder = neg_prompt_builder
        self._stop_token_ids: Optional[list] = None
        self._max_ctx_tokens = self._resolve_max_context()

    def _resolve_max_context(self) -> int:
        """Detect the model's maximum context length."""
        cfg = getattr(self.decoder.model, "config", None)
        if cfg is not None:
            for attr in ("max_position_embeddings", "n_positions", "seq_length"):
                val = getattr(cfg, attr, None)
                if val is not None:
                    return int(val)
        return 128_000  # safe fallback

    def _resolve_stop_token_ids(self) -> Optional[list]:
        """Resolve the token ID for ``}`` to enable early JSON stop."""
        if self._stop_token_ids is not None:
            return self._stop_token_ids
        try:
            ids = self.decoder.tokenizer.encode("}", add_special_tokens=False)
            if ids:
                self._stop_token_ids = [ids[-1]]
                return self._stop_token_ids
        except Exception:
            pass
        return None

    def _compute_chunk_budget(self, prompt_template: str) -> int:
        """Return the max number of *filing text tokens* that fit in one chunk.

        The budget accounts for the prompt template overhead, generation
        tokens, and a small safety margin.
        """
        tokenizer = self.decoder.tokenizer
        # Tokens consumed by the template itself (without the filing text)
        template_text = prompt_template.replace("{input_annual_report}", "")
        template_tokens = len(tokenizer.encode(template_text, add_special_tokens=True))
        # Budget for the filing text portion
        budget = self._max_ctx_tokens - template_tokens - self.config.max_new_tokens - 64
        return max(256, budget)

    def _chunk_filing(
        self,
        text: str,
        prompt_template: str,
    ) -> List[str]:
        """Split filing text into chunks that each fit the model's context.

        If ``chunk_size > 0`` in the config it is used directly as the max
        tokens per chunk; otherwise the budget is derived from the model's
        context window.  Consecutive chunks overlap by ``chunk_overlap``
        tokens so that sentences at boundaries are not cut mid-thought.

        Returns a list of *text strings* (decoded back from tokens) — one per
        chunk.
        """
        tokenizer = self.decoder.tokenizer
        auto_budget = self._compute_chunk_budget(prompt_template)
        chunk_tokens = self.config.chunk_size if self.config.chunk_size > 0 else auto_budget
        # Clamp to what actually fits in the context window
        chunk_tokens = min(chunk_tokens, auto_budget)
        overlap = min(self.config.chunk_overlap, chunk_tokens // 2)

        # Tokenize the entire filing text
        all_ids = tokenizer.encode(text, add_special_tokens=False)

        if len(all_ids) <= chunk_tokens:
            return [text]

        # Sliding-window chunking
        chunks: List[str] = []
        stride = max(1, chunk_tokens - overlap)
        start = 0
        while start < len(all_ids):
            end = min(start + chunk_tokens, len(all_ids))
            chunk_ids = all_ids[start:end]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
            if end >= len(all_ids):
                break
            start += stride

        logger.info(
            "Split filing (%d tokens) into %d chunks of ≤%d tokens (overlap=%d)",
            len(all_ids), len(chunks), chunk_tokens, overlap,
        )
        return chunks

    def score_filing(
        self,
        text: str,
        symbol: str,
        report_date: pd.Timestamp,
    ) -> FilingScoringResult:
        """Score a single filing across all categories.

        Long filings are split into chunks.  Each chunk is scored
        independently per category and the per-category scores are averaged
        across chunks.  For baseline mode ``alpha=0`` is used so the
        combined-logit formula reduces to standard generation.
        """
        alpha = 0.0
        if self.config.decoding_mode == "cad":
            if self.config.use_calibrator and self.calibrator is not None:
                cal_result = self.calibrator.calibrate_alpha(
                    symbol,
                    alpha_min=self.config.calibrator_alpha_min,
                    alpha_max=self.config.calibrator_alpha_max,
                    date=f"{report_date:%Y-%m-%d}",
                )
                alpha = cal_result.alpha
                logger.info(
                    "Calibrated alpha for %s: %.3f (entropy=%.3f)",
                    symbol, alpha, cal_result.entropy,
                )
            else:
                alpha = self.config.cad_alpha

        # Early-stop on "}" so we don't waste tokens after the JSON object closes
        stop_ids = self._resolve_stop_token_ids()

        cad_config = CADConfig(
            alpha=alpha,
            top_p=self.config.cad_top_p,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            stop_token_ids=stop_ids,
        )

        # Use the first prompt template to determine chunking (all templates
        # have similar overhead so one is sufficient for sizing)
        first_template = next(iter(self.category_prompts.values()))
        chunks = self._chunk_filing(text, first_template)
        num_chunks = len(chunks)

        # Accumulate per-category scores across chunks
        # {category: [score_chunk_0, score_chunk_1, ...]}
        chunk_scores: Dict[str, List[float]] = {cat: [] for cat in self.category_prompts}
        chunk_explanations: Dict[str, List[str]] = {cat: [] for cat in self.category_prompts}
        raw_generations: Dict[str, str] = {}

        n_categories = len(self.category_prompts)
        total_calls = num_chunks * n_categories
        pbar = tqdm(
            total=total_calls,
            desc=f"  {symbol}",
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        for ci, chunk_text in enumerate(chunks):
            for category, prompt_template in self.category_prompts.items():
                pbar.set_postfix_str(
                    f"chunk {ci + 1}/{num_chunks} {category}", refresh=True,
                )
                context_prompt = prompt_template.format(input_annual_report=chunk_text)
                prior_prompt = build_prior_prompt(
                    category, symbol, self.config.cad_prior_mode,
                    neg_prompt_builder=self.neg_prompt_builder,
                )

                generation = self.decoder.generate(context_prompt, prior_prompt, cad_config)

                gen_key = category if num_chunks == 1 else f"{category}_chunk{ci}"
                raw_generations[gen_key] = generation

                score, explanation = _parse_score_json(generation)
                chunk_scores[category].append(score)
                chunk_explanations[category].append(explanation)
                pbar.update(1)
        pbar.close()

        # Aggregate: mean of valid per-chunk scores for each category
        category_scores: Dict[str, float] = {}
        category_explanations: Dict[str, str] = {}
        for category in self.category_prompts:
            valid = [s for s in chunk_scores[category] if 0 <= s <= 100]
            category_scores[category] = (
                sum(valid) / len(valid) if valid else 0.0
            )
            category_explanations[category] = (
                chunk_explanations[category][0] if chunk_explanations[category] else ""
            )

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
            num_chunks=num_chunks,
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

        filings_pbar = tqdm(filings, desc="Scoring filings", unit="filing")
        for text, symbol, report_date in filings_pbar:
            key = (symbol.upper(), str(report_date))
            if key in cached_keys:
                continue

            filings_pbar.set_description(f"Scoring {symbol} @ {report_date}")
            result = self.score_filing(text, symbol, report_date)

            record = {
                "symbol": result.symbol,
                "report_date": str(result.report_date),
                "quality_score": result.quality_score,
                "alpha_used": result.alpha_used,
                "num_chunks": result.num_chunks,
            }
            for cat, score in result.category_scores.items():
                record[f"category_{cat}_score"] = score
            records.append(record)

            filings_pbar.set_postfix(q_score=f"{result.quality_score:.1f}")

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
