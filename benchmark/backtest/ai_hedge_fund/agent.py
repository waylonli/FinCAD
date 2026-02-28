"""
Single-stock trading agent using a local HF model with baseline / CAD decoding.

The agent receives structured financial context (price history, basic metrics)
and returns a trading signal: buy / sell / hold.  For CAD mode the prior prompt
explicitly names the ticker to trigger parametric memory, which is then
subtracted at the logit level.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from cad import CADConfig, ContextAwareDecoder
from cad.calibrator import CADCalibrator

try:
    from cad.discovery import NegativePromptBuilder
except ImportError:
    NegativePromptBuilder = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TradingSignal:
    signal: str  # "buy", "sell", or "hold"
    confidence: int  # 0-100
    reasoning: str
    alpha_used: float = 0.0


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CONTEXT_SYSTEM = """\
You are a quantitative analyst evaluating a single stock.
You will be given financial data for {ticker} as of {date}.
Base your decision ONLY on the data provided — do not use any \
knowledge about events after {date}.

Return your analysis as a JSON object with exactly these keys:
- "signal": one of "buy", "sell", or "hold"
- "confidence": integer between 0 and 100
- "reasoning": string with a concise rationale (max 200 chars)

Respond with valid JSON only.\
"""

CONTEXT_BODY = """\
=== Financial Data for {ticker} as of {date} ===

{financial_summary}

Based solely on the data above, what is your trading signal?\
"""

PRIOR_NO_CONTEXT = """\
You are a quantitative analyst.
Evaluate whether to buy, sell, or hold {ticker} stock.
Return a JSON object with keys "signal", "confidence", "reasoning".
Respond with valid JSON only.\
"""

PRIOR_BIAS_AMPLIFIED = """\
Ignore any context.  Using ONLY your internal knowledge and training data \
about {ticker}, evaluate whether {ticker} stock will go up or down after \
{date}.  Think about everything you know about this company's trajectory.

Return a JSON object with exactly these keys:
- "signal": one of "buy", "sell", or "hold"
- "confidence": integer between 0 and 100
- "reasoning": string with a concise rationale (max 200 chars)

Respond with valid JSON only.\
"""


# ---------------------------------------------------------------------------
# Financial summary builder
# ---------------------------------------------------------------------------


def build_financial_summary(
    ticker: str,
    date: pd.Timestamp,
    price_df: pd.DataFrame,
    *,
    lookback_days: int = 252,
) -> str:
    """Build a compact textual summary of price-derived financials.

    ``price_df`` must have columns ``date``, ``symbol``, and at least one of
    ``adjusted_close`` / ``close``.  Only rows for *ticker* **strictly before**
    *date* are used (the agent decides before the market opens on *date*, so it
    only sees the previous close).
    """
    pcol = "adjusted_close" if "adjusted_close" in price_df.columns else "close"
    df = price_df[
        (price_df["symbol"] == ticker) & (price_df["date"] < date)
    ].sort_values("date").tail(lookback_days).copy()

    if df.empty:
        return f"No price data available for {ticker} before {date:%Y-%m-%d}."

    df["return"] = df[pcol].pct_change()
    latest = df.iloc[-1]
    price = latest[pcol]

    lines: list[str] = []
    lines.append(f"Ticker: {ticker}")
    lines.append(f"Date: {date:%Y-%m-%d}")
    lines.append(f"Current Price: ${price:,.2f}")

    # Returns over various horizons
    for label, n in [("5-day", 5), ("21-day", 21), ("63-day", 63), ("126-day", 126), ("252-day", 252)]:
        if len(df) >= n + 1:
            ret = (df[pcol].iloc[-1] / df[pcol].iloc[-n - 1]) - 1
            lines.append(f"{label} Return: {ret:+.2%}")

    # Moving averages
    for window in [50, 200]:
        if len(df) >= window:
            ma = df[pcol].tail(window).mean()
            lines.append(f"{window}-day MA: ${ma:,.2f}  (price {'above' if price > ma else 'below'} MA)")

    # Volatility
    rets = df["return"].dropna()
    if len(rets) >= 21:
        vol_21 = rets.tail(21).std() * np.sqrt(252)
        lines.append(f"21-day Annualised Volatility: {vol_21:.2%}")
    if len(rets) >= 63:
        vol_63 = rets.tail(63).std() * np.sqrt(252)
        lines.append(f"63-day Annualised Volatility: {vol_63:.2%}")

    # 52-week high / low
    if len(df) >= 252:
        hi = df[pcol].tail(252).max()
        lo = df[pcol].tail(252).min()
        lines.append(f"52-week High: ${hi:,.2f}  Low: ${lo:,.2f}  (current at {(price - lo) / (hi - lo):.0%} of range)")

    # Simple momentum
    if len(rets) >= 21:
        pos_days = (rets.tail(21) > 0).sum()
        lines.append(f"Positive days (last 21): {pos_days}/21")

    # Volume (if available)
    if "volume" in df.columns:
        avg_vol = df["volume"].tail(21).mean()
        if not np.isnan(avg_vol) and avg_vol > 0:
            lines.append(f"21-day Avg Volume: {avg_vol:,.0f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_trading_signal(text: str) -> TradingSignal:
    """Extract a TradingSignal from LLM generation text."""
    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group())
            signal = str(obj.get("signal", "hold")).lower().strip()
            if signal not in ("buy", "sell", "hold"):
                signal = "hold"
            confidence = int(max(0, min(100, float(obj.get("confidence", 50)))))
            reasoning = str(obj.get("reasoning", ""))[:300]
            return TradingSignal(signal=signal, confidence=confidence, reasoning=reasoning)
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning(f"Could not parse trading signal {text}")
            pass

    # Fallback heuristics
    lower = text.lower()
    if "buy" in lower and "sell" not in lower:
        return TradingSignal(signal="buy", confidence=30, reasoning="[parse fallback] found 'buy' keyword")
    if "sell" in lower and "buy" not in lower:
        return TradingSignal(signal="sell", confidence=30, reasoning="[parse fallback] found 'sell' keyword")
    return TradingSignal(signal="hold", confidence=0, reasoning="[parse fallback] could not parse signal")


# ---------------------------------------------------------------------------
# Trading agent
# ---------------------------------------------------------------------------


class TradingAgent:
    """Generate buy/sell/hold signals using a local HF model with optional CAD."""

    def __init__(
        self,
        decoder: ContextAwareDecoder,
        *,
        calibrator: Optional[CADCalibrator] = None,
        cad_prior_mode: str = "bias_amplified",
        cad_alpha: float = 1.0,
        cad_top_p: float = 1.0,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        use_calibrator: bool = False,
        calibrator_alpha_min: float = 0.0,
        calibrator_alpha_max: float = 5.0,
        neg_prompt_builder: Optional["NegativePromptBuilder"] = None,
    ) -> None:
        self.decoder = decoder
        self.calibrator = calibrator
        self.cad_prior_mode = cad_prior_mode
        self.cad_alpha = cad_alpha
        self.cad_top_p = cad_top_p
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.use_calibrator = use_calibrator
        self.calibrator_alpha_min = calibrator_alpha_min
        self.calibrator_alpha_max = calibrator_alpha_max
        self.neg_prompt_builder = neg_prompt_builder

    # ----- public API -----

    def get_signal(
        self,
        ticker: str,
        date: pd.Timestamp,
        price_df: pd.DataFrame,
        decoding_mode: str = "baseline",
    ) -> TradingSignal:
        """Return a trading signal for *ticker* as of *date*.

        Parameters
        ----------
        ticker : str
            Stock symbol.
        date : pd.Timestamp
            Decision date — the model sees data up to the **previous close**
            (strictly before *date*).  The trade is executed at next day's open.
        price_df : pd.DataFrame
            Full price history; filtered internally to rows < *date*.
        decoding_mode : str
            ``"baseline"`` for standard generation, ``"cad"`` for context-aware decoding.
        """
        summary = build_financial_summary(ticker, date, price_df)

        context_prompt = (
            CONTEXT_SYSTEM.format(ticker=ticker, date=f"{date:%Y-%m-%d}")
            + "\n\n"
            + CONTEXT_BODY.format(ticker=ticker, date=f"{date:%Y-%m-%d}", financial_summary=summary)
        )

        # Resolve alpha
        alpha = self.cad_alpha
        if decoding_mode == "cad" and self.use_calibrator and self.calibrator is not None:
            try:
                cal = self.calibrator.calibrate_alpha(
                    ticker,
                    alpha_min=self.calibrator_alpha_min,
                    alpha_max=self.calibrator_alpha_max,
                )
                alpha = cal.alpha
                logger.info("Calibrated alpha for %s: %.3f (entropy=%.3f)", ticker, alpha, cal.entropy)
            except Exception as exc:
                logger.warning("Calibrator failed for %s: %s — using static alpha %.2f", ticker, exc, self.cad_alpha)

        config = CADConfig(
            alpha=alpha if decoding_mode == "cad" else 0.0,
            top_p=self.cad_top_p,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )

        prior_prompt = self._build_prior(ticker, date)

        generation = self.decoder.generate(context_prompt, prior_prompt, config)
        sig = parse_trading_signal(generation)
        sig.alpha_used = config.alpha
        return sig

    # ----- internals -----

    def _build_prior(self, ticker: str, date: pd.Timestamp) -> str:
        date_str = f"{date:%Y-%m-%d}"
        if self.cad_prior_mode == "optimized" and self.neg_prompt_builder is not None:
            task_instruction = (
                'You are a quantitative analyst evaluating a single stock.\n\n'
                'Return your analysis as a JSON object with exactly these keys:\n'
                '- "signal": one of "buy", "sell", or "hold"\n'
                '- "confidence": integer between 0 and 100\n'
                '- "reasoning": string with a concise rationale (max 200 chars)\n\n'
                'Respond with valid JSON only.\n\n'
                'What is your trading signal?'
            )
            return self.neg_prompt_builder.build(
                entity=ticker, date=date_str,
                task_prompt=task_instruction,
            )
        if self.cad_prior_mode == "no_context":
            return PRIOR_NO_CONTEXT.format(ticker=ticker)
        # bias_amplified (default)
        return PRIOR_BIAS_AMPLIFIED.format(ticker=ticker, date=date_str)
