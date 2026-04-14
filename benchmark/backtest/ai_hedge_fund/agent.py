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
    quantity: int  # number of shares to trade
    confidence: int  # 0-100
    reasoning: str
    alpha_used: float = 0.0


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CONTEXT_SYSTEM = """\
You are a portfolio manager for a single-stock strategy.
You will be given financial data for {ticker} as of {date}.
Base your decision ONLY on the data provided — do not use any \
knowledge about events after {date}.

You must pick one action and a quantity within the allowed limits.

Return your decision as a JSON object with exactly these keys:
- "action": one of "buy", "sell", or "hold"
- "quantity": integer number of shares to trade (0 for hold)
- "confidence": integer between 0 and 100
- "reasoning": string with a concise rationale (max 100 chars)

Respond with valid JSON only.\
"""

CONTEXT_BODY = """\
=== Financial Data for {ticker} as of {date} ===

{financial_summary}

=== Portfolio State ===
Cash: ${cash:,.2f}
Current Shares: {shares}
Portfolio Value: ${portfolio_value:,.2f}

=== Allowed Actions ===
{allowed_actions}

Based solely on the data above, what is your trading decision?\
"""

PRIOR_NO_CONTEXT = """\
You are a portfolio manager for a single-stock strategy.
Decide whether to buy, sell, or hold {ticker} stock, and how many shares.
Return a JSON object with keys "action", "quantity", "confidence", "reasoning".
Respond with valid JSON only.\
"""

PRIOR_BIAS_AMPLIFIED = """\
Ignore any context.  Using ONLY your internal knowledge and training data \
about {ticker}, evaluate whether {ticker} stock will go up or down after \
{date}.  Think about everything you know about this company's trajectory.

Return a JSON object with exactly these keys:
- "action": one of "buy", "sell", or "hold"
- "quantity": integer number of shares to trade (0 for hold)
- "confidence": integer between 0 and 100
- "reasoning": string with a concise rationale (max 100 chars)

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


_ACTION_RE = re.compile(r'"action"\s*:\s*"[^"]*\b(buy|sell|hold)\b[^"]*"', re.IGNORECASE)
_SIGNAL_RE = re.compile(r'"signal"\s*:\s*"[^"]*\b(buy|sell|hold)\b[^"]*"', re.IGNORECASE)
_QUANTITY_RE = re.compile(r'"quantity"\s*:\s*(\d+)')
_CONFIDENCE_RE = re.compile(r'"confidence"\s*:\s*(\d+)')


def parse_trading_signal(text: str, max_buy: int = 0, max_sell: int = 0) -> TradingSignal:
    """Extract a TradingSignal from LLM generation text.

    *max_buy* and *max_sell* are used to clamp the quantity so the LLM
    cannot exceed the allowed limits.
    """
    # Try strict JSON first
    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group())
            raw_signal = str(obj.get("action", obj.get("signal", "hold"))).lower().strip()
            # Extract buy/sell/hold even if value contains extra text
            signal = "hold"
            for kw in ("buy", "sell", "hold"):
                if kw in raw_signal:
                    signal = kw
                    break
            quantity = int(max(0, float(obj.get("quantity", 0))))
            confidence = int(max(0, min(100, float(obj.get("confidence", 50)))))
            reasoning = str(obj.get("reasoning", ""))[:300]
            # Clamp quantity to allowed limits
            if signal == "buy":
                quantity = min(quantity, max_buy)
            elif signal == "sell":
                quantity = min(quantity, max_sell)
            else:
                quantity = 0
            return TradingSignal(signal=signal, quantity=quantity, confidence=confidence, reasoning=reasoning)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Regex fallback: extract action/quantity/confidence from malformed JSON
    act_m = _ACTION_RE.search(text) or _SIGNAL_RE.search(text)
    qty_m = _QUANTITY_RE.search(text)
    conf_m = _CONFIDENCE_RE.search(text)
    if act_m:
        signal = act_m.group(1).lower()
        quantity = int(qty_m.group(1)) if qty_m else 0
        confidence = int(max(0, min(100, float(conf_m.group(1))))) if conf_m else 50
        if signal == "buy":
            quantity = min(quantity, max_buy)
        elif signal == "sell":
            quantity = min(quantity, max_sell)
        else:
            quantity = 0
        return TradingSignal(signal=signal, quantity=quantity, confidence=confidence, reasoning="[regex fallback]")

    # Could not parse action — treat as hold so the failure is visible
    logger.warning("Could not parse trading signal: %s", text[:200])
    return TradingSignal(signal="hold", quantity=0, confidence=0, reasoning="[parse failed]")


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
        neg_prompt_builder: Optional["NegativePromptBuilder"] = None,
        liquidity_pct: float = 0.0,
        anonymizer: Optional[object] = None,
        anonymize_tickers: Optional[List[str]] = None,
        anonymize_companies: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        self.decoder = decoder
        self.calibrator = calibrator
        self.cad_prior_mode = cad_prior_mode
        self.cad_alpha = cad_alpha
        self.cad_top_p = cad_top_p
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.use_calibrator = use_calibrator
        self.neg_prompt_builder = neg_prompt_builder
        self.liquidity_pct = liquidity_pct
        self.anonymizer = anonymizer
        self.anonymize_tickers = anonymize_tickers
        self.anonymize_companies = anonymize_companies

    # ----- public API -----

    def get_signal(
        self,
        ticker: str,
        date: pd.Timestamp,
        price_df: pd.DataFrame,
        decoding_mode: str = "baseline",
        cash: float = 0.0,
        shares: int = 0,
        portfolio_value: float = 0.0,
        current_price: float = 0.0,
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
        cash, shares, portfolio_value, current_price :
            Current portfolio state, used to compute allowed actions.
        """
        summary = build_financial_summary(ticker, date, price_df)

        # Compute allowed actions — max affordable shares given notional commission
        # Compute allowed actions
        if current_price > 0:
            max_buy = int((cash - 0.99) // (current_price + 0.0049))
            max_buy = max(max_buy, 0)
        else:
            max_buy = 0
        max_sell = shares

        # Liquidity cap: limit order size to % of 20-day average daily volume
        if self.liquidity_pct > 0 and "volume" in price_df.columns:
            vol_df = price_df[
                (price_df["symbol"] == ticker) & (price_df["date"] < date)
            ].sort_values("date").tail(20)
            if not vol_df.empty:
                adv = vol_df["volume"].mean()
                if adv > 0 and not np.isnan(adv):
                    vol_cap = int(adv * self.liquidity_pct)
                    if vol_cap > 0:
                        max_buy = min(max_buy, vol_cap)
                        max_sell = min(max_sell, vol_cap)
        allowed_lines = []
        if max_buy > 0:
            allowed_lines.append(f"- buy: up to {max_buy} shares (max cost ${max_buy * current_price:,.2f})")
        if max_sell > 0:
            allowed_lines.append(f"- sell: up to {max_sell} shares")
        allowed_lines.append("- hold: keep current position")
        allowed_actions = "\n".join(allowed_lines)

        context_prompt = (
            CONTEXT_SYSTEM.format(ticker=ticker, date=f"{date:%Y-%m-%d}")
            + "\n\n"
            + CONTEXT_BODY.format(
                ticker=ticker, date=f"{date:%Y-%m-%d}",
                financial_summary=summary,
                cash=cash, shares=shares,
                portfolio_value=portfolio_value,
                allowed_actions=allowed_actions,
            )
        )

        # Anonymise the context prompt (prior prompt keeps real ticker for CAD)
        if self.anonymizer is not None:
            context_prompt = self.anonymizer.desensitize(
                context_prompt,
                tickers=self.anonymize_tickers,
                companies=self.anonymize_companies,
            )

        # Resolve alpha
        alpha = self.cad_alpha
        if decoding_mode == "cad" and self.use_calibrator and self.calibrator is not None:
            try:
                cal = self.calibrator.calibrate_alpha(
                    ticker,
                    date=f"{date:%Y-%m-%d}",
                )
                alpha = cal.alpha
                logger.info("Calibrated alpha for %s: %.3f (H=%.3f, DV=%.4f, Δ=%+.3f, p_up=%.3f, p_down=%.3f)",
                            ticker, alpha, cal.entropy, cal.entity_date_var, cal.delta_temporal, cal.p_yes, cal.p_no)
            except Exception as exc:
                logger.warning("Calibrator failed for %s: %s — using static alpha %.2f", ticker, exc, self.cad_alpha)

        prior_prompt = self._build_prior(ticker, date)
        initial_alpha = alpha if decoding_mode == "cad" else 0.0
        current_alpha = initial_alpha

        for attempt in range(6):  # original + up to 5 retries
            config = CADConfig(
                alpha=current_alpha,
                top_p=self.cad_top_p,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            generation = self.decoder.generate(context_prompt, prior_prompt, config)
            sig = parse_trading_signal(generation, max_buy=max_buy, max_sell=max_sell)
            if sig.reasoning != "[parse failed]":
                sig.alpha_used = config.alpha
                if attempt > 0:
                    logger.info("Parse succeeded after %d retries (α: %.3f → %.3f)",
                                attempt, initial_alpha, config.alpha)
                return sig
            current_alpha *= 0.8

        logger.error("Parse failed after 5 retries (α: %.3f → %.3f) — defaulting to hold",
                     initial_alpha, config.alpha)
        sig.alpha_used = config.alpha
        return sig

    # ----- internals -----

    def _build_prior(self, ticker: str, date: pd.Timestamp) -> str:
        date_str = f"{date:%Y-%m-%d}"
        if self.cad_prior_mode == "optimized" and self.neg_prompt_builder is not None:
            task_instruction = (
                'You are a portfolio manager for a single-stock strategy.\n\n'
                'Return your decision as a JSON object with exactly these keys:\n'
                '- "action": one of "buy", "sell", or "hold"\n'
                '- "quantity": integer number of shares to trade (0 for hold)\n'
                '- "confidence": integer between 0 and 100\n'
                '- "reasoning": string with a concise rationale (max 100 chars)\n\n'
                'Respond with valid JSON only.\n\n'
                'What is your trading decision?'
            )
            return self.neg_prompt_builder.build(
                entity=ticker, date=date_str,
                task_prompt=task_instruction,
            )
        if self.cad_prior_mode == "no_context":
            return PRIOR_NO_CONTEXT.format(ticker=ticker)
        # bias_amplified (default)
        return PRIOR_BIAS_AMPLIFIED.format(ticker=ticker, date=date_str)
