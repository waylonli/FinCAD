"""
Single-stock backtest: compare baseline vs CAD on one ticker.

Inspired by the ai-hedge-fund approach — at each rebalance date the LLM
receives structured financial data and returns a buy/sell/hold signal.
The portfolio is just cash + shares of that one stock.

Examples
--------
Baseline (NVDA, 2018-2020):
    python -m benchmark.backtest.ai_hedge_fund.eval \\
        --model-name Qwen/Qwen2.5-7B-Instruct --use-chat-template \\
        --ticker NVDA --start-date 2018-01-01 --end-date 2020-01-01 \\
        --decoding-mode baseline

CAD with bias-amplified prior:
    python -m benchmark.backtest.ai_hedge_fund.eval \\
        --model-name Qwen/Qwen2.5-7B-Instruct --use-chat-template \\
        --ticker NVDA --start-date 2018-01-01 --end-date 2020-01-01 \\
        --decoding-mode cad --cad-alpha 1.0 --cad-prior-mode bias_amplified
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-stock backtest with baseline vs CAD comparison.",
    )

    # Model
    g = p.add_argument_group("Model")
    g.add_argument("--model-name", required=True, help="HF model id or local path")
    g.add_argument("--model-cache-dir", default="../pretrained_models")
    g.add_argument("--use-chat-template", action="store_true")
    g.add_argument("--attn-implementation", default=None)
    g.add_argument("--peft-adapter", default=None,
                   help="Path to PEFT/LoRA adapter directory (base model resolved from adapter_config.json)")

    # Decoding
    g = p.add_argument_group("Decoding")
    g.add_argument("--decoding-mode", default="baseline", choices=["baseline", "cad"])
    g.add_argument("--cad-alpha", type=float, default=1.0)
    g.add_argument("--cad-top-p", type=float, default=1.0)
    g.add_argument("--cad-prior-mode", default="bias_amplified",
                   choices=["no_context", "bias_amplified", "optimized"])
    g.add_argument("--optimized-instruction", type=str, default=None,
                   help="Path to optimized instruction JSON (required when --cad-prior-mode=optimized)")

    # Generation
    g = p.add_argument_group("Generation")
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--max-new-tokens", type=int, default=256)
    g.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility when temperature > 0")

    # Calibrator
    g = p.add_argument_group("Calibrator")
    g.add_argument("--use-calibrator", action="store_true")

    # Backtest
    g = p.add_argument_group("Backtest")
    g.add_argument("--ticker", default="NVDA", help="Stock ticker to backtest")
    g.add_argument("--start-date", default="2018-01-01")
    g.add_argument("--end-date", default="2020-01-01")
    g.add_argument("--rebalance-freq", default="M",
                   help="Rebalance frequency: M=monthly, Q=quarterly, W=weekly (pandas offset alias)")
    g.add_argument("--initial-capital", type=float, default=100_000.0)
    g.add_argument("--liquidity-pct", type=float, default=0.01,
                   help="Max order size as fraction of 20-day ADV (e.g. 0.01 = 1%%). "
                   "0 = no limit. Default: 0.01 (1%%).")

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--price-csv", default=None,
                   help="Path to local price CSV (with date, symbol, adjusted_close). "
                   "If omitted, tries abrdn-risk-factor-eval/data/price/price_data.csv")

    # Anonymisation (Experiment 3)
    g = p.add_argument_group("Anonymisation")
    g.add_argument("--anonymize", action="store_true",
                   help="Replace ticker and company name with generic labels in the "
                   "context prompt (for entity-anonymisation experiments).")
    g.add_argument("--entity-file", default=None,
                   help="Path to entity.json (ticker↔company mapping). "
                   "Default: <project_root>/utils/entity.json")

    # Output
    g = p.add_argument_group("Output")
    g.add_argument("--results-file", default=None, help="Per-decision JSONL")
    g.add_argument("--summary-file", default=None, help="Summary JSON")
    g.add_argument("--values-csv", default=None,
                   help="Daily portfolio values CSV (date, strategy, buy_and_hold) "
                   "for plotting backtesting curves")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------


def load_price_data(path: str | Path, ticker: str) -> pd.DataFrame:
    """Load and filter price data for a single ticker.

    If the CSV contains ``close``, ``adjusted_close``, and ``open`` but no
    ``adjusted_open``, we derive it:  ``adjusted_open = open * (adjusted_close / close)``.
    This ensures execution prices are on the same split-adjusted scale as
    valuation prices.
    """
    df = pd.read_csv(path, low_memory=False)

    # Normalise columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df["symbol"] = df["symbol"].astype(str).str.upper()

    # Filter
    df = df[df["symbol"] == ticker.upper()].sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No price data found for {ticker} in {path}")

    # Derive adjusted_open from the split/dividend adjustment factor
    if (
        "adjusted_open" not in df.columns
        and {"open", "close", "adjusted_close"}.issubset(df.columns)
    ):
        adj_factor = df["adjusted_close"] / df["close"]
        df["adjusted_open"] = df["open"] * adj_factor

    return df


# ---------------------------------------------------------------------------
# Commission model — notional-based (basis points of trade value)
# Note: commission is deducted mechanically from the portfolio but is NOT
# mentioned in the LLM prompt, to avoid introducing asymmetric context
# between the context prompt and the CAD prior prompt.
#
# 10 bps is a central estimate for large/mid-cap US equities covering
# commission + half-spread + market impact (Novy-Marx & Velikov 2016 RFS).
# This is split-invariant: fees scale with notional value, not share count,
# so split-adjusted prices do not inflate commissions.
# ---------------------------------------------------------------------------

COMMISSION_BPS = 10.0  # basis points of trade notional (one-way)


def compute_commission(shares: int, price: float = 0.0) -> float:
    """Notional-based commission: COMMISSION_BPS bps of trade value.

    Split-invariant — the fee depends only on notional value, not share count.
    Falls back to $0 if price is unknown (backward compat).
    """
    if shares <= 0 or price <= 0:
        return 0.0
    return shares * price * COMMISSION_BPS / 10_000


# ---------------------------------------------------------------------------
# Risk-free rate — historical average
# ---------------------------------------------------------------------------

RF_ANNUAL = 0.03  # 3% annualised


# ---------------------------------------------------------------------------
# Simple portfolio
# ---------------------------------------------------------------------------


class SimplePortfolio:
    """Cash + shares of a single stock, with commission tracking."""

    def __init__(self, initial_cash: float) -> None:
        self.cash = initial_cash
        self.shares = 0
        self.cost_basis = 0.0
        self.total_commission = 0.0

    def buy(self, price: float, qty: int) -> int:
        """Buy *qty* shares at *price*. Returns shares actually bought."""
        if qty <= 0:
            return 0
        cost = qty * price
        commission = compute_commission(qty, price)
        # Reduce qty if we can't afford it
        while qty > 0 and qty * price + compute_commission(qty, price) > self.cash:
            qty -= 1
        if qty <= 0:
            return 0
        cost = qty * price
        commission = compute_commission(qty, price)
        # Update cost basis (weighted avg, inclusive of commission)
        total_shares = self.shares + qty
        if total_shares > 0:
            self.cost_basis = (self.cost_basis * self.shares + cost + commission) / total_shares
        self.shares = total_shares
        self.cash -= cost + commission
        self.total_commission += commission
        return qty

    def sell(self, price: float, qty: int) -> int:
        """Sell *qty* shares at *price*. Returns shares actually sold."""
        qty = min(qty, self.shares)
        if qty <= 0:
            return 0
        proceeds = qty * price
        commission = compute_commission(qty, price)
        self.cash += proceeds - commission
        self.shares -= qty
        self.total_commission += commission
        if self.shares == 0:
            self.cost_basis = 0.0
        return qty

    def value(self, price: float) -> float:
        return self.cash + self.shares * price

    def summary(self, price: float) -> dict:
        return {
            "cash": self.cash,
            "shares": self.shares,
            "cost_basis": self.cost_basis,
            "total_commission": self.total_commission,
            "portfolio_value": self.value(price),
        }


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    values: pd.Series,
    initial_capital: float,
    rf_annual: float = RF_ANNUAL,
) -> dict:
    """Compute standard performance metrics from a daily portfolio value series."""
    returns = values.pct_change().dropna()
    if len(returns) < 2:
        return {}

    total_ret = values.iloc[-1] / initial_capital - 1
    n_days = len(returns)
    cagr = (1 + total_ret) ** (252 / n_days) - 1
    vol = returns.std() * np.sqrt(252)

    daily_rf = rf_annual / 252
    excess = returns - daily_rf
    sharpe = float(np.sqrt(252) * excess.mean() / excess.std()) if excess.std() > 1e-12 else 0.0

    downside = excess[excess < 0]
    if len(downside) > 0 and downside.std() > 1e-12:
        sortino = float(np.sqrt(252) * excess.mean() / downside.std())
    else:
        sortino = float("inf") if excess.mean() > 0 else 0.0

    cummax = values.cummax()
    drawdown = (values - cummax) / cummax
    max_dd = float(drawdown.min())

    return {
        "total_return": float(total_ret),
        "cagr": float(cagr),
        "annual_vol": float(vol),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "ending_value": float(values.iloc[-1]),
    }


# ---------------------------------------------------------------------------
# Buy-and-hold benchmark
# ---------------------------------------------------------------------------


def buy_and_hold(
    val_series: pd.Series,
    initial_capital: float,
    open_series: pd.Series | None = None,
) -> pd.Series:
    """Return daily portfolio values for a buy-and-hold strategy.

    Buys at the first day's **open** (via *open_series*) for a fair comparison
    with the LLM strategy which also executes at the open.  Moomoo commission
    is deducted on the initial purchase.  Daily values are then
    marked-to-market using *val_series* (adjusted_close).

    If *open_series* is ``None``, falls back to buying at the first day's
    adjusted_close (legacy behaviour).
    """
    if open_series is not None and len(open_series) > 0:
        entry_price = open_series.iloc[0]
    else:
        entry_price = val_series.iloc[0]
    shares = int(initial_capital // entry_price)
    commission = compute_commission(shares, entry_price)
    # Reduce shares if commission makes it unaffordable
    while shares > 0 and shares * entry_price + commission > initial_capital:
        shares -= 1
        commission = compute_commission(shares, entry_price)
    leftover = initial_capital - shares * entry_price - commission
    return shares * val_series + leftover


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------


def run_single_stock_backtest(
    agent,  # TradingAgent
    price_df: pd.DataFrame,
    ticker: str,
    start_date: str,
    end_date: str,
    rebalance_freq: str,
    initial_capital: float,
    decoding_mode: str,
    results_file=None,
) -> dict:
    """Run the single-stock backtest.

    Timing convention (avoids intraday look-ahead):
    - On each rebalance date the agent sees data up to the **previous close**
      (``build_financial_summary`` uses ``date < rebal_date``).
    - Trades are executed at the **next trading day's adjusted open** price
      (split-consistent with ``adjusted_close``).
    - Daily portfolio values are marked-to-market using ``adjusted_close``.

    Returns a dict with keys: strategy_metrics, benchmark_metrics,
    portfolio_values (list of {date, value}), decisions (list of dicts).
    """
    val_col = "adjusted_close" if "adjusted_close" in price_df.columns else "close"
    # Prefer adjusted_open (split-consistent with adjusted_close); fall back
    # to raw open, then to the valuation column.
    if "adjusted_open" in price_df.columns:
        exec_col = "adjusted_open"
    elif "open" in price_df.columns:
        exec_col = "open"
    else:
        exec_col = val_col

    sub = price_df.set_index("date").sort_index()
    ts_val = sub[val_col][start_date:end_date]           # for mark-to-market
    ts_open = sub[exec_col][start_date:end_date]         # for execution

    if ts_val.empty:
        raise ValueError(f"No price data for {ticker} in [{start_date}, {end_date}]")

    trading_days = ts_val.index

    # Build rebalance dates — snapped to actual trading days
    rebal_dates = pd.date_range(start_date, end_date, freq=rebalance_freq)
    rebal_dates = [
        trading_days[trading_days.searchsorted(d)]
        for d in rebal_dates
        if trading_days.searchsorted(d) < len(trading_days)
    ]
    rebal_set = set(rebal_dates)

    # Map each rebalance date → next trading day (for execution)
    day_list = list(trading_days)
    day_pos = {d: i for i, d in enumerate(day_list)}
    exec_date_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for rd in rebal_dates:
        idx = day_pos[rd] + 1
        if idx < len(day_list):
            exec_date_map[rd] = day_list[idx]
        # else: last day — no next day to execute, signal is dropped

    logger.info(
        "Backtest: %s from %s to %s — %d rebalance dates, %d trading days",
        ticker, start_date, end_date, len(rebal_dates), len(ts_val),
    )

    portfolio = SimplePortfolio(initial_capital)
    daily_values: list[dict] = []
    decisions: list[dict] = []
    pending_signal: Optional[TradingSignal] = None
    pending_rebal_date: Optional[pd.Timestamp] = None
    pending_exec_date: Optional[pd.Timestamp] = None
    results_fh = None
    if results_file:
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        results_fh = open(results_file, "w")

    try:
        for date in trading_days:
            val_price = ts_val[date]      # adjusted close — for valuation
            open_price = ts_open[date]    # open — for execution

            # Execute pending trade at today's open
            if pending_signal is not None and pending_exec_date == date:
                sig = pending_signal
                executed = 0
                if sig.signal == "buy" and sig.quantity > 0:
                    executed = portfolio.buy(open_price, sig.quantity)
                elif sig.signal == "sell" and sig.quantity > 0:
                    executed = portfolio.sell(open_price, sig.quantity)

                record = {
                    "decision_date": str(pending_rebal_date.date()),
                    "execution_date": str(date.date()),
                    "ticker": ticker,
                    "execution_price": round(open_price, 2),
                    "signal": sig.signal,
                    "requested_shares": sig.quantity,
                    "confidence": sig.confidence,
                    "reasoning": sig.reasoning,
                    "alpha_used": sig.alpha_used,
                    "executed_shares": executed,
                    "decoding_mode": decoding_mode,
                    **portfolio.summary(val_price),
                }
                decisions.append(record)

                logger.info(
                    "[%s] decide → [%s] exec  %s %s  conf=%d  shares=%+d  "
                    "exec_price=$%.2f  value=$%.0f  alpha=%.2f",
                    pending_rebal_date.date(), date.date(),
                    ticker, sig.signal.upper(), sig.confidence,
                    executed if sig.signal == "buy" else -executed,
                    open_price, portfolio.value(val_price), sig.alpha_used,
                )

                if results_fh:
                    results_fh.write(json.dumps(record, default=str) + "\n")
                    results_fh.flush()

                pending_signal = None
                pending_rebal_date = None
                pending_exec_date = None

            # Generate signal on rebalance date (sees data up to prev close)
            if date in rebal_set and date in exec_date_map:
                sig = agent.get_signal(
                    ticker, date, price_df,
                    decoding_mode=decoding_mode,
                    cash=portfolio.cash,
                    shares=portfolio.shares,
                    portfolio_value=portfolio.value(val_price),
                    current_price=val_price,
                )
                pending_signal = sig
                pending_rebal_date = date
                pending_exec_date = exec_date_map[date]

            daily_values.append({"date": date, "value": portfolio.value(val_price)})

    finally:
        if results_fh:
            results_fh.close()

    # Build value series
    idx = pd.DatetimeIndex([v["date"] for v in daily_values])
    val_series = pd.Series([v["value"] for v in daily_values], index=idx, name="strategy")

    bnh_series = buy_and_hold(ts_val, initial_capital, open_series=ts_open)
    bnh_series.name = "buy_and_hold"

    strategy_metrics = compute_metrics(val_series, initial_capital)
    benchmark_metrics = compute_metrics(bnh_series, initial_capital)

    # Combine into a single DataFrame for easy export / plotting
    values_df = pd.DataFrame({
        "date": val_series.index,
        "strategy": val_series.values,
        "buy_and_hold": bnh_series.reindex(val_series.index).values,
    })

    return {
        "strategy_metrics": strategy_metrics,
        "benchmark_metrics": benchmark_metrics,
        "portfolio_values": daily_values,
        "values_df": values_df,
        "decisions": decisions,
        "total_commission": portfolio.total_commission,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_summary(result: dict, args: argparse.Namespace) -> None:
    print("\n" + "=" * 64)
    print(f"  SINGLE-STOCK BACKTEST: {args.ticker}  [{args.decoding_mode.upper()}]")
    print(f"  Period: {args.start_date} → {args.end_date}")
    print(f"  Risk-free rate: {RF_ANNUAL:.1%}  (historical average)")
    print(f"  Commission: {COMMISSION_BPS:.0f} bps of trade notional (split-invariant)")
    if hasattr(args, 'liquidity_pct') and args.liquidity_pct > 0:
        print(f"  Liquidity cap: {args.liquidity_pct:.2%} of 20-day ADV")
    if hasattr(args, 'anonymize') and args.anonymize:
        print(f"  Anonymisation: ON (context prompt only)")
    print("=" * 64)

    def _fmt(metrics: dict, label: str) -> None:
        print(f"\n  {label}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                if "return" in k or "cagr" in k or "drawdown" in k:
                    print(f"    {k:20s}  {v:+.4%}")
                else:
                    print(f"    {k:20s}  {v:,.4f}")

    _fmt(result["strategy_metrics"], f"LLM Strategy ({args.decoding_mode})")
    _fmt(result["benchmark_metrics"], "Buy & Hold")

    n_decisions = len(result["decisions"])
    buys = sum(1 for d in result["decisions"] if d["signal"] == "buy")
    sells = sum(1 for d in result["decisions"] if d["signal"] == "sell")
    holds = sum(1 for d in result["decisions"] if d["signal"] == "hold")
    real_buys = sum(1 for d in result["decisions"] if d["signal"] == "buy" and d["executed_shares"] > 0)
    real_sells = sum(1 for d in result["decisions"] if d["signal"] == "sell" and d["executed_shares"] > 0)
    total_commission = result.get("total_commission", 0.0)
    print(f"\n  Decisions: {n_decisions} total — {buys} buy, {sells} sell, {holds} hold")
    print(f"  Executed trades (non-zero): {real_buys} buys, {real_sells} sells")
    print(f"  Total commission paid: ${total_commission:,.2f}")

    alphas = [d["alpha_used"] for d in result["decisions"]]
    if alphas:
        a = np.array(alphas)
        print(f"\n  Alpha stats (n={len(a)}):")
        print(f"    mean={a.mean():.3f}  std={a.std():.3f}  "
              f"min={a.min():.3f}  max={a.max():.3f}  median={np.median(a):.3f}")
    print("=" * 64 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # ---- Load price data ----
    price_path = args.price_csv
    if price_path is None:
        # Try abrdn data
        candidate = REPO_ROOT / "abrdn-risk-factor-eval" / "data" / "price" / "price_data.csv"
        if candidate.exists():
            price_path = str(candidate)
        else:
            print("ERROR: No --price-csv provided and default abrdn price data not found.", file=sys.stderr)
            sys.exit(1)

    logger.info("Loading prices from %s for %s", price_path, args.ticker)
    price_df = load_price_data(price_path, args.ticker)
    logger.info("Loaded %d price rows", len(price_df))

    # ---- Load model ----
    from adapters import AdapterInitConfig, TransformersAdapter
    from cad import CADConfig, ContextAwareDecoder
    from cad.calibrator import CADCalibrator
    from .agent import TradingAgent

    import random, numpy as np, torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Loading model %s ...", args.model_name)
    adapter = TransformersAdapter(
        AdapterInitConfig(
            model_name=args.model_name,
            use_chat_template=args.use_chat_template,
            cache_dir=args.model_cache_dir,
            attn_implementation=args.attn_implementation,
            peft_adapter=getattr(args, 'peft_adapter', None),
        )
    )

    decoder = ContextAwareDecoder(
        adapter.model,
        adapter.tokenizer,
        device=adapter.device,
        use_chat_template=args.use_chat_template,
    )

    neg_prompt_builder = None
    if args.cad_prior_mode == "optimized" or args.use_calibrator:
        from cad.discovery import NegativePromptBuilder
        if args.optimized_instruction is None:
            print("ERROR: --optimized-instruction is required when --cad-prior-mode=optimized or --use-calibrator.", file=sys.stderr)
            sys.exit(1)
        neg_prompt_builder = NegativePromptBuilder.from_file(args.optimized_instruction)

    calibrator = None
    if args.use_calibrator:
        logit_gap_profile = None
        if neg_prompt_builder is not None:
            logit_gap_profile = getattr(neg_prompt_builder.instruction, "logit_gap_profile", None)
        calibrator = CADCalibrator(
            adapter.model,
            adapter.tokenizer,
            device=adapter.device,
            use_chat_template=args.use_chat_template,
            optimized_instruction=neg_prompt_builder.instruction.instruction if neg_prompt_builder else "",
            logit_gap_profile=logit_gap_profile,
        )

    # ---- Anonymiser (optional) ----
    anonymizer = None
    if args.anonymize:
        from utils.anonymizer import LegacyAnonymizer
        entity_path = args.entity_file or str(PROJECT_ROOT / "utils" / "entity.json")
        anonymizer = LegacyAnonymizer(entity_file=entity_path)
        # Resolve company name for the target ticker
        company_name = None
        if args.ticker in anonymizer.tickers:
            idx = anonymizer.tickers.index(args.ticker)
            company_name = anonymizer.companies[idx]
        logger.info("Anonymisation ON — ticker=%s  company=%s", args.ticker, company_name)

    agent = TradingAgent(
        decoder,
        calibrator=calibrator,
        cad_prior_mode=args.cad_prior_mode,
        cad_alpha=args.cad_alpha,
        cad_top_p=args.cad_top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        use_calibrator=args.use_calibrator,
        neg_prompt_builder=neg_prompt_builder,
        liquidity_pct=args.liquidity_pct,
        anonymizer=anonymizer,
        anonymize_tickers=[args.ticker] if args.anonymize else None,
        anonymize_companies=[company_name] if args.anonymize and company_name else None,
    )

    # ---- Entity calibration (date-variance) ----
    if args.use_calibrator and calibrator is not None:
        logger.info("Calibrating entity date-variance for %s ...", args.ticker)
        calibrator.calibrate_entity(args.ticker)

    # ---- Run backtest ----
    result = run_single_stock_backtest(
        agent=agent,
        price_df=price_df,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        rebalance_freq=args.rebalance_freq,
        initial_capital=args.initial_capital,
        decoding_mode=args.decoding_mode,
        results_file=args.results_file,
    )

    # ---- Output ----
    print_summary(result, args)

    # ---- Export alpha per decision date ----
    decisions = result["decisions"]
    alphas = [d["alpha_used"] for d in decisions]
    alpha_stats = {}
    if alphas:
        a = np.array(alphas)
        alpha_stats = {
            "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()),
            "median": float(np.median(a)),
        }

    if args.results_file and alphas:
        alpha_csv = Path(args.results_file).with_suffix(".alpha.csv")
        alpha_df = pd.DataFrame([
            {"decision_date": d["decision_date"], "alpha": d["alpha_used"]}
            for d in decisions
        ])
        alpha_df.to_csv(alpha_csv, index=False)
        logger.info("Alpha per date written to %s (%d rows)", alpha_csv, len(alpha_df))

    if args.summary_file:
        out = Path(args.summary_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "ticker": args.ticker,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "decoding_mode": args.decoding_mode,
            "model_name": args.model_name,
            "cad_alpha": args.cad_alpha,
            "cad_prior_mode": args.cad_prior_mode,
            "rebalance_freq": args.rebalance_freq,
            "initial_capital": args.initial_capital,
            "rf_annual": RF_ANNUAL,
            "commission_bps": COMMISSION_BPS,
            "liquidity_pct": args.liquidity_pct,
            "anonymize": args.anonymize,
            "total_commission": result["total_commission"],
            "strategy": result["strategy_metrics"],
            "buy_and_hold": result["benchmark_metrics"],
            "n_decisions": len(decisions),
            "n_buys": sum(1 for d in decisions if d["signal"] == "buy"),
            "n_sells": sum(1 for d in decisions if d["signal"] == "sell"),
            "n_holds": sum(1 for d in decisions if d["signal"] == "hold"),
            "n_real_buys": sum(1 for d in decisions if d["signal"] == "buy" and d["executed_shares"] > 0),
            "n_real_sells": sum(1 for d in decisions if d["signal"] == "sell" and d["executed_shares"] > 0),
            "alpha_stats": alpha_stats,
        }
        with open(out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Summary written to %s", out)

    if args.values_csv:
        vpath = Path(args.values_csv)
        vpath.parent.mkdir(parents=True, exist_ok=True)
        result["values_df"].to_csv(vpath, index=False)
        logger.info("Daily values CSV written to %s (%d rows)",
                     vpath, len(result["values_df"]))


if __name__ == "__main__":
    main()
