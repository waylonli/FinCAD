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

    # Decoding
    g = p.add_argument_group("Decoding")
    g.add_argument("--decoding-mode", default="baseline", choices=["baseline", "cad"])
    g.add_argument("--cad-alpha", type=float, default=1.0)
    g.add_argument("--cad-top-p", type=float, default=1.0)
    g.add_argument("--cad-prior-mode", default="bias_amplified",
                   choices=["no_context", "bias_amplified"])

    # Generation
    g = p.add_argument_group("Generation")
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--max-new-tokens", type=int, default=256)

    # Calibrator
    g = p.add_argument_group("Calibrator")
    g.add_argument("--use-calibrator", action="store_true")
    g.add_argument("--calibrator-alpha-min", type=float, default=0.0)
    g.add_argument("--calibrator-alpha-max", type=float, default=5.0)

    # Backtest
    g = p.add_argument_group("Backtest")
    g.add_argument("--ticker", default="NVDA", help="Stock ticker to backtest")
    g.add_argument("--start-date", default="2018-01-01")
    g.add_argument("--end-date", default="2020-01-01")
    g.add_argument("--rebalance-freq", default="M",
                   help="Rebalance frequency: M=monthly, Q=quarterly, W=weekly (pandas offset alias)")
    g.add_argument("--initial-capital", type=float, default=100_000.0)

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--price-csv", default=None,
                   help="Path to local price CSV (with date, symbol, adjusted_close). "
                   "If omitted, tries abrdn-risk-factor-eval/data/price/price_data.csv")

    # Output
    g = p.add_argument_group("Output")
    g.add_argument("--results-file", default=None, help="Per-decision JSONL")
    g.add_argument("--summary-file", default=None, help="Summary JSON")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------


def load_price_data(path: str | Path, ticker: str) -> pd.DataFrame:
    """Load and filter price data for a single ticker."""
    df = pd.read_csv(path, low_memory=False)

    # Normalise columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df["symbol"] = df["symbol"].astype(str).str.upper()

    # Filter
    df = df[df["symbol"] == ticker.upper()].sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No price data found for {ticker} in {path}")
    return df


# ---------------------------------------------------------------------------
# Simple portfolio
# ---------------------------------------------------------------------------


class SimplePortfolio:
    """Cash + shares of a single stock."""

    def __init__(self, initial_cash: float) -> None:
        self.cash = initial_cash
        self.shares = 0
        self.cost_basis = 0.0

    def buy(self, price: float, fraction: float = 1.0) -> int:
        """Buy shares using *fraction* of available cash. Returns shares bought."""
        available = self.cash * fraction
        qty = int(available // price)
        if qty <= 0:
            return 0
        cost = qty * price
        # Update cost basis (weighted avg)
        total_shares = self.shares + qty
        if total_shares > 0:
            self.cost_basis = (self.cost_basis * self.shares + cost) / total_shares
        self.shares = total_shares
        self.cash -= cost
        return qty

    def sell(self, price: float, fraction: float = 1.0) -> int:
        """Sell *fraction* of current shares. Returns shares sold."""
        qty = int(self.shares * fraction)
        if qty <= 0:
            return 0
        self.cash += qty * price
        self.shares -= qty
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
            "portfolio_value": self.value(price),
        }


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    values: pd.Series,
    initial_capital: float,
    rf_annual: float = 0.0,
) -> dict:
    """Compute standard performance metrics from a daily portfolio value series."""
    returns = values.pct_change().dropna()
    if len(returns) < 2:
        return {}

    total_ret = values.iloc[-1] / values.iloc[0] - 1
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


def buy_and_hold(price_series: pd.Series, initial_capital: float) -> pd.Series:
    """Return daily portfolio values for a buy-and-hold strategy."""
    first_price = price_series.iloc[0]
    shares = int(initial_capital // first_price)
    leftover = initial_capital - shares * first_price
    return shares * price_series + leftover


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

    Returns a dict with keys: strategy_metrics, benchmark_metrics,
    portfolio_values (list of {date, value}), decisions (list of dicts).
    """
    pcol = "adjusted_close" if "adjusted_close" in price_df.columns else "close"
    ts = price_df.set_index("date")[pcol]
    ts = ts[start_date:end_date]

    if ts.empty:
        raise ValueError(f"No price data for {ticker} in [{start_date}, {end_date}]")

    # Build rebalance dates
    rebal_dates = pd.date_range(start_date, end_date, freq=rebalance_freq)
    # Only keep dates that fall on or after actual trading days
    trading_days = ts.index
    rebal_dates = [
        trading_days[trading_days.searchsorted(d)]
        for d in rebal_dates
        if trading_days.searchsorted(d) < len(trading_days)
    ]
    rebal_set = set(rebal_dates)

    logger.info(
        "Backtest: %s from %s to %s — %d rebalance dates, %d trading days",
        ticker, start_date, end_date, len(rebal_dates), len(ts),
    )

    portfolio = SimplePortfolio(initial_capital)
    daily_values: list[dict] = []
    decisions: list[dict] = []
    results_fh = None
    if results_file:
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        results_fh = open(results_file, "w")

    try:
        for date, price in ts.items():
            # Check if today is a rebalance date
            if date in rebal_set:
                sig = agent.get_signal(ticker, date, price_df, decoding_mode=decoding_mode)

                # Execute trade
                executed = 0
                if sig.signal == "buy":
                    executed = portfolio.buy(price)
                elif sig.signal == "sell":
                    executed = portfolio.sell(price)

                record = {
                    "date": str(date.date()),
                    "ticker": ticker,
                    "price": round(price, 2),
                    "signal": sig.signal,
                    "confidence": sig.confidence,
                    "reasoning": sig.reasoning,
                    "alpha_used": sig.alpha_used,
                    "executed_shares": executed,
                    "decoding_mode": decoding_mode,
                    **portfolio.summary(price),
                }
                decisions.append(record)

                logger.info(
                    "[%s] %s %s  conf=%d  shares=%+d  value=$%,.0f  alpha=%.2f",
                    date.date(), ticker, sig.signal.upper(), sig.confidence,
                    executed if sig.signal == "buy" else -executed,
                    portfolio.value(price), sig.alpha_used,
                )

                if results_fh:
                    results_fh.write(json.dumps(record, default=str) + "\n")
                    results_fh.flush()

            daily_values.append({"date": date, "value": portfolio.value(price)})

    finally:
        if results_fh:
            results_fh.close()

    # Build value series
    val_series = pd.Series(
        [v["value"] for v in daily_values],
        index=pd.DatetimeIndex([v["date"] for v in daily_values]),
        name="strategy",
    )
    bnh_series = buy_and_hold(ts, initial_capital)

    strategy_metrics = compute_metrics(val_series, initial_capital)
    benchmark_metrics = compute_metrics(bnh_series, initial_capital)

    return {
        "strategy_metrics": strategy_metrics,
        "benchmark_metrics": benchmark_metrics,
        "portfolio_values": daily_values,
        "decisions": decisions,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_summary(result: dict, args: argparse.Namespace) -> None:
    print("\n" + "=" * 64)
    print(f"  SINGLE-STOCK BACKTEST: {args.ticker}  [{args.decoding_mode.upper()}]")
    print(f"  Period: {args.start_date} → {args.end_date}")
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
    print(f"\n  Decisions: {n_decisions} total — {buys} buy, {sells} sell, {holds} hold")
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
    from steering.adapters import AdapterInitConfig, TransformersAdapter
    from cad import CADConfig, ContextAwareDecoder
    from cad.calibrator import CADCalibrator
    from .agent import TradingAgent

    logger.info("Loading model %s ...", args.model_name)
    adapter = TransformersAdapter(
        AdapterInitConfig(
            model_name=args.model_name,
            use_chat_template=args.use_chat_template,
            cache_dir=args.model_cache_dir,
            attn_implementation=args.attn_implementation,
        )
    )

    decoder = ContextAwareDecoder(
        adapter.model,
        adapter.tokenizer,
        device=adapter.device,
        use_chat_template=args.use_chat_template,
    )

    calibrator = None
    if args.use_calibrator:
        calibrator = CADCalibrator(
            adapter.model,
            adapter.tokenizer,
            device=adapter.device,
            use_chat_template=args.use_chat_template,
        )

    agent = TradingAgent(
        decoder,
        calibrator=calibrator,
        cad_prior_mode=args.cad_prior_mode,
        cad_alpha=args.cad_alpha,
        cad_top_p=args.cad_top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        use_calibrator=args.use_calibrator,
        calibrator_alpha_min=args.calibrator_alpha_min,
        calibrator_alpha_max=args.calibrator_alpha_max,
    )

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
            "strategy": result["strategy_metrics"],
            "buy_and_hold": result["benchmark_metrics"],
            "n_decisions": len(result["decisions"]),
            "n_buys": sum(1 for d in result["decisions"] if d["signal"] == "buy"),
            "n_sells": sum(1 for d in result["decisions"] if d["signal"] == "sell"),
            "n_holds": sum(1 for d in result["decisions"] if d["signal"] == "hold"),
        }
        with open(out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Summary written to %s", out)


if __name__ == "__main__":
    main()
