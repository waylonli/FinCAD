"""
Financial backtest benchmark with CAD integration (Step 2 evaluation).

Runs a quality-factor backtest on filings scored by a local HF model with
optional context-aware decoding.  Compares baseline (alpha=0) vs CAD to
measure the honesty drop in returns.

Usage examples
--------------
Precomputed scores (fast iteration):
    python -m benchmark.backtest.q_scores_eval.eval \\
        --score-mode precomputed \\
        --symbols mag7 --start-year 2014 --end-year 2024

On-demand baseline:
    python -m benchmark.backtest.q_scores_eval.eval \\
        --model-name Qwen/Qwen2.5-7B-Instruct --use-chat-template \\
        --score-mode on_demand --decoding-mode baseline \\
        --symbols mag7 --start-year 2014 --end-year 2024

On-demand CAD (bias-amplified prior):
    python -m benchmark.backtest.q_scores_eval.eval \\
        --model-name Qwen/Qwen2.5-7B-Instruct --use-chat-template \\
        --score-mode on_demand --decoding-mode cad \\
        --cad-alpha 1.5 --cad-prior-mode bias_amplified \\
        --symbols mag7 --start-year 2014 --end-year 2024
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .backtest import (
    BacktestConfig,
    BacktestResult,
    FactorRegressionInput,
    run_backtest,
)
from .data import (
    DEFAULT_DATA_ROOT,
    MAG7_TICKERS,
    collect_filing_paths,
    compute_returns,
    load_cached_scores,
    load_fama_french_factors,
    load_index_components,
    load_prices,
    read_filing,
)
from .scorer import HFFilingScorer, ScoringConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]

    p = argparse.ArgumentParser(
        description="Financial backtest benchmark with CAD integration.",
    )

    # Model
    g = p.add_argument_group("Model")
    g.add_argument("--model-name", type=str, default=None, help="HuggingFace model id (required for on_demand)")
    g.add_argument("--model-cache-dir", type=str, default="../pretrained_models")
    g.add_argument("--use-chat-template", action="store_true")
    g.add_argument("--attn-implementation", type=str, default=None,
                   help="Attention implementation, e.g. 'flash_attention_2'.")

    # Decoding
    g = p.add_argument_group("Decoding")
    g.add_argument("--decoding-mode", type=str, default="baseline", choices=["baseline", "cad"])
    g.add_argument("--cad-alpha", type=float, default=1.0)
    g.add_argument("--cad-top-p", type=float, default=1.0)
    g.add_argument("--cad-prior-mode", type=str, default="no_context", choices=["no_context", "bias_amplified", "optimized"])
    g.add_argument("--optimized-instruction", type=str, default=None,
                   help="Path to optimized instruction JSON (required when --cad-prior-mode=optimized)")

    # Generation
    g = p.add_argument_group("Generation")
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--max-new-tokens", type=int, default=512)

    # Chunking
    g = p.add_argument_group("Chunking")
    g.add_argument("--chunk-size", type=int, default=8192,
                   help="Max filing text tokens per chunk (default: 8192)")
    g.add_argument("--chunk-overlap", type=int, default=256,
                   help="Overlap tokens between consecutive chunks (default: 256)")

    # Calibrator
    g = p.add_argument_group("Calibrator")
    g.add_argument("--use-calibrator", action="store_true")
    g.add_argument("--calibrator-alpha-min", type=float, default=0.0)
    g.add_argument("--calibrator-alpha-max", type=float, default=5.0)

    # Scoring
    g = p.add_argument_group("Scoring")
    g.add_argument("--score-mode", type=str, default="precomputed", choices=["precomputed", "on_demand"])
    g.add_argument("--cached-scores-path", type=str, default=None, help="CSV with precomputed scores")
    g.add_argument("--score-cache-path", type=str, default=None, help="Path to cache on-demand scores")

    # Backtest
    g = p.add_argument_group("Backtest")
    g.add_argument("--symbols", type=str, default="all", help="Comma-separated tickers, 'mag7', or 'all' (scan reports dir)")
    g.add_argument("--start-year", type=int, default=2014)
    g.add_argument("--end-year", type=int, default=2024)
    g.add_argument("--top-quantile", type=float, default=0.05)
    g.add_argument("--max-filing-chars", type=int, default=60000)

    # Data
    g = p.add_argument_group("Data paths")
    g.add_argument("--data-root", type=str, default=str(repo_root / DEFAULT_DATA_ROOT))
    g.add_argument("--price-file", type=str, default="price/price_data.csv")
    g.add_argument("--ff-factors-file", type=str, default="fama_french_daily_2014_2024.csv")
    g.add_argument("--components-file", type=str, default="sp500_components.csv")
    g.add_argument("--benchmark-price-file", type=str, default="sp500_historical_2000_2024.csv")
    g.add_argument("--reports-dir", type=str, default="reports")

    # Output
    g = p.add_argument_group("Output")
    g.add_argument("--results-file", type=str, default=None, help="Per-filing JSONL output")
    g.add_argument("--summary-file", type=str, default=None, help="Backtest summary JSON")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger.info("Config: %s", args)

    data_root = Path(args.data_root)

    # ---- Resolve symbols ----
    if args.symbols.lower() == "all":
        reports_root = data_root / args.reports_dir
        symbols = sorted([
            d.name.upper() for d in reports_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
    elif args.symbols.lower() == "mag7":
        symbols = MAG7_TICKERS
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    logger.info("Symbols (%d): %s", len(symbols), symbols)

    # ---- Load price data ----
    price_path = data_root / args.price_file
    logger.info("Loading prices from %s", price_path)
    price_df = load_prices(
        price_path, symbols=symbols,
        start_year=args.start_year, end_year=args.end_year,
    )
    logger.info("Loaded %d price rows for %d symbols", len(price_df), price_df["symbol"].nunique())

    # ---- Load FF factors (optional) ----
    factors = None
    ff_path = data_root / args.ff_factors_file
    if ff_path.exists():
        logger.info("Loading Fama-French factors from %s", ff_path)
        ff_df = load_fama_french_factors(ff_path)
        factors = FactorRegressionInput(data=ff_df)

    # ---- Load index components (optional) ----
    components = None
    comp_path = data_root / args.components_file
    if comp_path.exists():
        logger.info("Loading index components from %s", comp_path)
        components = load_index_components(comp_path)

    # ---- Resolve benchmark ----
    bench_path = data_root / args.benchmark_price_file
    benchmark_price_path = str(bench_path) if bench_path.exists() else None

    # ---- Score data ----
    if args.score_mode == "precomputed":
        cached_path = args.cached_scores_path
        if cached_path is None:
            # Try default location
            cached_path = str(data_root / "checkpoints" / ".cached_q_scores.csv")
        logger.info("Loading precomputed scores from %s", cached_path)
        score_df = load_cached_scores(cached_path, symbols=symbols)
        logger.info("Loaded %d score rows", len(score_df))

    elif args.score_mode == "on_demand":
        if args.model_name is None:
            print("ERROR: --model-name is required for on_demand scoring.", file=sys.stderr)
            sys.exit(1)

        score_df = _run_on_demand_scoring(args, data_root, symbols)
    else:
        raise ValueError(f"Unknown score-mode: {args.score_mode}")

    # ---- Run backtest ----
    bt_config = BacktestConfig(
        start_year=args.start_year,
        end_year=args.end_year,
        top_quantile=args.top_quantile,
        name=f"quality_{args.decoding_mode}",
        benchmark_price_path=benchmark_price_path,
        components_path=str(comp_path) if comp_path.exists() else None,
    )

    logger.info("Running backtest (%s) ...", args.decoding_mode)
    result = run_backtest(
        price_df,
        score_df,
        bt_config,
        factors=factors,
        components=components,
    )

    # ---- Print summary ----
    _print_summary(result, args)

    # ---- Write outputs ----
    if args.results_file:
        _write_results_jsonl(result, score_df, args)
    if args.summary_file:
        _write_summary_json(result, args)


# ---------------------------------------------------------------------------
# On-demand scoring
# ---------------------------------------------------------------------------


def _auto_score_cache_path(args: argparse.Namespace) -> Path:
    """Build a cache filename that distinguishes model, decoding mode, and CAD settings.

    Examples:
        results/backtest/scores_qwen2.5-7b-instruct_baseline.csv
        results/backtest/scores_qwen2.5-7b-instruct_cad_a1.5_bias_amplified.csv
        results/backtest/scores_phi-3-mini_cad_a2.0_no_context_calibrated.csv
    """
    model_short = args.model_name.split("/")[-1].lower() if args.model_name else "unknown"
    parts = [model_short, args.decoding_mode]
    if args.decoding_mode == "cad":
        parts.append(f"a{args.cad_alpha}")
        parts.append(args.cad_prior_mode)
        if args.use_calibrator:
            parts.append("calibrated")
    name = "scores_" + "_".join(parts) + ".csv"
    return Path("results/backtest") / name


def _run_on_demand_scoring(
    args: argparse.Namespace,
    data_root: Path,
    symbols: list[str],
) -> pd.DataFrame:
    """Load a HF model, score filings, and return a score DataFrame."""
    from adapters import AdapterInitConfig, TransformersAdapter
    from cad import CADConfig, ContextAwareDecoder
    from cad.calibrator import CADCalibrator

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

    neg_prompt_builder = None
    if args.cad_prior_mode == "optimized" or args.use_calibrator:
        from cad.discovery import NegativePromptBuilder
        if args.optimized_instruction is None:
            print("ERROR: --optimized-instruction is required when --cad-prior-mode=optimized or --use-calibrator.", file=sys.stderr)
            sys.exit(1)
        neg_prompt_builder = NegativePromptBuilder.from_file(args.optimized_instruction)

    calibrator = None
    if args.use_calibrator:
        calibrator = CADCalibrator(
            adapter.model,
            adapter.tokenizer,
            neg_prompt_builder=neg_prompt_builder,
            device=adapter.device,
            use_chat_template=args.use_chat_template,
        )

    scoring_config = ScoringConfig(
        decoding_mode=args.decoding_mode,
        cad_alpha=args.cad_alpha,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        cad_top_p=args.cad_top_p,
        cad_prior_mode=args.cad_prior_mode,
        use_calibrator=args.use_calibrator,
        calibrator_alpha_min=args.calibrator_alpha_min,
        calibrator_alpha_max=args.calibrator_alpha_max,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    scorer = HFFilingScorer(decoder, scoring_config, calibrator=calibrator,
                            neg_prompt_builder=neg_prompt_builder)

    # Collect filings
    reports_root = data_root / args.reports_dir
    logger.info("Collecting filings from %s", reports_root)
    filings_map = collect_filing_paths(reports_root, symbols)

    from tqdm import tqdm

    all_paths = [
        (symbol, fpath, ts)
        for symbol, paths in filings_map.items()
        for fpath, ts in paths
    ]
    filings: list[tuple[str, str, pd.Timestamp]] = []
    skipped = 0
    pbar = tqdm(all_paths, desc="Reading filings", unit="file")
    for symbol, fpath, ts in pbar:
        pbar.set_description(f"Reading {symbol} {fpath.stem}")
        text = read_filing(fpath, max_chars=args.max_filing_chars)
        if not text.strip():
            logger.warning("Empty/unreadable filing %s – skipping", fpath)
            skipped += 1
            continue
        filings.append((text, symbol, ts))
    if skipped:
        logger.warning("Skipped %d unreadable filings out of %d total", skipped, len(all_paths))

    logger.info("Scoring %d filings ...", len(filings))

    cache_path = Path(args.score_cache_path) if args.score_cache_path else _auto_score_cache_path(args)
    logger.info("Score cache: %s", cache_path)
    score_df = scorer.score_filings_to_dataframe(filings, cache_path=cache_path)
    logger.info("Scored %d filings", len(score_df))
    return score_df


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_summary(result: BacktestResult, args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS  [{args.decoding_mode.upper()}]")
    print("=" * 60)

    def _fmt(metrics: dict, label: str) -> None:
        print(f"\n  {label}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                if "return" in k or "cagr" in k or "drawdown" in k:
                    print(f"    {k:24s}  {v:+.4%}")
                else:
                    print(f"    {k:24s}  {v:,.4f}")
            else:
                print(f"    {k:24s}  {v}")

    _fmt(result.performance, f"Strategy ({result.config.name})")

    if result.equal_weight_performance:
        _fmt(result.equal_weight_performance, "Equal-Weight Benchmark")
    if result.benchmark_performance:
        _fmt(result.benchmark_performance, f"Market Benchmark ({result.config.benchmark_symbol})")
    if result.excess_performance_vs_benchmark:
        _fmt(result.excess_performance_vs_benchmark, "Excess vs Market")

    if result.factor_regression:
        fr = result.factor_regression
        print(f"\n  Factor Regression (FF5):")
        print(f"    Alpha:         {fr.alpha:+.6f}  (t={fr.alpha_t:.2f}, p={fr.alpha_p:.4f})")
        print(f"    R-squared:     {fr.r_squared:.4f}")
        print(f"    Observations:  {fr.observations}")

    print(f"\n  Commission paid:  strategy=${result.portfolio_commission:,.2f}  eq_wt=${result.equal_weight_commission:,.2f}")
    print("=" * 60 + "\n")


def _write_results_jsonl(
    result: BacktestResult,
    score_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    out_path = Path(args.results_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for _, row in score_df.iterrows():
            record = row.to_dict()
            # Convert timestamps to strings for JSON
            for k, v in record.items():
                if isinstance(v, pd.Timestamp):
                    record[k] = str(v)
            record["decoding_mode"] = args.decoding_mode
            record["cad_alpha"] = args.cad_alpha
            record["cad_prior_mode"] = args.cad_prior_mode
            f.write(json.dumps(record, default=str) + "\n")
    logger.info("Wrote per-filing results to %s", out_path)


def _write_summary_json(
    result: BacktestResult,
    args: argparse.Namespace,
) -> None:
    out_path = Path(args.summary_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "decoding_mode": args.decoding_mode,
        "cad_alpha": args.cad_alpha,
        "cad_prior_mode": args.cad_prior_mode,
        "score_mode": args.score_mode,
        "symbols": args.symbols,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "top_quantile": args.top_quantile,
        "strategy_performance": result.performance,
        "equal_weight_performance": result.equal_weight_performance,
        "benchmark_performance": result.benchmark_performance,
        "excess_vs_benchmark": result.excess_performance_vs_benchmark,
        "portfolio_commission": result.portfolio_commission,
        "equal_weight_commission": result.equal_weight_commission,
    }

    if result.factor_regression:
        fr = result.factor_regression
        summary["factor_regression"] = {
            "alpha": fr.alpha,
            "alpha_t": fr.alpha_t,
            "alpha_p": fr.alpha_p,
            "r_squared": fr.r_squared,
            "adj_r_squared": fr.adj_r_squared,
            "observations": fr.observations,
        }

    if args.model_name:
        summary["model_name"] = args.model_name

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote backtest summary to %s", out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
