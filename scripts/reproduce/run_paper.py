#!/usr/bin/env python3
"""Run or print the canonical commands used for the FinCAD paper."""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = REPO_ROOT / "configs" / "paper" / "models.json"
EXPERIMENT_CONFIG = REPO_ROOT / "configs" / "paper" / "experiments.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    models = load_json(MODEL_CONFIG)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["discovery", "profile", "benchmarks", "backtest-is", "backtest-oos", "ranking"],
    )
    parser.add_argument("--model", required=True, choices=sorted(models))
    parser.add_argument(
        "--price-csv",
        default="dataset/backtest-data/price/price_data.csv",
    )
    parser.add_argument("--ticker", action="append", help="Override the default ticker set; repeat as needed")
    parser.add_argument("--output-root", default="results/reproduction")
    parser.add_argument("--discovery-file", default=None, help="Override the released discovery JSON")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def common_model_args(model: dict, args: argparse.Namespace) -> list[str]:
    values = ["--model-name", model["model_id"], "--use-chat-template"]
    if args.cache_dir:
        values += ["--model-cache-dir", args.cache_dir]
    return values


def backtest_command(
    *,
    model_key: str,
    model: dict,
    args: argparse.Namespace,
    experiment: dict,
    ticker: str,
    period_name: str,
    condition: str,
) -> list[str]:
    start_date, end_date = experiment["backtest"][period_name]
    stem = f"{condition}_{ticker}_{model_key}_{period_name}"
    out = Path(args.output_root) / "backtests"
    command = [
        sys.executable,
        "-m",
        "benchmark.backtest.ai_hedge_fund.eval",
        *common_model_args(model, args),
        "--ticker",
        ticker,
        "--price-csv",
        args.price_csv,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--rebalance-freq",
        experiment["backtest"]["rebalance_frequency"],
        "--initial-capital",
        str(experiment["backtest"]["initial_capital"]),
        "--liquidity-pct",
        str(experiment["backtest"]["liquidity_pct"]),
        "--temperature",
        str(experiment["backtest_temperature"]),
        "--seed",
        str(experiment["seed"]),
        "--max-new-tokens",
        str(experiment["backtest_max_new_tokens"]),
        "--results-file",
        str(out / f"{stem}.jsonl"),
        "--summary-file",
        str(out / f"{stem}_summary.json"),
        "--values-csv",
        str(out / f"{stem}_values.csv"),
    ]

    if condition == "cad":
        command += [
            "--decoding-mode",
            "cad",
            "--cad-prior-mode",
            "optimized",
            "--optimized-instruction",
            args.discovery_file or model["discovery_file"],
            "--use-calibrator",
        ]
    else:
        command += ["--decoding-mode", "baseline"]
        if condition == "anonymized":
            command += ["--anonymize", "--entity-file", "utils/entity.json"]
        elif condition == "prompt_injection":
            command += ["--prompt-no-future"]
    return command


def main() -> None:
    args = parse_args()
    models = load_json(MODEL_CONFIG)
    experiment = load_json(EXPERIMENT_CONFIG)
    model = models[args.model]
    output_root = Path(args.output_root)
    discovery_file = args.discovery_file or model["discovery_file"]
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    if args.stage == "discovery":
        cfg = experiment["discovery"]
        command = [
            sys.executable,
            "-m",
            "cad.discovery",
            "--model-name",
            model["model_id"],
            "--price-csv",
            args.price_csv,
            "--output-dir",
            str(output_root / "discovery"),
            "--optimizer",
            cfg["optimizer"],
            "--num-candidates",
            str(cfg["num_candidates"]),
            "--num-trials",
            str(cfg["num_trials"]),
            "--forward-days",
            str(cfg["forward_days"]),
            "--max-examples",
            str(cfg["max_examples"]),
            "--min-abs-return",
            str(cfg["min_abs_return"]),
            "--date-range-start",
            cfg["date_range_start"],
            "--date-range-end",
            cfg["date_range_end"],
        ]
        run(command, dry_run=args.dry_run)
        return

    if args.stage == "profile":
        profile_dir = output_root / "profiles"
        profile_copy = profile_dir / Path(discovery_file).name
        if args.dry_run:
            print(f"copy {discovery_file} -> {profile_copy}")
        else:
            profile_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / discovery_file, REPO_ROOT / profile_copy)
        command = [
            sys.executable,
            "-m",
            "cad.discovery.profiler",
            "--model-name",
            model["model_id"],
            "--use-chat-template",
            "--price-csv",
            args.price_csv,
            "--optimized-instruction",
            str(profile_copy),
            "--alpha-target",
            "3.0",
        ]
        if args.cache_dir:
            command += ["--cache-dir", args.cache_dir]
        run(command, dry_run=args.dry_run)
        return

    if args.stage == "benchmarks":
        if "benchmark_alpha" not in model:
            raise SystemExit(f"{args.model} is not one of the five Experiment 1 models")
        out = output_root / "benchmarks"
        if not args.dry_run:
            out.mkdir(parents=True, exist_ok=True)
        for benchmark, settings in experiment["general_benchmarks"].items():
            for mode in ("baseline", "cad"):
                command = [
                    sys.executable,
                    "-m",
                    f"benchmark.{benchmark}.eval",
                    *common_model_args(model, args),
                    "--temperature",
                    str(settings["temperature"]),
                    "--seed",
                    str(experiment["seed"]),
                    "--max-new-tokens",
                    str(settings["max_new_tokens"]),
                    "--batch-size",
                    str(settings["batch_size"]),
                    "--decoding-mode",
                    mode,
                    "--results-file",
                    str(out / f"{benchmark}_{args.model}_{mode}.jsonl"),
                ]
                if mode == "cad":
                    command += [
                        "--cad-alpha",
                        str(model["benchmark_alpha"]),
                        "--cad-prior-mode",
                        "optimized",
                        "--optimized-instruction",
                        discovery_file,
                    ]
                run(command, dry_run=args.dry_run)
        return

    if args.stage in {"backtest-is", "backtest-oos"}:
        period = "in_sample" if args.stage.endswith("is") else "out_of_sample"
        tickers = args.ticker or (
            experiment["backtest"]["mega_cap_tickers"]
            + experiment["backtest"]["robustness_tickers"]
        )
        for ticker in tickers:
            for condition in ("baseline", "cad"):
                run(
                    backtest_command(
                        model_key=args.model,
                        model=model,
                        args=args,
                        experiment=experiment,
                        ticker=ticker,
                        period_name=period,
                        condition=condition,
                    ),
                    dry_run=args.dry_run,
                )
        return

    # Experiment 3: four in-sample conditions plus strict OOS baseline on SPY.
    ticker = args.ticker[0] if args.ticker else experiment["backtest"]["ranking_ticker"]
    for condition in ("baseline", "anonymized", "prompt_injection", "cad"):
        run(
            backtest_command(
                model_key=args.model,
                model=model,
                args=args,
                experiment=experiment,
                ticker=ticker,
                period_name="in_sample",
                condition=condition,
            ),
            dry_run=args.dry_run,
        )
    run(
        backtest_command(
            model_key=args.model,
            model=model,
            args=args,
            experiment=experiment,
            ticker=ticker,
            period_name="out_of_sample",
            condition="baseline",
        ),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
