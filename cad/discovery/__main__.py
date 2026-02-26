"""CLI entry point: ``python -m cad.discovery``."""
from __future__ import annotations

import argparse
import logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize CAD negative-prompt instruction via DSPy.",
    )
    parser.add_argument(
        "--model-name", type=str, required=True,
        help="HuggingFace model id or local path",
    )
    parser.add_argument(
        "--price-csv", type=str, required=True,
        help="Path to price_data.csv with columns: date, symbol, adjusted_close",
    )
    parser.add_argument(
        "--optimizer", type=str, default="MIPROv2",
        choices=["MIPROv2", "COPRO"],
    )
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--num-trials", type=int, default=30)
    parser.add_argument("--forward-days", type=int, default=63,
                        help="Forward return horizon in trading days (default: 63)")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--min-abs-return", type=float, default=0.05)
    parser.add_argument("--sample-freq", type=str, default="QE")
    parser.add_argument("--date-range-start", type=str, default="2005-01-01")
    parser.add_argument("--date-range-end", type=str, default="2015-01-01")
    parser.add_argument("--output-dir", type=str, default="results/discovery")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from . import CalibrationDatasetConfig, DiscoveryConfig, run_optimization

    result = run_optimization(
        DiscoveryConfig(
            model_name=args.model_name,
            optimizer=args.optimizer,
            num_candidates=args.num_candidates,
            num_trials=args.num_trials,
            calibration=CalibrationDatasetConfig(
                price_csv=args.price_csv,
                date_range=(args.date_range_start, args.date_range_end),
                forward_days=args.forward_days,
                max_examples=args.max_examples,
                sample_freq=args.sample_freq,
                min_abs_return=args.min_abs_return,
            ),
            output_dir=args.output_dir,
        )
    )

    print(f"\nOptimized instruction (val score = {result.score:.2%}):")
    print("-" * 60)
    print(result.instruction)
    print("-" * 60)


if __name__ == "__main__":
    main()
