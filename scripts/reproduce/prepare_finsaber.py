#!/usr/bin/env python3
"""Download the FINSABER-V2 price partition and export FinCAD's input CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

DATASET_ID = "finsaber-team/FINSABER-V2-Data"
REQUIRED_COLUMNS = {
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", default="dataset/FINSABER-V2-Data")
    parser.add_argument(
        "--output",
        default="dataset/backtest-data/price/price_data.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            local_dir=args.download_dir,
            allow_patterns=["price_daily/**"],
        )
    )
    parquet_files = sorted((root / "price_daily").glob("year=*/part-*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No price_daily Parquet files found under {root}")

    frames = [pd.read_parquet(path) for path in parquet_files]
    prices = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_COLUMNS.difference(prices.columns)
    if missing:
        raise SystemExit(f"FINSABER price data is missing columns: {sorted(missing)}")

    prices["date"] = pd.to_datetime(prices["date"], utc=True).dt.tz_localize(None)
    prices["symbol"] = prices["symbol"].astype(str).str.upper()
    prices = prices.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output, index=False)
    print(f"Wrote {len(prices):,} rows to {output}")


if __name__ == "__main__":
    main()
