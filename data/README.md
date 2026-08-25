# Data

Data files are intentionally not committed to this repository. The paper's
daily market data is available from the
[FINSABER-V2 dataset](https://huggingface.co/datasets/finsaber-team/FINSABER-V2-Data).

FinCAD only needs the `price_daily` partition for adversarial discovery and
the single-stock backtests. The public dataset provides yearly Parquet files
with the required columns:

```text
date, symbol, cik, open, high, low, close, adjusted_close, volume
```

From the repository root, install the reproduction dependencies and create
the CSV expected by the released commands:

```bash
pip install -e ".[reproduction]"
python scripts/reproduce/prepare_finsaber.py
```

This downloads only `price_daily/**` and writes:

```text
dataset/backtest-data/price/price_data.csv
```

The directory remains ignored by Git. The backtest derives split-adjusted
open prices as `open * adjusted_close / close` when `adjusted_open` is not
already present.

Users are responsible for following the dataset card, model licences, and
the terms of the original data providers.
