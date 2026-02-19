"""Data loading utilities for the financial backtest benchmark.

Functions are adapted from ``abrdn-risk-factor-eval/src/data/loaders.py``
and ``abrdn-risk-factor-eval/src/analysis/prices.py`` to be self-contained
(no LangChain/OpenAI imports).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAG7_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

DEFAULT_DATA_ROOT = Path("dataset/backtest-data/")

MAX_ABS_DAILY_RETURN = 0.5

# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------


def load_prices(
    path: PathLike,
    *,
    symbols: Optional[Sequence[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    """Load price history CSV with columns ``date``, ``symbol``, ``adjusted_close``.

    Adapted from ``abrdn-risk-factor-eval/src/data/loaders.py::load_price_data``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Price data file not found: {path}")

    df = pd.read_csv(path)
    if "date" not in df.columns or "symbol" not in df.columns:
        raise ValueError("Expected columns 'date' and 'symbol' in price data.")

    df["date"] = pd.to_datetime(df["date"], utc=False)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df.sort_values(["symbol", "date"], inplace=True)

    if symbols is not None:
        symbols_set = {s.upper() for s in symbols}
        df = df[df["symbol"].isin(symbols_set)]

    if start_year is not None:
        df = df[df["date"].dt.year >= start_year]
    if end_year is not None:
        df = df[df["date"].dt.year <= end_year]

    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def compute_returns(
    price_df: pd.DataFrame,
    *,
    price_column: str = "adjusted_close",
) -> pd.DataFrame:
    """Compute simple daily returns with basic outlier handling.

    Adapted from ``abrdn-risk-factor-eval/src/analysis/prices.py::compute_price_returns``.
    """
    if "date" not in price_df.columns or "symbol" not in price_df.columns:
        raise ValueError("price_df must contain 'date' and 'symbol' columns")
    if price_column not in price_df.columns:
        raise ValueError(f"price_df must contain '{price_column}' column")

    data = price_df[["date", "symbol", price_column]].copy()
    data["date"] = pd.to_datetime(data["date"], utc=False)
    data = data[data[price_column] > 0]
    data.sort_values(["symbol", "date"], inplace=True)
    data["return"] = data.groupby("symbol")[price_column].pct_change()
    data.dropna(subset=["return"], inplace=True)

    outlier_mask = data["return"].abs() > MAX_ABS_DAILY_RETURN
    if outlier_mask.any():
        dropped = int(outlier_mask.sum())
        logger.warning(
            "Dropping %d price-return rows with |return| > %.2f",
            dropped,
            MAX_ABS_DAILY_RETURN,
        )
        data = data.loc[~outlier_mask]

    return data


# ---------------------------------------------------------------------------
# Cached scores
# ---------------------------------------------------------------------------


def load_cached_scores(
    path: PathLike,
    *,
    symbols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Read a ``.cached_q_scores.csv`` with quality_score columns.

    Expected columns: ``symbol``, ``report_date``, ``quality_score``
    (plus optional per-category scores).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cached score file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalise symbol column name
    if "symbol" not in df.columns:
        for alias in ("ticker", "company", "company_name"):
            if alias in df.columns:
                df.rename(columns={alias: "symbol"}, inplace=True)
                break
        else:
            raise ValueError("Could not find symbol column in cached scores.")

    df["symbol"] = df["symbol"].astype(str).str.upper()

    if "report_date" not in df.columns:
        for alias in ("as_of", "date", "reporting_date"):
            if alias in df.columns:
                df.rename(columns={alias: "report_date"}, inplace=True)
                break
        else:
            raise ValueError("Could not find report_date column in cached scores.")

    df["report_date"] = pd.to_datetime(df["report_date"], utc=False)

    if symbols is not None:
        symbols_set = {s.upper() for s in symbols}
        df = df[df["symbol"].isin(symbols_set)]

    df.sort_values(["symbol", "report_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Index components
# ---------------------------------------------------------------------------


def load_index_components(path: PathLike) -> pd.DataFrame:
    """Load historical index membership (date, tickers CSV).

    Adapted from ``abrdn-risk-factor-eval/src/data/loaders.py::load_index_components``.
    Returns a DataFrame with columns ``date`` and ``symbol``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Index components file not found: {path}")

    frame = pd.read_csv(path)
    if "date" not in frame.columns or "tickers" not in frame.columns:
        raise ValueError("Expected columns 'date' and 'tickers' in index components file.")

    frame["date"] = pd.to_datetime(frame["date"], utc=False, errors="coerce")
    frame.dropna(subset=["date"], inplace=True)
    frame["tickers"] = frame["tickers"].fillna("")
    exploded = frame.assign(tickers=frame["tickers"].str.split(",")).explode("tickers")
    exploded["symbol"] = exploded["tickers"].astype(str).str.strip().str.upper()
    exploded = exploded[exploded["symbol"] != ""]
    exploded = exploded[["date", "symbol"]].drop_duplicates()
    exploded.sort_values(["date", "symbol"], inplace=True)
    exploded.reset_index(drop=True, inplace=True)
    return exploded


# ---------------------------------------------------------------------------
# Fama-French factors
# ---------------------------------------------------------------------------


def load_fama_french_factors(
    path: PathLike,
    *,
    frequency: str = "daily",
) -> pd.DataFrame:
    """Load a local Fama-French five-factor CSV.

    Adapted from ``abrdn-risk-factor-eval/src/data/loaders.py::load_fama_french_factors``.
    Expects columns: ``date``, ``mkt_rf``, ``smb``, ``hml``, ``rmw``, ``cma``, ``rf``.
    Values are converted from percent to decimal form.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fama-French factor file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    if "date" not in df.columns:
        raise ValueError("Fama-French factor data must include a 'date' column.")

    df["date"] = pd.to_datetime(df["date"], utc=False)

    freq = frequency.lower()
    if freq in {"monthly", "m"}:
        df.set_index("date", inplace=True)
        df = df.resample("D").ffill().reset_index()
    elif freq not in {"daily", "d"}:
        raise ValueError("frequency must be 'daily' or 'monthly'")

    factor_columns = [c for c in df.columns if c != "date"]
    for col in factor_columns:
        df[col] = df[col].astype(float) / 100.0

    return df


# ---------------------------------------------------------------------------
# Filing collection
# ---------------------------------------------------------------------------


def collect_filing_paths(
    reports_root: PathLike,
    symbols: Sequence[str],
) -> Dict[str, List[Tuple[Path, pd.Timestamp]]]:
    """Scan ``reports_root/{SYMBOL}/`` for filings.

    Expects files named ``{DATE}.pdf`` or ``{DATE}.txt`` inside each symbol
    directory, where ``DATE`` is parseable by ``pd.to_datetime``.

    Returns a mapping from symbol to a sorted list of ``(path, timestamp)``
    tuples.
    """
    reports_root = Path(reports_root)
    result: Dict[str, List[Tuple[Path, pd.Timestamp]]] = {}

    for symbol in symbols:
        symbol = symbol.upper()
        sym_dir = reports_root / symbol
        if not sym_dir.is_dir():
            logger.warning("No filing directory found for %s at %s", symbol, sym_dir)
            continue
        filings: List[Tuple[Path, pd.Timestamp]] = []
        for fpath in sorted(sym_dir.iterdir()):
            if not fpath.is_file():
                continue
            stem = fpath.stem
            try:
                ts = pd.to_datetime(stem)
            except (ValueError, TypeError):
                continue
            filings.append((fpath, ts))
        filings.sort(key=lambda t: t[1])
        if filings:
            result[symbol] = filings
        else:
            logger.warning("No parseable filings found for %s in %s", symbol, sym_dir)

    return result


def read_filing(path: PathLike, max_chars: int = 60_000) -> str:
    """Read a filing from disk, truncating to *max_chars*.

    Supports plain text and PDF (via ``PyPDF2`` / ``pypdf`` if available).
    For PDFs, the extracted text is cached as a ``.txt`` file next to the
    original so that subsequent reads skip PDF parsing entirely.
    """
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return _read_pdf_cached(path, max_chars)
    return path.read_text(encoding="utf-8", errors="replace")[:max_chars]


def _read_pdf_cached(path: Path, max_chars: int) -> str:
    """Read a PDF, caching the full extracted text as a sibling ``.txt``."""
    txt_path = path.with_suffix(".txt")

    # Use cached text if it exists and is newer than the PDF
    if txt_path.exists() and txt_path.stat().st_mtime >= path.stat().st_mtime:
        return txt_path.read_text(encoding="utf-8", errors="replace")[:max_chars]

    # Parse PDF
    text = _extract_pdf_text(path)

    # Cache the full extracted text for future runs
    try:
        txt_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not cache PDF text to %s: %s", txt_path, exc)

    return text[:max_chars]


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError:
            raise ImportError(
                "pypdf or PyPDF2 is required to read PDF filings. "
                "Install one with: pip install pypdf"
            )
    import logging

    logger = logging.getLogger(__name__)
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        logger.warning("Cannot open PDF %s: %s", path, exc)
        return ""
    texts: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Skipping page %d of %s: %s", i, path.name, exc)
            continue
        texts.append(text)
    return "".join(texts)
