"""Simplified backtesting engine for the quality-factor signal.

Extracted and adapted from ``abrdn-risk-factor-eval/src/analysis/backtest.py``,
``signals.py``, ``utils.py``, and ``factors.py``.  This version is long-only,
self-contained, and does not depend on LangChain or OpenAI.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RebalanceSpec = Sequence[Tuple[int, int]]


# =========================================================================
# Configuration & result dataclasses
# =========================================================================


@dataclass
class BacktestConfig:
    """Configuration for the long-only quality factor backtest."""

    start_year: int
    end_year: Optional[int] = None
    rebalance_month_days: RebalanceSpec = ((1, 1), (4, 1), (7, 1), (10, 1))
    signal_column: str = "quality_score"
    descending: bool = True
    top_k: Optional[int] = None
    top_quantile: Optional[float] = 0.2
    price_column: str = "adjusted_close"
    benchmark_symbol: Optional[str] = "SPY"
    long_gross_exposure: float = 1.0
    min_assets: int = 5
    risk_free_rate: float = 0.0
    periods_per_year: int = 252
    name: str = "quality_factor"
    initial_capital: float = 1_000_000.0
    commission_bps: float = 10.0  # bps of trade notional (Frazzini et al. 2018)
    max_gross_exposure: float = 1.0
    benchmark_price_path: Optional[str] = None
    benchmark_price_column: str = "close"
    components_path: Optional[str] = None
    include_equal_weight: bool = True
    include_benchmark: bool = True

    def __post_init__(self) -> None:
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if self.top_quantile is not None and not (0 < self.top_quantile <= 1):
            raise ValueError("top_quantile must lie in (0, 1]")


@dataclass
class BacktestResult:
    config: BacktestConfig
    portfolio_returns: pd.Series
    portfolio_value: pd.Series
    performance: Dict[str, float]
    weights: pd.DataFrame
    signals_panel: pd.DataFrame
    equal_weight_returns: Optional[pd.Series] = None
    equal_weight_value: Optional[pd.Series] = None
    equal_weight_performance: Optional[Dict[str, float]] = None
    benchmark_returns: Optional[pd.Series] = None
    benchmark_cumulative: Optional[pd.Series] = None
    benchmark_performance: Optional[Dict[str, float]] = None
    excess_returns_vs_benchmark: Optional[pd.Series] = None
    excess_performance_vs_benchmark: Optional[Dict[str, float]] = None
    factor_regression: Optional["FactorRegressionResult"] = None
    score_data: Optional[pd.DataFrame] = None
    portfolio_commission: float = 0.0
    equal_weight_commission: float = 0.0


# =========================================================================
# Factor regression  (extracted from abrdn factors.py)
# =========================================================================


@dataclass
class FactorRegressionInput:
    data: pd.DataFrame
    factor_columns: Sequence[str] = ("mkt_rf", "smb", "hml", "rmw", "cma")
    risk_free_column: str = "rf"


@dataclass
class FactorRegressionResult:
    alpha: float
    alpha_t: float
    alpha_p: float
    coefficients: pd.Series
    t_stats: pd.Series
    p_values: pd.Series
    r_squared: float
    adj_r_squared: float
    regression_table: pd.DataFrame
    observations: int


def run_factor_regression(
    portfolio_returns: pd.Series,
    factors: FactorRegressionInput,
    *,
    risk_free_rate: float = 0.0,
) -> FactorRegressionResult:
    """Regress strategy excess returns against the Fama-French factors."""
    try:
        import statsmodels.api as sm  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for factor regression. "
            "Install with: pip install statsmodels"
        ) from exc

    factor_df = factors.data.copy()
    factor_df.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_")
        for col in factor_df.columns
    ]
    if "date" not in factor_df.columns:
        raise ValueError("Factor data must include a 'date' column")

    factor_df["date"] = pd.to_datetime(factor_df["date"], utc=False)
    factor_df.sort_values("date", inplace=True)
    factor_df.set_index("date", inplace=True)

    joined = pd.concat(
        [portfolio_returns.rename("strategy_return"), factor_df],
        axis=1,
        join="inner",
    ).dropna()

    if joined.empty:
        raise ValueError("No overlapping observations between strategy and factor data")

    target = joined["strategy_return"].copy()
    if factors.risk_free_column in joined.columns:
        target = target - joined[factors.risk_free_column]
    else:
        target = target - risk_free_rate / 252

    X = joined[list(factors.factor_columns)].copy()
    X = sm.add_constant(X)

    results = sm.OLS(target, X).fit()

    coefficients = results.params
    t_stats = results.tvalues
    p_values = results.pvalues

    return FactorRegressionResult(
        alpha=float(coefficients.get("const", np.nan)),
        alpha_t=float(t_stats.get("const", np.nan)),
        alpha_p=float(p_values.get("const", np.nan)),
        coefficients=coefficients,
        t_stats=t_stats,
        p_values=p_values,
        r_squared=float(results.rsquared),
        adj_r_squared=float(results.rsquared_adj),
        regression_table=pd.DataFrame(
            {"coef": coefficients, "t_stat": t_stats, "p_value": p_values}
        ),
        observations=int(results.nobs),
    )


# =========================================================================
# Rebalance schedule utilities  (extracted from abrdn utils.py)
# =========================================================================


def build_rebalance_schedule(
    trading_days: Iterable[pd.Timestamp],
    *,
    start_year: int,
    end_year: Optional[int],
    rebalance_month_days: RebalanceSpec,
) -> pd.DataFrame:
    trading_array = pd.Index(sorted(pd.to_datetime(list(trading_days))))
    if trading_array.empty:
        raise ValueError("Trading calendar is empty")

    last_year = end_year or int(trading_array[-1].year)
    start_date = pd.Timestamp(year=start_year, month=1, day=1)
    end_date = pd.Timestamp(year=last_year, month=12, day=31)

    rebalance_dates: List[pd.Timestamp] = []
    for year in range(start_year, last_year + 1):
        for month, day in rebalance_month_days:
            candidate = pd.Timestamp(year=year, month=month, day=day)
            if candidate < start_date or candidate > end_date:
                continue
            next_day = _next_trading_day(candidate, trading_array, include_current=True)
            if next_day is not None:
                rebalance_dates.append(next_day)

    rebalance_dates = sorted(set(rebalance_dates))
    return pd.DataFrame({"rebalance_date": rebalance_dates})


def _next_trading_day(
    date: Optional[pd.Timestamp],
    trading_days: pd.Index,
    *,
    include_current: bool,
) -> Optional[pd.Timestamp]:
    if date is None:
        return None
    side = "left" if include_current else "right"
    idx = trading_days.searchsorted(pd.Timestamp(date), side=side)
    if idx >= len(trading_days):
        return None
    return pd.Timestamp(trading_days[idx])


# =========================================================================
# Signal alignment  (extracted from abrdn signals.py)
# =========================================================================


def prepare_signal_panel(
    score_df: pd.DataFrame,
    rebalance_schedule: pd.DataFrame,
    *,
    signal_column: str,
) -> pd.DataFrame:
    """Align company-level scores to each rebalance date via merge_asof."""
    if "symbol" not in score_df.columns:
        raise ValueError("score_df must contain a 'symbol' column")
    if "report_date" not in score_df.columns:
        raise ValueError("score_df must contain a 'report_date' column")
    if signal_column not in score_df.columns:
        raise ValueError(f"score_df must contain signal column '{signal_column}'")
    if "rebalance_date" not in rebalance_schedule.columns:
        raise ValueError("rebalance_schedule must contain 'rebalance_date'")

    score_df = score_df.copy()
    score_df["symbol"] = score_df["symbol"].astype(str).str.upper()
    symbols = score_df["symbol"].unique()

    frames: List[pd.DataFrame] = []
    for symbol in symbols:
        symbol_scores = score_df[score_df["symbol"] == symbol].sort_values("report_date")
        if symbol_scores.empty:
            continue
        merge_scores = symbol_scores[["report_date", signal_column]].drop_duplicates(
            subset=["report_date"], keep="last"
        )
        symbol_rebalance = rebalance_schedule.copy()
        symbol_rebalance["symbol"] = symbol
        merged = pd.merge_asof(
            symbol_rebalance.sort_values("rebalance_date"),
            merge_scores,
            left_on="rebalance_date",
            right_on="report_date",
            direction="backward",
        )
        frames.append(merged)

    if not frames:
        return pd.DataFrame(columns=["rebalance_date", "symbol", "report_date", signal_column])

    result = pd.concat(frames, ignore_index=True)
    result.dropna(subset=[signal_column, "report_date"], inplace=True)
    result.sort_values(["rebalance_date", signal_column], ascending=[True, False], inplace=True)
    return result


# =========================================================================
# Component universe  (extracted from abrdn backtest.py)
# =========================================================================


class _ComponentUniverse:
    """Efficient lookup of index constituents for arbitrary dates."""

    def __init__(self, component_frame: pd.DataFrame) -> None:
        if component_frame.empty:
            self._dates: np.ndarray = np.array([], dtype="datetime64[ns]")
            self._sets: List[FrozenSet[str]] = []
        else:
            prepared = (
                component_frame.sort_values(["date", "symbol"])
                .groupby("date")
                .agg({"symbol": lambda s: frozenset(str(sym).upper() for sym in s if str(sym).strip())})
            )
            self._dates = prepared.index.to_numpy(dtype="datetime64[ns]")
            self._sets = list(prepared["symbol"].values)
        self._cache: Dict[pd.Timestamp, FrozenSet[str]] = {}

    def allowed(self, date: pd.Timestamp) -> FrozenSet[str]:
        if not self._sets:
            return frozenset()
        ts = pd.Timestamp(date)
        cached = self._cache.get(ts)
        if cached is not None:
            return cached
        idx = self._dates.searchsorted(ts.to_datetime64(), side="right") - 1
        result = frozenset() if idx < 0 else self._sets[idx]
        self._cache[ts] = result
        return result


def _filter_signals_to_universe(
    signals_panel: pd.DataFrame,
    universe: _ComponentUniverse,
    *,
    available_symbols: Optional[Set[str]] = None,
) -> pd.DataFrame:
    if signals_panel.empty:
        return signals_panel

    filtered_frames: List[pd.DataFrame] = []
    for rebalance_date, group in signals_panel.groupby("rebalance_date"):
        allowed = set(universe.allowed(pd.Timestamp(rebalance_date)))
        if available_symbols is not None:
            allowed &= available_symbols
        if not allowed:
            continue
        subset = group[group["symbol"].isin(allowed)]
        if not subset.empty:
            filtered_frames.append(subset)

    if not filtered_frames:
        logger.warning("No symbols remain after applying component universe filter.")
        return signals_panel.head(0).copy()

    return pd.concat(filtered_frames, ignore_index=True)


# =========================================================================
# Core backtest
# =========================================================================


@dataclass
class _PortfolioWeights:
    rebalance_date: pd.Timestamp
    symbol: str
    weight: float
    signal_value: float
    report_date: pd.Timestamp


def run_backtest(
    price_data: pd.DataFrame,
    score_data: pd.DataFrame,
    config: BacktestConfig,
    *,
    factors: Optional[FactorRegressionInput] = None,
    components: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """Orchestrate the long-only quality factor backtest."""

    price_df = price_data.copy()
    price_df["date"] = pd.to_datetime(price_df["date"], utc=False)
    price_df["symbol"] = price_df["symbol"].astype(str).str.upper()

    if config.price_column not in price_df.columns:
        raise ValueError(f"price_data must include column '{config.price_column}'")

    available_price_symbols = {str(s).upper() for s in price_df["symbol"].unique()}

    trading_days = np.array(sorted(price_df["date"].unique()))
    rebalance_schedule = build_rebalance_schedule(
        trading_days,
        start_year=config.start_year,
        end_year=config.end_year,
        rebalance_month_days=config.rebalance_month_days,
    )
    if rebalance_schedule.empty:
        raise ValueError("No rebalance dates fall within the available trading calendar.")

    # ---- Component universe ----
    component_universe: Optional[_ComponentUniverse] = None
    if components is not None and not components.empty:
        components = components[components["symbol"].isin(available_price_symbols)]
        if not components.empty:
            component_universe = _ComponentUniverse(components)

    # ---- Score alignment ----
    score_df = score_data.copy()
    score_df["symbol"] = score_df["symbol"].astype(str).str.upper()
    score_df["report_date"] = pd.to_datetime(score_df["report_date"], utc=False)

    if config.signal_column not in score_df.columns:
        raise ValueError(f"score_data must include signal column '{config.signal_column}'")

    signals_panel = prepare_signal_panel(
        score_df,
        rebalance_schedule,
        signal_column=config.signal_column,
    )
    if component_universe is not None:
        signals_panel = _filter_signals_to_universe(
            signals_panel,
            component_universe,
            available_symbols=available_price_symbols,
        )
    if signals_panel.empty:
        raise ValueError(
            "Signal panel is empty; ensure score coverage overlaps with rebalance dates."
        )

    # ---- Build weights ----
    weights = _build_weight_schedule(signals_panel, config)
    if weights.empty:
        raise ValueError("No portfolio weights generated; check configuration.")

    # ---- Price matrix ----
    price_pivot = (
        price_df.pivot(index="date", columns="symbol", values=config.price_column)
        .sort_index()
        .ffill()
        .bfill()
    )

    from .data import compute_returns

    returns = compute_returns(price_df, price_column=config.price_column)

    # ---- Strategy simulation ----
    strat_returns, strat_values, strat_commission = _apply_weight_schedule(
        weights, returns, trading_days, price_pivot, config, series_name=config.name,
    )
    if strat_returns.empty:
        raise ValueError("Portfolio returns are empty; check price data span.")

    performance = _compute_performance_metrics(strat_returns, config)

    # ---- Equal-weight benchmark ----
    eqw_returns: Optional[pd.Series] = None
    eqw_cum: Optional[pd.Series] = None
    eqw_perf: Optional[Dict[str, float]] = None
    eqw_commission = 0.0
    if config.include_equal_weight:
        eq_ret, eq_val, eq_comm = _run_equal_weight_benchmark(
            signals_panel,
            returns,
            trading_days,
            price_pivot,
            config,
            component_universe=component_universe,
            available_symbols=available_price_symbols,
        )
        if not eq_ret.empty:
            eqw_returns = eq_ret
            eqw_cum = eq_val
            eqw_perf = _compute_performance_metrics(eq_ret, config)
            eqw_commission = eq_comm

    # ---- Market benchmark ----
    benchmark_returns: Optional[pd.Series] = None
    benchmark_cum: Optional[pd.Series] = None
    benchmark_perf: Optional[Dict[str, float]] = None
    excess_returns_vs_benchmark: Optional[pd.Series] = None
    excess_perf_vs_benchmark: Optional[Dict[str, float]] = None
    if config.include_benchmark and config.benchmark_symbol:
        benchmark_returns = _compute_benchmark_returns(
            returns,
            config.benchmark_symbol,
            strat_returns.index,
            benchmark_price_path=config.benchmark_price_path,
            benchmark_price_column=config.benchmark_price_column,
        )
        if benchmark_returns is not None and not benchmark_returns.empty:
            benchmark_cum = config.initial_capital * (1.0 + benchmark_returns).cumprod()
            benchmark_perf = _compute_performance_metrics(benchmark_returns, config)
            strat_aligned, bench_aligned = strat_returns.align(benchmark_returns, join="inner")
            if not strat_aligned.empty:
                excess_returns_vs_benchmark = strat_aligned - bench_aligned
            if benchmark_perf is not None:
                excess_perf_vs_benchmark = {
                    k: performance[k] - benchmark_perf[k]
                    for k in performance
                    if k in benchmark_perf and pd.notna(performance[k]) and pd.notna(benchmark_perf[k])
                }

    # ---- Factor regression ----
    factor_result: Optional[FactorRegressionResult] = None
    if factors is not None:
        factor_result = run_factor_regression(
            strat_returns, factors, risk_free_rate=config.risk_free_rate,
        )

    return BacktestResult(
        config=config,
        portfolio_returns=strat_returns,
        portfolio_value=strat_values,
        performance=performance,
        weights=weights,
        signals_panel=signals_panel,
        score_data=score_df.copy(),
        equal_weight_returns=eqw_returns,
        equal_weight_value=eqw_cum,
        equal_weight_performance=eqw_perf,
        benchmark_returns=benchmark_returns,
        benchmark_cumulative=benchmark_cum,
        benchmark_performance=benchmark_perf,
        excess_returns_vs_benchmark=excess_returns_vs_benchmark,
        excess_performance_vs_benchmark=excess_perf_vs_benchmark,
        factor_regression=factor_result,
        portfolio_commission=strat_commission,
        equal_weight_commission=eqw_commission,
    )


# =========================================================================
# Weight schedule
# =========================================================================


def _build_weight_schedule(signals_panel: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    records: List[_PortfolioWeights] = []
    for rebalance_date, group in signals_panel.groupby("rebalance_date"):
        selected = _select_long_assets(group, config)
        for symbol, details in selected.items():
            records.append(
                _PortfolioWeights(
                    rebalance_date=rebalance_date,
                    symbol=symbol,
                    weight=details["weight"],
                    signal_value=details["signal"],
                    report_date=details["report_date"],
                )
            )
    if not records:
        return pd.DataFrame(
            columns=["rebalance_date", "symbol", "weight", "signal_value", "report_date"]
        )
    weights = pd.DataFrame([vars(r) for r in records])
    # Enforce max gross exposure
    for rebalance_date, group in weights.groupby("rebalance_date"):
        gross = group["weight"].abs().sum()
        if config.max_gross_exposure > 0 and gross > config.max_gross_exposure and gross > 0:
            scale = config.max_gross_exposure / gross
            weights.loc[group.index, "weight"] = group["weight"] * scale
    weights.sort_values(["rebalance_date", "weight"], ascending=[True, False], inplace=True)
    return weights


def _select_long_assets(
    group: pd.DataFrame, config: BacktestConfig
) -> Dict[str, Dict[str, Any]]:
    universe = group.dropna(subset=[config.signal_column])
    if universe.empty:
        return {}

    ascending = not config.descending
    ordered = universe.sort_values(config.signal_column, ascending=ascending)

    if config.top_k is not None:
        long_count = min(config.top_k, len(ordered))
    elif config.top_quantile is not None:
        long_count = max(1, int(np.floor(len(ordered) * config.top_quantile)))
    else:
        long_count = len(ordered)

    long_slice = ordered.head(long_count)
    long_symbols = list(dict.fromkeys(long_slice["symbol"].tolist()))

    selections: Dict[str, Dict[str, Any]] = {}
    if long_symbols:
        w = config.long_gross_exposure / len(long_symbols)
        for sym in long_symbols:
            row = long_slice[long_slice["symbol"] == sym].iloc[-1]
            selections[sym] = {
                "weight": w,
                "signal": float(row[config.signal_column]),
                "report_date": pd.Timestamp(row["report_date"]),
            }
    return selections


# =========================================================================
# NAV simulation
# =========================================================================


def _apply_weight_schedule(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    trading_days: np.ndarray,
    price_pivot: pd.DataFrame,
    config: BacktestConfig,
    *,
    series_name: Optional[str] = None,
) -> Tuple[pd.Series, pd.Series, float]:
    sorted_weights = weights.sort_values("rebalance_date")
    rebalance_dates = sorted_weights["rebalance_date"].unique()
    if len(rebalance_dates) == 0:
        empty = pd.Series(dtype=float)
        return empty, empty, 0.0

    trading_index = pd.Index(pd.to_datetime(trading_days))
    price_pivot = price_pivot.sort_index().ffill()

    positions: Dict[str, float] = {}
    cash: float = config.initial_capital
    prev_value: float = config.initial_capital
    daily_returns: List[Tuple[pd.Timestamp, float]] = []
    daily_values: List[Tuple[pd.Timestamp, float]] = []
    total_commission = 0.0

    for idx, rebalance_date in enumerate(rebalance_dates):
        period_weights = sorted_weights[sorted_weights["rebalance_date"] == rebalance_date]
        next_rebalance = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else None

        start = _next_trading_day(rebalance_date, trading_index, include_current=False)
        if start is None or start not in price_pivot.index:
            continue

        start_loc = trading_index.get_indexer([start])
        if start_loc.size == 0 or start_loc[0] == -1:
            continue
        start_idx = start_loc[0]

        if next_rebalance is not None:
            next_start = _next_trading_day(next_rebalance, trading_index, include_current=False)
            if next_start is None:
                end_idx = len(trading_index)
            else:
                end_loc = trading_index.get_indexer([next_start])
                end_idx = end_loc[0] if end_loc.size and end_loc[0] != -1 else len(trading_index)
        else:
            end_idx = len(trading_index)

        period_dates = trading_index[start_idx:end_idx]
        if not len(period_dates):
            continue

        period_weights_series = period_weights.set_index("symbol")["weight"]
        symbols = sorted(set(positions.keys()) | set(period_weights_series.index))
        start_prices = price_pivot.reindex(index=[start], columns=symbols).iloc[0]
        if start_prices.isna().any():
            start_prices = start_prices.ffill().bfill()
        if start_prices.isna().any() or (start_prices <= 0).any():
            continue

        equity = prev_value
        if equity <= 0:
            raise ValueError("Portfolio equity is non-positive; cannot apply weights.")

        target_weights = pd.Series(0.0, index=symbols)
        target_weights.update(period_weights_series)

        current_shares = pd.Series({sym: positions.get(sym, 0.0) for sym in symbols})
        target_shares = target_weights * equity / start_prices
        trade_shares = target_shares - current_shares

        # Sells first
        for sym, trade in trade_shares[trade_shares < -1e-8].items():
            price = float(start_prices[sym])
            shares = float(-trade)
            commission = _calculate_commission(price, shares, config)
            total_commission += commission
            cash += shares * price - commission
            positions[sym] = current_shares[sym] + trade
            if abs(positions[sym]) < 1e-8:
                positions.pop(sym, None)

        # Buys
        for sym, trade in trade_shares[trade_shares > 1e-8].items():
            price = float(start_prices[sym])
            shares = float(trade)
            commission = _calculate_commission(price, shares, config)
            total_commission += commission
            cash -= shares * price + commission
            positions[sym] = current_shares.get(sym, 0.0) + trade
            if abs(positions[sym]) < 1e-8:
                positions.pop(sym, None)

        prev_value = equity

        for date in period_dates:
            if date not in price_pivot.index:
                continue
            price_row = price_pivot.reindex(index=[date], columns=list(positions.keys())).iloc[0]
            if price_row.isna().any():
                price_row = price_row.ffill().bfill()
            if price_row.isna().any() or (price_row <= 0).any():
                continue
            value = sum(positions[sym] * price_row[sym] for sym in positions) + cash
            if value <= 0:
                raise ValueError(
                    "Portfolio value became non-positive; check leverage and price data."
                )
            if prev_value <= 0:
                raise ValueError("Previous portfolio value is non-positive.")
            daily_return = (value - prev_value) / prev_value
            daily_returns.append((pd.Timestamp(date), float(daily_return)))
            daily_values.append((pd.Timestamp(date), float(value)))
            prev_value = value

    if not daily_returns:
        empty = pd.Series(dtype=float)
        return empty, empty, total_commission

    ret_dates, ret_vals = zip(*daily_returns)
    val_dates, val_vals = zip(*daily_values)
    returns_series = pd.Series(ret_vals, index=pd.DatetimeIndex(ret_dates)).sort_index()
    values_series = pd.Series(val_vals, index=pd.DatetimeIndex(val_dates)).sort_index()
    returns_series.name = series_name or config.name
    values_series.name = f"{returns_series.name}_value"
    values_series = values_series[~values_series.index.duplicated(keep="last")]
    returns_series = returns_series[~returns_series.index.duplicated(keep="last")]
    returns_series = returns_series.loc[values_series.index]
    return returns_series, values_series, total_commission


# =========================================================================
# Equal-weight benchmark
# =========================================================================


def _run_equal_weight_benchmark(
    signals_panel: pd.DataFrame,
    returns: pd.DataFrame,
    trading_days: np.ndarray,
    price_pivot: pd.DataFrame,
    config: BacktestConfig,
    *,
    component_universe: Optional[_ComponentUniverse] = None,
    available_symbols: Optional[Set[str]] = None,
) -> Tuple[pd.Series, pd.Series, float]:
    sorted_panel = signals_panel.sort_values("rebalance_date")
    rebalance_dates = sorted_panel["rebalance_date"].unique()
    price_symbols = {str(col).upper() for col in price_pivot.columns}

    records: List[_PortfolioWeights] = []
    for rebalance_date in rebalance_dates:
        if component_universe is not None:
            allowed = set(component_universe.allowed(pd.Timestamp(rebalance_date)))
        else:
            allowed = set(
                sorted_panel.loc[sorted_panel["rebalance_date"] == rebalance_date, "symbol"].tolist()
            )
        if available_symbols is not None:
            allowed &= available_symbols
        allowed &= price_symbols
        if not allowed:
            continue
        w = 1.0 / len(allowed)
        for symbol in sorted(allowed):
            records.append(
                _PortfolioWeights(
                    rebalance_date=rebalance_date,
                    symbol=symbol,
                    weight=w,
                    signal_value=0.0,
                    report_date=rebalance_date,
                )
            )

    if not records:
        empty = pd.Series(dtype=float)
        return empty, empty, 0.0

    weights_df = pd.DataFrame([vars(r) for r in records])
    return _apply_weight_schedule(
        weights_df, returns, trading_days, price_pivot, config, series_name="equal_weight",
    )


# =========================================================================
# Benchmark returns
# =========================================================================


def _compute_benchmark_returns(
    returns: pd.DataFrame,
    benchmark_symbol: str,
    index: Iterable[pd.Timestamp],
    *,
    benchmark_price_path: Optional[str] = None,
    benchmark_price_column: str = "close",
) -> Optional[pd.Series]:
    from pathlib import Path

    # Try external benchmark CSV first
    if benchmark_price_path is not None:
        bpath = Path(benchmark_price_path)
        if bpath.exists():
            try:
                series = _load_benchmark_returns_from_csv(bpath, benchmark_price_column)
                if series is not None and not series.empty:
                    target_index = pd.DatetimeIndex(index)
                    aligned = series.reindex(target_index).dropna()
                    if not aligned.empty:
                        return aligned.rename(benchmark_symbol.upper())
            except Exception as exc:
                logger.warning("Failed to load benchmark from %s: %s", bpath, exc)

    # Fall back to symbol in returns data
    symbol_mask = returns["symbol"].str.upper() == benchmark_symbol.upper()
    bench = returns.loc[symbol_mask, ["date", "return"]].copy()
    if bench.empty:
        return None
    bench.set_index("date", inplace=True)
    bench = bench.loc[bench.index.isin(index)]
    return bench["return"].rename(benchmark_symbol.upper())


def _load_benchmark_returns_from_csv(
    path: "Path", price_column: str
) -> Optional[pd.Series]:
    from pathlib import Path

    frame = pd.read_csv(path)
    if frame.empty:
        return None

    frame.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_")
        for col in frame.columns
    ]
    if "date" not in frame.columns:
        raise ValueError("Benchmark price data must include a 'date' column")

    col = price_column.strip().lower()
    if col not in frame.columns:
        raise ValueError(
            f"Benchmark missing column '{price_column}'. Available: {list(frame.columns)}"
        )

    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%y", errors="coerce")
    if frame["date"].isna().any():
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame.dropna(subset=["date", col], inplace=True)
    frame.sort_values("date", inplace=True)
    frame.drop_duplicates(subset="date", keep="last", inplace=True)

    frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame.dropna(subset=[col], inplace=True)

    frame["return"] = frame[col].pct_change()
    return frame.set_index("date")["return"].dropna()


# =========================================================================
# Performance metrics
# =========================================================================


def _compute_performance_metrics(returns: pd.Series, config: BacktestConfig) -> Dict[str, float]:
    if returns.empty:
        return {
            "total_return": float("nan"),
            "cagr": float("nan"),
            "annual_vol": float("nan"),
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
        }

    total_return = float((1.0 + returns).prod() - 1.0)
    periods = returns.shape[0]
    ppy = config.periods_per_year
    cagr = (1.0 + total_return) ** (ppy / periods) - 1.0 if periods > 0 else np.nan

    mean_return = returns.mean()
    std_dev = returns.std(ddof=0)
    annual_vol = std_dev * np.sqrt(ppy)

    sharpe = np.nan
    if std_dev > 0:
        sharpe = ((mean_return - config.risk_free_rate / ppy) / std_dev) * np.sqrt(ppy)

    downside = returns[returns < config.risk_free_rate / ppy]
    downside_std = downside.std(ddof=0)
    sortino = np.nan
    if downside_std > 0:
        sortino = (mean_return - config.risk_free_rate / ppy) / downside_std * np.sqrt(ppy)

    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()
    max_dd = float((cumulative / running_max - 1.0).min())

    return {
        "initial_capital": config.initial_capital,
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "ending_value": config.initial_capital * (1.0 + total_return),
    }


# =========================================================================
# Commission
# =========================================================================


def _calculate_commission(price: float, shares: float, config: BacktestConfig) -> float:
    quantity = abs(shares)
    if quantity == 0 or price <= 0:
        return 0.0
    return quantity * price * config.commission_bps / 10_000
