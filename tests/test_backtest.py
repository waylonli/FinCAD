import pandas as pd
import pytest

from benchmark.backtest.ai_hedge_fund.agent import build_financial_summary
from benchmark.backtest.ai_hedge_fund.eval import compute_commission


def test_financial_summary_excludes_decision_date_and_future_rows():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-06-27", "2018-06-28", "2018-06-29", "2018-07-02"]),
            "symbol": ["NVDA"] * 4,
            "adjusted_close": [10.0, 11.0, 9999.0, 20000.0],
        }
    )
    summary = build_financial_summary("NVDA", pd.Timestamp("2018-06-29"), prices)

    assert "Current Price: $11.00" in summary
    assert "9,999" not in summary
    assert "20,000" not in summary


def test_commission_is_ten_basis_points_of_notional():
    assert compute_commission(100, 25.0) == pytest.approx(2.50)
