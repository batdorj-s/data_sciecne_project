"""Proposal 5.1-ийн 4-р сарын суурь таамагтай тулгана."""

import pytest

from finagent.classify import load_ledger
from finagent.forecast import (
    backtest,
    baseline_forecast,
    confidence_level,
    linear_trend,
    mape,
    simulate_scenario,
    weighted_average,
)

M = 1_000_000


@pytest.fixture(scope="module")
def fc(sample_files):
    return baseline_forecast(load_ledger(sample_files))


def test_helpers():
    assert weighted_average([1, 2, 3]) == pytest.approx(14 / 6)
    assert linear_trend([10, 20, 30]) == 40
    assert linear_trend([5]) == 5


def test_april_forecast_matches_proposal(fc):
    assert fc.target_month == "2026-04"
    assert fc.revenue.base / M == pytest.approx(61.7, abs=0.05)
    assert (fc.revenue.low / M, fc.revenue.high / M) == pytest.approx((61.7, 86.8), abs=0.05)
    assert fc.expenses["payroll"].base / M == pytest.approx(25.6, abs=0.05)
    assert fc.expenses["payroll"].low / M == pytest.approx(23.3, abs=0.05)
    assert fc.expenses["project"].high / M == pytest.approx(20.4, abs=0.05)
    other = sum(fc.expenses[k].base for k in ("office", "rent", "opex"))
    assert other / M == pytest.approx(7.7, abs=0.05)
    assert fc.profit_base / M == pytest.approx(10.8, abs=0.05)
    assert fc.margin_base == pytest.approx(0.174, abs=0.001)
    assert fc.profit_low <= fc.profit_base <= fc.profit_high


def test_scenario_project_cost_minus_10pct(fc):
    sc = simulate_scenario(fc, expense_pct={"project": -10})
    assert sc["profit_delta"] / M == pytest.approx(1.76, abs=0.01)
    assert sc["margin"] == pytest.approx(0.203, abs=0.001)
    assert simulate_scenario(fc)["profit_delta"] == pytest.approx(0)


def test_confidence():
    assert confidence_level(3, reconciled=False, high_issues=2)[0] == "low"
    assert confidence_level(4, reconciled=True, high_issues=0)[0] == "medium"
    assert confidence_level(8, reconciled=True, high_issues=0)[0] == "high"


def test_backtest_and_mape():
    bt = backtest([10, 12, 14, 16, 18])
    assert len(bt) == 2 and bt["covered"].all()
    assert mape([100, 200], [110, 180]) == pytest.approx(0.1)
