"""Пилотын гар тайлантай (tailan_orlog_zarlag.xlsx, mungun_ursgal.xlsx) тулгах тест."""

import pytest

from finagent.classify import load_ledger
from finagent.reports import (
    TOTAL,
    cash_flow_statement,
    cost_breakdown,
    income_statement,
    monthly_summary,
)


@pytest.fixture(scope="module")
def ledger(sample_files):
    return load_ledger(sample_files)


def test_monthly_summary_matches_manual_report(ledger):
    s = monthly_summary(ledger)
    assert s["revenue"].tolist() == [41_797_000, 56_550_000, 71_777_000]
    assert s["payroll"].tolist() == [26_699_360, 26_741_429, 24_435_492]
    assert s["project"].tolist() == [19_495_776, 13_021_000, 20_090_750]
    assert s["operating_profit"].round(0).tolist() == [-14_886_153, 10_896_056, 19_213_028]
    assert s["margin"].round(3).tolist() == [-0.356, 0.193, 0.268]
    assert s["txn_count"].sum() == 185
    # нийт зардал ба ҮАЗ мөр тусдаа багана байх ёстой
    assert s["total_expenses"].tolist() == pytest.approx([56_683_153, 45_653_944, 52_563_972])
    assert s["opex"].tolist() == [5_528_817, 1_186_409, 4_742_630]


def test_quarter_operating_profit_is_15_2m_not_4_1m(ledger):
    inc = income_statement(ledger)
    assert inc.loc["НИЙТ ОРЛОГО", TOTAL] == pytest.approx(170_124_000)
    assert inc.loc["ҮЙЛ АЖИЛЛАГААНЫ АШИГ / АЛДАГДАЛ", TOTAL] == pytest.approx(15_222_931)
    assert inc.loc["Ашгийн маржин", TOTAL] == pytest.approx(0.0895, abs=1e-3)


def test_cash_flow_reconciliation_gap_in_january(ledger, balances):
    cf = cash_flow_statement(ledger, balances)
    gap = cf.loc["Тулгалтын зөрүү"]
    # 1-р сарын хуулга дутуу (48 сая ₮-ийн дамжин өнгөрсөн зээл байхгүй) тул зөрнө
    assert gap["2026-01"] == pytest.approx(35_955_492, abs=1)
    assert abs(gap["2026-02"]) < 10_000 and abs(gap["2026-03"]) < 10_000
    assert cf.loc["Санхүүгийн цэвэр мөнгөн урсгал", "2026-03"] == pytest.approx(-4_249_000)


def test_cost_breakdown_shares_sum_to_one(ledger):
    cb = cost_breakdown(ledger)
    assert cb["share"][TOTAL].sum() == pytest.approx(1.0)
    assert cb["amount"][TOTAL].sum() == pytest.approx(154_901_069)
