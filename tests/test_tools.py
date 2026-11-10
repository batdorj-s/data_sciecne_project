import json

import pytest

from finagent.pipeline import Analysis
from finagent.tools import TOOL_SPECS, TOOLS, call_tool


@pytest.fixture(scope="module")
def analysis(sample_files, balances):
    return Analysis.from_files(sample_files, balances)


def test_specs_match_functions():
    assert [s["name"] for s in TOOL_SPECS] == list(TOOLS)
    assert len(TOOLS) == 6


@pytest.mark.parametrize("name", list(TOOLS))
def test_every_tool_returns_small_json(analysis, name):
    args = {"line": "project"} if name == "get_line_trend" else {}
    out = call_tool(analysis, name, args)
    assert "error" not in out
    payload = json.dumps(out, ensure_ascii=False)
    assert len(payload) < 8_000


def test_quarter_overview(analysis):
    q = call_tool(analysis, "get_quarter_overview")
    assert q["quarter"] == {"revenue": 170_124_000, "total_expenses": 154_901_069,
                            "operating_profit": 15_222_931, "margin": 0.089}
    assert q["months"][0]["operating_profit"] == -14_886_153
    assert q["confidence"] == "low" and q["reconciled"] is False


def test_concentration_74_46_28(analysis):
    c = call_tool(analysis, "get_revenue_concentration")
    assert [m["largest_share"] for m in c["months"]] == [0.739, 0.46, 0.283]
    assert c["largest_share_trend"] == "буурч байна"


def test_scenario_and_validation(analysis):
    sc = call_tool(analysis, "simulate_scenario", {"expense_pct": {"project": -10}})
    assert sc["profit_delta"] == 1_763_500 and sc["margin"] == 0.203
    assert "error" in call_tool(analysis, "simulate_scenario", {"expense_pct": {"foo": 1}})
    assert "error" in call_tool(analysis, "get_line_trend", {"line": "nope"})
    assert "error" in call_tool(analysis, "unknown_tool")


def test_no_personal_names_leak(analysis):
    """Агент руу очих бүх гаралтад харилцагчийн жинхэнэ нэр байх ёсгүй."""
    blob = json.dumps([call_tool(analysis, n, {"line": "project"} if n == "get_line_trend"
                                 else {}) for n in TOOLS], ensure_ascii=False)
    for cp in analysis.ledger["counterparty"].unique():
        if not str(cp).startswith("CP-"):
            assert cp not in blob


def test_raw_names_are_pseudonymized_before_tools(analysis, balances):
    led = analysis.ledger.copy()
    name = "БАДМААРАГ ЧУЛУУНБААТАР"
    mask = led["counterparty"] == "CP-010"
    led.loc[mask, "counterparty"] = name
    led.loc[mask, "description"] = led.loc[mask, "description"] + ",БАДМААРАГ Ч"
    raw = Analysis(led, balances)
    blob = json.dumps([call_tool(raw, n, {"line": "project"} if n == "get_line_trend" else {})
                       for n in TOOLS], ensure_ascii=False)
    assert "БАДМААРАГ" not in blob and "ЧУЛУУНБААТАР" not in blob


def test_excel_export(analysis):
    import io

    import pandas as pd

    from finagent.export import to_excel

    xls = pd.ExcelFile(io.BytesIO(to_excel(analysis)))
    assert "Орлого-зарлага" in xls.sheet_names and "Гүйлгээ" in xls.sheet_names
    assert len(pd.read_excel(xls, "Гүйлгээ")) == 185
