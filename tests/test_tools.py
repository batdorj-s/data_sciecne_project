"""Хэрэгслүүд: proposal-ийн "Хэрэгслийн жишээ (Компани А, 2026 оны I улирал)"-тай тулгана."""

import json

import pytest

from finagent.pipeline import Analysis
from finagent.tools import TOOL_SPECS, TOOLS, call_tool, resolve_line


@pytest.fixture(scope="module")
def analysis(sample_files, balances):
    return Analysis.from_files(sample_files, balances)


def test_specs_match_functions():
    assert [s["name"] for s in TOOL_SPECS] == list(TOOLS)
    assert len(TOOLS) == 6


@pytest.mark.parametrize("name", list(TOOLS))
def test_every_tool_returns_small_json(analysis, name):
    args = {"line": "Төслийн зардал"} if name == "get_line_trend" else {}
    out = call_tool(analysis, name, args)
    assert "error" not in out
    assert len(json.dumps(out, ensure_ascii=False)) < 6_000


# --- Proposal-ийн жишээ 1–6 --------------------------------------------------------------
def test_example1_quarter_overview(analysis):
    q = call_tool(analysis, "get_quarter_overview")
    assert q["сар"] == ["1", "2", "3"]
    assert q["орлого"] == [41.80, 56.55, 71.78]
    assert q["зардал"] == [56.68, 45.65, 52.56]
    assert q["ашиг"] == [-14.89, 10.90, 19.21]
    assert q["маржин_%"] == [-35.6, 19.3, 26.8]
    assert q["улирал"] == {"орлого": 170.12, "ашиг": 15.22, "маржин_%": 8.9}
    assert q["тулгалтын_зөрүү"] == 35.96
    assert q["итгэл"] == "бага"
    assert set(q["анхааруулга"]) == {"өндөр", "дунд", "бага"}


def test_example2_line_trend_projects(analysis):
    t = call_tool(analysis, "get_line_trend", {"line": "Төслийн зардал"})
    assert t["сарын_дүн"] == [19.50, 13.02, 20.09]
    assert t["улирал"] == 52.61
    assert t["хамгийн_том_5"] == [["1/14", 13.82, "Төсөл Б"], ["3/10", 7.50, "Төсөл А"],
                                  ["2/16", 6.00, "Төсөл А"], ["3/25", 5.90, "Төсөл А"],
                                  ["1/13", 5.00, "Зар сурталчилгаа"]]
    assert t["төслөөр"] == {"Төсөл А": 22.19, "Төсөл Б": 15.70}
    # 42% нь Төсөл А-д төвлөрсөн
    assert round(t["төслөөр"]["Төсөл А"] / t["улирал"] * 100) == 42


def test_example3_baseline_forecast(analysis):
    f = call_tool(analysis, "get_baseline_forecast")
    assert f["сар"] == "2026-04"
    assert f["орлого"] == {"суурь": 61.70, "доод": 61.70, "дээд": 86.77}
    assert f["зардал"] == {"суурь": 50.95, "доод": 47.09, "дээд": 54.36}
    assert f["ашиг"] == {"суурь": 10.76, "маржин_%": 17.4}


def test_example4_simulate_scenario(analysis):
    s = call_tool(analysis, "simulate_scenario",
                  {"revenue_change_pct": 0, "cost_change_pct": {"Төслийн зардал": -10}})
    assert {k: s[k] for k in ("суурь_ашиг", "сценари_ашиг", "өөрчлөлт", "суурь_маржин_%",
                              "сценари_маржин_%")} == {
        "суурь_ашиг": 10.76, "сценари_ашиг": 12.52, "өөрчлөлт": 1.76,
        "суурь_маржин_%": 17.4, "сценари_маржин_%": 20.3}


def test_example5_revenue_concentration(analysis):
    c = call_tool(analysis, "get_revenue_concentration")
    assert c["орлогын_төлбөрийн_тоо"] == [6, 5, 8]
    assert c["хамгийн_том_төлбөр"] == [30.0, 26.0, 20.0]
    assert c["хамгийн_томын_хувь_%"] == [74, 46, 28]
    assert c["чиг"] == "буурч байна"


def test_example6_data_issues_high(analysis):
    d = call_tool(analysis, "get_data_issues", {"level": "Өндөр"})
    rows = {(x["дүрэм"], x["огноо"], x["дүн"]) for x in d["жагсаалт"] if isinstance(x, dict)}
    assert ("R2 Ангилагдаагүй", "1/14", 17.60) in rows
    assert ("R7 Орлого давхар бүртгэгдэх", "3/12", 20.00) in rows
    assert d["тулгалт"] == {"зөрүү": 35.96, "үүссэн_сар": "1"}
    assert d["тоо"] == len([x for x in d["жагсаалт"] if isinstance(x, dict)])


# --- Оролтын хувилбарууд ба алдаа ------------------------------------------------------------
def test_line_aliases():
    assert resolve_line("Төслийн зардал") == resolve_line("project") == "project"
    assert resolve_line("Нийт орлого") == "revenue"
    assert resolve_line("ҮАЗ") == "opex"


def test_validation_errors(analysis):
    assert "error" in call_tool(analysis, "simulate_scenario", {"cost_change_pct": {"foo": 1}})
    assert "error" in call_tool(analysis, "get_line_trend", {"line": "nope"})
    assert "error" in call_tool(analysis, "get_data_issues", {"level": "маш өндөр"})
    assert "error" in call_tool(analysis, "unknown_tool")
    all_cost = call_tool(analysis, "simulate_scenario", {"cost_change_pct": {"Бүх зардал": -5}})
    assert all_cost["өөрчлөлт"] == pytest.approx(50.95 * 0.05, abs=0.01)


def test_no_personal_or_client_names_leak(analysis):
    blob = json.dumps([call_tool(analysis, n, {"line": "Төслийн зардал"}
                                 if n == "get_line_trend" else {}) for n in TOOLS],
                      ensure_ascii=False)
    for cp in analysis.ledger["counterparty"].unique():
        if not str(cp).startswith("CP-"):
            assert cp not in blob
    assert "MCA" not in blob and "RUBY" not in blob  # захиалагчийн брэнд → Төсөл А/Б


def test_raw_names_are_pseudonymized_before_tools(analysis, balances):
    led = analysis.ledger.copy()
    name = "БАДМААРАГ ЧУЛУУНБААТАР"
    mask = led["counterparty"] == "CP-010"
    led.loc[mask, "counterparty"] = name
    led.loc[mask, "description"] = led.loc[mask, "description"] + ",БАДМААРАГ Ч"
    raw = Analysis(led, balances)
    blob = json.dumps([call_tool(raw, n, {"line": "Төслийн зардал"}
                                 if n == "get_line_trend" else {}) for n in TOOLS],
                      ensure_ascii=False)
    assert "БАДМААРАГ" not in blob and "ЧУЛУУНБААТАР" not in blob


def test_excel_export(analysis):
    import io

    import pandas as pd

    from finagent.export import to_excel

    xls = pd.ExcelFile(io.BytesIO(to_excel(analysis)))
    assert "Орлого-зарлага" in xls.sheet_names and "Гүйлгээ" in xls.sheet_names
    assert len(pd.read_excel(xls, "Гүйлгээ")) == 185
