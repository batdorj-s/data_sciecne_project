import pytest

from finagent.guardrails import (
    check_faithfulness,
    check_recommendation,
    enforce_confidence,
    extract_numbers,
    validate_structure,
)
from finagent.pipeline import Analysis
from finagent.tools import TOOLS, call_tool


@pytest.fixture(scope="module")
def results(sample_files, balances):
    a = Analysis.from_files(sample_files, balances)
    out = [call_tool(a, n, {"line": "revenue"} if n == "get_line_trend" else {}) for n in TOOLS]
    out.append(call_tool(a, "simulate_scenario", {"expense_pct": {"project": -10}}))
    return out


def test_extract_numbers_units_and_ranges():
    nums = extract_numbers("Орлого 61.7–86.8 сая ₮, маржин 20.3%, 1,763,500 ₮, 2026-04 сар")
    vals = [round(v) if u != "%" else v for _, v, u in nums]
    assert vals == [61_700_000, 86_800_000, 20.3, 1_763_500]
    assert extract_numbers("(14.9) сая")[0][1] == pytest.approx(-14_900_000)
    arrow = extract_numbers("ашиг 10.8 → 12.5 сая ₮")
    assert [round(v / 1e5) for _, v, _ in arrow] == [108, 125]


def test_proposal_example_is_faithful(results):
    text = ("Тогтвортой–өсөлт: орлого 61.7–86.8 сая ₮, суурь ашиг 10.8 сая ₮. "
            "1-р сард нэг төлбөр сарын орлогын 73.9%-ийг эзэлсэн; 74% → 46% → 28% болж буурсан. "
            "Төслийн зардлыг 10% бууруулбал ашиг 10.8 → 12.5 сая ₮ (+1.8 сая), маржин 20.3%.")
    rep = check_faithfulness(text, results)
    assert rep.ok, rep.unmatched
    assert rep.score == 1.0


def test_hallucinated_number_is_caught(results):
    rep = check_faithfulness("Дараа сарын орлого 95.3 сая ₮ болно, маржин 41%.", results)
    assert not rep.ok
    assert len(rep.unmatched) == 2
    # хэрэгслийн хоёр тооны зөрүүтэй санамсаргүй таарч болохгүй
    assert not check_faithfulness("ашиг 15.0 сая ₮ болно", results).ok


def _rec(impact=1_763_500):
    tactic = {"action": "Төслийн зардлыг 10% бууруулах", "kpi": "төслийн зардал / орлого",
              "impact_mnt": impact, "priority": "high"}
    return {"trend": "stable_growth", "confidence": "high", "drivers": [], "risks": [],
            "tactics": [tactic] * 3, "data_warnings": []}


def test_recommendation_checks(results):
    assert validate_structure(_rec()) == []
    assert check_recommendation(_rec(), results).ok
    assert not check_recommendation(_rec(impact=9_999_999), results).ok
    assert "тактик 3–6" in validate_structure({**_rec(), "tactics": []})[-1]
    assert enforce_confidence(_rec(), "low")["confidence"] == "low"
