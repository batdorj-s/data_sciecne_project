import pandas as pd
import pytest

from finagent.checks import issues_frame, reconciled, run_checks
from finagent.classify import load_ledger


@pytest.fixture(scope="module")
def ledger(sample_files):
    return load_ledger(sample_files)


@pytest.fixture(scope="module")
def issues(ledger, balances):
    return run_checks(ledger, balances)


def _rules(issues, rule):
    return [i for i in issues if i.rule == rule]


def test_pilot_known_errors_are_caught(issues):
    # 1-р сарын дутуу файл → үлдэгдэл 35.96 сая ₮ зөрнө
    r8 = _rules(issues, "R8")
    assert len(r8) == 1 and r8[0].month == "2026-01" and r8[0].severity == "high"
    assert r8[0].amount == pytest.approx(35_955_492, abs=1)
    # 17.6 сая ₮ ангилагдаагүй
    r3 = _rules(issues, "R2")
    assert len(r3) == 1 and r3[0].amount == 17_600_000
    assert not reconciled(issues)


def test_rule_coverage(issues):
    found = {i.rule for i in issues}
    assert {"R7", "R2", "R3", "R4", "R5", "R6", "R8", "R9"} <= found
    r9 = {i.month: i.amount for i in _rules(issues, "R9")}
    assert r9 == {"2026-01": 20_600_000, "2026-03": 20_500_000}
    assert not _rules(issues, "R1")  # жишээ өгөгдөл бүтцийн хувьд зөв
    assert len(_rules(issues, "R3")) == 1  # Chimege subscription давхар
    # draft: "орлого давхар бүртгэгдэх эрсдэл (5)", 3/12-ны 20 сая ₮ өндөр түвшин
    r7 = _rules(issues, "R7")
    assert len(r7) == 5
    assert [(i.month, i.amount) for i in r7 if i.severity == "high"] == [("2026-03", 20_000_000)]


def test_sorted_by_severity(issues):
    order = {"high": 0, "medium": 1, "low": 2}
    sev = [order[i.severity] for i in issues]
    assert sev == sorted(sev)


def test_structure_rule_and_missing_balances():
    led = pd.DataFrame({
        "txn_id": ["x-1"], "date": [pd.NaT], "month": ["2026-04"], "counterparty": [None],
        "amount": [100.0], "direction": ["out"], "signed_amount": [-100.0],
        "src_category": ["Зарлага"], "src_subcategory": ["Оффис"], "description": ["a"],
        "activity": ["operating"], "line": ["office"], "class_source": ["rule"],
        "source_file": ["f.xlsx"], "source_row": [4],
    })
    iss = run_checks(led, None)
    assert {i.rule for i in iss} == {"R1", "R8"}
    df = issues_frame(iss)
    assert list(df.columns)[:3] == ["rule", "title", "severity"]
