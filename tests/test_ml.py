import pytest

from finagent.checks import run_checks
from finagent.classify import load_ledger
from finagent.ml.anomaly import detect, detection_recall, inject_errors
from finagent.ml.classifier import apply_model, evaluate, suggest, train
from finagent.pipeline import Analysis


@pytest.fixture(scope="module")
def ledger(sample_files):
    return load_ledger(sample_files)


def test_classifier_cv_baseline(ledger):
    ev = evaluate(ledger, seed=0)
    # 185 мөрийн жишиг; proposal-ийн босго 1,000+ шошготой, 4+ компани дээр хэмжигдэнэ
    assert ev.macro_f1 > 0.75
    assert ev.financing_recall > 0.8


def test_suggest_and_apply_on_bank_rows(ledger, tmp_path):
    from openpyxl import Workbook

    model = train(ledger)
    s = suggest(model, ledger.head(10))
    assert set(s.columns) == {"txn_id", "predicted", "proba", "confident"}
    wb = Workbook()
    ws = wb.active
    ws.append(["Огноо", "Харилцагч", "Дебит", "Кредит", "Гүйлгээний утга"])
    ws.append(["2026-04-14", "CP-005", 84_200, None, "OFFICE ZARDAL"])
    ws.append(["2026-04-15", "CP-006", 13_000, None, "Tools zahialga"])
    p = tmp_path / "bank_2026_04.xlsx"
    wb.save(p)
    led = apply_model(model, load_ledger([p]), threshold=0.0)
    assert led["class_source"].tolist() == ["model", "model"]
    assert (led["direction"] == "out").all()
    a = Analysis.from_files([p], classifier=model)
    assert set(a.ledger["class_source"]) <= {"model", "rule"}


def test_injected_error_recall(ledger):
    totals = []
    for seed in range(3):
        bad, truth = inject_errors(ledger, 50, seed=seed)
        r = detection_recall(run_checks(bad) + detect(bad), truth)
        assert r["duplicate"] == 1.0 and r["unclassified"] == 1.0
        totals.append(r["total"])
    assert sum(totals) / len(totals) >= 0.85


def test_anomaly_issues_in_analysis(sample_files, balances):
    a = Analysis.from_files(sample_files, balances)
    a1 = [i for i in a.issues if i.rule == "A1"]
    assert 0 < len(a1) <= 20 and all(i.severity == "low" for i in a1)
    assert a.confidence[0] == "low"  # A1 (low) итгэлийн түвшинд нөлөөлөхгүй
