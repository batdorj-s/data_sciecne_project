import pytest

from finagent.classify import classify_row, load_ledger
from finagent.schema import Line


@pytest.fixture(scope="module")
def ledger(sample_files):
    return load_ledger(sample_files)


@pytest.mark.parametrize(
    "cat,sub,desc,expected",
    [
        ("Үйл ажиллагааны орлого", "Касс", "Кассаас банк руу", ("in", Line.SALES)),
        ("Зарлага", "Зээл", "ZEEL TULUW", ("out", Line.LOAN_REPAYMENT)),
        ("Зарлага", "Цалин", "", ("out", Line.PAYROLL)),
        ("Эздийн зээл", "Зээл", "", ("in", Line.OWNER_LOAN_IN)),
        ("Дотоод шилжүүлэг", "Касс", "Банкнаас касс руу", ("out", Line.TRANSFER_TO_CASH)),
        ("Зарлага", "Бусад", "КАСС РУУ ХИЙВ", ("out", Line.UNCLASSIFIED_OUT)),
        ("Шинэ ангилал", None, None, ("out", Line.UNCLASSIFIED_OUT)),
    ],
)
def test_classify_row(cat, sub, desc, expected):
    assert classify_row(cat, sub, desc) == expected


def test_ledger_shape(ledger):
    assert len(ledger) == 185
    assert ledger["txn_id"].is_unique
    assert (ledger["amount"] > 0).all()
    assert set(ledger["month"]) == {"2026-01", "2026-02", "2026-03"}


def test_loan_payments_are_financing_not_expense(ledger):
    loans = ledger[ledger["line"] == "loan_repayment"]
    assert (loans["activity"] == "financing").all()
    by_month = loans.groupby("month")["amount"].sum().round(0).to_dict()
    assert by_month == {"2026-01": 1_640_000, "2026-02": 10_205_000, "2026-03": 30_628_800}


def test_overrides_mark_manual(sample_files):
    first = load_ledger(sample_files)
    tid = first.loc[first["line"] == "unclassified_out", "txn_id"].iloc[0]
    led = load_ledger(sample_files, overrides={tid: "transfer_to_cash"})
    row = led.set_index("txn_id").loc[tid]
    assert row["line"] == "transfer_to_cash"
    assert row["activity"] == "internal"
    assert row["class_source"] == "manual"
