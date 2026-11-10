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


def _xlsx(tmp_path, name, rows, header):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["Дансны хуулга 2026 оны 4-р сар"])
    ws.append(header)
    for r in rows:
        ws.append(r)
    p = tmp_path / name
    wb.save(p)
    return p


def test_txn_id_is_stable_when_rows_are_inserted(sample_files, tmp_path):
    from openpyxl import load_workbook

    before = load_ledger(sample_files[:1])
    wb = load_workbook(sample_files[0])
    wb.active.insert_rows(4)
    for col, v in enumerate(["3-Jan", "CP-999", 1000, "Зарлага", "Оффис", "шинэ мөр"], 1):
        wb.active.cell(row=4, column=col, value=v)
    p = tmp_path / "transactions_2026_01.xlsx"
    wb.save(p)
    after = load_ledger([p])
    assert len(after) == len(before) + 1
    assert set(before["txn_id"]) <= set(after["txn_id"])
    assert before["txn_id"].is_unique  # давхар Chimege гүйлгээ -2 дагавар авна


def test_bank_export_debit_credit(tmp_path):
    p = _xlsx(tmp_path, "khan_2026_04.xlsx", [
        ["2026-04-02", "CP-001", None, 5_000_000, "Нэхэмжлэх 12 төлбөр"],
        ["2026-04-05", "CP-002", 3_000_000, None, "4 сарын түрээс"],
        ["2026-04-20", "CP-003", 20_000_000, None, "4 САРЫН 20 ЦАЛИН"],
        ["2026-04-21", "CP-004", 800_000, None, "ZEEL TULULT"],
        ["2026-04-22", "CP-005", 1_000_000, None, "Банкнаас касс руу"],
        ["2026-04-23", "CP-006", 50_000, None, "Тодорхойгүй"],
    ], ["Гүйлгээний огноо", "Харьцсан дансны нэр", "Дебит", "Кредит", "Гүйлгээний утга"])
    led = load_ledger([p])
    assert led["line"].tolist() == ["sales", "rent", "payroll", "loan_repayment",
                                    "transfer_to_cash", "unclassified_out"]
    assert led["direction"].tolist() == ["in"] + ["out"] * 5
    assert (led["amount"] > 0).all()
    assert led["signed_amount"].sum() == 5_000_000 - 3_000_000 - 20_000_000 - 800_000 \
        - 1_000_000 - 50_000


def test_bank_export_signed_amount(tmp_path):
    p = _xlsx(tmp_path, "golomt_2026_04.xlsx", [
        ["2026-04-02", "CP-001", 2_500_000, "Борлуулалт"],
        ["2026-04-03", "CP-002", -120_000, "Банкны шимтгэл"],
    ], ["Огноо", "Харилцагч", "Гүйлгээний дүн", "Утга"])
    led = load_ledger([p])
    assert led["direction"].tolist() == ["in", "out"]
    assert led["line"].tolist() == ["sales", "opex"]
