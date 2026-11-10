from datetime import date

import pytest

from finagent.anonymize import anonymize_frame, scrub_text
from finagent.ingest import IngestError, parse_date, read_many, read_statement


def test_parse_date_variants():
    assert parse_date("5-Jan", 2026) == date(2026, 1, 5)
    assert parse_date("2026-03-31", None) == date(2026, 3, 31)
    assert parse_date("12-Feb-25", None) == date(2025, 2, 12)
    assert parse_date(None, 2026) is None
    with pytest.raises(IngestError):
        parse_date("5-Jan", None)


def test_read_sample_months(sample_files):
    parsed = read_many(sample_files)
    assert [(p.year, p.month, len(p.frame)) for p in parsed] == [
        (2026, 1, 53), (2026, 2, 61), (2026, 3, 71),
    ]
    # Доорх нийлбэрийн томьёо гүйлгээнд орох ёсгүй
    assert all(p.frame["amount"].notna().all() for p in parsed)
    assert all(not p.warnings for p in parsed)


def test_missing_header_raises(tmp_path):
    from openpyxl import Workbook

    wb = Workbook()
    wb.active.append(["Огноо", "Мөнгө"])
    path = tmp_path / "bad.xlsx"
    wb.save(path)
    with pytest.raises(IngestError):
        read_statement(path)


def test_anonymize_replaces_names_and_truncations():
    import pandas as pd

    df = pd.DataFrame({
        "counterparty": ["БАЯРМАГНАЙ БИЛГҮҮН", "МОНГОЛ КОНТЕНТ ХХК"],
        "description": ["UB PARKET ZARDAL,БАЯРМАГНАЙ БИЛГ", "ЗАР СУРТАЛЧИЛГАА"],
    })
    out, book = anonymize_frame(df)
    assert list(out["counterparty"]) == ["CP-001", "CP-002"]
    assert out["description"][0] == "UB PARKET ZARDAL,*** ***"
    assert out["description"][1] == "ЗАР СУРТАЛЧИЛГАА"
    assert book["МОНГОЛ КОНТЕНТ ХХК"] == "CP-002"
    assert scrub_text("КАСС РУУ ХИЙВ-ЦЭЭХҮҮ", {"ЦЭЭХҮҮ"}) == "КАСС РУУ ХИЙВ-***"
