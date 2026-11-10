"""Аппын төлөв: өгөгдлийн эх сурвалж, Analysis-ийг кэшлэх, хүний засвар (overrides)."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from finagent.classify import apply_overrides
from finagent.ingest import IngestError, read_many
from finagent.pipeline import Analysis, load_balances
from finagent.store import Store

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"

KEY_FILES = "uploaded_files"  # list[(name, bytes)]
KEY_BALANCES = "balances"  # DataFrame | None
KEY_OVERRIDES = "overrides"  # dict[txn_id, line]
KEY_SOURCE = "source"  # "sample" | "upload" | "saved"
KEY_COMPANY = "company"  # хадгалсан компанийн нэр (source == "saved")
KEY_DB_VERSION = "db_version"  # хадгалах бүрт нэмэгдэж кэшийг шинэчилнэ


def _init() -> None:
    st.session_state.setdefault(KEY_SOURCE, "sample")
    st.session_state.setdefault(KEY_FILES, [])
    st.session_state.setdefault(KEY_BALANCES, None)
    st.session_state.setdefault(KEY_OVERRIDES, {})
    st.session_state.setdefault(KEY_COMPANY, None)
    st.session_state.setdefault(KEY_DB_VERSION, 0)


@st.cache_resource
def get_store() -> Store:
    """DATABASE_URL (анхдагч: data/finagent.sqlite)."""
    return Store()


@st.cache_data(show_spinner="Хадгалсан өгөгдлийг ачаалж байна…")
def _analyze_saved(company: str, overrides: tuple[tuple[str, str], ...],
                   _version: int) -> Analysis | None:
    store = get_store()
    ledger = store.load_ledger(company)
    if ledger.empty:
        return None
    a = Analysis(apply_overrides(ledger, dict(overrides)), store.load_balances(company))
    _ = a.summary, a.issues, a.forecast, a.confidence, a.safe_ledger
    return a


def bump_db_version() -> None:
    st.session_state[KEY_DB_VERSION] = st.session_state.get(KEY_DB_VERSION, 0) + 1
    _analyze_saved.clear()


def sample_files() -> list[tuple[str, bytes]]:
    return [(p.name, p.read_bytes()) for p in sorted(SAMPLE_DIR.glob("transactions_*.xlsx"))]


def sample_balances() -> pd.DataFrame:
    return load_balances(SAMPLE_DIR / "balances.csv")


@st.cache_data(show_spinner="Тооцоолж байна…")
def _analyze(files: tuple[tuple[str, bytes], ...], balances_csv: str | None,
             overrides: tuple[tuple[str, str], ...]) -> Analysis:
    balances = pd.read_csv(io.StringIO(balances_csv), dtype={"month": str}) \
        if balances_csv else None
    a = Analysis.from_files(list(files), balances, dict(overrides))
    # cached_property-г урьдчилан тооцож кэшэд хадгална
    _ = a.summary, a.issues, a.forecast, a.confidence, a.safe_ledger
    return a


def current_files() -> list[tuple[str, bytes]]:
    _init()
    return sample_files() if st.session_state[KEY_SOURCE] == "sample" \
        else st.session_state[KEY_FILES]


def current_balances() -> pd.DataFrame | None:
    _init()
    if st.session_state[KEY_SOURCE] == "sample":
        return sample_balances()
    return st.session_state[KEY_BALANCES]


def get_analysis() -> Analysis | None:
    """Одоогийн өгөгдлөөр Analysis буцаана; файл байхгүй бол None."""
    _init()
    if st.session_state[KEY_SOURCE] == "saved":
        company = st.session_state[KEY_COMPANY]
        if not company:
            return None
        overrides = tuple(sorted(st.session_state[KEY_OVERRIDES].items()))
        return _analyze_saved(company, overrides, st.session_state[KEY_DB_VERSION])
    files = current_files()
    if not files:
        return None
    balances = current_balances()
    bal_csv = None
    if balances is not None and not balances.dropna(subset=["opening", "closing"]).empty:
        bal_csv = balances.dropna(subset=["opening", "closing"]).to_csv(index=False)
    overrides = tuple(sorted(st.session_state[KEY_OVERRIDES].items()))
    try:
        return _analyze(tuple(files), bal_csv, overrides)
    except IngestError as e:
        st.error(f"Файл уншихад алдаа гарлаа: {e}")
        return None


def months_of(files: list[tuple[str, bytes]]) -> list[str]:
    try:
        parsed = read_many(files)
    except IngestError:
        return []
    return [f"{p.year:04d}-{p.month:02d}" for p in parsed if p.year and p.month]


def set_override(txn_id: str, line: str) -> None:
    _init()
    st.session_state[KEY_OVERRIDES][txn_id] = line


def clear_overrides() -> None:
    st.session_state[KEY_OVERRIDES] = {}
