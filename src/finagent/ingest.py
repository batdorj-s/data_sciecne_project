"""Сарын гүйлгээний хуулга (.xlsx / .csv) уншиж нэгдсэн хүснэгт болгох.

Файлын бүтэц (Data Dictionary):
    [гарчиг мөр: "Гүйлгээний дэлгэрэнгүй - 2026 оны 1-р сар"]
    [хоосон мөр]
    Өдөр | Харилцагч | Дүн | Ангилал | Дэд ангилал | Тайлбар
    ...

Толгой мөрийг "Өдөр" гэсэн нүдээр олно; эхний бүрэн хоосон мөрөөр хүснэгт дуусна
(доорх нийлбэрийн томьёог алгасна). Оныг гарчиг, файлын нэр эсвэл `year` параметрээс авна.

Энэ модуль чиглэл (орлого/зарлага) тогтоохгүй — үүнийг `classify` хийнэ.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import BinaryIO

import pandas as pd

_MONTHS_EN = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
    )
}
_TITLE_RE = re.compile(r"(\d{4})\s*оны\s*(\d{1,2})")
_FILENAME_RE = re.compile(r"(\d{4})[_-](\d{1,2})")


# Баганын нэрийн хувилбарууд: бэлтгэсэн хүснэгт ба банкны хуулгын экспорт
COLUMN_ALIASES: dict[str, list[str]] = {
    "date_raw": ["Өдөр", "Огноо", "Гүйлгээний огноо", "Date", "Transaction date"],
    "counterparty": ["Харилцагч", "Харьцсан дансны нэр", "Харьцсан данс", "Хүлээн авагч",
                     "Counterparty"],
    "amount_raw": ["Дүн", "Гүйлгээний дүн", "Amount"],
    "debit_raw": ["Дебит", "Гарсан", "Зарлага", "Debit"],
    "credit_raw": ["Кредит", "Орсон", "Орлого", "Credit"],
    "src_category": ["Ангилал", "Category"],
    "src_subcategory": ["Дэд ангилал", "Subcategory"],
    "description": ["Тайлбар", "Гүйлгээний утга", "Утга", "Description"],
}


def _norm_header(v) -> str:
    return " ".join(str(v).split()).lower() if pd.notna(v) else ""


_ALIAS_LOOKUP = {a.lower(): key for key, names in COLUMN_ALIASES.items() for a in names}


def _map_header(cells: list) -> dict[int, str]:
    """Баганын индекс → дотоод нэр. Нэг нэрийг эхний тохиолдлоор авна."""
    out: dict[int, str] = {}
    for i, c in enumerate(cells):
        key = _ALIAS_LOOKUP.get(_norm_header(c))
        if key and key not in out.values():
            out[i] = key
    return out


def _is_header(mapping: dict[int, str]) -> bool:
    keys = set(mapping.values())
    return "date_raw" in keys and ("amount_raw" in keys or {"debit_raw", "credit_raw"} <= keys)


class IngestError(ValueError):
    """Файлын бүтэц буруу үед."""


@dataclass
class ParsedFile:
    name: str
    year: int | None
    month: int | None
    frame: pd.DataFrame  # дотоод нэртэй баганууд + direction_raw + source_row
    warnings: list[str] = field(default_factory=list)
    fmt: str = "prepared"  # "prepared" (ангилалтай) | "bank" (банкны экспорт)


def _period_from_text(text: str) -> tuple[int | None, int | None]:
    if m := _TITLE_RE.search(text):
        return int(m.group(1)), int(m.group(2))
    return None, None


def _period_from_filename(name: str) -> tuple[int | None, int | None]:
    if m := _FILENAME_RE.search(Path(name).stem):
        y, mo = int(m.group(1)), int(m.group(2))
        if 2000 <= y <= 2100 and 1 <= mo <= 12:
            return y, mo
    return None, None


def parse_date(value, year: int | None) -> date | None:
    """'5-Jan', '2026-01-05', datetime зэрэг хэлбэрийг хүлээн авна."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if not s:
        return None
    m = re.fullmatch(r"(\d{1,2})[-\s/.]([A-Za-z]{3})[A-Za-z]*(?:[-\s/.](\d{2,4}))?", s)
    if m:
        day, mon, yr = int(m.group(1)), _MONTHS_EN.get(m.group(2).lower()), m.group(3)
        if mon is None:
            return None
        y = int(yr) + (2000 if yr and len(yr) == 2 else 0) if yr else year
        if y is None:
            raise IngestError(f"'{s}' огноонд он алга; файлын гарчиг эсвэл нэрэнд он оруулна уу.")
        return date(y, mon, day)
    try:
        return pd.to_datetime(s, dayfirst=False).date()
    except (ValueError, TypeError):
        return None


def _read_grid(source: str | Path | BinaryIO, name: str) -> pd.DataFrame:
    if name.lower().endswith(".csv"):
        return pd.read_csv(source, header=None, dtype=object)
    return pd.read_excel(source, header=None, dtype=object, engine="openpyxl")


def read_statement(
    source: str | Path | BinaryIO | bytes,
    name: str | None = None,
    year: int | None = None,
) -> ParsedFile:
    """Нэг сарын хуулга уншина."""
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    if name is None:
        name = Path(source).name if isinstance(source, (str, Path)) else "uploaded.xlsx"

    grid = _read_grid(source, name)
    header_idx, mapping = None, {}
    for i in range(min(len(grid), 30)):
        m = _map_header(grid.iloc[i].tolist())
        if _is_header(m):
            header_idx, mapping = i, m
            break
    if header_idx is None:
        raise IngestError(f"{name}: огноо ба дүнгийн (эсвэл дебит/кредит) баганатай толгой мөр "
                          "олдсонгүй.")

    title = " ".join(str(c) for c in grid.iloc[:header_idx].stack().tolist())
    y, mo = _period_from_text(title)
    if y is None:
        y, mo = _period_from_filename(name)
    y = year or y

    body = grid.iloc[header_idx + 1:, list(mapping)].copy()
    body.columns = list(mapping.values())
    # Эхний бүрэн хоосон мөрөөр таслана (доор нь нийлбэрийн томьёо байдаг)
    blank = body.isna().all(axis=1).to_numpy()
    if blank.any():
        body = body.iloc[: int(blank.argmax())]
    for c in COLUMN_ALIASES:
        if c not in body:
            body[c] = None
    # Excel-ийн мөрийн дугаар (1-ээс), алдааг хэрэглэгчид заахад хэрэгтэй
    body["source_row"] = body.index + 1
    fmt = "prepared" if "src_category" in mapping.values() else "bank"

    warnings: list[str] = []
    if y is None:
        warnings.append("Он тодорхойгүй: гарчиг/файлын нэрэнд он байхгүй.")

    def _parse(v):
        try:
            return parse_date(v, y)
        except IngestError as e:
            warnings.append(str(e))
            return None

    body["date"] = body["date_raw"].map(_parse)
    amount = pd.to_numeric(body["amount_raw"], errors="coerce")
    debit = pd.to_numeric(body["debit_raw"], errors="coerce").fillna(0).abs()
    credit = pd.to_numeric(body["credit_raw"], errors="coerce").fillna(0).abs()
    has_dc = body["debit_raw"].notna() | body["credit_raw"].notna()
    # Чиглэл: дебит/кредит багана эсвэл тэмдэгтэй дүнгээс; бэлтгэсэн хүснэгтэд None
    direction = pd.Series([None] * len(body), index=body.index, dtype=object)
    direction[has_dc & (credit > 0)] = "in"
    direction[has_dc & (debit > 0)] = "out"
    amount = amount.where(~has_dc, credit - debit)
    if fmt == "bank":
        direction = direction.where(direction.notna(),
                                    amount.map(lambda v: None if pd.isna(v)
                                               else ("out" if v < 0 else "in")))
    body["direction_raw"] = direction
    body["amount"] = amount.abs()
    for c in ["counterparty", "src_category", "src_subcategory", "description"]:
        body[c] = body[c].map(lambda v: " ".join(str(v).split()) if pd.notna(v) else None)

    if mo is None and body["date"].notna().any():
        first = body["date"].dropna().iloc[0]
        y, mo = y or first.year, first.month

    return ParsedFile(name=name, year=y, month=mo, frame=body.reset_index(drop=True),
                      warnings=warnings, fmt=fmt)


def read_many(files: list[tuple[str, bytes]] | list[Path]) -> list[ParsedFile]:
    out = []
    for f in files:
        if isinstance(f, tuple):
            out.append(read_statement(f[1], name=f[0]))
        else:
            out.append(read_statement(f))
    return sorted(out, key=lambda p: (p.year or 0, p.month or 0, p.name))
