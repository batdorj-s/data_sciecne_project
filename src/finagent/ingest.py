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

from .schema import RAW_COLUMNS, REQUIRED_RAW

_MONTHS_EN = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
    )
}
_TITLE_RE = re.compile(r"(\d{4})\s*оны\s*(\d{1,2})")
_FILENAME_RE = re.compile(r"(\d{4})[_-](\d{1,2})")


class IngestError(ValueError):
    """Файлын бүтэц буруу үед."""


@dataclass
class ParsedFile:
    name: str
    year: int | None
    month: int | None
    frame: pd.DataFrame  # RAW_COLUMNS-ийн англи нэртэй баганууд + source_row
    warnings: list[str] = field(default_factory=list)


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
    header_idx = None
    for i in range(min(len(grid), 30)):
        cells = [str(c).strip() for c in grid.iloc[i].tolist() if pd.notna(c)]
        if "Өдөр" in cells and "Дүн" in cells:
            header_idx = i
            break
    if header_idx is None:
        raise IngestError(f"{name}: 'Өдөр', 'Дүн' баганатай толгой мөр олдсонгүй.")

    title = " ".join(str(c) for c in grid.iloc[:header_idx].stack().tolist())
    y, mo = _period_from_text(title)
    if y is None:
        y, mo = _period_from_filename(name)
    y = year or y

    header = [str(c).strip() if pd.notna(c) else "" for c in grid.iloc[header_idx].tolist()]
    missing = [c for c in REQUIRED_RAW if c not in header]
    if missing:
        raise IngestError(f"{name}: заавал байх багана алга: {', '.join(missing)}")

    body = grid.iloc[header_idx + 1 :].copy()
    body.columns = header
    # Эхний бүрэн хоосон мөрөөр таслана (доор нь нийлбэрийн томьёо байдаг)
    blank = body.isna().all(axis=1).to_numpy()
    if blank.any():
        body = body.iloc[: int(blank.argmax())]
    body = body[[c for c in RAW_COLUMNS if c in header]].rename(columns=RAW_COLUMNS)
    for c in RAW_COLUMNS.values():
        if c not in body:
            body[c] = None
    # Excel-ийн мөрийн дугаар (1-ээс), алдааг хэрэглэгчид заахад хэрэгтэй
    body["source_row"] = body.index + 1

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
    body["amount"] = pd.to_numeric(body["amount_raw"], errors="coerce")
    for c in ["counterparty", "src_category", "src_subcategory", "description"]:
        body[c] = body[c].map(lambda v: " ".join(str(v).split()) if pd.notna(v) else None)

    if mo is None and body["date"].notna().any():
        first = body["date"].dropna().iloc[0]
        y, mo = y or first.year, first.month

    return ParsedFile(name=name, year=y, month=mo, frame=body.reset_index(drop=True),
                      warnings=warnings)


def read_many(files: list[tuple[str, bytes]] | list[Path]) -> list[ParsedFile]:
    out = []
    for f in files:
        if isinstance(f, tuple):
            out.append(read_statement(f[1], name=f[0]))
        else:
            out.append(read_statement(f))
    return sorted(out, key=lambda p: (p.year or 0, p.month or 0, p.name))
