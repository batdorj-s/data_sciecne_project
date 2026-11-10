"""Эх ангиллыг нэгдсэн схемд буулгах (DM1-ийн дүрэмд суурилсан эхний хувилбар).

Пилотын гол алдаа нь зээлийн төлбөр зардалд, эздийн зээл орлогод холилдсон явдал байсан.
Тиймээс энд үйл ажиллагааны (ашигт нөлөөлөх) ба санхүүгийн/дотоод гүйлгээг тодорхой салгана.

Эх файлд дүнгийн тэмдэг байхгүй тул чиглэлийг (орсон/гарсан) ангиллаас тогтооно.
Дараагийн шатанд (Sprint 3) TF-IDF + LogReg загвар `class_source="model"` байдлаар нэмэгдэнэ;
хүний засвар `overrides`-ээр орж `class_source="manual"` болно.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import pandas as pd

from .ingest import ParsedFile
from .schema import LINE_ACTIVITY, TXN_COLUMNS, Col, Line

IN, OUT = "in", "out"


def _norm(s: str | None) -> str:
    return " ".join((s or "").lower().split())


# (эх ангилал) -> чиглэл; дэд ангиллаас үл хамаарах
_CATEGORY_LINE: dict[str, tuple[str, Line]] = {
    "үйл ажиллагааны орлого": (IN, Line.SALES),
    "үндсэн үйл ажиллагааны орлого": (IN, Line.SALES),
    "борлуулалтын орлого": (IN, Line.SALES),
    "бусад орлого": (IN, Line.OTHER_INCOME),
    "эздийн зээл": (IN, Line.OWNER_LOAN_IN),
    "зээлийн орлого": (IN, Line.LOAN_RETURN_IN),
    "зээлийн буцаалт": (IN, Line.LOAN_RETURN_IN),
    "татвар хураамж": (OUT, Line.OPEX),
}

_EXPENSE_CATEGORIES = {"зарлага", "үйл ажиллагааны зардал", "зардал"}

# Зарлагын дэд ангилал -> мөр
_EXPENSE_SUB: dict[str, Line] = {
    "цалин": Line.PAYROLL,
    "төсөл": Line.PROJECT,
    "маркетинг": Line.PROJECT,
    "оффис": Line.OFFICE,
    "түрээс": Line.RENT,
    "үаз": Line.OPEX,
    "банкны шимтгэл": Line.OPEX,
    "татвар": Line.OPEX,
    "зээл": Line.LOAN_REPAYMENT,
}


def classify_row(category: str | None, subcategory: str | None,
                 description: str | None) -> tuple[str, Line]:
    """(чиглэл, мөр) буцаана."""
    cat, sub, desc = _norm(category), _norm(subcategory), _norm(description)

    if cat in _CATEGORY_LINE:
        return _CATEGORY_LINE[cat]

    if cat == "дотоод шилжүүлэг":
        if "касснаас банк" in desc or "кассаас банк" in desc:
            return IN, Line.TRANSFER_FROM_CASH
        return OUT, Line.TRANSFER_TO_CASH

    if cat in _EXPENSE_CATEGORIES:
        return OUT, _EXPENSE_SUB.get(sub, Line.UNCLASSIFIED_OUT)

    if "орлого" in cat:
        return IN, Line.UNCLASSIFIED_IN
    return OUT, Line.UNCLASSIFIED_OUT


# Ангилалгүй банкны хуулгад зориулсан түлхүүр үгийн дүрэм (DM1 загварын жишиг)
_KEYWORDS: list[tuple[str, str | None, Line]] = [
    # (regex, чиглэл [None = аль ч], мөр)
    (r"касс", "out", Line.TRANSFER_TO_CASH),
    (r"цалин|tsalin|salary|ндш|ndsh", "out", Line.PAYROLL),
    (r"түрээс|turees|rent", "out", Line.RENT),
    (r"түрээс|turees|rent", "in", Line.OTHER_INCOME),
    (r"зээл|zeel|loan", "out", Line.LOAN_REPAYMENT),
    (r"эздийн зээл|ezdiin zeel", "in", Line.OWNER_LOAN_IN),
    (r"шимтгэл|shimtgel|fee|татвар|tatvar|ehs|ebarimt", "out", Line.OPEX),
    (r"оффис|office|кофе|coffee|ус захиалга|us zahialga", "out", Line.OFFICE),
    (r"төсөл|tusul|tusliin|зураг авалт|zurag awalt|sponsor|зар сурталчилгаа", "out",
     Line.PROJECT),
    (r"subscription|эрх|erh|taxi|такси", "out", Line.OPEX),
]


def classify_by_keywords(description: str | None, direction: str) -> Line:
    desc = _norm(description)
    for pattern, d, line in _KEYWORDS:
        if (d is None or d == direction) and re.search(pattern, desc):
            return line
    return Line.SALES if direction == IN else Line.UNCLASSIFIED_OUT


def _txn_key(month: str | None, r, seen: dict[str, int]) -> str:
    """Мөрийн дугаараас хамааралгүй тогтвортой id: огноо, дүн, тайлбар, ангиллын hash.
    Яг ижил гүйлгээ давтагдвал -2, -3 дагавар авна."""
    raw = f"{r.date}|{r.amount}|{r.description}|{r.src_category}|{r.src_subcategory}"
    h = hashlib.sha1(raw.encode()).hexdigest()[:8]
    base = f"{month or 'NA'}-{h}"
    seen[base] = seen.get(base, 0) + 1
    return base if seen[base] == 1 else f"{base}-{seen[base]}"


def build_ledger(
    files: Iterable[ParsedFile],
    overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Уншсан файлуудаас нэгдсэн гүйлгээний хүснэгт (ledger) үүсгэнэ.

    overrides: {txn_id: line} — хүний баталсан ангилал.
    """
    frames = []
    for pf in files:
        df = pf.frame
        if df.empty:
            continue
        month_fallback = f"{pf.year:04d}-{pf.month:02d}" if pf.year and pf.month else None
        rows = []
        seen: dict[str, int] = {}
        for r in df.itertuples(index=False):
            amount = abs(r.amount) if pd.notna(r.amount) else r.amount
            raw_dir = getattr(r, "direction_raw", None)
            if r.src_category:
                direction, line = classify_row(r.src_category, r.src_subcategory, r.description)
                if raw_dir in (IN, OUT) and raw_dir != direction:
                    # Банкны чиглэл үнэн; зөрвөл ангиллыг түлхүүр үгээр дахин тогтооно
                    direction, line = raw_dir, classify_by_keywords(r.description, raw_dir)
            else:
                direction = raw_dir if raw_dir in (IN, OUT) else OUT
                line = classify_by_keywords(r.description, direction)
            d = pd.Timestamp(r.date) if r.date is not None else pd.NaT
            month = d.strftime("%Y-%m") if pd.notna(d) else month_fallback
            rows.append({
                Col.TXN_ID: _txn_key(month, r, seen),
                Col.DATE: d,
                Col.MONTH: month,
                Col.COUNTERPARTY: r.counterparty,
                Col.AMOUNT: amount,
                Col.DIRECTION: direction,
                Col.SRC_CATEGORY: r.src_category,
                Col.SRC_SUBCATEGORY: r.src_subcategory,
                Col.DESCRIPTION: r.description,
                Col.LINE: line.value,
                Col.CLASS_SOURCE: "rule",
                Col.SOURCE_FILE: pf.name,
                Col.SOURCE_ROW: int(r.source_row),
            })
        frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame(columns=TXN_COLUMNS)
    ledger = pd.concat(frames, ignore_index=True)

    ledger[Col.AMOUNT] = ledger[Col.AMOUNT].astype(float)
    ledger = apply_overrides(ledger, overrides)
    return ledger[TXN_COLUMNS].sort_values([Col.DATE, Col.SOURCE_ROW], kind="stable").reset_index(
        drop=True)


# Мөр бүрийн мөнгөний чиглэл (хүн ангиллыг өөрчлөхөд чиглэл дагаж өөрчлөгдөнө)
_LINE_DIRECTION = {
    Line.SALES: IN, Line.OTHER_INCOME: IN, Line.OWNER_LOAN_IN: IN, Line.LOAN_RETURN_IN: IN,
    Line.PASSTHROUGH_LOAN_IN: IN, Line.TRANSFER_FROM_CASH: IN, Line.UNCLASSIFIED_IN: IN,
}


def apply_overrides(ledger: pd.DataFrame, overrides: dict[str, str] | None) -> pd.DataFrame:
    """Хүний баталсан ангиллыг хэрэглэж, activity/direction/signed_amount-ийг дахин тооцно."""
    ledger = ledger.copy()
    if overrides:
        mask = ledger[Col.TXN_ID].isin(overrides)
        ledger.loc[mask, Col.LINE] = ledger.loc[mask, Col.TXN_ID].map(overrides)
        ledger.loc[mask, Col.CLASS_SOURCE] = "manual"
        ledger.loc[mask, Col.DIRECTION] = ledger.loc[mask, Col.LINE].map(
            lambda x: _LINE_DIRECTION.get(Line(x), OUT))
    ledger[Col.ACTIVITY] = ledger[Col.LINE].map(lambda x: LINE_ACTIVITY[Line(x)].value)
    ledger[Col.SIGNED] = ledger[Col.AMOUNT].where(ledger[Col.DIRECTION] == IN, -ledger[Col.AMOUNT])
    return ledger


def load_ledger(paths, overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """Файлын замуудаас шууд ledger үүсгэх туслах функц."""
    from .ingest import read_many

    return build_ledger(read_many(list(paths)), overrides)
