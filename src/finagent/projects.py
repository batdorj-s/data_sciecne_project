"""Төслийн зардлыг төсөл (харилцагчийн захиалга) бүрээр бүлэглэх.

Маркетингийн агентлагийн төслийн зардлын тайлбарт ихэвчлэн захиалагчийн брэнд нэр
давтагдана ("MCA EVENT ...", "ZURAG AWALTIIN HOOL RUBY TUSUL"). Төслийн мөрийн тайлбараас
≥ 3 удаа давтагдсан латин үгийг төсөл гэж үзээд, нийт дүнгээр нь эрэмбэлж
"Төсөл А", "Төсөл Б" ... гэж нууцална (захиалагчийн нэр агент руу очихгүй).

Гараар тохируулах бол `detect_projects(..., keywords={"MCA": "Төсөл А"})`.
"""

from __future__ import annotations

import re

import pandas as pd

from .schema import Col, Line

MIN_MENTIONS = 3
LETTERS = "АБВГДЕЖЗИКЛМНОП"

# Төсөл биш, ерөнхий үйл ажиллагааг заадаг (галигласан) үгс
STOPWORDS = {
    "ZURAG", "AWALT", "AWALTIIN", "TUSLIIN", "TUSUL", "ZARDAL", "TAXI", "HOOL", "BENZIN",
    "MASHIN", "OFFICE", "TULBUR", "TULULT", "ZAHIALGA", "ERH", "VOICE", "SET", "ZASAL", "THE",
    "AND", "FOR", "PAYMENT", "PAYMEN", "SPONSOR", "DOTOOD", "UULZALT", "ZURAGCHIN", "TSALIN",
    "HAAN", "KHAN", "BANK", "SNACK", "COFFEE", "OVER", "DAY",
}


def _tokens(text: str | None) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z]{3,}", text or "") if w.upper() not in STOPWORDS]


def detect_projects(ledger: pd.DataFrame, keywords: dict[str, str] | None = None
                    ) -> pd.Series:
    """Төслийн мөрийн гүйлгээ бүрийн төслийн нэрийг (Төсөл А …) буцаана; бусад нь NaN."""
    rows = ledger[ledger[Col.LINE] == Line.PROJECT.value]
    out = pd.Series(pd.NA, index=ledger.index, dtype="object")
    if rows.empty:
        return out
    toks = rows[Col.DESCRIPTION].map(lambda d: {t.upper() for t in _tokens(d)})
    if keywords:
        cands = {k.upper(): v for k, v in keywords.items()}
    else:
        counts: dict[str, int] = {}
        for ts in toks:
            for t in ts:
                counts[t] = counts.get(t, 0) + 1
        cands = {t: None for t, c in counts.items() if c >= MIN_MENTIONS}
    totals = {c: float(rows.loc[toks.map(lambda ts, c=c: c in ts), Col.AMOUNT].sum())
              for c in cands}
    ranked = sorted(totals, key=totals.get, reverse=True)
    names = {c: cands[c] or f"Төсөл {LETTERS[i]}" for i, c in enumerate(ranked)
             if i < len(LETTERS)}
    for idx, ts in toks.items():
        hit = [c for c in ranked if c in ts]
        if hit:
            out.loc[idx] = names.get(hit[0])
    return out


def project_totals(ledger: pd.DataFrame, min_share: float = 0.10) -> dict[str, float]:
    """Төслийн зардлын ≥ min_share эзэлсэн төслүүдийн нийт дүн (₮), их нь эхэндээ."""
    proj = detect_projects(ledger)
    rows = ledger[ledger[Col.LINE] == Line.PROJECT.value]
    total = float(rows[Col.AMOUNT].sum())
    if not total:
        return {}
    sums = rows[Col.AMOUNT].groupby(proj.loc[rows.index]).sum().sort_values(ascending=False)
    return {k: float(v) for k, v in sums.items() if v / total >= min_share}
