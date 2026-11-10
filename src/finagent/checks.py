"""R1–R8 шалгалтын дүрэм ба үлдэгдлийн тулгалт (DM2-ийн дүрэмд суурилсан хэсэг).

Шалгалт агентаас ӨМНӨ ажиллана. Өндөр түвшний анхааруулга эсвэл тулгалт зөрвөл
агентийн дүгнэлт "бага итгэл" гэж заавал тэмдэглэгдэнэ (proposal 2.4).

| Код | Дүрэм | Түвшин |
|-----|-------|--------|
| R1  | Бүтэц: заавал талбар хоосон, огноо/дүн буруу | high |
| R2  | Орлого давхар бүртгэгдэх эрсдэл: банк→касс шилжүүлгийн дараа касс→банк орлого | medium |
| R3  | Ангилагдаагүй гүйлгээ | high (≥ 1 сая ₮) / medium |
| R4  | Давхардсан гүйлгээ (огноо, дүн, тайлбар ижил) | medium |
| R5  | Зээлийн хүү ба үндсэн төлбөр холилдсон | low |
| R6  | Нэг өдөр орж-гарсан ижил дүн | low |
| R7  | Хувийн шинжтэй зардал | low |
| R8  | Дансны үлдэгдлийн тулгалт | high |
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import pandas as pd

from .reports import cash_flow_statement
from .schema import Activity, Col, Line

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

RULES: dict[str, str] = {
    "R1": "Бүтэц ба формат",
    "R2": "Орлого давхар бүртгэгдэх эрсдэл",
    "R3": "Ангилагдаагүй гүйлгээ",
    "R4": "Давхардсан гүйлгээ",
    "R5": "Зээлийн хүү ба үндсэн төлбөр холилдсон",
    "R6": "Нэг өдөр орж-гарсан ижил дүн",
    "R7": "Хувийн шинжтэй зардал",
    "R8": "Дансны үлдэгдлийн тулгалт",
}

# Тохируулж болох босго
LARGE_UNCLASSIFIED = 1_000_000
CASH_CYCLE_DAYS = 7
RECONCILE_TOLERANCE = 10_000  # банкны шимтгэл г.м. жижиг зөрүүг зөвшөөрнө

_INTEREST_RE = re.compile(r"\b(?:h[vu]u|хүү|hvv|huu|interest)\b", re.IGNORECASE)
_PERSONAL_RE = re.compile(
    r"birthday|champagne|cake|төрсөн өдөр|бэлэг|gift|wine|дарс|гоо сайхан|salon|"
    r"single day pass|fitness|фитнес|гэр бүл",
    re.IGNORECASE,
)


@dataclass
class Issue:
    rule: str
    severity: str
    message: str
    month: str | None = None
    amount: float | None = None
    txn_ids: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return RULES[self.rule]

    def to_dict(self) -> dict:
        return {**asdict(self), "title": self.title}


def _r1_structure(led: pd.DataFrame) -> list[Issue]:
    issues = []
    for r in led.itertuples(index=False):
        problems = []
        if pd.isna(r.date):
            problems.append("огноо")
        if not r.counterparty:
            problems.append("харилцагч")
        if not r.src_category:
            problems.append("ангилал")
        if pd.isna(r.amount) or r.amount == 0:
            problems.append("дүн")
        if problems:
            issues.append(Issue("R1", "high", f"Хоосон/буруу талбар: {', '.join(problems)} "
                                f"({r.source_file}, мөр {r.source_row})", r.month, r.amount,
                                [r.txn_id]))
    return issues


def _r2_cash_cycle(led: pd.DataFrame) -> list[Issue]:
    """Банкнаас касс руу гарсан мөнгө хэдэн хоногийн дотор "касс→банк орлого" болж буцаж
    орвол тэр нь шинэ борлуулалт биш, нэг мөнгө байж болно."""
    out_cash = led[led[Col.LINE] == Line.TRANSFER_TO_CASH.value]
    in_sales = led[(led[Col.LINE] == Line.SALES.value)
                   & led[Col.SRC_SUBCATEGORY].fillna("").str.lower().eq("касс")]
    issues = []
    for t in out_cash.itertuples(index=False):
        window = in_sales[(in_sales[Col.DATE] >= t.date)
                          & (in_sales[Col.DATE] <= t.date + pd.Timedelta(days=CASH_CYCLE_DAYS))
                          & (in_sales[Col.AMOUNT] <= t.amount)]
        if window.empty:
            continue
        at_risk = min(t.amount, float(window[Col.AMOUNT].sum()))
        issues.append(Issue(
            "R2", "medium",
            f"{t.date:%m-%d}-нд банкнаас касс руу {t.amount:,.0f} ₮ гарсны дараа "
            f"{CASH_CYCLE_DAYS} хоногт кассаас банк руу {len(window)} удаа "
            f"{window[Col.AMOUNT].sum():,.0f} ₮ орлогоор орсон — нэг мөнгө орлогод давхар "
            "тооцогдож болзошгүй.",
            t.month, at_risk, [t.txn_id, *window[Col.TXN_ID].tolist()]))
    return issues


def _r3_unclassified(led: pd.DataFrame) -> list[Issue]:
    rows = led[led[Col.ACTIVITY] == Activity.UNCLASSIFIED.value]
    return [Issue("R3", "high" if r.amount >= LARGE_UNCLASSIFIED else "medium",
                  f"Ангилагдаагүй: '{r.src_category} / {r.src_subcategory}' — {r.description}",
                  r.month, r.amount, [r.txn_id])
            for r in rows.itertuples(index=False)]


def _r4_duplicates(led: pd.DataFrame) -> list[Issue]:
    key = [Col.DATE, Col.AMOUNT, Col.DESCRIPTION]
    dup = led[led.duplicated(key, keep=False) & led[Col.DESCRIPTION].notna()]
    issues = []
    for _, g in dup.groupby(key, sort=True):
        first = g.iloc[0]
        issues.append(Issue("R4", "medium",
                            f"{len(g)} ижил гүйлгээ: {first[Col.DATE]:%m-%d}, "
                            f"{first[Col.AMOUNT]:,.0f} ₮, '{first[Col.DESCRIPTION]}'",
                            first[Col.MONTH], float(g[Col.AMOUNT].iloc[1:].sum()),
                            g[Col.TXN_ID].tolist()))
    return issues


def _r5_interest(led: pd.DataFrame) -> list[Issue]:
    rows = led[(led[Col.LINE] == Line.LOAN_REPAYMENT.value)
               & led[Col.DESCRIPTION].fillna("").str.contains(_INTEREST_RE)]
    return [Issue("R5", "low",
                  f"Зээлийн хүүгийн төлбөр санхүүгийн үйл ажиллагаанд орсон: '{r.description}'. "
                  "Хүүг ҮА зардал руу салгаж ангилна уу.", r.month, r.amount, [r.txn_id])
            for r in rows.itertuples(index=False)]


def _r6_same_day_round_trip(led: pd.DataFrame) -> list[Issue]:
    issues = []
    for (d, amt), g in led.groupby([Col.DATE, Col.AMOUNT]):
        if set(g[Col.DIRECTION]) >= {"in", "out"}:
            issues.append(Issue("R6", "low",
                                f"{d:%m-%d}-нд {amt:,.0f} ₮ орж, мөн гарсан "
                                f"({', '.join(g[Col.SRC_CATEGORY].astype(str).unique())}).",
                                g[Col.MONTH].iloc[0], float(amt), g[Col.TXN_ID].tolist()))
    return issues


def _r7_personal(led: pd.DataFrame) -> list[Issue]:
    rows = led[(led[Col.DIRECTION] == "out")
               & led[Col.DESCRIPTION].fillna("").str.contains(_PERSONAL_RE)]
    return [Issue("R7", "low", f"Хувийн шинжтэй байж болзошгүй зардал: '{r.description}'",
                  r.month, r.amount, [r.txn_id])
            for r in rows.itertuples(index=False)]


def _r8_reconcile(led: pd.DataFrame, balances: pd.DataFrame | None) -> list[Issue]:
    if balances is None or balances.empty:
        return [Issue("R8", "medium", "Банкны эхний/эцсийн үлдэгдэл оруулаагүй тул тулгалт "
                      "хийгдсэнгүй.")]
    cf = cash_flow_statement(led, balances)
    issues = []
    for month, gap in cf.loc["Тулгалтын зөрүү"].drop("Нийт").items():
        if pd.isna(gap):
            issues.append(Issue("R8", "medium", f"{month}: банкны үлдэгдэл алга.", month))
        elif abs(gap) > RECONCILE_TOLERANCE:
            issues.append(Issue(
                "R8", "high",
                f"{month}: гүйлгээгээр тооцсон эцсийн үлдэгдэл банкныхаас {gap:,.0f} ₮ зөрж "
                "байна — гүйлгээний файл дутуу эсвэл буруу байж болзошгүй.", month, float(gap)))
    return issues


def run_checks(ledger: pd.DataFrame, balances: pd.DataFrame | None = None) -> list[Issue]:
    issues: list[Issue] = []
    issues += _r1_structure(ledger)
    issues += _r2_cash_cycle(ledger)
    issues += _r3_unclassified(ledger)
    issues += _r4_duplicates(ledger)
    issues += _r5_interest(ledger)
    issues += _r6_same_day_round_trip(ledger)
    issues += _r7_personal(ledger)
    issues += _r8_reconcile(ledger, balances)
    return sorted(issues, key=lambda i: (SEVERITY_ORDER[i.severity], i.rule, i.month or ""))


def issues_frame(issues: list[Issue]) -> pd.DataFrame:
    cols = ["rule", "title", "severity", "month", "amount", "message", "txn_ids"]
    return pd.DataFrame([i.to_dict() for i in issues], columns=cols)


def reconciled(issues: list[Issue]) -> bool:
    return not any(i.rule == "R8" and i.severity == "high" for i in issues)
