"""Агентийн хэрэгслүүд (proposal 2.2).

Агент өгөгдөлд шууд хандахгүй — зөвхөн эдгээр функцийг дуудаж жижиг JSON авна.
Мөнгөн дүнг бүхэл ₮-өөр, хувийг 0–1 бутархайгаар (3 оронтой) буцаана.
Харилцагчийн нэр хэзээ ч гарахгүй: зөвхөн `CP-###` код эсвэл "харилцагч #1".

`TOOL_SPECS` нь LLM-ийн tool use-д шууд дамжуулах JSON schema.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from .forecast import simulate_scenario as _simulate
from .pipeline import Analysis
from .schema import EXPENSE_LINES, Col, Line, line_label


def _money(x) -> int | None:
    return None if x is None or pd.isna(x) else int(round(float(x)))


def _pct(x) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x), 3)


# ---------------------------------------------------------------------------------------
def get_quarter_overview(a: Analysis) -> dict:
    """Улирлын сар бүрийн орлого, зардал, ашиг, маржин, мөнгөн урсгал, анхааруулгын тоо."""
    s = a.summary.tail(3)
    issues_by_month: dict[str, dict[str, int]] = {}
    for i in a.issues:
        bucket = issues_by_month.setdefault(i.month or "all", {"high": 0, "medium": 0, "low": 0})
        bucket[i.severity] += 1
    months = []
    for m, r in s.iterrows():
        months.append({
            "month": m,
            "revenue": _money(r["revenue"]),
            "total_expenses": _money(r["total_expenses"]),
            "operating_profit": _money(r["operating_profit"]),
            "margin": _pct(r["margin"]),
            "operating_cash_flow": _money(r["operating_cash_flow"]),
            "financing_cash_flow": _money(r["financing_cash_flow"]),
            "net_cash_change": _money(r["net_cash_change"]),
            "transactions": int(r["txn_count"]),
            "issues": issues_by_month.get(m, {"high": 0, "medium": 0, "low": 0}),
        })
    rev, prof = s["revenue"].sum(), s["operating_profit"].sum()
    level, reasons = a.confidence
    return {
        "months": months,
        "quarter": {
            "revenue": _money(rev),
            "total_expenses": _money(s["total_expenses"].sum()),
            "operating_profit": _money(prof),
            "margin": _pct(prof / rev if rev else None),
        },
        "reconciled": a.reconciled,
        "confidence": level,
        "confidence_reasons": reasons,
        "basis": "мөнгөн суурь (НББОУС-ийн тайланг орлохгүй)",
    }


def get_line_trend(a: Analysis, line: str, top_n: int = 5) -> dict:
    """Нэг мөрийн сар бүрийн дүн ба хамгийн том гүйлгээнүүд."""
    valid = {ln.value for ln in Line} | {"revenue"}
    if line not in valid:
        return {"error": f"Тодорхойгүй мөр '{line}'. Боломжит: {sorted(valid)}"}
    led = a.safe_ledger
    if line == "revenue":
        rows = led[led[Col.LINE].isin([Line.SALES.value, Line.OTHER_INCOME.value])]
        label = "Нийт орлого"
    else:
        rows = led[led[Col.LINE] == line]
        label = line_label(line)
    by_month = rows.groupby(Col.MONTH)[Col.AMOUNT].sum().reindex(a.months, fill_value=0.0)
    changes = by_month.pct_change().replace([float("inf"), float("-inf")], pd.NA)
    top = rows.nlargest(top_n, Col.AMOUNT)
    return {
        "line": line,
        "label": label,
        "monthly": [{"month": m, "amount": _money(v), "change_pct": _pct(c)}
                    for (m, v), c in zip(by_month.items(), changes, strict=True)],
        "top_transactions": [{
            "txn_id": r.txn_id, "date": r.date.strftime("%Y-%m-%d"),
            "counterparty": r.counterparty,
            "amount": _money(r.amount), "description": r.description,
        } for r in top.itertuples(index=False)],
    }


def get_baseline_forecast(a: Analysis) -> dict:
    """Мөр бүрийн дараа сарын суурь, доод, дээд таамаг."""
    fc = a.forecast
    level, reasons = a.confidence

    def _lf(f):
        return {"label": f.label, "base": _money(f.base), "low": _money(f.low),
                "high": _money(f.high), "history": [_money(v) for v in f.history]}

    return {
        "target_month": fc.target_month,
        "method": fc.method,
        "months_used": fc.months_used,
        "revenue": _lf(fc.revenue),
        "expenses": {k: _lf(v) for k, v in fc.expenses.items()},
        "total_expenses_base": _money(fc.expenses_base),
        "profit": {"base": _money(fc.profit_base), "low": _money(fc.profit_low),
                   "high": _money(fc.profit_high)},
        "margin_base": _pct(fc.margin_base),
        "confidence": level,
        "confidence_reasons": reasons,
    }


def simulate_scenario(a: Analysis, revenue_pct: float = 0.0,
                      expense_pct: dict[str, float] | None = None) -> dict:
    """Орлого, зардлын % өөрчлөлтийн үед дараа сарын ашиг, маржин."""
    allowed = {ln.value for ln in EXPENSE_LINES} | {"all"}
    bad = set(expense_pct or {}) - allowed
    if bad:
        return {"error": f"Тодорхойгүй зардлын мөр: {sorted(bad)}. Боломжит: {sorted(allowed)}"}
    r = _simulate(a.forecast, revenue_pct, expense_pct)
    return {
        "target_month": r["target_month"],
        "assumptions": r["assumptions"],
        "revenue": _money(r["revenue"]),
        "total_expenses": _money(r["total_expenses"]),
        "profit": _money(r["profit"]),
        "margin": _pct(r["margin"]),
        "base_profit": _money(r["base_profit"]),
        "profit_delta": _money(r["profit_delta"]),
    }


def get_revenue_concentration(a: Analysis) -> dict:
    """Хамгийн том орлогын (борлуулалт) хувь, сарын доторх цаг хугацаа."""
    sales = a.ledger[a.ledger[Col.LINE] == Line.SALES.value]
    out = []
    for m in a.months:
        g = sales[sales[Col.MONTH] == m].sort_values(Col.AMOUNT, ascending=False)
        total = g[Col.AMOUNT].sum()
        if g.empty or not total:
            out.append({"month": m, "sales": 0, "payments": 0})
            continue
        top = g.iloc[0]
        late = g[g[Col.DATE].dt.day > 20][Col.AMOUNT].sum()
        out.append({
            "month": m,
            "sales": _money(total),
            "payments": int(len(g)),
            "largest_payment": _money(top[Col.AMOUNT]),
            "largest_share": _pct(top[Col.AMOUNT] / total),
            "largest_day": int(top[Col.DATE].day),
            "top3_share": _pct(g[Col.AMOUNT].head(3).sum() / total),
            "after_day_20_share": _pct(late / total),
        })
    shares = [m["largest_share"] for m in out if "largest_share" in m]
    trend = None
    if len(shares) >= 2:
        trend = "буурч байна" if shares[-1] < shares[0] else "өсөж байна"
    return {"months": out, "largest_share_trend": trend}


def get_data_issues(a: Analysis, min_severity: str = "low") -> dict:
    """Шалгалтын дүрмээр илэрсэн анхааруулга (түвшинээр)."""
    order = {"high": 0, "medium": 1, "low": 2}
    keep = [i for i in a.issues if order[i.severity] <= order.get(min_severity, 2)]
    counts = {"high": 0, "medium": 0, "low": 0}
    for i in a.issues:
        counts[i.severity] += 1
    return {
        "counts": counts,
        "reconciled": a.reconciled,
        "issues": [{"rule": i.rule, "title": i.title, "severity": i.severity, "month": i.month,
                    "amount": _money(i.amount), "message": i.message} for i in keep],
    }


# ---------------------------------------------------------------------------------------
TOOLS: dict[str, Callable[..., dict]] = {
    "get_quarter_overview": get_quarter_overview,
    "get_line_trend": get_line_trend,
    "get_baseline_forecast": get_baseline_forecast,
    "simulate_scenario": simulate_scenario,
    "get_revenue_concentration": get_revenue_concentration,
    "get_data_issues": get_data_issues,
}

_EXPENSE_KEYS = [ln.value for ln in EXPENSE_LINES]

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_quarter_overview",
        "description": "Сүүлийн 3 сарын орлого, зардал, ҮА ашиг, маржин, мөнгөн урсгал, "
                       "анхааруулгын тоо, итгэлийн түвшин. Шинжилгээг үүгээр эхлүүл.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_line_trend",
        "description": "Нэг мөрийн сар бүрийн дүн, өөрчлөлт, хамгийн том гүйлгээнүүд. "
                       "Өөрчлөлтийн шалтгаан хайхад.",
        "input_schema": {
            "type": "object",
            "properties": {
                "line": {"type": "string",
                         "enum": ["revenue"] + [ln.value for ln in Line]},
                "top_n": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["line"],
        },
    },
    {
        "name": "get_baseline_forecast",
        "description": "Дараа сарын орлого, зардлын мөр бүрийн суурь/доод/дээд таамаг, "
                       "ашгийн хүрээ.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "simulate_scenario",
        "description": "Орлого ба зардлын мөрийн % өөрчлөлтийн үед дараа сарын ашиг, маржин. "
                       "Тактик бүрийн ₮ нөлөөг үүгээр тооцно.",
        "input_schema": {
            "type": "object",
            "properties": {
                "revenue_pct": {"type": "number", "description": "жишээ: 5 = +5%"},
                "expense_pct": {
                    "type": "object",
                    "properties": {k: {"type": "number"} for k in [*_EXPENSE_KEYS, "all"]},
                    "additionalProperties": False,
                },
            },
        },
    },
    {
        "name": "get_revenue_concentration",
        "description": "Сар бүрийн хамгийн том борлуулалтын төлбөрийн хувь, топ-3 хувь, "
                       "сарын сүүлд орсон хувь. Орлогын эрсдэл үнэлэхэд.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_data_issues",
        "description": "R1–R9 шалгалтын анхааруулга. Итгэлийн түвшин тогтооход.",
        "input_schema": {
            "type": "object",
            "properties": {"min_severity": {"type": "string", "enum": ["high", "medium", "low"]}},
        },
    },
]


def call_tool(a: Analysis, name: str, args: dict | None = None) -> dict:
    if name not in TOOLS:
        return {"error": f"Тодорхойгүй хэрэгсэл '{name}'"}
    try:
        return TOOLS[name](a, **(args or {}))
    except TypeError as e:
        return {"error": f"Буруу аргумент: {e}"}
