"""Агентийн хэрэгслүүд (proposal 2.2, "Хэрэгслийн жишээ").

Агент өгөгдөлд шууд хандахгүй — зөвхөн эдгээр функцийг дуудаж жижиг JSON авна.
Гаралтын хэлбэр proposal-ийн жишээтэй ижил:
    - түлхүүр нь монгол, мөнгөн дүн **сая ₮**, 2 оронтой (41.80), хувь нь `_%` (17.4)
    - харилцагчийн нэр гарахгүй (CP-### код), төслийн нэрийг "Төсөл А" болгон нууцална
Оролтод мөрийн монгол нэр ("Төслийн зардал") болон дотоод түлхүүр ("project") хоёулаа болно.

`TOOL_SPECS` нь LLM-ийн tool use-д шууд дамжуулах JSON schema.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pandas as pd

from .checks import RULES
from .forecast import simulate_scenario as _simulate
from .pipeline import Analysis
from .projects import detect_projects, project_totals
from .schema import EXPENSE_LINES, Col, Line, line_label

UNIT = "сая ₮"
LEVELS = {"high": "өндөр", "medium": "дунд", "low": "бага"}
LEVEL_IN = {"өндөр": "high", "дунд": "medium", "бага": "low", "high": "high",
            "medium": "medium", "low": "low"}

# Дүрмийн богино нэр (proposal-ийн жишээ: "R2 Ангилагдаагүй", "R7 Орлого давхар бүртгэгдэх")
RULE_SHORT = {
    "R1": "Бүтэц", "R2": "Ангилагдаагүй", "R3": "Давхардал", "R4": "Зээлийн хүү холилдсон",
    "R5": "Нэг өдөр орж-гарсан", "R6": "Хувийн шинжтэй зардал",
    "R7": "Орлого давхар бүртгэгдэх", "R8": "Үлдэгдэл тулгарахгүй", "R9": "Кассаар дамжсан",
    "A1": "Хэвийн бус гүйлгээ",
}

# Сэдвийн шошго (төсөлгүй том гүйлгээнд)
_TOPICS = [(r"зар сурталчилгаа|zar surtalchilgaa|advert", "Зар сурталчилгаа"),
           (r"зураг авалт|zurag awalt", "Зураг авалт"),
           (r"цалин|tsalin", "Цалин"), (r"түрээс|turees", "Түрээс")]


def _mln(x) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x) / 1e6, 2)


def _pct(x, digits: int = 1) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x) * 100, digits)


def _month_no(m: str) -> str:
    return str(int(m.split("-")[1]))


def _short_date(d) -> str:
    return f"{d.month}/{d.day}"


def _line_aliases() -> dict[str, str]:
    out = {"revenue": "revenue", "нийт орлого": "revenue", "орлого": "revenue"}
    for ln in Line:
        label = line_label(ln)
        for alias in {ln.value, label, label.split(" (")[0]}:
            out[alias.lower()] = ln.value
    out.update({"үаз": Line.OPEX.value, "оффис": Line.OFFICE.value, "цалин": Line.PAYROLL.value,
                "төсөл": Line.PROJECT.value, "түрээс": Line.RENT.value})
    return out


LINE_ALIASES = _line_aliases()
EXPENSE_LABELS = {line_label(ln).split(" (")[0]: ln.value for ln in EXPENSE_LINES}


def resolve_line(name: str) -> str | None:
    return LINE_ALIASES.get(str(name).strip().lower())


# ---------------------------------------------------------------------------------------
def get_quarter_overview(a: Analysis) -> dict:
    """Улирлын сар бүрийн орлого, зардал, ашиг, маржин, мөнгөн урсгал, анхааруулгын тоо."""
    s = a.summary.tail(3)
    rev, prof = s["revenue"].sum(), s["operating_profit"].sum()
    counts = {"өндөр": 0, "дунд": 0, "бага": 0}
    for i in a.issues:
        counts[LEVELS[i.severity]] += 1
    gap = _reconciliation_gap(a)
    level, reasons = a.confidence
    return {
        "сар": [_month_no(m) for m in s.index],
        "орлого": [_mln(v) for v in s["revenue"]],
        "зардал": [_mln(v) for v in s["total_expenses"]],
        "ашиг": [_mln(v) for v in s["operating_profit"]],
        "маржин_%": [_pct(v) for v in s["margin"]],
        "үйл_ажиллагааны_мөнгөн_урсгал": [_mln(v) for v in s["operating_cash_flow"]],
        "санхүүгийн_мөнгөн_урсгал": [_mln(v) for v in s["financing_cash_flow"]],
        "улирал": {"орлого": _mln(rev), "ашиг": _mln(prof),
                   "маржин_%": _pct(prof / rev if rev else None)},
        "анхааруулга": counts,
        "тулгалтын_зөрүү": gap["зөрүү"] if gap else None,
        "итгэл": LEVELS[level],
        "итгэлийн_шалтгаан": reasons,
        "нэгж": UNIT,
    }


def _reconciliation_gap(a: Analysis) -> dict | None:
    r8 = [i for i in a.issues if i.rule == "R8" and i.amount]
    if not r8:
        return None
    return {"зөрүү": _mln(sum(abs(i.amount) for i in r8)),
            "үүссэн_сар": ", ".join(_month_no(i.month) for i in r8 if i.month)}


def _topic(desc: str | None) -> str | None:
    d = (desc or "").lower()
    return next((label for pat, label in _TOPICS if re.search(pat, d)), None)


def get_line_trend(a: Analysis, line: str, top_n: int = 5) -> dict:
    """Нэг мөрийн сар бүрийн дүн, хамгийн том гүйлгээнүүд, (төслийн мөрөнд) төслөөр."""
    key = resolve_line(line)
    if key is None:
        options = sorted({line_label(ln).split(" (")[0] for ln in Line} | {"Нийт орлого"})
        return {"error": f"Тодорхойгүй мөр '{line}'. Боломжит: {options}"}
    led = a.safe_ledger
    if key == "revenue":
        rows = led[led[Col.LINE].isin([Line.SALES.value, Line.OTHER_INCOME.value])]
        label = "Нийт орлого"
    else:
        rows = led[led[Col.LINE] == key]
        label = line_label(key).split(" (")[0]
    months = a.months[-3:]
    by_month = rows.groupby(Col.MONTH)[Col.AMOUNT].sum().reindex(months, fill_value=0.0)
    q_rows = rows[rows[Col.MONTH].isin(months)]
    projects = detect_projects(led)
    top = q_rows.nlargest(top_n, Col.AMOUNT)
    out: dict[str, Any] = {
        "мөр": label,
        "сар": [_month_no(m) for m in months],
        "сарын_дүн": [_mln(v) for v in by_month],
        "улирал": _mln(q_rows[Col.AMOUNT].sum()),
        "хамгийн_том_5": [[_short_date(r[Col.DATE]), _mln(r[Col.AMOUNT]),
                           projects.get(i) if pd.notna(projects.get(i))
                           else (_topic(r[Col.DESCRIPTION]) or r[Col.COUNTERPARTY])]
                          for i, r in top.iterrows()],
    }
    if key == Line.PROJECT.value:
        out["төслөөр"] = {k: _mln(v) for k, v in project_totals(q_rows).items()}
    out["нэгж"] = UNIT
    return out


def get_baseline_forecast(a: Analysis) -> dict:
    """Дараа сарын орлого, зардлын (нийт ба мөр бүрийн) суурь, доод, дээд таамаг."""
    fc = a.forecast
    level, reasons = a.confidence

    def _rng(base, low, high):
        return {"суурь": _mln(base), "доод": _mln(low), "дээд": _mln(high)}

    return {
        "сар": fc.target_month,
        "арга": "жинлэсэн дундаж 1:2:3 + шугаман чиг хандлага",
        "орлого": _rng(fc.revenue.base, fc.revenue.low, fc.revenue.high),
        "зардал": _rng(fc.expenses_base, fc.expenses_low, fc.expenses_high),
        "ашиг": {"суурь": _mln(fc.profit_base), "маржин_%": _pct(fc.margin_base)},
        "мөрөөр": {line_label(k).split(" (")[0]: _rng(f.base, f.low, f.high)
                   for k, f in fc.expenses.items()},
        "итгэл": LEVELS[level],
        "итгэлийн_шалтгаан": reasons,
        "нэгж": UNIT,
    }


def simulate_scenario(a: Analysis, revenue_change_pct: float = 0.0,
                      cost_change_pct: dict[str, float] | None = None) -> dict:
    """Орлого, зардлын мөрийн % өөрчлөлтийн үед дараа сарын ашиг, маржин."""
    expense_pct: dict[str, float] = {}
    bad = []
    for name, pct in (cost_change_pct or {}).items():
        key = "all" if str(name).strip().lower() in {"all", "бүх зардал", "нийт зардал"} \
            else resolve_line(name)
        if key not in {*(ln.value for ln in EXPENSE_LINES), "all"}:
            bad.append(name)
        else:
            expense_pct[key] = float(pct)
    if bad:
        return {"error": f"Тодорхойгүй зардлын мөр: {bad}. Боломжит: "
                         f"{sorted(EXPENSE_LABELS)} эсвэл 'Бүх зардал'"}
    r = _simulate(a.forecast, float(revenue_change_pct), expense_pct)
    return {
        "суурь_ашиг": _mln(r["base_profit"]),
        "сценари_ашиг": _mln(r["profit"]),
        "өөрчлөлт": _mln(r["profit_delta"]),
        "суурь_маржин_%": _pct(a.forecast.margin_base),
        "сценари_маржин_%": _pct(r["margin"]),
        "сценари_орлого": _mln(r["revenue"]),
        "сценари_зардал": _mln(r["total_expenses"]),
        "нэгж": UNIT,
    }


def get_revenue_concentration(a: Analysis) -> dict:
    """Хамгийн том орлогын (борлуулалт) хувь, сарын доторх цаг хугацаа."""
    sales = a.ledger[a.ledger[Col.LINE] == Line.SALES.value]
    months = a.months[-3:]
    n, largest, share, day, top3, late = [], [], [], [], [], []
    for m in months:
        g = sales[sales[Col.MONTH] == m].sort_values(Col.AMOUNT, ascending=False)
        total = g[Col.AMOUNT].sum()
        n.append(int(len(g)))
        if g.empty or not total:
            for col in (largest, share, day, top3, late):
                col.append(None)
            continue
        largest.append(_mln(g[Col.AMOUNT].iloc[0]))
        share.append(round(float(g[Col.AMOUNT].iloc[0] / total) * 100))
        day.append(int(g[Col.DATE].iloc[0].day))
        top3.append(round(float(g[Col.AMOUNT].head(3).sum() / total) * 100))
        late.append(round(float(g[g[Col.DATE].dt.day > 20][Col.AMOUNT].sum() / total) * 100))
    valid = [s for s in share if s is not None]
    return {
        "сар": [_month_no(m) for m in months],
        "орлогын_төлбөрийн_тоо": n,
        "хамгийн_том_төлбөр": largest,
        "хамгийн_томын_хувь_%": share,
        "хамгийн_том_төлбөрийн_өдөр": day,
        "топ3_хувь_%": top3,
        "20-ноос_хойш_орсон_хувь_%": late,
        "чиг": None if len(valid) < 2 else ("буурч байна" if valid[-1] < valid[0]
                                             else "өсөж байна"),
        "нэгж": UNIT,
    }


def get_data_issues(a: Analysis, level: str | None = None, limit: int = 15) -> dict:
    """Шалгалтын дүрмээр илэрсэн анхааруулга (түвшинээр)."""
    sev = LEVEL_IN.get(str(level).strip().lower()) if level else None
    if level and sev is None:
        return {"error": f"Тодорхойгүй түвшин '{level}'. Боломжит: Өндөр, Дунд, Бага"}
    keep = [i for i in a.issues if sev is None or i.severity == sev]
    dates = a.safe_ledger.set_index(Col.TXN_ID)[Col.DATE]
    items = []
    for i in keep[:limit]:
        d = dates.get(i.txn_ids[0]) if i.txn_ids else None
        items.append({
            "дүрэм": f"{i.rule} {RULE_SHORT.get(i.rule, RULES.get(i.rule, ''))}",
            "огноо": _short_date(d) if d is not None and pd.notna(d)
            else (_month_no(i.month) + "-р сар" if i.month else None),
            "дүн": _mln(abs(i.amount)) if i.amount else None,
            "түвшин": LEVELS[i.severity],
            "тайлбар": i.message[:160],
        })
    if len(keep) > limit:
        items.append(f"… бусад {len(keep) - limit}")
    return {
        "тоо": len(keep),
        "жагсаалт": items,
        "тулгалт": _reconciliation_gap(a) or {"зөрүү": 0.0, "үүссэн_сар": None},
        "нэгж": UNIT,
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

_LINE_NAMES = ["Нийт орлого", "Борлуулалтын орлого", *EXPENSE_LABELS, "Зээлийн төлбөр"]

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_quarter_overview",
        "description": "Улирлын сар бүрийн орлого, зардал, ашиг, маржин, мөнгөн урсгал, "
                       "анхааруулгын тоо, тулгалтын зөрүү, итгэлийн түвшин (сая ₮). "
                       "Шинжилгээг үүгээр эхлүүл.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_line_trend",
        "description": "Нэг мөрийн (жишээ нь 'Төслийн зардал') сар бүрийн дүн, хамгийн том 5 "
                       "гүйлгээ; төслийн зардалд төслөөр задаргаа. Өөрчлөлтийн шалтгаан хайхад.",
        "input_schema": {
            "type": "object",
            "properties": {
                "line": {"type": "string", "description": "Мөрийн нэр",
                         "enum": _LINE_NAMES},
            },
            "required": ["line"],
        },
    },
    {
        "name": "get_baseline_forecast",
        "description": "Дараа сарын орлого, нийт зардал, мөр бүрийн суурь/доод/дээд таамаг, "
                       "суурь ашиг ба маржин. Таамаглалын хүрээнд ашигла.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "simulate_scenario",
        "description": "Орлого, зардлын өөрчлөлтийн (%) үед дараа сарын ашиг, маржин. "
                       "Тактик бүрийн ₮ нөлөөг заавал үүгээр тооц.",
        "input_schema": {
            "type": "object",
            "properties": {
                "revenue_change_pct": {"type": "number", "description": "5 = орлого +5%"},
                "cost_change_pct": {
                    "type": "object",
                    "description": "Зардлын мөр → % өөрчлөлт, ж: {\"Төслийн зардал\": -10}",
                    "properties": {k: {"type": "number"}
                                   for k in [*EXPENSE_LABELS, "Бүх зардал"]},
                    "additionalProperties": False,
                },
            },
        },
    },
    {
        "name": "get_revenue_concentration",
        "description": "Сар бүрийн орлогын төлбөрийн тоо, хамгийн том төлбөр ба түүний хувь, "
                       "сарын доторх цаг хугацаа. Орлогын эрсдэл үнэлэхэд.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_data_issues",
        "description": "Шалгалтын дүрмээр (R1–R9, A1) илэрсэн анхааруулга ба тулгалтын зөрүү. "
                       "Итгэлийн түвшин тогтооход.",
        "input_schema": {
            "type": "object",
            "properties": {"level": {"type": "string", "enum": ["Өндөр", "Дунд", "Бага"]}},
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
