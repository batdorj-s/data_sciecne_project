"""Дараа сарын суурь таамаг ба сценарийн тооцоо (DM3-ийн суурь түвшин).

Өгөгдлийн уртаас хамаарсан шатлалтай аргачлал (proposal 4.1, DM3):
    3–5 сар : жинлэсэн дундаж (1:2:3) + шугаман чиг хандлагын хүрээ   ← одоо
    6+ сар  : экспоненциал тэгшитгэл (ETS)                           ← дараагийн шат
    12+ сар : улирлын давтагдалтай загвар                             ← дараагийн шат

Суурь  = сүүлийн 3 сарын жинлэсэн дундаж (сүүлийн сард их жин).
Чиг    = сүүлийн утга + сарын дундаж өөрчлөлт.
Хүрээ  = [min(суурь, чиг), max(суурь, чиг)] — нэг тоо биш, тодорхойгүй байдлыг харуулна.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .reports import monthly_summary
from .schema import EXPENSE_LINES, line_label

WEIGHTS = (1, 2, 3)
METHOD = "weighted_avg_1_2_3 + linear_trend"


@dataclass
class LineForecast:
    line: str
    label: str
    history: list[float]
    base: float
    low: float
    high: float
    trend: float

    def to_dict(self) -> dict:
        return asdict(self)


def next_month(month: str) -> str:
    return (pd.Period(month, freq="M") + 1).strftime("%Y-%m")


def weighted_average(values: list[float], weights=WEIGHTS) -> float:
    vals = list(values)[-len(weights):]
    w = list(weights)[-len(vals):]
    return float(np.dot(vals, w) / sum(w))


def linear_trend(values: list[float]) -> float:
    """Сүүлийн утга + сарын дундаж өөрчлөлт (3 цэгт OLS налуутай тэнцүү)."""
    vals = list(values)[-len(WEIGHTS):]
    if len(vals) < 2:
        return float(vals[-1])
    slope = (vals[-1] - vals[0]) / (len(vals) - 1)
    return float(vals[-1] + slope)


def forecast_series(values: list[float], line: str = "", floor: float = 0.0) -> LineForecast:
    base = weighted_average(values)
    trend = max(linear_trend(values), floor)
    return LineForecast(line=line, label=line_label(line), history=[float(v) for v in values],
                        base=base, low=min(base, trend), high=max(base, trend), trend=trend)


def confidence_level(n_months: int, reconciled: bool, high_issues: int) -> tuple[str, list[str]]:
    """Итгэлийн түвшин ба шалтгаан (proposal 2.4)."""
    reasons = []
    if n_months < 3:
        reasons.append(f"өгөгдөл {n_months} сар (< 3)")
    elif n_months < 6:
        reasons.append(f"өгөгдөл {n_months} сар — статистик загварт хүрэлцэхгүй")
    if not reconciled:
        reasons.append("дансны үлдэгдэл тулгарахгүй")
    if high_issues:
        reasons.append(f"өндөр түвшний {high_issues} анхааруулга")
    if n_months < 3 or not reconciled or high_issues:
        return "low", reasons
    if n_months < 6:
        return "medium", reasons
    return "high", reasons


@dataclass
class Forecast:
    target_month: str
    method: str
    months_used: list[str]
    revenue: LineForecast
    expenses: dict[str, LineForecast]

    @property
    def opex_base(self) -> float:
        return sum(f.base for f in self.expenses.values())

    @property
    def profit_base(self) -> float:
        return self.revenue.base - self.opex_base

    @property
    def profit_low(self) -> float:
        return self.revenue.low - sum(f.high for f in self.expenses.values())

    @property
    def profit_high(self) -> float:
        return self.revenue.high - sum(f.low for f in self.expenses.values())

    @property
    def margin_base(self) -> float | None:
        return self.profit_base / self.revenue.base if self.revenue.base else None

    def to_dict(self) -> dict:
        return {
            "target_month": self.target_month,
            "method": self.method,
            "months_used": self.months_used,
            "revenue": self.revenue.to_dict(),
            "expenses": {k: v.to_dict() for k, v in self.expenses.items()},
            "opex_base": self.opex_base,
            "profit_base": self.profit_base,
            "profit_low": self.profit_low,
            "profit_high": self.profit_high,
            "margin_base": self.margin_base,
        }


def baseline_forecast(ledger: pd.DataFrame, window: int = 3) -> Forecast:
    s = monthly_summary(ledger).tail(window)
    if s.empty:
        raise ValueError("Таамаглахад өгөгдөл алга.")
    rev = forecast_series(s["revenue"].tolist(), "revenue")
    rev.label = "Нийт орлого"
    exp = {ln.value: forecast_series(s[ln.value].tolist(), ln.value) for ln in EXPENSE_LINES}
    return Forecast(target_month=next_month(s.index[-1]), method=METHOD,
                    months_used=s.index.tolist(), revenue=rev, expenses=exp)


def simulate_scenario(fc: Forecast, revenue_pct: float = 0.0,
                      expense_pct: dict[str, float] | None = None) -> dict:
    """Орлого / зардлын мөрийн % өөрчлөлтийн үед дараа сарын ашиг, маржин.

    expense_pct: {"project": -10} гэх мэт; "all" түлхүүр бүх зардалд үйлчилнэ.
    """
    expense_pct = expense_pct or {}
    revenue = fc.revenue.base * (1 + revenue_pct / 100)
    lines = {}
    for key, f in fc.expenses.items():
        pct = expense_pct.get(key, expense_pct.get("all", 0.0))
        lines[key] = f.base * (1 + pct / 100)
    opex = sum(lines.values())
    profit = revenue - opex
    return {
        "target_month": fc.target_month,
        "assumptions": {"revenue_pct": revenue_pct, "expense_pct": expense_pct},
        "revenue": revenue,
        "opex": opex,
        "expenses": lines,
        "profit": profit,
        "margin": profit / revenue if revenue else None,
        "base_profit": fc.profit_base,
        "profit_delta": profit - fc.profit_base,
    }


def backtest(values: list[float], min_train: int = 3) -> pd.DataFrame:
    """Rolling-origin backtest: naive (өмнөх сар) vs жинлэсэн дундаж.

    BO4 / DM3-ийн MAPE, coverage-ийг 6+ сарын өгөгдөл хуримтлагдахад хэмжинэ.
    """
    rows = []
    for t in range(min_train, len(values)):
        hist, actual = values[:t], values[t]
        f = forecast_series(hist)
        rows.append({
            "step": t, "actual": actual, "naive": hist[-1], "base": f.base,
            "low": f.low, "high": f.high, "covered": f.low <= actual <= f.high,
        })
    return pd.DataFrame(rows)


def mape(actual, predicted) -> float:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    mask = a != 0
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask]))) if mask.any() else float("nan")

