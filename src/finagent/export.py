"""Тайланг Excel файл болгон гаргах."""

from __future__ import annotations

import io

import pandas as pd

from .checks import issues_frame
from .pipeline import Analysis
from .reports import cash_flow_statement, cost_breakdown, income_statement
from .schema import line_label


def to_excel(a: Analysis) -> bytes:
    buf = io.BytesIO()
    fc = a.forecast
    fc_rows = [{"Мөр": fc.revenue.label, "Суурь": fc.revenue.base, "Доод": fc.revenue.low,
                "Дээд": fc.revenue.high}]
    fc_rows += [{"Мөр": f.label, "Суурь": f.base, "Доод": f.low, "Дээд": f.high}
                for f in fc.expenses.values()]
    fc_rows.append({"Мөр": "ҮА ашиг", "Суурь": fc.profit_base, "Доод": fc.profit_low,
                    "Дээд": fc.profit_high})
    ledger = a.ledger.assign(line_label=a.ledger["line"].map(line_label))
    ledger["date"] = ledger["date"].dt.date

    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        income_statement(a.ledger).to_excel(xw, sheet_name="Орлого-зарлага")
        cash_flow_statement(a.ledger, a.balances).to_excel(xw, sheet_name="Мөнгөн урсгал")
        cost_breakdown(a.ledger).to_excel(xw, sheet_name="Зардлын задаргаа")
        pd.DataFrame(fc_rows).to_excel(xw, sheet_name=f"Таамаг {fc.target_month}", index=False)
        issues_frame(a.issues).drop(columns="txn_ids").to_excel(
            xw, sheet_name="Анхааруулга", index=False)
        ledger.to_excel(xw, sheet_name="Гүйлгээ", index=False)
    return buf.getvalue()
