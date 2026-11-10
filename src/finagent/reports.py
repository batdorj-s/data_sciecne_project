"""Мөнгөн суурьтай удирдлагын тайлангууд.

Бүх функц ledger (classify.build_ledger-ийн гаралт) авч, сар бүрээр багана гаргана.
Ашгийг зөвхөн үйл ажиллагааны мөрөөр тооцно; санхүүгийн болон дотоод гүйлгээ
мөнгөн урсгалд л харагдана.
"""

from __future__ import annotations

import pandas as pd

from .schema import EXPENSE_LINES, FINANCING_LINES, INCOME_LINES, Activity, Col, Line, line_label

TOTAL = "Нийт"


def months(ledger: pd.DataFrame) -> list[str]:
    return sorted(ledger[Col.MONTH].dropna().unique().tolist())


def line_matrix(ledger: pd.DataFrame, signed: bool = False) -> pd.DataFrame:
    """мөр × сар дүнгийн хүснэгт (бүх мөр, байхгүй бол 0)."""
    value = Col.SIGNED if signed else Col.AMOUNT
    mat = ledger.pivot_table(index=Col.LINE, columns=Col.MONTH, values=value, aggfunc="sum",
                             fill_value=0.0)
    mat = mat.reindex(index=[ln.value for ln in Line], columns=months(ledger), fill_value=0.0)
    return mat.astype(float)


def monthly_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    """Сар бүрийн гол үзүүлэлт (index = сар)."""
    m = line_matrix(ledger)
    s = line_matrix(ledger, signed=True)
    by_act = ledger.pivot_table(index=Col.ACTIVITY, columns=Col.MONTH, values=Col.SIGNED,
                                aggfunc="sum", fill_value=0.0).reindex(
        index=[a.value for a in Activity], columns=months(ledger), fill_value=0.0)

    revenue = m.loc[[ln.value for ln in INCOME_LINES]].sum()
    opex = m.loc[[ln.value for ln in EXPENSE_LINES]].sum()
    out = pd.DataFrame({
        "revenue": revenue,
        "sales": m.loc[Line.SALES.value],
        "total_expenses": opex,
        "operating_profit": revenue - opex,
        "operating_cash_flow": by_act.loc[Activity.OPERATING.value],
        "financing_cash_flow": by_act.loc[Activity.FINANCING.value],
        "internal_cash_flow": by_act.loc[Activity.INTERNAL.value],
        "unclassified_cash_flow": by_act.loc[Activity.UNCLASSIFIED.value],
        "net_cash_change": s.sum(),
        "txn_count": ledger.groupby(Col.MONTH).size().reindex(months(ledger), fill_value=0),
    })
    out["margin"] = (out["operating_profit"] / out["revenue"]).where(out["revenue"] != 0)
    for ln in EXPENSE_LINES:
        out[ln.value] = m.loc[ln.value]
    out.index.name = "month"
    return out


def _with_total(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TOTAL] = df.sum(axis=1)
    return df


def income_statement(ledger: pd.DataFrame) -> pd.DataFrame:
    """Орлого, зарлагын тайлан (мөнгөн суурь). Мөр = үзүүлэлт, багана = сар + Нийт."""
    m = line_matrix(ledger)
    rows: list[tuple[str, pd.Series]] = []
    for ln in INCOME_LINES:
        rows.append((line_label(ln), m.loc[ln.value]))
    revenue = m.loc[[ln.value for ln in INCOME_LINES]].sum()
    rows.append(("НИЙТ ОРЛОГО", revenue))
    for ln in EXPENSE_LINES:
        rows.append((line_label(ln), m.loc[ln.value]))
    opex = m.loc[[ln.value for ln in EXPENSE_LINES]].sum()
    rows.append(("НИЙТ ҮА ЗАРЛАГА", opex))
    rows.append(("ҮЙЛ АЖИЛЛАГААНЫ АШИГ / АЛДАГДАЛ", revenue - opex))
    df = _with_total(pd.DataFrame({k: v for k, v in rows}).T)
    margin = (df.loc["ҮЙЛ АЖИЛЛАГААНЫ АШИГ / АЛДАГДАЛ"] / df.loc["НИЙТ ОРЛОГО"]).where(
        df.loc["НИЙТ ОРЛОГО"] != 0)
    df.loc["Ашгийн маржин"] = margin
    return df


def cash_flow_statement(ledger: pd.DataFrame,
                        balances: pd.DataFrame | None = None) -> pd.DataFrame:
    """Мөнгөн урсгалын тайлан. balances: month, opening, closing (банкны хуулгаас)."""
    s = line_matrix(ledger, signed=True)
    rows: list[tuple[str, pd.Series]] = []
    op_lines = INCOME_LINES + EXPENSE_LINES
    for ln in op_lines:
        rows.append((line_label(ln), s.loc[ln.value]))
    op = s.loc[[ln.value for ln in op_lines]].sum()
    rows.append(("ҮА цэвэр мөнгөн урсгал", op))
    for ln in FINANCING_LINES:
        rows.append((line_label(ln), s.loc[ln.value]))
    fin = s.loc[[ln.value for ln in FINANCING_LINES]].sum()
    rows.append(("Санхүүгийн цэвэр мөнгөн урсгал", fin))
    internal = s.loc[[Line.TRANSFER_TO_CASH.value, Line.TRANSFER_FROM_CASH.value]].sum()
    rows.append(("Дотоод шилжүүлэг (цэвэр)", internal))
    uncl = s.loc[[Line.UNCLASSIFIED_IN.value, Line.UNCLASSIFIED_OUT.value]].sum()
    rows.append(("Ангилагдаагүй (цэвэр)", uncl))
    net = op + fin + internal + uncl
    rows.append(("МӨНГӨН ХӨРӨНГИЙН ЦЭВЭР ӨӨРЧЛӨЛТ", net))
    df = _with_total(pd.DataFrame({k: v for k, v in rows}).T)

    if balances is not None and not balances.empty:
        b = balances.set_index("month").reindex(s.columns)
        opening, closing = b["opening"], b["closing"]
        df.loc["Эхний үлдэгдэл (банк)"] = opening.tolist() + [opening.iloc[0]]
        df.loc["Эцсийн үлдэгдэл (банк)"] = closing.tolist() + [closing.iloc[-1]]
        expected = (opening + net).tolist()
        df.loc["Эцсийн үлдэгдэл (гүйлгээгээр)"] = expected + [opening.iloc[0] + net.sum()]
        df.loc["Тулгалтын зөрүү"] = df.loc["Эцсийн үлдэгдэл (банк)"] - df.loc[
            "Эцсийн үлдэгдэл (гүйлгээгээр)"]
    return df


def cost_breakdown(ledger: pd.DataFrame) -> pd.DataFrame:
    """Үйл ажиллагааны зардлын задаргаа: дүн ба хувь (зээлгүй)."""
    m = line_matrix(ledger).loc[[ln.value for ln in EXPENSE_LINES]]
    m = _with_total(m)
    share = m / m.sum()
    out = pd.concat({"amount": m, "share": share}, axis=1)
    out.index = [line_label(i) for i in out.index]
    return out
