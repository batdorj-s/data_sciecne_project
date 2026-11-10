"""Тайлан: орлого-зарлага, мөнгөн урсгал, зардлын задаргаа + Excel татах."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from ui import cash_basis_note, month_label, page_header, require_analysis

from finagent.export import to_excel
from finagent.reports import cash_flow_statement, cost_breakdown, income_statement

a = require_analysis()
page_header("Тайлан", "Сар бүр ба улирлын нийлбэр, ₮")


def _rename(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: month_label(c) for c in df.columns if c[:2] == "20"})


def _fmt_money(df: pd.DataFrame, pct_rows: tuple[str, ...] = ()) -> pd.io.formats.style.Styler:
    def f(v, row):
        if pd.isna(v):
            return "—"
        if row in pct_rows:
            return f"{v * 100:.1f}%".replace("-", "−")
        return f"({abs(v):,.0f})" if v < 0 else f"{v:,.0f}"

    out = df.copy().astype(object)
    for r in out.index:
        out.loc[r] = [f(v, r) for v in df.loc[r]]
    bold = {"НИЙТ ОРЛОГО", "НИЙТ ҮА ЗАРЛАГА", "ҮЙЛ АЖИЛЛАГААНЫ АШИГ / АЛДАГДАЛ",
            "ҮА цэвэр мөнгөн урсгал", "Санхүүгийн цэвэр мөнгөн урсгал",
            "МӨНГӨН ХӨРӨНГИЙН ЦЭВЭР ӨӨРЧЛӨЛТ", "Тулгалтын зөрүү"}
    return out.style.apply(
        lambda row: ["font-weight: 700" if row.name in bold else "" for _ in row], axis=1)


t1, t2, t3 = st.tabs(["Орлого, зарлага", "Мөнгөн урсгал", "Зардлын задаргаа"])
with t1:
    st.dataframe(_fmt_money(_rename(income_statement(a.ledger)), ("Ашгийн маржин",)),
                 use_container_width=True, height=460)
    st.caption("Зээлийн төлбөр, эздийн зээл, дотоод шилжүүлэг ашигт тооцогдохгүй — "
               "зөвхөн мөнгөн урсгалд харагдана.")
with t2:
    st.dataframe(_fmt_money(_rename(cash_flow_statement(a.ledger, a.balances))),
                 use_container_width=True, height=780)
    if a.balances is None:
        st.info("Банкны үлдэгдэл оруулбал тулгалтын мөрүүд нэмэгдэнэ.")
with t3:
    cb = cost_breakdown(a.ledger)
    amount, share = _rename(cb["amount"]), _rename(cb["share"])
    table = pd.concat({"₮": amount, "%": share}, axis=1).swaplevel(axis=1).sort_index(
        axis=1, level=0, sort_remaining=False)
    st.dataframe(
        table.style.format(lambda v: f"{v:,.0f}" if v > 1 else f"{v * 100:.1f}%"),
        use_container_width=True,
    )

st.download_button("Excel тайлан татах", data=to_excel(a),
                   file_name=f"tailan_{a.months[0]}_{a.months[-1]}.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                   icon="⬇️")
cash_basis_note()
