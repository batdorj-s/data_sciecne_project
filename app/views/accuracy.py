"""Таамгийн нарийвчлал (BO4): хадгалсан таамгийг бодит дүнтэй тулгана."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from state import KEY_COMPANY, KEY_SOURCE, get_store
from ui import mnt, month_label, page_header, pct, require_analysis

from finagent.forecast import backtest, mape
from finagent.store import forecast_accuracy

a = require_analysis()
page_header("Таамгийн нарийвчлал",
            "Сар бүр хадгалсан таамгийг дараагийн сарын бодит дүнтэй харьцуулна. "
            "Зорилт (BO4): 6+ сарын дараа MAPE ≤ 20%, хүрээнд багтах ≥ 70%, naive-аас бага алдаа.")

store = get_store()
names = store.list_companies()
company = st.session_state.get(KEY_COMPANY) if st.session_state.get(KEY_SOURCE) == "saved" \
    else None
company = st.selectbox("Компани", names, index=names.index(company) if company in names else 0) \
    if names else None

if company:
    saved = store.load_forecasts(company)
    df, m = forecast_accuracy(saved, a.summary)
    if df.empty:
        st.info("Харьцуулах бодит дүн алга. Таамаг хадгалсны дараа тухайн сарын файлыг оруулж, "
                "**Хадгалах** дарна уу.", icon="⏳")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Орлогын MAPE", pct(m["mape"]), help="Зорилт ≤ 20%")
        c2.metric("Хүрээнд багтсан", pct(m["coverage"], 0), help="Зорилт ≥ 70%")
        c3.metric("Naive MAPE", pct(m.get("naive_mape")),
                  help="Өмнөх сарын дүнг таамаг гэж үзсэн жишиг; загвар үүнээс бага "
                       "алдаатай байх ёстой.")
        show = df.assign(
            Сар=df["month"].map(month_label),
            Бодит=df["revenue_actual"].map(mnt), Суурь=df["revenue_base"].map(mnt),
            Хүрээ=[f"{mnt(lo)} … {mnt(hi)}" for lo, hi in zip(df["revenue_low"],
                                                             df["revenue_high"], strict=True)],
            Алдаа=df["revenue_error"].map(pct),
            Багтсан=df["covered"].map({True: "✅", False: "❌"}),
            Naive=df["naive_revenue"].map(mnt),
        )[["Сар", "Бодит", "Суурь", "Хүрээ", "Алдаа", "Багтсан", "Naive"]]
        st.dataframe(show, hide_index=True, use_container_width=True)
    if not saved.empty:
        with st.expander(f"Хадгалсан таамаг ({len(saved)})"):
            st.dataframe(saved[["made_at", "target_month", "months_used", "revenue_base",
                                "revenue_low", "revenue_high", "profit_base"]],
                         hide_index=True, use_container_width=True)
else:
    st.info("Хадгалсан компани алга. **Өгөгдөл** хуудаснаас хадгална уу.")

st.subheader("Түүхэн backtest")
rev = a.summary["revenue"].tolist()
if len(rev) < 4:
    st.caption(f"Одоо {len(rev)} сарын өгөгдөл байна. Rolling-origin backtest хийхэд дор хаяж "
               "4 сар хэрэгтэй (3 сараар сургаж, 4 дэх сарыг таамаглана).")
else:
    bt = backtest(rev)
    c1, c2, c3 = st.columns(3)
    c1.metric("MAPE (жинлэсэн дундаж)", pct(mape(bt["actual"], bt["base"])))
    c2.metric("MAPE (naive)", pct(mape(bt["actual"], bt["naive"])))
    c3.metric("Хүрээнд багтсан", pct(bt["covered"].mean(), 0))
    st.dataframe(pd.DataFrame({
        "Сар": [month_label(a.months[i]) for i in bt["step"]],
        "Бодит": bt["actual"].map(mnt), "Суурь": bt["base"].map(mnt),
        "Naive": bt["naive"].map(mnt), "Багтсан": bt["covered"].map({True: "✅", False: "❌"}),
    }), hide_index=True, use_container_width=True)
