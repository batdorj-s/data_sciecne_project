"""Өгөгдөл оруулах: жишээ өгөгдөл эсвэл өөрийн сарын хуулга + банкны үлдэгдэл."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from state import (
    KEY_BALANCES,
    KEY_FILES,
    KEY_SOURCE,
    clear_overrides,
    get_analysis,
    months_of,
)
from ui import mnt, month_label, page_header

page_header(
    "Өгөгдөл оруулах",
    "Сар бүрийн банкны гүйлгээний файл (.xlsx) оруулахад систем ангилж, шалгаж, тайлан гаргана.",
)

source = st.radio(
    "Өгөгдлийн эх сурвалж",
    options=["sample", "upload"],
    format_func=lambda v: {"sample": "Жишээ: Компани А, 2026 оны I улирал (нууцалсан)",
                           "upload": "Өөрийн файл оруулах"}[v],
    index=0 if st.session_state.get(KEY_SOURCE, "sample") == "sample" else 1,
    horizontal=True,
)
if source != st.session_state.get(KEY_SOURCE):
    st.session_state[KEY_SOURCE] = source
    clear_overrides()

if source == "upload":
    with st.container(border=True):
        st.markdown("**1. Сарын хуулга** — нэг файл = нэг сар. Багана: Өдөр, Харилцагч, Дүн, "
                    "Ангилал, Дэд ангилал, Тайлбар.")
        uploads = st.file_uploader("Файлууд", type=["xlsx", "csv"], accept_multiple_files=True,
                                   label_visibility="collapsed")
        if uploads:
            files = [(u.name, u.getvalue()) for u in uploads]
            if files != st.session_state.get(KEY_FILES):
                st.session_state[KEY_FILES] = files
                clear_overrides()

    files = st.session_state.get(KEY_FILES, [])
    months = months_of(files)
    if months:
        with st.container(border=True):
            st.markdown("**2. Банкны үлдэгдэл** — сар бүрийн эхний ба эцсийн үлдэгдэл "
                        "(R8 тулгалтад). Хоосон орхивол тулгалт хийгдэхгүй.")
            prev = st.session_state.get(KEY_BALANCES)
            base = pd.DataFrame({"month": months, "opening": [None] * len(months),
                                 "closing": [None] * len(months)})
            if prev is not None:
                base = base.drop(columns=["opening", "closing"]).merge(prev, on="month",
                                                                       how="left")
            edited = st.data_editor(
                base, hide_index=True, use_container_width=True, disabled=["month"],
                column_config={
                    "month": st.column_config.TextColumn("Сар"),
                    "opening": st.column_config.NumberColumn("Эхний үлдэгдэл, ₮", format="%,.2f"),
                    "closing": st.column_config.NumberColumn("Эцсийн үлдэгдэл, ₮",
                                                             format="%,.2f"),
                },
                key="balances_editor",
            )
            st.session_state[KEY_BALANCES] = edited

a = get_analysis()
if a is None:
    st.stop()

st.divider()
st.subheader("Уншсан өгөгдөл")
cols = st.columns(len(a.months) or 1)
for col, m in zip(cols, a.months, strict=False):
    row = a.summary.loc[m]
    with col, st.container(border=True):
        st.markdown(f"**{m[:4]} · {month_label(m)}**")
        st.metric("Гүйлгээ", int(row["txn_count"]))
        st.caption(f"Орлого {mnt(row['revenue'])} · Зардал {mnt(row['total_expenses'])}")

high = sum(1 for i in a.issues if i.severity == "high")
c1, c2, c3 = st.columns(3)
c1.metric("Нийт гүйлгээ", len(a.ledger))
c2.metric("Анхааруулга", len(a.issues), delta=f"{high} өндөр" if high else None,
          delta_color="inverse")
c3.metric("Үлдэгдэл тулгалт", "Тулгарсан" if a.reconciled else "Зөрүүтэй")

for pf in a.files:
    for w in pf.warnings:
        st.warning(f"{pf.name}: {w}")

st.page_link("views/dashboard.py", label="Дашбоард руу орох", icon="📊")
