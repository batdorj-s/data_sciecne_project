"""Өгөгдөл оруулах: жишээ өгөгдөл эсвэл өөрийн сарын хуулга + банкны үлдэгдэл."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from state import (
    KEY_BALANCES,
    KEY_COMPANY,
    KEY_FILES,
    KEY_OVERRIDES,
    KEY_SOURCE,
    bump_db_version,
    clear_overrides,
    get_analysis,
    get_store,
    months_of,
)
from ui import mnt, month_label, page_header

page_header(
    "Өгөгдөл оруулах",
    "Сар бүрийн банкны гүйлгээний файл (.xlsx) оруулахад систем ангилж, шалгаж, тайлан гаргана.",
)

SOURCES = ["sample", "upload", "saved"]
source = st.radio(
    "Өгөгдлийн эх сурвалж",
    options=SOURCES,
    format_func=lambda v: {"sample": "Жишээ: Компани А, 2026 оны I улирал (нууцалсан)",
                           "upload": "Өөрийн файл оруулах",
                           "saved": "Хадгалсан компани"}[v],
    index=SOURCES.index(st.session_state.get(KEY_SOURCE, "sample")),
    horizontal=True,
)
if source != st.session_state.get(KEY_SOURCE):
    st.session_state[KEY_SOURCE] = source
    clear_overrides()

if source == "saved":
    names = get_store().list_companies()
    if not names:
        st.info("Хадгалсан компани алга. Жишээ эсвэл өөрийн файлаа ачаалаад доороос хадгална уу.")
        st.stop()
    cur = st.session_state.get(KEY_COMPANY)
    company = st.selectbox("Компани", names, index=names.index(cur) if cur in names else 0)
    if company != cur:
        st.session_state[KEY_COMPANY] = company
        clear_overrides()

if source == "upload":
    with st.container(border=True):
        st.markdown("**1. Сарын хуулга** — нэг файл = нэг сар. Бэлтгэсэн хүснэгт (Өдөр, "
                    "Харилцагч, Дүн, Ангилал, Дэд ангилал, Тайлбар) эсвэл банкны экспорт "
                    "(Гүйлгээний огноо, Дебит, Кредит, Гүйлгээний утга).")
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

# --- Хадгалах ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("**Хадгалах** — сарын өгөгдөл, үлдэгдэл, засвар болон дараа сарын таамгийг "
                "хадгална. Дараагийн сарын файл ороход таамгийг бодит дүнтэй харьцуулна.")
    default = st.session_state.get(KEY_COMPANY) if source == "saved" else (
        "Компани А (жишээ)" if source == "sample" else st.session_state.get("upload_company"))
    c1, c2 = st.columns([3, 1])
    name = c1.text_input("Компанийн нэр", value=default or "", placeholder="ж: Компани А")
    c2.write("")
    if c2.button("Хадгалах", type="primary", disabled=not name.strip()):
        store = get_store()
        n = store.save_ledger(name.strip(), a.ledger, a.balances,
                              st.session_state.get(KEY_OVERRIDES))
        naive = float(a.summary["revenue"].iloc[-1])
        store.save_forecast(name.strip(), a.forecast, naive_revenue=naive)
        bump_db_version()
        st.session_state["upload_company"] = name.strip()
        st.success(f"{name.strip()}: {n} гүйлгээ ба {a.forecast.target_month}-ийн таамаг "
                   "хадгалагдлаа.")

st.page_link("views/dashboard.py", label="Дашбоард руу орох", icon="📊")
