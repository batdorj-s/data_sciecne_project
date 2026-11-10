"""Шалгалт (R1–R9) ба ангиллын хүний баталгаажуулалт."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from state import KEY_OVERRIDES, clear_overrides, get_classifier, set_override
from ui import SEVERITY_ICON, SEVERITY_LABEL, mnt, page_header, require_analysis

from finagent.checks import RULES
from finagent.schema import Line, line_label

a = require_analysis()
page_header("Шалгалт ба ангилал",
            "Агент ажиллахаас өмнө өгөгдлийн алдааг илрүүлж, ангиллыг хүн батална.")

counts = {s: sum(1 for i in a.issues if i.severity == s) for s in ("high", "medium", "low")}
c = st.columns(4)
for col, s in zip(c, ("high", "medium", "low"), strict=False):
    col.metric(f"{SEVERITY_ICON[s]} {SEVERITY_LABEL[s]}", counts[s])
c[3].metric("Үлдэгдлийн тулгалт", "Тулгарсан" if a.reconciled else "Зөрүүтэй")

tab_issues, tab_ledger, tab_model = st.tabs(
    ["Анхааруулга", "Гүйлгээ ба ангилал", "Загварын санал (DM1)"])

with tab_issues:
    f1, f2 = st.columns(2)
    sev = f1.multiselect("Түвшин", ["high", "medium", "low"], default=["high", "medium", "low"],
                         format_func=lambda s: SEVERITY_LABEL[s])
    rules = f2.multiselect("Дүрэм", list(RULES), default=list(RULES),
                           format_func=lambda r: f"{r} · {RULES[r]}")
    shown = [i for i in a.issues if i.severity in sev and i.rule in rules]
    if not shown:
        st.success("Сонгосон шүүлтүүрээр анхааруулга алга.")
    for i in shown:
        with st.expander(f"{SEVERITY_ICON[i.severity]} **{i.rule} · {i.title}** — "
                         f"{i.month or ''} {('· ' + mnt(i.amount)) if i.amount else ''}"):
            st.write(i.message)
            if i.txn_ids:
                rows = a.ledger[a.ledger["txn_id"].isin(i.txn_ids)]
                st.dataframe(rows[["txn_id", "date", "counterparty", "amount", "src_category",
                                   "src_subcategory", "description"]],
                             hide_index=True, use_container_width=True)

with tab_ledger:
    st.caption("**Нэгдсэн ангилал** баганыг засаж хадгалахад тайлан, таамаг шууд шинэчлэгдэнэ. "
               "Засвар 'manual' гэж тэмдэглэгдэж, ангиллын загварын (DM1) шошго болно.")
    lines = [ln.value for ln in Line]
    only_flag = st.toggle("Зөвхөн анхааруулгатай / ангилагдаагүй гүйлгээ", value=False)
    flagged = {t for i in a.issues for t in i.txn_ids}
    view = a.ledger.copy()
    if only_flag:
        view = view[view["txn_id"].isin(flagged) | (view["activity"] == "unclassified")]
    view = view[["txn_id", "date", "counterparty", "amount", "direction", "src_category",
                 "src_subcategory", "description", "line", "activity", "class_source"]]
    edited = st.data_editor(
        view, hide_index=True, use_container_width=True, height=520,
        disabled=[c for c in view.columns if c != "line"],
        column_config={
            "date": st.column_config.DateColumn("Огноо", format="YYYY-MM-DD"),
            "amount": st.column_config.NumberColumn("Дүн, ₮", format="%,.0f"),
            "direction": st.column_config.TextColumn("Чиглэл"),
            "line": st.column_config.SelectboxColumn(
                "Нэгдсэн ангилал", options=lines, required=True,
                help="; ".join(f"{k} = {line_label(k)}" for k in lines)),
            "activity": "Төрөл", "class_source": "Эх",
        },
        key="ledger_editor",
    )
    changed = edited[edited["line"] != view["line"]]
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Засварыг хадгалах", type="primary", disabled=changed.empty):
        for r in changed.itertuples(index=False):
            set_override(r.txn_id, r.line)
        st.rerun()
    if b2.button("Засвар цэвэрлэх", disabled=not st.session_state.get(KEY_OVERRIDES)):
        clear_overrides()
        st.rerun()
    if not changed.empty:
        st.info(f"{len(changed)} гүйлгээний ангилал өөрчлөгдсөн — хадгална уу.")
    ov = st.session_state.get(KEY_OVERRIDES, {})
    if ov:
        st.caption(f"Хадгалсан засвар: {len(ov)}")
        st.dataframe(pd.DataFrame(
            [{"txn_id": k, "line": v, "ангилал": line_label(v)} for k, v in ov.items()]),
            hide_index=True)

with tab_model:
    from finagent.ml.classifier import CONFIDENCE_THRESHOLD, suggest

    st.caption("TF-IDF + Logistic Regression загвар ангилагдаагүй, загвараар ангилсан болон "
               "түлхүүр үгээр ангилсан гүйлгээнд санал гаргана. Итгэл "
               f"≥ {CONFIDENCE_THRESHOLD:.0%} бол автоматаар хэрэглэгдсэн ('model'); бусдыг "
               "хүн батална. Баталсан засвар нь дараагийн сургалтын шошго болно.")
    led = a.ledger
    target = led[(led["activity"] == "unclassified") | (led["class_source"] == "model")
                 | led["src_category"].isna()]
    if target.empty:
        st.success("Загварын санал шаардлагатай гүйлгээ алга — бүх гүйлгээ нягтлангийн "
                   "ангилалтай.")
    else:
        sug = suggest(get_classifier(), target).merge(
            target[["txn_id", "date", "amount", "direction", "description", "line",
                    "class_source"]], on="txn_id")
        sug.insert(0, "Батлах", ~sug["confident"] & (sug["predicted"] != sug["line"]))
        edited_s = st.data_editor(
            sug[["Батлах", "txn_id", "date", "amount", "direction", "description", "line",
                 "predicted", "proba", "class_source"]],
            hide_index=True, use_container_width=True, disabled=["txn_id", "date", "amount",
                                                                  "direction", "description",
                                                                  "line", "proba",
                                                                  "class_source"],
            column_config={
                "proba": st.column_config.ProgressColumn("Итгэл", min_value=0, max_value=1,
                                                         format="percent"),
                "predicted": st.column_config.SelectboxColumn(
                    "Загварын санал", options=[ln.value for ln in Line]),
                "line": "Одоогийн", "amount": st.column_config.NumberColumn(format="%,.0f"),
            },
            key="model_editor",
        )
        chosen = edited_s[edited_s["Батлах"]]
        if st.button(f"Сонгосон {len(chosen)} саналыг батлах", type="primary",
                     disabled=chosen.empty):
            for r in chosen.itertuples(index=False):
                set_override(r.txn_id, r.predicted)
            st.rerun()
