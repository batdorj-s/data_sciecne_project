"""Дашбоард: улирлын гол үзүүлэлт, чиг хандлага, зардлын бүтэц, орлогын төвлөрөл."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from ui import (
    C_EXPENSE,
    C_LOSS,
    C_PROFIT,
    C_REVENUE,
    C_SERIES,
    cash_basis_note,
    confidence_banner,
    mnt,
    month_label,
    page_header,
    pct,
    require_analysis,
    style_layout,
)

from finagent.schema import EXPENSE_LINES, line_label
from finagent.tools import get_revenue_concentration

a = require_analysis()
s = a.summary.tail(3)
fc = a.forecast
labels = [month_label(m) for m in s.index]
target = month_label(fc.target_month)

page_header("Дашбоард", f"{s.index[0]} – {s.index[-1]} · дараа сар: {fc.target_month}")
confidence_banner(*a.confidence)

# --- KPI -------------------------------------------------------------------------------
rev, prof = s["revenue"].sum(), s["operating_profit"].sum()
last, prev = s.iloc[-1], s.iloc[-2] if len(s) > 1 else None
high = sum(1 for i in a.issues if i.severity == "high")
k1, k2, k3, k4 = st.columns(4)
with k1, st.container(border=True):
    st.metric("Улирлын орлого, ₮", mnt(rev, unit=False),
              delta=pct(last["revenue"] / prev["revenue"] - 1) + " сүүлийн сар"
              if prev is not None and prev["revenue"] else None)
with k2, st.container(border=True):
    st.metric("ҮА ашиг, ₮", mnt(prof, unit=False))
    st.caption(f"маржин {pct(prof / rev if rev else None)}")
with k3, st.container(border=True):
    st.metric(f"{target}: суурь ашиг, ₮", mnt(fc.profit_base, unit=False))
    st.caption(f"маржин {pct(fc.margin_base)} · зардал {mnt(fc.expenses_low)} … "
               f"{mnt(fc.expenses_high)}")
with k4, st.container(border=True):
    st.metric("Анхааруулга", len(a.issues))
    st.caption(f"🔴 {high} өндөр түвшин" if high else "өндөр түвшин алга")

# --- Орлого ба зардал + таамаг ------------------------------------------------------------
c1, c2 = st.columns([3, 2])
with c1, st.container(border=True):
    fig = go.Figure()
    fig.add_bar(name="Орлого", x=labels, y=s["revenue"] / 1e6, marker_color=C_REVENUE,
                hovertemplate="%{x}: %{y:.1f} сая ₮<extra>Орлого</extra>")
    fig.add_bar(name="ҮА зардал", x=labels, y=s["total_expenses"] / 1e6, marker_color=C_EXPENSE,
                hovertemplate="%{x}: %{y:.1f} сая ₮<extra>ҮА зардал</extra>")
    # Таамаг: суурь + хүрээ (error bar), тод бус дүүргэлттэй
    fig.add_bar(
        name=f"Орлогын таамаг ({target})", x=[target], y=[fc.revenue.base / 1e6],
        marker=dict(color="rgba(42,120,214,0.25)", line=dict(color=C_REVENUE, width=1.5)),
        error_y=dict(type="data", symmetric=False,
                     array=[(fc.revenue.high - fc.revenue.base) / 1e6],
                     arrayminus=[(fc.revenue.base - fc.revenue.low) / 1e6], color=C_REVENUE),
        hovertemplate=(f"суурь %{{y:.1f}} сая ₮<br>хүрээ {fc.revenue.low / 1e6:.1f}–"
                       f"{fc.revenue.high / 1e6:.1f}<extra></extra>"),
    )
    fig.add_bar(
        name=f"Зардлын таамаг ({target})", x=[target], y=[fc.expenses_base / 1e6],
        marker=dict(color="rgba(235,104,52,0.25)", line=dict(color=C_EXPENSE, width=1.5)),
        hovertemplate="суурь %{y:.1f} сая ₮<extra></extra>",
    )
    fig.update_layout(title="Орлого ба үйл ажиллагааны зардал, сая ₮", barmode="group",
                      bargap=0.3, bargroupgap=0.08)
    st.plotly_chart(style_layout(fig), use_container_width=True)

with c2, st.container(border=True):
    y = s["operating_profit"] / 1e6
    fig = go.Figure(go.Bar(
        x=labels + [target], y=list(y) + [fc.profit_base / 1e6],
        marker=dict(
            color=[C_PROFIT if v >= 0 else C_LOSS for v in y] + ["rgba(27,175,122,0.25)"],
            line=dict(color=[C_PROFIT if v >= 0 else C_LOSS for v in y] + [C_PROFIT], width=1),
        ),
        text=[pct(m, 0) for m in s["margin"]] + [pct(fc.margin_base, 0)],
        textposition="outside",
        hovertemplate="%{x}: %{y:.1f} сая ₮<extra>ҮА ашиг</extra>",
    ))
    fig.update_layout(title="Үйл ажиллагааны ашиг ба маржин, сая ₮")
    st.plotly_chart(style_layout(fig, legend=False), use_container_width=True)

# --- Зардлын бүтэц ба мөнгөн урсгал --------------------------------------------------------
c3, c4 = st.columns(2)
with c3, st.container(border=True):
    fig = go.Figure()
    for color, ln in zip(C_SERIES, EXPENSE_LINES, strict=False):
        fig.add_bar(name=line_label(ln).split(" (")[0], x=labels, y=s[ln.value] / 1e6,
                    marker=dict(color=color, line=dict(color="white", width=2)),
                    hovertemplate="%{x}: %{y:.1f} сая ₮<extra>%{fullData.name}</extra>")
    fig.update_layout(title="Зардлын бүтэц, сая ₮", barmode="stack", bargap=0.45)
    st.plotly_chart(style_layout(fig), use_container_width=True)

with c4, st.container(border=True):
    fig = go.Figure()
    flows = [("ҮА мөнгөн урсгал", "operating_cash_flow", C_SERIES[0]),
             ("Санхүүгийн", "financing_cash_flow", C_SERIES[1]),
             ("Дотоод + ангилагдаагүй", None, C_SERIES[2])]
    for name, col, color in flows:
        vals = (s["internal_cash_flow"] + s["unclassified_cash_flow"]) if col is None else s[col]
        fig.add_bar(name=name, x=labels, y=vals / 1e6, marker_color=color,
                    hovertemplate="%{x}: %{y:.1f} сая ₮<extra>" + name + "</extra>")
    fig.add_scatter(name="Цэвэр өөрчлөлт", x=labels, y=s["net_cash_change"] / 1e6,
                    mode="lines+markers", line=dict(color="#0f172a", width=2),
                    marker=dict(size=9, line=dict(color="white", width=2)),
                    hovertemplate="%{x}: %{y:.1f} сая ₮<extra>Цэвэр өөрчлөлт</extra>")
    fig.update_layout(title="Мөнгөн урсгал үйл ажиллагаагаар, сая ₮", barmode="relative",
                      bargap=0.45)
    st.plotly_chart(style_layout(fig), use_container_width=True)

# --- Орлогын төвлөрөл ---------------------------------------------------------------------
conc = get_revenue_concentration(a)
with st.container(border=True):
    st.markdown("**Орлогын төвлөрөл** — хамгийн том нэг төлбөр сарын борлуулалтын хэдэн хувь вэ")
    cols = st.columns(len(conc["сар"]))
    for i, col in enumerate(cols):
        label = f"{conc['сар'][i]}-р сар"
        share = conc["хамгийн_томын_хувь_%"][i]
        if share is None:
            col.metric(label, "—")
            continue
        col.metric(label, f"{share}%",
                   help=f"{conc['орлогын_төлбөрийн_тоо'][i]} төлбөр, хамгийн том нь "
                        f"{conc['хамгийн_том_төлбөр'][i]:.1f} сая ₮ "
                        f"({conc['хамгийн_том_төлбөрийн_өдөр'][i]}-нд); "
                        f"топ-3 = {conc['топ3_хувь_%'][i]}%")
        col.progress(min(1.0, share / 100))

cash_basis_note()
