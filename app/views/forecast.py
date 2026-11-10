"""Дараа сарын суурь таамаг ба сценарийн тооцоо."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui import (
    C_EXPENSE,
    C_REVENUE,
    cash_basis_note,
    confidence_banner,
    mnt,
    month_label,
    page_header,
    pct,
    require_analysis,
    style_layout,
)

from finagent.forecast import simulate_scenario
from finagent.schema import line_label

SHORT = {"payroll": "Цалин", "project": "Төсөл", "office": "Оффис", "rent": "Түрээс",
         "opex": "ҮАЗ"}

a = require_analysis()
fc = a.forecast
target = month_label(fc.target_month)
page_header(f"Таамаг ба сценари · {fc.target_month}",
            "Суурь = жинлэсэн дундаж (1:2:3); хүрээ = суурь ба шугаман чиг хандлагын хооронд.")
confidence_banner(*a.confidence)

k1, k2, k3 = st.columns(3)
with k1, st.container(border=True):
    st.metric("Орлого (суурь), ₮", mnt(fc.revenue.base, unit=False))
    st.caption(f"хүрээ {mnt(fc.revenue.low)} … {mnt(fc.revenue.high)}")
with k2, st.container(border=True):
    st.metric("ҮА зардал (суурь), ₮", mnt(fc.expenses_base, unit=False))
    st.caption("мөр бүрийн суурь таамгийн нийлбэр")
with k3, st.container(border=True):
    st.metric("ҮА ашиг (суурь), ₮", mnt(fc.profit_base, unit=False))
    st.caption(f"маржин {pct(fc.margin_base)} · хүрээ {mnt(fc.profit_low)} … "
               f"{mnt(fc.profit_high)}")

# --- Түүх + таамгийн хүрээ ------------------------------------------------------------------
with st.container(border=True):
    hist_x = [month_label(m) for m in fc.months_used]
    fig = go.Figure()
    for name, f, color, fill in (("Орлого", fc.revenue, C_REVENUE, "rgba(42,120,214,0.18)"),):
        fig.add_scatter(name=name, x=hist_x, y=[v / 1e6 for v in f.history], mode="lines+markers",
                        line=dict(color=color, width=2), marker=dict(size=9),
                        hovertemplate="%{x}: %{y:.1f} сая ₮<extra>" + name + "</extra>")
        fig.add_scatter(name=f"{name}: таамгийн хүрээ", x=[hist_x[-1], target, target, hist_x[-1]],
                        y=[f.history[-1] / 1e6, f.high / 1e6, f.low / 1e6, f.history[-1] / 1e6],
                        fill="toself", fillcolor=fill, line=dict(width=0), hoverinfo="skip")
        fig.add_scatter(name=f"{name}: суурь", x=[hist_x[-1], target],
                        y=[f.history[-1] / 1e6, f.base / 1e6], mode="lines+markers",
                        line=dict(color=color, width=2, dash="dash"), marker=dict(size=9),
                        hovertemplate=(f"{target}: суурь %{{y:.1f}} сая ₮ (хүрээ "
                                       f"{f.low / 1e6:.1f}–{f.high / 1e6:.1f})<extra></extra>"))
    exp_hist = [sum(e.history[i] for e in fc.expenses.values()) for i in range(len(hist_x))]
    fig.add_scatter(name="ҮА зардал", x=hist_x + [target],
                    y=[v / 1e6 for v in exp_hist] + [fc.expenses_base / 1e6],
                    mode="lines+markers", line=dict(color=C_EXPENSE, width=2),
                    marker=dict(size=9),
                    hovertemplate="%{x}: %{y:.1f} сая ₮<extra>ҮА зардал</extra>")
    fig.update_layout(title="Орлого ба зардал, сая ₮ · тасархай = таамаг")
    st.plotly_chart(style_layout(fig, height=360), use_container_width=True)

rows = [{"Мөр": fc.revenue.label, **{month_label(m): v for m, v in
                                    zip(fc.months_used, fc.revenue.history, strict=True)},
         "Суурь": fc.revenue.base, "Доод": fc.revenue.low, "Дээд": fc.revenue.high}]
for f in fc.expenses.values():
    rows.append({"Мөр": f.label, **{month_label(m): v for m, v in
                                   zip(fc.months_used, f.history, strict=True)},
                 "Суурь": f.base, "Доод": f.low, "Дээд": f.high})
rows.append({"Мөр": "ҮА ашиг", "Суурь": fc.profit_base, "Доод": fc.profit_low,
             "Дээд": fc.profit_high})
st.dataframe(pd.DataFrame(rows).set_index("Мөр").astype(float)
             .style.format("{:,.0f}", na_rep="—"),
             use_container_width=True)

# --- Сценари -------------------------------------------------------------------------------
st.subheader("Сценарийн тооцоо")
st.caption("Тактик бүрийн ₮ нөлөөг тооцох хэрэгсэл (агентийн `simulate_scenario`-тай ижил).")
with st.container(border=True):
    left, right = st.columns([2, 3])
    with left:
        rev_pct = st.slider("Орлогын өөрчлөлт, %", -50, 50, 0, 1)
        exp_pct = {}
        for key, f in fc.expenses.items():
            exp_pct[key] = st.slider(f"{line_label(key)}, %", -50, 50, 0, 1,
                                     key=f"sc_{key}",
                                     help=f"суурь {mnt(f.base)}")
    sc = simulate_scenario(fc, rev_pct, {k: v for k, v in exp_pct.items() if v})
    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Ашиг", mnt(sc["profit"]), delta=mnt(sc["profit_delta"]))
        m2.metric("Маржин", pct(sc["margin"]),
                  delta=f"{(sc['margin'] - (fc.margin_base or 0)) * 100:+.1f} pp"
                  if sc["margin"] is not None else None)
        m3.metric("Орлого", mnt(sc["revenue"]))
        fig = go.Figure(go.Waterfall(
            x=["Суурь ашиг", "Орлого"] + [SHORT[k] for k in fc.expenses] + ["Сценари"],
            measure=["absolute", "relative"] + ["relative"] * len(fc.expenses) + ["total"],
            y=[fc.profit_base / 1e6, (sc["revenue"] - fc.revenue.base) / 1e6]
              + [-(sc["expenses"][k] - f.base) / 1e6 for k, f in fc.expenses.items()] + [0],
            connector=dict(line=dict(color="rgba(100,116,139,0.4)")),
            increasing=dict(marker_color="#1baf7a"), decreasing=dict(marker_color="#e34948"),
            totals=dict(marker_color=C_REVENUE),
            hovertemplate="%{x}: %{y:+.2f} сая ₮<extra></extra>",
        ))
        fig.update_layout(title="Ашгийн өөрчлөлт, сая ₮")
        st.plotly_chart(style_layout(fig, height=320, legend=False), use_container_width=True)

cash_basis_note()
