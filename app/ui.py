"""UI туслахууд: тоо форматлах, график загвар, нийтлэг блокууд."""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Категорийн өнгө тогтмол дарааллаар (dataviz reference palette, CVD-д шалгагдсан)
C_REVENUE = "#2a78d6"  # 1 цэнхэр
C_EXPENSE = "#eb6834"  # 2 улбар
C_PROFIT = "#1baf7a"  # 3 ногоовтор — эерэг туйл
C_LOSS = "#e34948"  # сөрөг туйл
C_MUTED = "#94a3b8"
C_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]

SEVERITY_LABEL = {"high": "Өндөр", "medium": "Дунд", "low": "Бага"}
SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}
CONFIDENCE_LABEL = {"low": "Бага итгэл", "medium": "Дунд итгэл", "high": "Өндөр итгэл"}

MONTH_MN = {1: "1-р сар", 2: "2-р сар", 3: "3-р сар", 4: "4-р сар", 5: "5-р сар", 6: "6-р сар",
            7: "7-р сар", 8: "8-р сар", 9: "9-р сар", 10: "10-р сар", 11: "11-р сар",
            12: "12-р сар"}


def month_label(m: str) -> str:
    try:
        return MONTH_MN[int(m.split("-")[1])]
    except (IndexError, ValueError, KeyError):
        return m


def _round_half_up(v: float, digits: int) -> Decimal:
    return Decimal(repr(v)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)


def mnt(x: float | None, digits: int = 1, unit: bool = True) -> str:
    """41_797_000 → '41.8 сая ₮'; 56_550_000 → '56.6 сая ₮'; сөрөг → '−14.9 сая ₮'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    sign = "−" if x < 0 else ""
    v = abs(float(x))
    cur = " ₮" if unit else ""
    if v >= 1e9:
        return f"{sign}{_round_half_up(v / 1e9, digits)} тэрбум{cur}"
    if v >= 1e6:
        return f"{sign}{_round_half_up(v / 1e6, digits)} сая{cur}"
    return f"{sign}{v:,.0f}{cur}"


def pct(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:.{digits}f}%".replace("-", "−")


def millions(df: pd.DataFrame) -> pd.DataFrame:
    return df / 1e6


def style_layout(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=40, b=8),
        title=dict(font=dict(size=15), x=0, xanchor="left", y=0.98, yanchor="top"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0) if legend else None,
        showlegend=legend,
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.25)", zeroline=True,
                     zerolinecolor="rgba(100,116,139,0.5)")
    return fig


def page_header(title: str, caption: str | None = None) -> None:
    st.title(title)
    if caption:
        st.caption(caption)


def confidence_banner(level: str, reasons: list[str]) -> None:
    text = f"**{CONFIDENCE_LABEL[level]}**" + (f" — {'; '.join(reasons)}" if reasons else "")
    {"low": st.warning, "medium": st.info, "high": st.success}[level](text, icon="🧭")


def cash_basis_note() -> None:
    st.caption("Мөнгөн суурьтай удирдлагын тайлан. НББОУС-ийн санхүүгийн тайланг орлохгүй.")


def require_analysis():
    from state import get_analysis

    a = get_analysis()
    if a is None:
        st.info("Эхлээд **Өгөгдөл** хуудаснаас сарын хуулгаа оруулна уу, эсвэл жишээ өгөгдөл "
                "сонгоно уу.", icon="📂")
        st.stop()
    return a
