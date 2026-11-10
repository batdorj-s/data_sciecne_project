"""AI агент: Claude хэрэгслүүдийг дуудаж дараа сарын зөвлөмж бичнэ; хамгаалалтаар шалгагдана."""

from __future__ import annotations

import hashlib
import json
import os

import streamlit as st
from ui import confidence_banner, mnt, page_header, require_analysis

from finagent.guardrails import check_faithfulness
from finagent.schema import EXPENSE_LINES, Line
from finagent.tools import TOOL_SPECS, call_tool

TREND = {"growth": "📈 Өсөлт", "stable": "➡️ Тогтвортой", "decline": "📉 Бууралт",
         "stable_growth": "↗️ Тогтвортой–өсөлт", "stable_decline": "↘️ Тогтвортой–бууралт"}
PRIORITY = {"high": "🔴 Өндөр", "medium": "🟠 Дунд", "low": "🟡 Бага"}
CONF = {"low": "Бага", "medium": "Дунд", "high": "Өндөр"}

a = require_analysis()
page_header("AI агент", "Агент тоо зохиохгүй: зөвхөн хэрэгслүүдийн JSON-г ашиглан дүгнэлт, "
                        "тактик бичнэ. Тоо бүрийг автоматаар тулгана.")
confidence_banner(*a.confidence)

data_key = hashlib.sha256(a.ledger.to_json().encode() + str(a.balances).encode()).hexdigest()
has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

with st.container(border=True):
    c1, c2 = st.columns([3, 1])
    c1.markdown("**Дараа сарын зөвлөмж** — Claude Opus 5.5 хэрэгслүүдийг дуудаж, тактик бүрийн "
                "₮ нөлөөг `simulate_scenario`-оор тооцно. Ойролцоогоор 30–90 секунд.")
    effort = c2.selectbox("Effort", ["low", "medium", "high"], index=1,
                          help="Өндөр бол илүү нарийн, гэхдээ удаан, үнэтэй.")
    if not has_key:
        st.warning("`ANTHROPIC_API_KEY` тохируулаагүй байна. Терминалд "
                   "`export ANTHROPIC_API_KEY=...` гэж тохируулаад аппыг дахин эхлүүлнэ үү.",
                   icon="🔑")
    if st.button("Агент ажиллуулах", type="primary", disabled=not has_key):
        from finagent.agent import run_agent

        with st.spinner("Агент хэрэгслүүдийг дуудаж байна…"):
            st.session_state["agent_result"] = (data_key, run_agent(a, effort=effort))

stored = st.session_state.get("agent_result")
if stored and stored[0] == data_key:
    r = stored[1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Хэрэгслийн дуудлага", len(r.tool_calls))
    m2.metric("Faithfulness", f"{r.faithfulness.score:.0%}" if r.faithfulness else "—")
    m3.metric("Хугацаа", f"{r.seconds:.0f} сек")
    m4.metric("Токен (орох/гарах)", f"{r.input_tokens:,} / {r.output_tokens:,}")

    if r.error:
        st.error(r.error)
    if r.shown and r.recommendation:
        rec = r.recommendation
        with st.container(border=True):
            st.markdown(f"### {TREND.get(rec['trend'], rec['trend'])} · "
                        f"итгэл: **{CONF.get(rec['confidence'], rec['confidence'])}**")
            if rec.get("summary"):
                st.write(rec["summary"])
            d, k = st.columns(2)
            with d:
                st.markdown("**Өсөлтийн хүчин зүйл**")
                for x in rec.get("drivers", []):
                    st.markdown(f"- {x}")
            with k:
                st.markdown("**Бууралтын эрсдэл**")
                for x in rec.get("risks", []):
                    st.markdown(f"- {x}")
        st.markdown("#### Тактик")
        for i, t in enumerate(rec.get("tactics", []), 1):
            with st.container(border=True):
                left, right = st.columns([4, 1])
                left.markdown(f"**{i}. {t['action']}**  \nKPI: {t['kpi']}")
                right.metric("Ашгийн нөлөө", mnt(t["impact_mnt"]))
                right.caption(PRIORITY.get(t["priority"], t["priority"]))
        if rec.get("data_warnings"):
            st.warning("**Өгөгдлийн анхааруулга**\n\n" + "\n".join(
                f"- {w}" for w in rec["data_warnings"]))
        st.caption("Агент зөвхөн санал болгоно; шийдвэрийг эзэн гаргана.")
    elif r.recommendation and not r.shown:
        with st.expander("Хамгаалалтад баригдсан зөвлөмж (хэрэглэгчид харуулахгүй)"):
            if r.faithfulness and r.faithfulness.unmatched:
                st.write("Таарахгүй тоо:", ", ".join(r.faithfulness.unmatched))
            if r.structure_errors:
                st.write("Бүтцийн алдаа:", "; ".join(r.structure_errors))
            st.json(r.recommendation, expanded=False)
    with st.expander("Агентийн хэрэгслийн дуудлагууд"):
        for c in r.tool_calls:
            st.code(json.dumps({"tool": c.name, "input": c.input}, ensure_ascii=False),
                    language="json")

st.divider()
st.subheader("Хэрэгслүүд (агентийн харах өгөгдөл)")
results = []
for spec in TOOL_SPECS:
    name = spec["name"]
    with st.expander(f"`{name}` — {spec['description']}"):
        args: dict = {}
        if name == "get_line_trend":
            args["line"] = st.selectbox("line", ["revenue"] + [ln.value for ln in Line],
                                        key="lt_line")
        elif name == "simulate_scenario":
            c1, c2 = st.columns(2)
            args["revenue_pct"] = c1.number_input("revenue_pct", value=0.0, step=1.0)
            key = c2.selectbox("expense line", [ln.value for ln in EXPENSE_LINES] + ["all"],
                               index=1)
            val = c2.number_input("expense_pct", value=-10.0, step=1.0)
            args["expense_pct"] = {key: val}
        out = call_tool(a, name, args)
        results.append(out)
        st.json(out, expanded=1)

with st.expander("Faithfulness шалгалтыг гараар турших"):
    fc = a.forecast
    example = (f"Орлого {fc.revenue.low / 1e6:.1f}–{fc.revenue.high / 1e6:.1f} сая ₮, "
               f"суурь ашиг {fc.profit_base / 1e6:.1f} сая ₮. Төслийн зардлыг 10% бууруулбал "
               "ашиг 15.0 сая ₮ болно.")
    text = st.text_area("Агентийн гаралт (туршилт)", value=example, height=110)
    rep = check_faithfulness(text, results)
    if rep.ok:
        st.success(f"Бүх {rep.checked} тоо хэрэгслийн үр дүнтэй таарлаа.")
    else:
        st.error(f"Faithfulness {rep.score:.0%}: таарахгүй тоо — {', '.join(rep.unmatched)}")
