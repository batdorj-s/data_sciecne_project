"""AI агент: хэрэгслүүдийн гаралт (агент харах өгөгдөл) ба хамгаалалтын шалгалт.

LLM дуудлага Sprint 4-д нэмэгдэнэ. Одоогоор агентад очих яг ижил JSON-г харуулж,
faithfulness шалгалтыг гараар туршиж болно.
"""

from __future__ import annotations

import json

import streamlit as st
from ui import confidence_banner, page_header, require_analysis

from finagent.guardrails import check_faithfulness
from finagent.schema import EXPENSE_LINES, Line
from finagent.tools import TOOL_SPECS, call_tool

a = require_analysis()
page_header("AI агент", "Агент тоо зохиохгүй: зөвхөн доорх хэрэгслүүдийн JSON-г ашиглан "
                        "дүгнэлт, тактик бичнэ.")
st.info("**Sprint 4:** LLM агентийн холболт (tool use) энд нэмэгдэнэ. Одоогоор хэрэгслүүд, "
        "нууцлал, faithfulness шалгалт бэлэн.", icon="🛠️")
confidence_banner(*a.confidence)

st.subheader("Хэрэгслүүд")
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
        st.code(json.dumps({"tool": name, "input": args}, ensure_ascii=False), language="json")
        st.json(out, expanded=1)

st.subheader("Faithfulness шалгалт")
st.caption("Агентийн бичсэн текстийн ₮, сая, % тоо бүр дээрх хэрэгслийн үр дүнд байх ёстой. "
           "Таарахгүй тоо байвал гаралтыг хэрэглэгчид харуулахгүй.")
fc = a.forecast
example = (f"Орлого {fc.revenue.low / 1e6:.1f}–{fc.revenue.high / 1e6:.1f} сая ₮, "
           f"суурь ашиг {fc.profit_base / 1e6:.1f} сая ₮. Төслийн зардлыг 10% бууруулбал "
           "ашиг 15.0 сая ₮ болно.")
text = st.text_area("Агентийн гаралт (туршилт)", value=example, height=110)
rep = check_faithfulness(text, results)
if rep.ok:
    st.success(f"Бүх {rep.checked} тоо хэрэгслийн үр дүнтэй таарлаа (faithfulness 100%).")
else:
    st.error(f"Faithfulness {rep.score:.0%}: таарахгүй тоо — {', '.join(rep.unmatched)}")
