"""Санхүүгийн агент AI — вэб аппын оролт.

    streamlit run app/Home.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Санхүүгийн агент AI", page_icon="📊", layout="wide")

pages = {
    "Өгөгдөл": [
        st.Page("views/data.py", title="Өгөгдөл оруулах", icon="📂", default=True),
    ],
    "Үр дүн": [
        st.Page("views/dashboard.py", title="Дашбоард", icon="📊"),
        st.Page("views/reports.py", title="Тайлан", icon="🧾"),
        st.Page("views/checks.py", title="Шалгалт ба ангилал", icon="🔍"),
        st.Page("views/forecast.py", title="Таамаг ба сценари", icon="📈"),
        st.Page("views/accuracy.py", title="Таамгийн нарийвчлал", icon="🎯"),
        st.Page("views/agent.py", title="AI агент", icon="🤖"),
    ],
}

with st.sidebar:
    st.markdown("### Санхүүгийн агент AI")
    st.caption("Сар бүр нэг файл → тайлан, дашбоард, дараа сарын тактик")

st.navigation(pages).run()
