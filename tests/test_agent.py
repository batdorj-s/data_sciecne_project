"""Агентийн давталтыг хуурамч Claude клиентээр шалгана (API дуудахгүй)."""

from types import SimpleNamespace as NS

import pytest

from finagent.agent import SUBMIT_TOOL, run_agent
from finagent.pipeline import Analysis


def _use(i, name, inp):
    return NS(type="tool_use", id=f"tu_{i}", name=name, input=inp)


def _resp(*blocks, stop="tool_use"):
    return NS(content=list(blocks), stop_reason=stop,
              usage=NS(input_tokens=100, output_tokens=50))


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})
        return self.responses.pop(0)


@pytest.fixture(scope="module")
def analysis(sample_files, balances):
    return Analysis.from_files(sample_files, balances)


def _rec(impact, text="Төслийн зардлыг 10% бууруулах", confidence="high"):
    t = {"action": text, "kpi": "төслийн зардал / орлого", "impact_mln": impact,
         "priority": "high"}
    return {"trend": "stable_growth", "confidence": confidence,
            "summary": "Орлого 61.7–86.8 сая ₮, суурь ашиг 10.8 сая ₮.",
            "drivers": ["Хамгийн том төлбөрийн хувь 74% → 46% → 28% болж буурсан."],
            "risks": [], "tactics": [t, t, t], "data_warnings": ["Үлдэгдэл тулгарахгүй."]}


def _script(*submits):
    out = [
        _resp(_use(1, "get_quarter_overview", {}), _use(2, "get_baseline_forecast", {})),
        _resp(_use(3, "get_revenue_concentration", {}),
              _use(4, "simulate_scenario", {"cost_change_pct": {"Төслийн зардал": -10}})),
    ]
    out += [_resp(_use(10 + i, SUBMIT_TOOL, r)) for i, r in enumerate(submits)]
    return out


def test_happy_path_shows_faithful_recommendation(analysis):
    client = FakeClient(_script(_rec(1.76)))
    r = run_agent(analysis, client=client)
    assert r.shown and r.error is None
    assert r.faithfulness.ok
    # хэрэгслийн итгэл "low" тул агентийн "high"-ийг бууруулна
    assert r.recommendation["confidence"] == "low"
    assert [c.name for c in r.tool_calls][:2] == ["get_quarter_overview", "get_baseline_forecast"]
    req = client.calls[0]
    assert req["model"] == "claude-opus-5-5"
    assert {t["name"] for t in req["tools"]} >= {SUBMIT_TOOL, "simulate_scenario"}
    # tool_result нь tool_use-ийн id-тай таарна
    last_user = client.calls[-1]["messages"][-1]
    assert [b["tool_use_id"] for b in last_user["content"]] == ["tu_3", "tu_4"]


def test_hallucinated_number_gets_one_retry_then_passes(analysis):
    client = FakeClient(_script(_rec(9.99), _rec(1.76)))
    r = run_agent(analysis, client=client)
    assert r.shown
    retry_msg = client.calls[-1]["messages"][-1]["content"][0]
    assert retry_msg["is_error"] and "9.99" in retry_msg["content"]


def test_persistent_hallucination_is_hidden(analysis):
    client = FakeClient(_script(_rec(9.99), _rec(8.88)))
    r = run_agent(analysis, client=client)
    assert not r.shown
    assert "харуулахгүй" in r.error


def test_refusal_stops(analysis):
    r = run_agent(analysis, client=FakeClient([_resp(stop="refusal")]))
    assert not r.shown and "татгалз" in r.error
