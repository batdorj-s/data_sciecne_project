import pytest

from finagent.forecast import baseline_forecast
from finagent.pipeline import Analysis
from finagent.reports import monthly_summary
from finagent.store import Store, forecast_accuracy


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 't.sqlite'}")


@pytest.fixture(scope="module")
def analysis(sample_files, balances):
    return Analysis.from_files(sample_files, balances)


def test_ledger_roundtrip_and_month_replace(store, analysis, balances):
    tid = analysis.ledger["txn_id"].iloc[0]
    n = store.save_ledger("Компани А", analysis.ledger, balances, {tid: "transfer_to_cash"})
    assert n == 185
    back = store.load_ledger("Компани А")
    assert len(back) == 185
    assert back["amount"].sum() == pytest.approx(analysis.ledger["amount"].sum())
    assert monthly_summary(back)["operating_profit"].sum() == pytest.approx(15_222_931)
    # ижил сарыг дахин хадгалахад давхардахгүй
    store.save_ledger("Компани А", analysis.ledger[analysis.ledger["month"] == "2026-03"])
    assert len(store.load_ledger("Компани А")) == 185
    assert store.load_balances("Компани А")["month"].tolist() == ["2026-01", "2026-02", "2026-03"]
    labels = store.load_overrides("Компани А")
    assert labels["line"].tolist() == ["transfer_to_cash"]
    assert store.list_companies() == ["Компани А"]
    # компани тусдаа
    assert store.load_ledger("Компани Б").empty


def test_forecast_vs_actual(store, analysis):
    led = analysis.ledger
    early = led[led["month"] < "2026-03"]
    fc = baseline_forecast(early)
    assert fc.target_month == "2026-03"
    last = monthly_summary(early)["revenue"].iloc[-1]
    store.save_forecast("Компани А", fc, naive_revenue=last)
    df, m = forecast_accuracy(store.load_forecasts("Компани А"), analysis.summary)
    assert df["month"].tolist() == ["2026-03"]
    assert m["n"] == 1 and 0 <= m["mape"] < 1
    assert "naive_mape" in m


def test_feedback(store):
    store.add_feedback("Компани А", "tactic", "Төслийн зардал -10%", "feasible")
    store.add_feedback("Компани А", "issue", "R7:2026-03-001", "false")
    assert len(store.load_feedback("Компани А")) == 2
    assert store.load_feedback("Компани А", "tactic")["value"].tolist() == ["feasible"]


def test_feedback_summary_uses_latest_vote(store):
    from finagent.store import feedback_summary

    store.add_feedback("К", "tactic", "t1", "not_feasible")
    store.add_feedback("К", "tactic", "t1", "feasible")
    store.add_feedback("К", "tactic", "t2", "not_feasible")
    store.add_feedback("К", "issue", "R7:2026-03:x", "false")
    store.add_feedback("К", "issue", "R3:2026-01:y", "real")
    s = feedback_summary(store.load_feedback("К"))
    assert s["tactic_n"] == 2 and s["tactic_feasible"] == 0.5
    assert s["issue_precision"] == 0.5
    assert s["issue_precision_by_rule"] == {"R3": 1.0, "R7": 0.0}
