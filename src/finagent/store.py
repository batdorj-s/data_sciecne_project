"""Өгөгдөл хадгалах давхарга (SQLAlchemy Core).

Анхдагчаар `data/finagent.sqlite`; PostgreSQL-д шилжихдээ `DATABASE_URL`-ийг
`postgresql+psycopg://user:pass@host/db` болгоход хангалттай.

Хүснэгтүүд:
    companies     — компани (хэрэглэгч бүрийн өгөгдөл тусдаа)
    transactions  — нэгдсэн ledger (сараар солигдоно)
    balances      — банкны эхний/эцсийн үлдэгдэл
    overrides     — хүний баталсан ангилал (DM1-ийн шошго)
    forecasts     — гаргасан таамаг бүрийн хуулбар (BO4: бодит дүнтэй харьцуулна)
    feedback      — тактик / анхааруулгын үнэлгээ (BO3, DM2)
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    insert,
    select,
)
from sqlalchemy.engine import Engine

from .forecast import Forecast, mape
from .schema import TXN_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = f"sqlite:///{ROOT / 'data' / 'finagent.sqlite'}"

meta = MetaData()

companies = Table(
    "companies", meta,
    Column("id", Integer, primary_key=True),
    Column("name", String(200), nullable=False, unique=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

transactions = Table(
    "transactions", meta,
    Column("company_id", Integer, ForeignKey("companies.id"), nullable=False, index=True),
    Column("txn_id", String(64), nullable=False),
    Column("date", DateTime),
    Column("month", String(7), nullable=False, index=True),
    Column("counterparty", String(300)),
    Column("amount", Float, nullable=False),
    Column("direction", String(3), nullable=False),
    Column("signed_amount", Float, nullable=False),
    Column("src_category", String(200)),
    Column("src_subcategory", String(200)),
    Column("description", Text),
    Column("activity", String(20), nullable=False),
    Column("line", String(40), nullable=False),
    Column("class_source", String(10), nullable=False),
    Column("source_file", String(300)),
    Column("source_row", Integer),
    UniqueConstraint("company_id", "txn_id"),
)

balances = Table(
    "balances", meta,
    Column("company_id", Integer, ForeignKey("companies.id"), nullable=False),
    Column("month", String(7), nullable=False),
    Column("opening", Float),
    Column("closing", Float),
    UniqueConstraint("company_id", "month"),
)

overrides = Table(
    "overrides", meta,
    Column("company_id", Integer, ForeignKey("companies.id"), nullable=False),
    Column("txn_id", String(64), nullable=False),
    Column("line", String(40), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("company_id", "txn_id"),
)

forecasts = Table(
    "forecasts", meta,
    Column("id", Integer, primary_key=True),
    Column("company_id", Integer, ForeignKey("companies.id"), nullable=False, index=True),
    Column("made_at", DateTime(timezone=True), nullable=False),
    Column("target_month", String(7), nullable=False),
    Column("method", String(100)),
    Column("months_used", String(200)),
    Column("revenue_base", Float), Column("revenue_low", Float), Column("revenue_high", Float),
    Column("profit_base", Float), Column("profit_low", Float), Column("profit_high", Float),
    Column("naive_revenue", Float),
    Column("payload", Text),
)

feedback = Table(
    "feedback", meta,
    Column("id", Integer, primary_key=True),
    Column("company_id", Integer, ForeignKey("companies.id"), nullable=False, index=True),
    Column("kind", String(20), nullable=False),  # "tactic" | "issue"
    Column("ref", String(200), nullable=False),  # тактикийн текст / issue key
    Column("value", String(20), nullable=False),  # "feasible"/"not_feasible", "real"/"false"
    Column("note", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


def _now() -> datetime:
    return datetime.now(UTC)


class Store:
    def __init__(self, url: str | None = None, engine: Engine | None = None):
        url = url or os.environ.get("DATABASE_URL") or DEFAULT_URL
        if url.startswith("sqlite:///"):
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self.engine = engine or create_engine(url, future=True)
        meta.create_all(self.engine)

    # --- компани --------------------------------------------------------------------
    def company_id(self, name: str, create: bool = True) -> int | None:
        with self.engine.begin() as c:
            row = c.execute(select(companies.c.id).where(companies.c.name == name)).first()
            if row:
                return row[0]
            if not create:
                return None
            return c.execute(insert(companies).values(name=name, created_at=_now())
                             ).inserted_primary_key[0]

    def list_companies(self) -> list[str]:
        """Гүйлгээ хадгалсан компаниуд (зөвхөн үнэлгээтэй компани орохгүй)."""
        q = (select(companies.c.name).where(
            select(transactions.c.txn_id).where(transactions.c.company_id == companies.c.id)
            .exists()).order_by(companies.c.name))
        with self.engine.connect() as c:
            return [r[0] for r in c.execute(q)]

    # --- ledger ---------------------------------------------------------------------
    def save_ledger(self, company: str, ledger: pd.DataFrame,
                    bal: pd.DataFrame | None = None,
                    manual: dict[str, str] | None = None) -> int:
        """Ledger-ийн сарууд хадгалагдсан бол солино. Хадгалсан мөрийн тоо буцаана."""
        cid = self.company_id(company)
        months = sorted(ledger["month"].dropna().unique())
        rows = ledger[TXN_COLUMNS].copy()
        rows["date"] = rows["date"].map(lambda d: None if pd.isna(d) else d.to_pydatetime())
        records = [{**r, "company_id": cid} for r in rows.to_dict("records")]
        with self.engine.begin() as c:
            c.execute(delete(transactions).where(transactions.c.company_id == cid,
                                                 transactions.c.month.in_(months)))
            if records:
                c.execute(insert(transactions), records)
            if bal is not None and not bal.empty:
                b = bal.dropna(subset=["opening", "closing"])
                c.execute(delete(balances).where(balances.c.company_id == cid,
                                                 balances.c.month.in_(b["month"].tolist())))
                if not b.empty:
                    c.execute(insert(balances), [
                        {"company_id": cid, "month": r.month, "opening": float(r.opening),
                         "closing": float(r.closing)} for r in b.itertuples(index=False)])
            for tid, line in (manual or {}).items():
                c.execute(delete(overrides).where(overrides.c.company_id == cid,
                                                  overrides.c.txn_id == tid))
                c.execute(insert(overrides).values(company_id=cid, txn_id=tid, line=line,
                                                   created_at=_now()))
        return len(records)

    def load_ledger(self, company: str) -> pd.DataFrame:
        cid = self.company_id(company, create=False)
        if cid is None:
            return pd.DataFrame(columns=TXN_COLUMNS)
        with self.engine.connect() as c:
            df = pd.read_sql(select(transactions).where(transactions.c.company_id == cid), c)
        df = df.drop(columns="company_id")
        df["date"] = pd.to_datetime(df["date"])
        return df[TXN_COLUMNS].sort_values(["date", "source_row"], kind="stable").reset_index(
            drop=True)

    def load_balances(self, company: str) -> pd.DataFrame | None:
        cid = self.company_id(company, create=False)
        if cid is None:
            return None
        with self.engine.connect() as c:
            df = pd.read_sql(select(balances.c.month, balances.c.opening, balances.c.closing)
                             .where(balances.c.company_id == cid).order_by(balances.c.month), c)
        return df if not df.empty else None

    def load_overrides(self, company: str | None = None) -> pd.DataFrame:
        """Хүний баталсан ангилал + тухайн гүйлгээний текст (DM1-ийн шошготой өгөгдөл)."""
        q = (select(companies.c.name.label("company"), overrides.c.txn_id, overrides.c.line,
                    transactions.c.description, transactions.c.counterparty,
                    transactions.c.amount, transactions.c.direction)
             .join(companies, companies.c.id == overrides.c.company_id)
             .join(transactions, (transactions.c.company_id == overrides.c.company_id)
                   & (transactions.c.txn_id == overrides.c.txn_id), isouter=True))
        if company:
            q = q.where(companies.c.name == company)
        with self.engine.connect() as c:
            return pd.read_sql(q, c)

    # --- таамаг ---------------------------------------------------------------------
    def save_forecast(self, company: str, fc: Forecast, naive_revenue: float | None = None) -> int:
        cid = self.company_id(company)
        with self.engine.begin() as c:
            return c.execute(insert(forecasts).values(
                company_id=cid, made_at=_now(), target_month=fc.target_month, method=fc.method,
                months_used=",".join(fc.months_used),
                revenue_base=fc.revenue.base, revenue_low=fc.revenue.low,
                revenue_high=fc.revenue.high, profit_base=fc.profit_base,
                profit_low=fc.profit_low, profit_high=fc.profit_high,
                naive_revenue=naive_revenue, payload=json.dumps(fc.to_dict(), ensure_ascii=False),
            )).inserted_primary_key[0]

    def load_forecasts(self, company: str) -> pd.DataFrame:
        cid = self.company_id(company, create=False)
        if cid is None:
            return pd.DataFrame()
        with self.engine.connect() as c:
            return pd.read_sql(select(forecasts).where(forecasts.c.company_id == cid)
                               .order_by(forecasts.c.made_at), c)

    # --- үнэлгээ --------------------------------------------------------------------
    def add_feedback(self, company: str, kind: str, ref: str, value: str,
                     note: str | None = None) -> None:
        cid = self.company_id(company)
        with self.engine.begin() as c:
            c.execute(insert(feedback).values(company_id=cid, kind=kind, ref=ref, value=value,
                                              note=note, created_at=_now()))

    def load_feedback(self, company: str, kind: str | None = None) -> pd.DataFrame:
        cid = self.company_id(company, create=False)
        if cid is None:
            return pd.DataFrame(columns=["kind", "ref", "value", "note", "created_at"])
        q = select(feedback.c.kind, feedback.c.ref, feedback.c.value, feedback.c.note,
                   feedback.c.created_at).where(feedback.c.company_id == cid)
        if kind:
            q = q.where(feedback.c.kind == kind)
        with self.engine.connect() as c:
            return pd.read_sql(q, c)


def feedback_summary(fb: pd.DataFrame) -> dict:
    """Сүүлийн үнэлгээгээр (ref бүрт нэг) BO3 ба DM2-ийн хэмжүүр."""
    if fb.empty:
        return {}
    last = fb.sort_values("created_at").groupby(["kind", "ref"]).tail(1)
    out = {}
    t = last[last["kind"] == "tactic"]
    if not t.empty:
        out["tactic_n"] = len(t)
        out["tactic_feasible"] = float((t["value"] == "feasible").mean())
    i = last[last["kind"] == "issue"]
    if not i.empty:
        out["issue_n"] = len(i)
        out["issue_precision"] = float((i["value"] == "real").mean())
        rule = i["ref"].str.split(":").str[0]
        out["issue_precision_by_rule"] = (i["value"] == "real").groupby(rule).mean().to_dict()
    return out


def forecast_accuracy(saved: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Хадгалсан таамгийг бодит дүнтэй тулгана (BO4). Нэг сарт сүүлийн таамгийг авна."""
    if saved.empty:
        return pd.DataFrame(), {}
    latest = saved.sort_values("made_at").groupby("target_month").tail(1)
    rows = []
    for r in latest.itertuples(index=False):
        if r.target_month not in summary.index:
            continue
        act = summary.loc[r.target_month]
        rows.append({
            "month": r.target_month,
            "revenue_actual": act["revenue"], "revenue_base": r.revenue_base,
            "revenue_low": r.revenue_low, "revenue_high": r.revenue_high,
            "revenue_error": (r.revenue_base - act["revenue"]) / act["revenue"]
            if act["revenue"] else None,
            "covered": r.revenue_low <= act["revenue"] <= r.revenue_high,
            "naive_revenue": r.naive_revenue,
            "profit_actual": act["operating_profit"], "profit_base": r.profit_base,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}
    metrics = {
        "n": len(df),
        "mape": mape(df["revenue_actual"], df["revenue_base"]),
        "coverage": float(df["covered"].mean()),
    }
    if df["naive_revenue"].notna().all():
        metrics["naive_mape"] = mape(df["revenue_actual"], df["naive_revenue"])
    return df, metrics
