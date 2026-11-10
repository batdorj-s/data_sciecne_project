"""DM2: Хэвийн бус гүйлгээ илрүүлэх (Isolation Forest) ба шалгалтын recall-ийг хэмжих.

Шинж чанар: log(дүн), тухайн мөрийн (line) дунджаас хазайлт (log орон зайд z-score),
харилцагчийн давтамж, сарын өдөр, чиглэл. Шошго шаардахгүй; нягтлангийн "бодит / худал"
тэмдэглэлээр нарийвчлалыг (≥ 60%) хэмжинэ.

`inject_errors` нь давхардал, хэт том дүн, буруу ангилал зэргийг зориуд суулгаж
R1–R9 + Isolation Forest-ийн recall-ийг (зорилт ≥ 0.90) хэмжинэ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..checks import Issue
from ..schema import Col, Line

CONTAMINATION = 0.04
MIN_ROWS = 30
ROBUST_Z = 3.5  # median/MAD-аар тооцсон log(дүн)-ийн хазайлтын босго
MIN_LINE_ROWS = 5
CP_RATIO = 10.0  # харилцагчийн бусад гүйлгээний медианаас хэд дахин их бол


def robust_z(ledger: pd.DataFrame) -> pd.Series:
    """Ангилал (line) бүрийн дотор log(дүн)-ийн робаст z-score (Iglewicz–Hoaglin)."""
    log_amt = np.log1p(ledger[Col.AMOUNT].astype(float).fillna(0))
    grp = log_amt.groupby(ledger[Col.LINE])
    med = grp.transform("median")
    mad = (log_amt - med).abs().groupby(ledger[Col.LINE]).transform("median")
    n = grp.transform("size")
    z = 0.6745 * (log_amt - med) / mad.replace(0, np.nan)
    return z.where(n >= MIN_LINE_ROWS).fillna(0)


def counterparty_ratio(ledger: pd.DataFrame) -> pd.Series:
    """Дүн / тухайн харилцагчийн ижил ангиллын БУСАД гүйлгээний медиан (≥ 2 бусад)."""
    amt = ledger[Col.AMOUNT].astype(float)
    out = pd.Series(np.nan, index=ledger.index)
    for _, idx in ledger.groupby([Col.COUNTERPARTY, Col.LINE]).groups.items():
        if len(idx) < 3:
            continue
        vals = amt.loc[idx]
        for i in idx:
            others = vals.drop(i)
            med = others.median()
            if med > 0:
                out.loc[i] = vals.loc[i] / med
    return out


def features(ledger: pd.DataFrame) -> pd.DataFrame:
    log_amt = np.log1p(ledger[Col.AMOUNT].astype(float).fillna(0))
    grp = log_amt.groupby(ledger[Col.LINE])
    std = grp.transform("std").fillna(0).replace(0, 1.0)
    z = (log_amt - grp.transform("median")) / std
    freq = ledger[Col.COUNTERPARTY].map(ledger[Col.COUNTERPARTY].value_counts()).fillna(1)
    return pd.DataFrame({
        "log_amount": log_amt,
        "line_z": z.fillna(0),
        "cp_freq": np.log1p(freq.astype(float)),
        "day": pd.to_datetime(ledger[Col.DATE]).dt.day.fillna(15).astype(float),
        "incoming": (ledger[Col.DIRECTION] == "in").astype(float),
    }, index=ledger.index)


def detect(ledger: pd.DataFrame, contamination: float = CONTAMINATION,
           seed: int = 0) -> list[Issue]:
    if len(ledger) < MIN_ROWS:
        return []
    X = features(ledger)
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed)
    rz = robust_z(ledger).to_numpy()
    ratio = counterparty_ratio(ledger).to_numpy()
    flags = ((model.fit_predict(X) == -1) | (np.abs(rz) >= ROBUST_Z)
             | (np.nan_to_num(ratio) >= CP_RATIO))
    scores = -model.score_samples(X)
    issues = []
    for (_, r), x, s in zip(ledger[flags].iterrows(), X[flags].itertuples(), scores[flags],
                            strict=True):
        why = []
        if abs(rz[ledger.index.get_loc(_)]) >= ROBUST_Z or abs(x.line_z) >= 2:
            why.append(f"ангиллын ердийн дүнгээс {'их' if x.line_z > 0 else 'бага'}")
        k = ledger.index.get_loc(_)
        if not np.isnan(ratio[k]) and ratio[k] >= CP_RATIO:
            why.append(f"энэ харилцагчийн ердийн дүнгээс {ratio[k]:.0f} дахин их")
        if x.cp_freq <= np.log1p(1):
            why.append("харилцагч ганц удаа гарсан")
        issues.append(Issue(
            "A1", "low",
            f"Хэвийн бус гүйлгээ (оноо {s:.2f}): {r[Col.AMOUNT]:,.0f} ₮, "
            f"{r[Col.LINE]} — '{r[Col.DESCRIPTION]}'" + (f" ({'; '.join(why)})" if why else ""),
            r[Col.MONTH], float(r[Col.AMOUNT]), [r[Col.TXN_ID]]))
    return issues


# --- Зориуд алдаа суулгах ----------------------------------------------------------------
ERROR_TYPES = ("duplicate", "outlier_amount", "unclassified", "loan_as_expense")


def inject_errors(ledger: pd.DataFrame, n: int = 50, seed: int = 0
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ledger-т n алдаа суулгана. (шинэ ledger, truth[txn_id, type]) буцаана."""
    rng = np.random.default_rng(seed)
    led = ledger.copy()
    truth = []
    ops = led[led[Col.ACTIVITY] == "operating"]
    for i in range(n):
        kind = ERROR_TYPES[i % len(ERROR_TYPES)]
        src = ops.iloc[int(rng.integers(len(ops)))]
        if kind == "duplicate":
            row = src.copy()
            row[Col.TXN_ID] = f"{src[Col.TXN_ID]}-dup{i}"
            led = pd.concat([led, row.to_frame().T], ignore_index=True)
            truth.append((row[Col.TXN_ID], kind))
            continue
        row = src.copy()
        row[Col.TXN_ID] = f"inj-{i}"
        row[Col.DESCRIPTION] = f"{src[Col.DESCRIPTION]} #{i}"
        if kind == "outlier_amount":
            row[Col.AMOUNT] = float(src[Col.AMOUNT]) * float(rng.uniform(15, 40))
        elif kind == "unclassified":
            row[Col.LINE], row[Col.ACTIVITY] = Line.UNCLASSIFIED_OUT.value, "unclassified"
            row[Col.DIRECTION] = "out"
        elif kind == "loan_as_expense":
            row[Col.DESCRIPTION] = f"ZEELIIN HVV TULULT #{i}"
            row[Col.LINE], row[Col.ACTIVITY] = Line.LOAN_REPAYMENT.value, "financing"
            row[Col.DIRECTION] = "out"
        row[Col.SIGNED] = row[Col.AMOUNT] if row[Col.DIRECTION] == "in" else -row[Col.AMOUNT]
        led = pd.concat([led, row.to_frame().T], ignore_index=True)
        truth.append((row[Col.TXN_ID], kind))
    led[Col.AMOUNT] = led[Col.AMOUNT].astype(float)
    led[Col.SIGNED] = led[Col.SIGNED].astype(float)
    led[Col.DATE] = pd.to_datetime(led[Col.DATE])
    return led, pd.DataFrame(truth, columns=["txn_id", "type"])


def detection_recall(issues: list[Issue], truth: pd.DataFrame) -> pd.Series:
    """Алдааны төрөл бүрийн recall + нийт."""
    flagged = {t for i in issues for t in i.txn_ids}
    hit = truth["txn_id"].isin(flagged)
    out = hit.groupby(truth["type"]).mean()
    out["total"] = hit.mean()
    return out
