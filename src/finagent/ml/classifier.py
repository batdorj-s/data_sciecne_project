"""DM1: Гүйлгээ ангилах загвар (жишиг: TF-IDF + Logistic Regression).

Шинж чанар (proposal 4.4): тайлбарын тэмдэгт ба үгийн n-gram (кирилл/латин галиглал, тасарсан үгэнд
тэсвэртэй), мөнгөний чиглэл, log(дүн), тэгш тоо эсэх, сарын өдөр.

Шошго: ledger-ийн `line` (бэлтгэсэн хүснэгтийн нягтлангийн ангиллаас + хүний засвар).
Итгэл босгоос доош бол хүн батална (`class_source="model"` зөвхөн итгэлтэй таамагт).

    python -m finagent.ml.classifier   # жишээ өгөгдөл дээр CV үнэлгээ
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..schema import FINANCING_LINES, Col

CONFIDENCE_THRESHOLD = 0.6
FINANCING = {ln.value for ln in FINANCING_LINES}


def features(ledger: pd.DataFrame) -> pd.DataFrame:
    amt = ledger[Col.AMOUNT].astype(float).fillna(0)
    return pd.DataFrame({
        "text": ledger[Col.DESCRIPTION].fillna("").str.lower(),
        "direction": ledger[Col.DIRECTION].fillna("out"),
        "log_amount": np.log1p(amt),
        "round": ((amt % 100_000) == 0).astype(float),
        "day": pd.to_datetime(ledger[Col.DATE]).dt.day.fillna(15).astype(float),
    }, index=ledger.index)


def build_model() -> Pipeline:
    pre = ColumnTransformer([
        ("text", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1,
                                 sublinear_tf=True), "text"),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True),
         "text"),
        ("dir", OneHotEncoder(handle_unknown="ignore"), ["direction"]),
        ("num", StandardScaler(), ["log_amount", "round", "day"]),
    ])
    # C=20: 3 seed-ийн CV дунджаар сонгосон (185 мөр — өгөгдөл нэмэгдэхэд дахин тохируулна)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=20.0)
    return Pipeline([("features", pre), ("clf", clf)])


@dataclass
class Evaluation:
    macro_f1: float
    financing_recall: float
    n: int
    classes: list[str]
    report: str


def _trainable(ledger: pd.DataFrame, min_count: int) -> pd.DataFrame:
    counts = ledger[Col.LINE].value_counts()
    return ledger[ledger[Col.LINE].isin(counts[counts >= min_count].index)]


def evaluate(ledger: pd.DataFrame, folds: int = 3, seed: int = 0) -> Evaluation:
    """Stratified CV. Proposal-ийн босго: macro-F1 ≥ 0.85, санхүүгийн recall ≥ 0.95
    (компаниар хуваасан CV-г 4+ компанийн өгөгдөл цугларсны дараа хийнэ)."""
    data = _trainable(ledger, folds)
    y = data[Col.LINE]
    pred = cross_val_predict(build_model(), features(data), y,
                             cv=StratifiedKFold(folds, shuffle=True, random_state=seed))
    fin = y.isin(FINANCING)
    fin_recall = recall_score(fin, pd.Series(pred, index=y.index).isin(FINANCING),
                              zero_division=0)
    return Evaluation(
        macro_f1=float(f1_score(y, pred, average="macro")),
        financing_recall=float(fin_recall),
        n=len(data),
        classes=sorted(y.unique()),
        report=classification_report(y, pred, zero_division=0),
    )


def train(ledger: pd.DataFrame, min_count: int = 2) -> Pipeline:
    data = _trainable(ledger, min_count)
    model = build_model()
    model.fit(features(data), data[Col.LINE])
    return model


def suggest(model: Pipeline, ledger: pd.DataFrame,
            threshold: float = CONFIDENCE_THRESHOLD) -> pd.DataFrame:
    """Гүйлгээ бүрийн загварын ангилал ба итгэл. `confident` = босгоос дээш."""
    if ledger.empty:
        return pd.DataFrame(columns=["txn_id", "predicted", "proba", "confident"])
    proba = model.predict_proba(features(ledger))
    classes = model.classes_
    best = proba.argmax(axis=1)
    return pd.DataFrame({
        "txn_id": ledger[Col.TXN_ID].to_numpy(),
        "predicted": classes[best],
        "proba": proba.max(axis=1),
        "confident": proba.max(axis=1) >= threshold,
    })


def apply_model(model: Pipeline, ledger: pd.DataFrame,
                threshold: float = CONFIDENCE_THRESHOLD) -> pd.DataFrame:
    """Эх ангилалгүй (банкны экспорт) мөрөнд итгэлтэй таамгийг хэрэглэнэ."""
    from ..classify import line_direction

    target = ledger[ledger[Col.SRC_CATEGORY].isna() & (ledger[Col.CLASS_SOURCE] == "rule")]
    if target.empty:
        return ledger
    sug = suggest(model, target, threshold)
    # Банкны чиглэлтэй зөрчилдөх таамгийг (зарлагыг "орлого" гэх мэт) хэрэглэхгүй
    actual_dir = dict(zip(target[Col.TXN_ID], target[Col.DIRECTION], strict=True))
    sug = sug[sug["confident"] & (sug["predicted"].map(line_direction)
                                  == sug["txn_id"].map(actual_dir))]
    from ..classify import apply_overrides

    out = apply_overrides(ledger, dict(zip(sug["txn_id"], sug["predicted"], strict=True)))
    out.loc[out[Col.TXN_ID].isin(sug["txn_id"]), Col.CLASS_SOURCE] = "model"
    return out


if __name__ == "__main__":
    from pathlib import Path

    from ..classify import load_ledger

    root = Path(__file__).resolve().parents[3]
    led = load_ledger(sorted((root / "data" / "sample").glob("transactions_*.xlsx")))
    ev = evaluate(led)
    print(f"n={ev.n}  macro-F1={ev.macro_f1:.3f}  санхүүгийн recall={ev.financing_recall:.3f}")
    print(ev.report)
