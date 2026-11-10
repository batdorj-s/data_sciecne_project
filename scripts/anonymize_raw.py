"""Бодит хуулгыг нууцалж `data/sample/`-д жишээ өгөгдөл болгон бичих.

Хэрэглээ:
    python scripts/anonymize_raw.py sar_1.xlsx sar_2.xlsx sar_3.xlsx [--out data/sample]

Кодын дэвтэр (нэр → код) `data/raw/codebook.csv`-д хадгалагдана; энэ нь git-д ордоггүй.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from finagent.anonymize import anonymize_frame
from finagent.ingest import read_many

ROOT = Path(__file__).resolve().parents[1]

# Пилот компанийн өөрийн нэрийг "Компани А" болгоно
EXTRA_TERMS = {
    "ВАЙРАЛ ПИКСЕЛЬ": "КОМПАНИ А",
    "VIRAL PIXEL": "KOMPANI A",
    "PXL": "KMA",
}
HEADER = ["Өдөр", "Харилцагч", "Дүн", "Ангилал", "Дэд ангилал", "Тайлбар"]


def write_statement(df: pd.DataFrame, year: int, month: int, path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append([f"Гүйлгээний дэлгэрэнгүй - {year} оны {month}-р сар"])
    ws.append([])
    ws.append(HEADER)
    for r in df.itertuples(index=False):
        ws.append([r.date_raw, r.counterparty, r.amount, r.src_category, r.src_subcategory,
                   r.description])
    wb.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "sample")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    codebook: dict[str, str] = {}
    for pf in read_many(args.files):
        df = pf.frame.copy()
        # Компанийн өөрийн нэрийг харилцагч дотор ч мөн солино
        for term, repl in EXTRA_TERMS.items():
            df["counterparty"] = df["counterparty"].str.replace(term, repl, regex=False)
        anon, codebook = anonymize_frame(df, codebook, EXTRA_TERMS)
        target = args.out / f"transactions_{pf.year}_{pf.month:02d}.xlsx"
        write_statement(anon, pf.year, pf.month, target)
        print(f"{pf.name} -> {target.relative_to(ROOT)} ({len(anon)} мөр)")

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sorted(codebook.items(), key=lambda kv: kv[1]), columns=["name", "code"]).to_csv(
        raw_dir / "codebook.csv", index=False
    )
    print(f"Кодын дэвтэр: data/raw/codebook.csv ({len(codebook)} харилцагч)")


if __name__ == "__main__":
    main()
