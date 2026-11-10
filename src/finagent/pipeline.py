"""Нэг удаагийн шинжилгээ: файл → ledger → шалгалт → тайлан → таамаг.

Вэб апп болон агентийн хэрэгслүүд яг ижил `Analysis` объект ашиглана — тоо нэг л газар
тооцогдоно.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd

from .anonymize import anonymize_frame
from .checks import Issue, reconciled, run_checks
from .classify import build_ledger
from .forecast import Forecast, baseline_forecast, confidence_level
from .ingest import ParsedFile, read_many
from .reports import monthly_summary


def load_balances(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"month": str})


@dataclass
class Analysis:
    ledger: pd.DataFrame
    balances: pd.DataFrame | None = None
    files: list[ParsedFile] = field(default_factory=list)

    @classmethod
    def from_files(cls, files: list[ParsedFile] | list[Path] | list[tuple[str, bytes]],
                   balances: pd.DataFrame | None = None,
                   overrides: dict[str, str] | None = None) -> Analysis:
        parsed = files if files and isinstance(files[0], ParsedFile) else read_many(files)
        return cls(build_ledger(parsed, overrides), balances, list(parsed))

    @cached_property
    def safe_ledger(self) -> pd.DataFrame:
        """Харилцагчийг кодоор, тайлбар дахь хувь хүний нэрийг *** болгосон ledger.
        Агент руу зөвхөн үүнээс гаралтай мэдээлэл илгээнэ."""
        return anonymize_frame(self.ledger)[0]

    @cached_property
    def summary(self) -> pd.DataFrame:
        return monthly_summary(self.ledger)

    @cached_property
    def issues(self) -> list[Issue]:
        # Нууцалсан ledger дээр ажиллуулна: мессеж дэх тайлбарт хувь хүний нэр гарахгүй
        return run_checks(self.safe_ledger, self.balances)

    @cached_property
    def forecast(self) -> Forecast:
        return baseline_forecast(self.ledger)

    @property
    def months(self) -> list[str]:
        return self.summary.index.tolist()

    @property
    def reconciled(self) -> bool:
        return reconciled(self.issues)

    @cached_property
    def confidence(self) -> tuple[str, list[str]]:
        high = sum(1 for i in self.issues if i.severity == "high")
        return confidence_level(len(self.months), self.reconciled, high)
