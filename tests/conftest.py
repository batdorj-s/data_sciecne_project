from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"


@pytest.fixture(scope="session")
def sample_files() -> list[Path]:
    return sorted(SAMPLE.glob("transactions_*.xlsx"))


@pytest.fixture(scope="session")
def balances() -> pd.DataFrame:
    return pd.read_csv(SAMPLE / "balances.csv", dtype={"month": str})
