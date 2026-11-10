from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"


@pytest.fixture(scope="session")
def sample_files() -> list[Path]:
    return sorted(SAMPLE.glob("transactions_*.xlsx"))
