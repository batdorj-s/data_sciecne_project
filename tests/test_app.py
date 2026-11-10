"""Вэб аппын хуудас бүр жишээ өгөгдөл дээр алдаагүй ачаалагдах ёстой."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = ["views/dashboard.py", "views/reports.py", "views/checks.py", "views/forecast.py",
         "views/agent.py"]


@pytest.fixture
def app():
    sys.path.insert(0, str(APP))
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=60)
    at.run()
    yield at
    sys.path.remove(str(APP))


def test_home_loads_sample(app):
    assert not app.exception
    assert any("185" in m.value for m in app.metric)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(app, page):
    app.switch_page(page).run()
    assert not app.exception, app.exception
