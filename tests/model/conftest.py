import json
from pathlib import Path
from datetime import date

import pytest

from prediction_bot.shared.schemas import AssetOHLCV

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_records():
    raw = json.loads((FIXTURE_DIR / "sample_ohlcv_panel.json").read_text())
    return [AssetOHLCV(**r) for r in raw]


@pytest.fixture
def tmp_model_path(tmp_path):
    return tmp_path / "model.joblib"
