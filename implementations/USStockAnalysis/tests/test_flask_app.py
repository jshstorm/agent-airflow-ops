import json
import sys
from pathlib import Path
import types

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "USStockAnalysis"))


def test_history_summary(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    history_dir = data_dir / "history"
    history_dir.mkdir(parents=True)

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("HISTORY_DIR", str(history_dir))

    yf_stub = types.ModuleType("yfinance")

    class DummyTicker:
        def __init__(self, *args, **kwargs):
            self.fast_info = types.SimpleNamespace(last_price=None, previous_close=None)
            self.info = {}

        def history(self, period=None):
            return pd.DataFrame()

    yf_stub.Ticker = DummyTicker
    sys.modules["yfinance"] = yf_stub

    if "flask_app" in sys.modules:
        del sys.modules["flask_app"]
    import flask_app

    # Prepare history file
    date = "2026-01-01"
    picks = [
        {"ticker": "AAPL", "price": 100},
        {"ticker": "MSFT", "price": 200},
    ]
    with open(history_dir / f"picks_{date}.json", "w", encoding="utf-8") as f:
        json.dump(picks, f)

    def fake_snapshot(ticker):
        price_map = {"AAPL": 110.0, "MSFT": 220.0}
        return (price_map.get(ticker, 110.0), 100.0)

    monkeypatch.setattr(flask_app, "_get_price_snapshot", fake_snapshot)

    client = flask_app.app.test_client()
    resp = client.get(f"/api/us/history/{date}")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data["summary"]["avg_return_pct"] == 10.0
    assert data["summary"]["win_rate"] == 100.0
