import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "StockAI"))


def test_summary_endpoint(tmp_path):
    if "flask_app" in sys.modules:
        del sys.modules["flask_app"]
    import flask_app

    base_dir = tmp_path / "stockai"
    base_dir.mkdir()
    flask_app.BASE_DIR = str(base_dir)

    results = pd.DataFrame([
        {
            "ticker": "000001",
            "name": "TEST",
            "investment_grade": "S급 (즉시 매수)",
            "final_investment_score": 80,
            "current_date": "2026-01-01",
        },
        {
            "ticker": "000002",
            "name": "TEST2",
            "investment_grade": "A급 (적극 매수)",
            "final_investment_score": 70,
            "current_date": "2026-01-01",
        },
    ])
    results.to_csv(base_dir / "wave_transition_analysis_results.csv", index=False)

    prices = []
    for i in range(20):
        prices.append({
            "ticker": "069500",
            "date": f"2026-01-{i+1:02d}",
            "close": 100 + i,
        })
    pd.DataFrame(prices).to_csv(base_dir / "daily_prices.csv", index=False)

    client = flask_app.app.test_client()
    resp = client.get("/api/kr/summary")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data["total_stocks"] == 2
    assert data["s_grade_count"] == 1
    assert data["a_grade_count"] == 1
