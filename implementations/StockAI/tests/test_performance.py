import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "StockAI"))


def test_performance_tracker(tmp_path, monkeypatch):
    if "track_performance" in sys.modules:
        del sys.modules["track_performance"]
    import track_performance

    base_dir = tmp_path / "stockai"
    base_dir.mkdir()
    history_dir = base_dir / "history"
    history_dir.mkdir()

    track_performance.BASE_DIR = str(base_dir)
    track_performance.HISTORY_DIR = str(history_dir)
    track_performance.RESULTS_FILE = str(base_dir / "wave_transition_analysis_results.csv")
    track_performance.PERFORMANCE_FILE = str(base_dir / "performance_report.csv")

    results = pd.DataFrame([
        {
            "ticker": "000001",
            "name": "TEST",
            "current_price": 100,
            "final_investment_score": 90,
            "investment_grade": "S급 (즉시 매수)",
            "wave_stage": "2단계",
            "rsi": 60,
        }
    ])
    results.to_csv(track_performance.RESULTS_FILE, index=False)

    prices = pd.DataFrame([
        {"ticker": "000001", "date": "2026-01-01", "close": 110},
    ])
    prices.to_csv(base_dir / "daily_prices.csv", index=False)

    tracker = track_performance.PerformanceTracker()
    history_file = tracker.save_recommendations()
    assert history_file

    perf = tracker.calculate_performance(days=0)
    assert not perf.empty
    assert round(perf.iloc[0]["return_pct"], 2) == 10.0
