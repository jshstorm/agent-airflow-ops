import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "USStockAnalysis"))


def _write_csv(path, columns):
    df = pd.DataFrame([{col: 1 for col in columns}])
    df.to_csv(path, index=False)


def test_validate_outputs_ok(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    monkeypatch.setenv("DATA_DIR", str(data_dir))

    # Required CSVs
    _write_csv(data_dir / "smart_money_picks_v2.csv", ["ticker", "name", "composite_score", "rank", "grade"])
    _write_csv(data_dir / "us_daily_prices.csv", ["ticker", "date", "current_price", "volume"])
    _write_csv(data_dir / "us_volume_analysis.csv", ["ticker", "supply_demand_score", "supply_demand_stage"])
    _write_csv(data_dir / "us_13f_holdings.csv", ["ticker"])
    _write_csv(data_dir / "us_etf_flows.csv", ["ticker", "flow_1w", "flow_1m"])
    _write_csv(data_dir / "us_sector_heatmap.csv", ["sector", "change_1d", "change_5d"])

    # Required JSONs
    for name in [
        "etf_flow_analysis.json",
        "macro_analysis.json",
        "ai_summaries.json",
        "options_flow.json",
        "portfolio_risk.json",
        "weekly_calendar.json",
    ]:
        with open(data_dir / name, "w", encoding="utf-8") as f:
            json.dump({}, f)

    if "data_validation" in sys.modules:
        del sys.modules["data_validation"]
    import data_validation
    report = data_validation.validate_outputs(str(data_dir))

    assert report["ok"] is True
