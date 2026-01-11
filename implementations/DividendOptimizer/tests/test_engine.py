import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "DividendOptimizer"))

from engine import DividendEngine


def test_get_themes_has_entries():
    engine = DividendEngine(data_dir=str(ROOT / "DividendOptimizer"))
    themes = engine.get_themes()
    assert isinstance(themes, list)
    assert themes, "themes should not be empty"
    assert "id" in themes[0]
    assert "title" in themes[0]


def test_filter_universe_respects_tags():
    engine = DividendEngine(data_dir=str(ROOT / "DividendOptimizer"))
    allowed = ["dividend_quality"]
    banned = []
    result = engine._filter_universe(allowed, banned)
    assert isinstance(result, list)
    if not engine.dividend_data:
        assert result == []
        return
    assert result, "filtered universe should not be empty"
    for ticker in result:
        tags = engine.symbol_tags.get(ticker, [])
        assert "dividend_quality" in tags
