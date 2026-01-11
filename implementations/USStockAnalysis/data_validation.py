#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation utilities for pipeline outputs.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from us_config import get_data_dir


SCHEMA_REQUIREMENTS: Dict[str, Dict[str, List[str]]] = {
    "smart_money_picks_v2.csv": {
        "required": ["ticker", "name", "composite_score", "rank", "grade"],
        "optional": [
            "current_price", "target_upside", "rsi", "ma_signal",
            "recommendation", "sd_score", "tech_score", "fund_score",
        ],
    },
    "us_daily_prices.csv": {
        "required": ["ticker", "date", "current_price", "volume"],
        "optional": ["open", "high", "low", "close", "name"],
    },
    "us_volume_analysis.csv": {
        "required": ["ticker", "supply_demand_score", "supply_demand_stage"],
        "optional": ["obv_trend", "ad_trend", "mfi"],
    },
    "us_13f_holdings.csv": {
        "required": ["ticker"],
        "optional": ["institution_count", "total_value"],
    },
    "us_etf_flows.csv": {
        "required": ["ticker", "flow_1w", "flow_1m"],
        "optional": ["aum", "name"],
    },
    "us_sector_heatmap.csv": {
        "required": ["sector", "change_1d", "change_5d"],
        "optional": ["ticker"],
    },
}

JSON_REQUIRED = [
    "etf_flow_analysis.json",
    "macro_analysis.json",
    "ai_summaries.json",
    "options_flow.json",
    "portfolio_risk.json",
    "weekly_calendar.json",
]


def _resolve_file(path: str, data_dir: str) -> str:
    candidate = os.path.join(data_dir, path)
    if os.path.exists(candidate):
        return candidate
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, path)


def _validate_csv(path: str, required: List[str], optional: List[str]) -> Tuple[bool, List[str]]:
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception as exc:
        return False, [f"failed to read: {exc}"]

    missing_required = [col for col in required if col not in df.columns]
    missing_optional = [col for col in optional if col not in df.columns]

    issues = []
    if missing_required:
        issues.append(f"missing required columns: {missing_required}")
    if missing_optional:
        issues.append(f"missing optional columns: {missing_optional}")

    return len(missing_required) == 0, issues


def validate_outputs(data_dir: str = None) -> Dict:
    if data_dir is None:
        data_dir = get_data_dir()

    report = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": data_dir,
        "files": {},
        "ok": True,
    }

    # CSV validations
    for filename, schema in SCHEMA_REQUIREMENTS.items():
        path = _resolve_file(filename, data_dir)
        if not os.path.exists(path):
            report["files"][filename] = {
                "status": "missing",
                "path": path,
                "issues": ["file not found"],
            }
            report["ok"] = False
            continue

        ok, issues = _validate_csv(path, schema["required"], schema["optional"])
        report["files"][filename] = {
            "status": "ok" if ok else "invalid",
            "path": path,
            "issues": issues,
        }
        if not ok:
            report["ok"] = False

    # JSON existence checks
    for filename in JSON_REQUIRED:
        path = _resolve_file(filename, data_dir)
        if not os.path.exists(path):
            report["files"][filename] = {
                "status": "missing",
                "path": path,
                "issues": ["file not found"],
            }
            report["ok"] = False
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                json.load(f)
            report["files"][filename] = {"status": "ok", "path": path, "issues": []}
        except Exception as exc:
            report["files"][filename] = {
                "status": "invalid",
                "path": path,
                "issues": [f"failed to parse: {exc}"],
            }
            report["ok"] = False

    return report


def write_report(report: Dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
