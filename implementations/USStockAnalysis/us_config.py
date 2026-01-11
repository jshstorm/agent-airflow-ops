#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared configuration helpers for USStockAnalysis.
"""

import os
from datetime import timedelta


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _abs_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(BASE_DIR, path))


def get_data_dir() -> str:
    data_dir = os.getenv("DATA_DIR", "data")
    return _abs_path(data_dir)


def ensure_data_dir() -> str:
    data_dir = get_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_history_dir(data_dir: str = None) -> str:
    custom = os.getenv("HISTORY_DIR")
    if custom:
        return _abs_path(custom)
    base = data_dir or get_data_dir()
    return os.path.join(base, "history")


def resolve_history_dir() -> str:
    preferred = get_history_dir()
    legacy = os.path.join(BASE_DIR, "history")
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(legacy):
        return legacy
    return preferred


def ensure_history_dir(data_dir: str = None) -> str:
    history_dir = get_history_dir(data_dir)
    os.makedirs(history_dir, exist_ok=True)
    return history_dir


def get_cache_ttl_seconds() -> int:
    return int(os.getenv("CACHE_TTL_SECONDS", "60"))


def get_price_cache_ttl_seconds() -> int:
    return int(os.getenv("PRICE_CACHE_TTL_SECONDS", "45"))


def get_rate_limit_delay() -> float:
    return float(os.getenv("YF_RATE_LIMIT_DELAY", "0.2"))


def get_history_retention_days() -> int:
    return int(os.getenv("HISTORY_RETENTION_DAYS", "120"))


def get_validation_report_name() -> str:
    return os.getenv("PIPELINE_VALIDATION_REPORT", "pipeline_validation.json")


def get_summary_report_name() -> str:
    return os.getenv("PIPELINE_SUMMARY_REPORT", "pipeline_summary.md")
