#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US Stock Analysis - Full Pipeline Update Script
Runs all analysis scripts in sequence
"""

import os
import sys
import subprocess
import time
import argparse
import logging
from datetime import datetime, timedelta

from data_validation import validate_outputs, write_report
from us_config import (
    ensure_data_dir,
    get_data_dir,
    resolve_history_dir,
    get_validation_report_name,
    get_summary_report_name,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Script execution order with descriptions and timeouts
SCRIPTS = [
    # Part 1: Data Collection
    ("create_us_daily_prices.py", "Price Data Collection", 600),
    ("analyze_volume.py", "Volume Analysis", 300),
    ("analyze_13f.py", "Institutional Holdings", 600),
    ("analyze_etf_flows.py", "ETF Fund Flows", 300),
    
    # Part 2: Analysis/Screening
    ("smart_money_screener_v2.py", "Smart Money Screening", 600),
    ("sector_heatmap.py", "Sector Heatmap", 300),
    ("options_flow.py", "Options Flow", 300),
    ("insider_tracker.py", "Insider Tracking", 300),
    ("portfolio_risk.py", "Portfolio Risk", 300),
    
    # Part 3: AI Analysis
    ("macro_analyzer.py", "Macro Analysis", 300),
    ("ai_summary_generator.py", "AI Summaries", 900),
    ("final_report_generator.py", "Final Report", 60),
    ("economic_calendar.py", "Economic Calendar", 300),
]

# Quick mode skips these scripts
AI_SCRIPTS = ["ai_summary_generator.py", "macro_analyzer.py"]

SCRIPTS_WITH_DIR = {
    "analyze_volume.py",
    "analyze_13f.py",
    "analyze_etf_flows.py",
    "smart_money_screener_v2.py",
    "sector_heatmap.py",
    "options_flow.py",
    "insider_tracker.py",
    "portfolio_risk.py",
    "macro_analyzer.py",
    "ai_summary_generator.py",
    "final_report_generator.py",
    "economic_calendar.py",
}


def run_script(script_name: str, description: str, timeout: int, data_dir: str) -> bool:
    """Run a single script with timeout"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {description}")
    logger.info(f"Script: {script_name}")
    logger.info(f"Timeout: {timeout}s")
    logger.info('='*60)
    
    start_time = time.time()
    
    try:
        cmd = [sys.executable, script_name]
        if script_name in SCRIPTS_WITH_DIR:
            cmd.extend(["--dir", data_dir])

        env = os.environ.copy()
        env["DATA_DIR"] = data_dir

        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
            env=env
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"SUCCESS: {description} ({elapsed:.1f}s)")
            if result.stdout:
                # Print last few lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    logger.info(f"  > {line}")
            return True
        else:
            logger.error(f"FAILED: {description}")
            if result.stderr:
                logger.error(f"Error: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"TIMEOUT: {description} exceeded {timeout}s")
        return False
    except FileNotFoundError:
        logger.error(f"NOT FOUND: {script_name}")
        return False
    except Exception as e:
        logger.error(f"ERROR: {description} - {e}")
        return False


def cleanup_history(history_dir: str, retention_days: int) -> int:
    if retention_days <= 0:
        return 0

    if not os.path.exists(history_dir):
        return 0

    cutoff = datetime.now() - timedelta(days=retention_days)
    removed = 0

    for filename in os.listdir(history_dir):
        if not filename.startswith("picks_") or not filename.endswith(".json"):
            continue
        date_str = filename.replace("picks_", "").replace(".json", "")
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        if file_date < cutoff:
            os.remove(os.path.join(history_dir, filename))
            removed += 1
    return removed


def write_summary(summary_path: str, results: list, validation: dict, history_removed: int) -> None:
    lines = [
        "# US Stock Analysis Pipeline Summary",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Validation status: {'OK' if validation.get('ok') else 'ISSUES'}",
        f"- History cleanup: removed {history_removed} files",
        "",
        "## Script Results",
        "",
    ]

    for script, desc, status in results:
        lines.append(f"- {desc}: {status} ({script})")

    lines.append("")
    lines.append("## Validation Issues")
    lines.append("")

    issues = []
    for filename, info in validation.get("files", {}).items():
        if info.get("status") != "ok":
            issues.append(f"- {filename}: {info.get('status')} ({', '.join(info.get('issues', []))})")

    lines.extend(issues if issues else ["- none"])
    lines.append("")

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description='US Stock Analysis Pipeline')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick mode: skip AI-heavy scripts')
    parser.add_argument('--skip', nargs='+', default=[], 
                       help='Scripts to skip')
    parser.add_argument('--only', nargs='+', default=[], 
                       help='Only run these scripts')
    parser.add_argument('--dir', default='.', 
                       help='Working directory')
    parser.add_argument('--data-dir', default=None,
                       help='Data directory (defaults to DATA_DIR or ./data)')
    args = parser.parse_args()
    
    # Change to working directory
    if args.dir != '.':
        os.chdir(args.dir)
    
    data_dir = args.data_dir or get_data_dir()
    data_dir = os.path.abspath(data_dir)
    os.environ["DATA_DIR"] = data_dir
    ensure_data_dir()

    logger.info(f"\n{'#'*60}")
    logger.info(f"US Stock Analysis Pipeline")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Mode: {'Quick' if args.quick else 'Full'}")
    logger.info(f"Data Dir: {data_dir}")
    logger.info(f"{'#'*60}")
    
    start_time = time.time()
    results = []
    
    for script_name, description, timeout in SCRIPTS:
        # Skip logic
        if args.only and script_name not in args.only:
            continue
        
        if script_name in args.skip:
            logger.info(f"SKIPPED: {description}")
            results.append((script_name, description, 'skipped'))
            continue
        
        if args.quick and script_name in AI_SCRIPTS:
            logger.info(f"SKIPPED (quick mode): {description}")
            results.append((script_name, description, 'skipped'))
            continue
        
        # Check if script exists
        if not os.path.exists(script_name):
            logger.warning(f"Script not found: {script_name}")
            results.append((script_name, description, 'not found'))
            continue
        
        # Run script
        success = run_script(script_name, description, timeout, data_dir)
        results.append((script_name, description, 'success' if success else 'failed'))
    
    # Summary
    total_time = time.time() - start_time
    
    logger.info(f"\n{'#'*60}")
    logger.info(f"Pipeline Complete")
    logger.info(f"Total Time: {total_time/60:.1f} minutes")
    logger.info(f"{'#'*60}")
    
    logger.info("\nResults Summary:")
    logger.info("-" * 50)
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for script, desc, status in results:
        emoji = {"success": "ok", "failed": "FAIL", "skipped": "skip", "not found": "?"}
        logger.info(f"  [{emoji.get(status, '?'):4}] {desc}")
        
        if status == 'success':
            success_count += 1
        elif status == 'failed':
            failed_count += 1
        else:
            skipped_count += 1
    
    logger.info("-" * 50)
    logger.info(f"Success: {success_count} | Failed: {failed_count} | Skipped: {skipped_count}")

    # Validation report
    validation = validate_outputs(data_dir)
    validation_path = os.path.join(data_dir, get_validation_report_name())
    write_report(validation, validation_path)
    logger.info(f"Validation report: {validation_path}")

    # History cleanup (only when env provided)
    history_removed = 0
    if os.getenv("HISTORY_RETENTION_DAYS"):
        history_dir = resolve_history_dir()
        retention_days = int(os.getenv("HISTORY_RETENTION_DAYS", "0"))
        history_removed = cleanup_history(history_dir, retention_days)
        logger.info(f"History cleanup removed {history_removed} files from {history_dir}")

    # Summary report
    summary_path = os.path.join(data_dir, get_summary_report_name())
    write_summary(summary_path, results, validation, history_removed)
    logger.info(f"Summary report: {summary_path}")
    
    # Return exit code based on failures
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
