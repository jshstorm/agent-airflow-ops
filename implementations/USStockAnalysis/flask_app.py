#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US Stock Analysis - Flask Web Server
Serves the analysis dashboard and API endpoints
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request
from us_config import (
    get_data_dir,
    resolve_history_dir,
    get_cache_ttl_seconds,
    get_price_cache_ttl_seconds,
    get_rate_limit_delay,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = get_data_dir()
HISTORY_DIR = resolve_history_dir()
os.makedirs(DATA_DIR, exist_ok=True)
if os.path.abspath(HISTORY_DIR).startswith(os.path.abspath(DATA_DIR)):
    os.makedirs(HISTORY_DIR, exist_ok=True)
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))

# Sector mapping for stocks
SECTOR_MAP = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Consumer Cyclical',
    'NVDA': 'Technology', 'META': 'Technology', 'TSLA': 'Consumer Cyclical', 'BRK-B': 'Financial',
    'JPM': 'Financial', 'V': 'Financial', 'JNJ': 'Healthcare', 'UNH': 'Healthcare',
    'HD': 'Consumer Cyclical', 'PG': 'Consumer Defensive', 'MA': 'Financial', 'DIS': 'Communication',
    'PYPL': 'Financial', 'ADBE': 'Technology', 'NFLX': 'Communication', 'CRM': 'Technology',
    'AMD': 'Technology', 'INTC': 'Technology', 'CSCO': 'Technology', 'PFE': 'Healthcare',
    'XOM': 'Energy', 'CVX': 'Energy', 'KO': 'Consumer Defensive', 'PEP': 'Consumer Defensive',
}

# Cache for price and API data
_price_cache: Dict[str, Dict] = {}
_data_cache: Dict[str, Dict] = {}
_cache_timestamp: datetime = datetime.min
_last_request_time: float = 0.0

CACHE_TTL_SECONDS = get_cache_ttl_seconds()
PRICE_CACHE_TTL_SECONDS = get_price_cache_ttl_seconds()
RATE_LIMIT_DELAY = get_rate_limit_delay()


def _resolve_data_path(filename: str) -> str:
    primary = os.path.join(DATA_DIR, filename)
    if os.path.exists(primary):
        return primary
    return os.path.join(BASE_DIR, filename)


def _throttle() -> None:
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    _last_request_time = time.time()


def _get_cached(key: str, ttl: int) -> Optional[Dict]:
    cached = _data_cache.get(key)
    if not cached:
        return None
    if (datetime.now() - cached["ts"]).total_seconds() > ttl:
        return None
    return cached["value"]


def _set_cached(key: str, value: Dict) -> None:
    _data_cache[key] = {"value": value, "ts": datetime.now()}


def _fetch_history(ticker: str, period: str, retries: int = 2) -> pd.DataFrame:
    for attempt in range(retries + 1):
        try:
            _throttle()
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                return hist
        except Exception as exc:
            if attempt == retries:
                raise exc
            time.sleep(0.5 * (attempt + 1))
    return pd.DataFrame()


def _get_price_snapshot(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    cached = _price_cache.get(ticker)
    if cached and (datetime.now() - cached["ts"]).total_seconds() < PRICE_CACHE_TTL_SECONDS:
        return cached["price"], cached["prev_close"]

    try:
        _throttle()
        stock = yf.Ticker(ticker)
        info = stock.fast_info
        current = getattr(info, "last_price", None)
        prev = getattr(info, "previous_close", None) or current
    except Exception:
        current = None
        prev = None

    _price_cache[ticker] = {"price": current, "prev_close": prev, "ts": datetime.now()}
    return current, prev


def get_sector(ticker: str) -> str:
    """Get sector for a ticker"""
    if ticker in SECTOR_MAP:
        return SECTOR_MAP[ticker]
    cached = _get_cached(f"sector:{ticker}", CACHE_TTL_SECONDS)
    if cached:
        return cached.get("sector", "Unknown")
    try:
        _throttle()
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', 'Unknown')
        _set_cached(f"sector:{ticker}", {"sector": sector})
        return sector
    except Exception:
        return 'Unknown'


def load_csv_data(filename: str) -> Optional[pd.DataFrame]:
    """Load CSV file from data directory"""
    filepath = _resolve_data_path(filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None


def load_json_data(filename: str) -> Optional[Dict]:
    """Load JSON file from data directory"""
    filepath = _resolve_data_path(filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices: pd.Series) -> Dict:
    """Calculate MACD indicator"""
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return {
        'macd': macd.tolist(),
        'signal': signal.tolist(),
        'histogram': histogram.tolist()
    }


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std: int = 2) -> Dict:
    """Calculate Bollinger Bands"""
    ma = prices.rolling(window=period).mean()
    std_dev = prices.rolling(window=period).std()
    upper = ma + (std_dev * std)
    lower = ma - (std_dev * std)
    return {
        'upper': upper.tolist(),
        'middle': ma.tolist(),
        'lower': lower.tolist()
    }


# ============================================
# Page Routes
# ============================================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


# ============================================
# API Routes
# ============================================

@app.route('/api/us/portfolio')
def get_portfolio():
    """Get market indices data"""
    try:
        cached = _get_cached("market_indices", CACHE_TTL_SECONDS)
        if cached:
            return jsonify(cached)

        indices = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ 100',
            'DIA': 'Dow Jones',
            'IWM': 'Russell 2000',
            '^VIX': 'VIX',
            'GC=F': 'Gold',
            'CL=F': 'Oil',
            'BTC-USD': 'Bitcoin'
        }

        market_indices = []
        for ticker, name in indices.items():
            try:
                hist = _fetch_history(ticker, period='2d')
                if len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2]
                    change = ((current / prev) - 1) * 100
                    market_indices.append({
                        'ticker': ticker,
                        'name': name,
                        'price': round(current, 2),
                        'change': round(change, 2)
                    })
            except Exception:
                continue

        payload = {
            'market_indices': market_indices,
            'timestamp': datetime.now().isoformat()
        }
        _set_cached("market_indices", payload)
        return jsonify(payload)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/smart-money')
def get_smart_money():
    """Get smart money picks"""
    try:
        df = load_csv_data('smart_money_picks_v2.csv')

        if df is None:
            return jsonify({
                'top_picks': [],
                'message': 'Run smart_money_screener_v2.py first'
            })

        # Get current prices for top picks
        top_picks = []
        for _, row in df.head(20).iterrows():
            ticker = row['ticker']
            current_price, prev_close = _get_price_snapshot(ticker)
            if current_price is None:
                current_price = row.get('current_price', 0)
            if prev_close:
                change_pct = ((current_price / prev_close) - 1) * 100
            else:
                change_pct = 0

            sector = get_sector(ticker)

            top_picks.append({
                'rank': int(row.get('rank', 0)),
                'ticker': ticker,
                'name': row.get('name', ticker),
                'sector': sector,
                'composite_score': round(row.get('composite_score', 0), 1),
                'grade': row.get('grade', 'N/A'),
                'current_price': round(current_price, 2) if current_price else 0,
                'change_pct': round(change_pct, 2),
                'target_upside': round(row.get('target_upside', 0), 1),
                'rsi': round(row.get('rsi', 50), 1),
                'ma_signal': row.get('ma_signal', 'Neutral'),
                'recommendation': row.get('recommendation', 'hold'),
                'sd_score': round(row.get('sd_score', 50), 1),
                'tech_score': round(row.get('tech_score', 50), 1),
                'fund_score': round(row.get('fund_score', 50), 1),
            })

        return jsonify({
            'top_picks': top_picks,
            'total_count': len(df),
            'last_updated': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in get_smart_money: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/etf-flows')
def get_etf_flows():
    """Get ETF fund flows data"""
    try:
        df = load_csv_data('us_etf_flows.csv')
        analysis = load_json_data('etf_flow_analysis.json')

        if df is None:
            return jsonify({
                'flows': [],
                'message': 'Run analyze_etf_flows.py first'
            })

        flows = []
        for _, row in df.head(20).iterrows():
            flows.append({
                'ticker': row.get('ticker', ''),
                'name': row.get('name', ''),
                'flow_1w': row.get('flow_1w', 0),
                'flow_1m': row.get('flow_1m', 0),
                'aum': row.get('aum', 0),
                'category': row.get('category', 'Unknown')
            })

        return jsonify({
            'flows': flows,
            'analysis': analysis.get('ai_analysis', '') if analysis else '',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/macro-analysis')
def get_macro_analysis():
    """Get macro analysis data"""
    try:
        lang = request.args.get('lang', 'ko')
        cache_key = f"macro_analysis:{lang}"
        cached = _get_cached(cache_key, CACHE_TTL_SECONDS)
        if cached:
            return jsonify(cached)

        # Try to load cached analysis
        if lang == 'en':
            analysis = load_json_data('macro_analysis_en.json')
        else:
            analysis = load_json_data('macro_analysis.json')

        if analysis is None:
            # Generate fresh macro data
            from macro_analyzer import MacroDataCollector
            collector = MacroDataCollector()
            macro_data = collector.get_current_macro_data()

            payload = {
                'macro_indicators': macro_data,
                'ai_analysis': 'Run macro_analyzer.py to generate AI analysis',
                'timestamp': datetime.now().isoformat()
            }
            _set_cached(cache_key, payload)
            return jsonify(payload)

        payload = {
            'macro_indicators': analysis.get('macro_indicators', {}),
            'ai_analysis': analysis.get('ai_analysis', ''),
            'news': analysis.get('news', []),
            'timestamp': analysis.get('timestamp', datetime.now().isoformat())
        }
        _set_cached(cache_key, payload)
        return jsonify(payload)
    except Exception as e:
        logger.error(f"Error in get_macro_analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/stock-chart/<ticker>')
def get_stock_chart(ticker: str):
    """Get OHLC chart data for a stock"""
    try:
        period = request.args.get('period', '1y')

        # Map period to yfinance format
        period_map = {
            '1mo': '1mo', '3mo': '3mo', '6mo': '6mo',
            '1y': '1y', '2y': '2y', '5y': '5y'
        }
        yf_period = period_map.get(period, '1y')

        hist = _fetch_history(ticker, period=yf_period)

        if hist.empty:
            return jsonify({'error': 'No data available', 'candles': []})

        # Format for Lightweight Charts
        candles = []
        for date, row in hist.iterrows():
            candles.append({
                'time': date.strftime('%Y-%m-%d'),
                'open': round(row['Open'], 2),
                'high': round(row['High'], 2),
                'low': round(row['Low'], 2),
                'close': round(row['Close'], 2),
                'volume': int(row['Volume'])
            })

        # Get company info
        try:
            _throttle()
            stock = yf.Ticker(ticker)
            info = stock.info
            company_name = info.get('longName', '') or info.get('shortName', '') or ticker
        except Exception:
            company_name = ticker

        return jsonify({
            'ticker': ticker,
            'name': company_name,
            'period': period,
            'candles': candles
        })
    except Exception as e:
        return jsonify({'error': str(e), 'candles': []}), 500


@app.route('/api/us/technical-indicators/<ticker>')
def get_technical_indicators(ticker: str):
    """Get technical indicators for a stock"""
    try:
        period = request.args.get('period', '6mo')

        hist = _fetch_history(ticker, period=period)

        if hist.empty or len(hist) < 30:
            return jsonify({'error': 'Insufficient data'})

        close = hist['Close']

        # Calculate RSI
        rsi = calculate_rsi(close)
        rsi_data = [{'time': d.strftime('%Y-%m-%d'), 'value': round(v, 2)}
                   for d, v in zip(hist.index, rsi) if not np.isnan(v)]

        # Calculate MACD
        macd_result = calculate_macd(close)
        macd_data = []
        for i, d in enumerate(hist.index):
            if i >= 25:  # Need enough data for MACD
                macd_data.append({
                    'time': d.strftime('%Y-%m-%d'),
                    'macd': round(macd_result['macd'][i], 3),
                    'signal': round(macd_result['signal'][i], 3),
                    'histogram': round(macd_result['histogram'][i], 3)
                })

        # Calculate Bollinger Bands
        bb = calculate_bollinger_bands(close)
        bb_data = []
        for i, d in enumerate(hist.index):
            if i >= 19:  # Need 20 days for BB
                bb_data.append({
                    'time': d.strftime('%Y-%m-%d'),
                    'upper': round(bb['upper'][i], 2),
                    'middle': round(bb['middle'][i], 2),
                    'lower': round(bb['lower'][i], 2)
                })

        # Calculate support/resistance (simple version)
        recent = close.tail(60)
        support = round(recent.min(), 2)
        resistance = round(recent.max(), 2)

        return jsonify({
            'ticker': ticker,
            'rsi': rsi_data,
            'macd': macd_data,
            'bollinger': bb_data,
            'support': support,
            'resistance': resistance,
            'current_rsi': round(rsi.iloc[-1], 1) if not np.isnan(rsi.iloc[-1]) else 50
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/ai-summary/<ticker>')
def get_ai_summary(ticker: str):
    """Get AI summary for a specific stock"""
    try:
        summaries = load_json_data('ai_summaries.json')

        if summaries and ticker in summaries:
            return jsonify({
                'ticker': ticker,
                'summary': summaries[ticker]
            })

        # Generate a basic summary if not available
        df = load_csv_data('smart_money_picks_v2.csv')
        if df is not None:
            row = df[df['ticker'] == ticker]
            if not row.empty:
                row = row.iloc[0]
                summary = {
                    'grade': row.get('grade', 'N/A'),
                    'composite_score': row.get('composite_score', 0),
                    'recommendation': row.get('recommendation', 'hold'),
                    'target_upside': row.get('target_upside', 0),
                    'key_factors': [
                        f"Technical: {row.get('ma_signal', 'Neutral')}",
                        f"RSI: {row.get('rsi', 50):.1f}",
                        f"Supply/Demand Score: {row.get('sd_score', 50):.1f}"
                    ]
                }
                return jsonify({'ticker': ticker, 'summary': summary})

        return jsonify({
            'ticker': ticker,
            'summary': {'message': 'No AI summary available'}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/realtime-prices', methods=['POST'])
def get_realtime_prices():
    """Get realtime prices for multiple tickers"""
    try:
        data = request.json or {}
        tickers = data.get('tickers', [])

        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400

        result = {}
        for ticker in tickers[:30]:  # Limit to 30 tickers
            price, prev_close = _get_price_snapshot(ticker)
            if price is None:
                continue
            change = 0
            change_pct = 0
            if prev_close and prev_close > 0:
                change = price - prev_close
                change_pct = (change / prev_close) * 100

            result[ticker] = {
                'price': round(price, 2),
                'change': round(change, 2),
                'change_pct': round(change_pct, 2)
            }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/history-dates')
def get_history_dates():
    """Get list of available history dates"""
    try:
        history_dir = HISTORY_DIR
        if not os.path.exists(history_dir):
            return jsonify({'dates': []})

        dates = []
        for filename in os.listdir(history_dir):
            if filename.startswith('picks_') and filename.endswith('.json'):
                date_str = filename.replace('picks_', '').replace('.json', '')
                dates.append(date_str)

        dates.sort(reverse=True)
        return jsonify({'dates': dates[:30]})  # Last 30 days
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/history/<date>')
def get_history(date: str):
    """Get historical picks for a specific date"""
    try:
        history_file = os.path.join(HISTORY_DIR, f'picks_{date}.json')

        if not os.path.exists(history_file):
            return jsonify({'error': 'No history for this date'}), 404

        with open(history_file, 'r') as f:
            picks = json.load(f)

        # Calculate performance since recommendation
        for pick in picks:
            ticker = pick.get('ticker')
            rec_price = pick.get('price', 0)

            if ticker and rec_price > 0:
                try:
                    current, _ = _get_price_snapshot(ticker)
                    if current:
                        pick['current_price'] = round(current, 2)
                        pick['return_pct'] = round(((current / rec_price) - 1) * 100, 2)
                except Exception:
                    pick['current_price'] = None
                    pick['return_pct'] = None

        returns = [p['return_pct'] for p in picks if p.get('return_pct') is not None]
        summary = {
            'total_count': len(picks),
            'valid_count': len(returns),
            'avg_return_pct': round(float(np.mean(returns)), 2) if returns else None,
            'median_return_pct': round(float(np.median(returns)), 2) if returns else None,
            'win_rate': round(float(sum(1 for r in returns if r > 0) / len(returns) * 100), 1)
            if returns else None,
        }

        return jsonify({
            'date': date,
            'picks': picks,
            'summary': summary
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/us/sector-heatmap')
def get_sector_heatmap():
    """Get sector performance heatmap data"""
    try:
        cached = _get_cached("sector_heatmap", CACHE_TTL_SECONDS)
        if cached:
            return jsonify(cached)

        df = load_csv_data('us_sector_heatmap.csv')

        if df is None:
            # Generate basic sector data
            sectors = {
                'XLK': 'Technology', 'XLF': 'Financial', 'XLV': 'Healthcare',
                'XLE': 'Energy', 'XLY': 'Consumer Cyclical', 'XLP': 'Consumer Defensive',
                'XLI': 'Industrial', 'XLB': 'Materials', 'XLRE': 'Real Estate',
                'XLU': 'Utilities', 'XLC': 'Communication'
            }

            sector_data = []
            for ticker, name in sectors.items():
                try:
                    hist = _fetch_history(ticker, period='5d')
                    if len(hist) >= 2:
                        change_1d = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100
                        change_5d = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                        sector_data.append({
                            'sector': name,
                            'ticker': ticker,
                            'change_1d': round(change_1d, 2),
                            'change_5d': round(change_5d, 2)
                        })
                except Exception:
                    continue

            payload = {'sectors': sector_data}
            _set_cached("sector_heatmap", payload)
            return jsonify(payload)

        payload = {'sectors': df.to_dict('records')}
        _set_cached("sector_heatmap", payload)
        return jsonify(payload)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=5002, debug=True, use_reloader=False)
