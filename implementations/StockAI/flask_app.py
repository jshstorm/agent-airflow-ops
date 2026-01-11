#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StockAI - Flask Web Server
국내 주식 파동 분석 대시보드 API 서버

사용법:
    python flask_app.py
    # http://localhost:5003
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))


def load_analysis_results() -> pd.DataFrame:
    """분석 결과 로드"""
    filepath = os.path.join(BASE_DIR, 'wave_transition_analysis_results.csv')
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, dtype={'ticker': str})
        df['ticker'] = df['ticker'].str.zfill(6)
        return df
    return pd.DataFrame()


def load_price_data(ticker: str = None) -> pd.DataFrame:
    """가격 데이터 로드"""
    filepath = os.path.join(BASE_DIR, 'daily_prices.csv')
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, dtype={'ticker': str})
        df['ticker'] = df['ticker'].str.zfill(6)
        if ticker:
            df = df[df['ticker'] == ticker.zfill(6)]
        return df
    return pd.DataFrame()


def get_market_status() -> Dict:
    """시장 상태 판단 (KODEX 200 기준)"""
    df = load_price_data('069500')  # KODEX 200

    if df.empty:
        return {'status': 'UNKNOWN', 'message': 'No market data'}

    df = df.sort_values('date', ascending=False)
    if len(df) < 20:
        return {'status': 'UNKNOWN', 'message': 'Insufficient data'}

    latest = df.iloc[0]
    close = float(latest['close'])

    # 이동평균 계산
    ma20 = float(df['close'].head(20).mean())
    ma50 = float(df['close'].head(50).mean()) if len(df) >= 50 else ma20
    ma200 = float(df['close'].head(200).mean()) if len(df) >= 200 else ma50

    # 시장 상태 판단
    if close > ma20 > ma50:
        status = 'RISK_ON'
        message = '상승 추세 - 적극 매수 가능'
        color = '#4ade80'
    elif close < ma20 < ma50:
        status = 'RISK_OFF'
        message = '하락 추세 - 방어적 투자 권장'
        color = '#ef4444'
    else:
        status = 'NEUTRAL'
        message = '횡보 구간 - 관망 또는 선별 투자'
        color = '#facc15'

    # 5일 변화율
    change_5d = ((close / float(df.iloc[4]['close'])) - 1) * 100 if len(df) > 4 else 0

    return {
        'status': status,
        'message': message,
        'color': color,
        'price': round(close, 2),
        'ma20': round(ma20, 2),
        'ma50': round(ma50, 2),
        'change_5d': round(change_5d, 2)
    }


# ============================================
# Page Routes
# ============================================

@app.route('/')
def index():
    """메인 대시보드 페이지"""
    return render_template('index.html')


# ============================================
# API Routes
# ============================================

@app.route('/api/kr/summary')
def get_summary():
    """시장 요약 데이터"""
    try:
        df = load_analysis_results()

        if df.empty:
            return jsonify({
                'error': 'No analysis data. Run python run_analysis.py first.',
                'total_stocks': 0
            })

        # 요약 통계
        total = int(len(df))
        s_grade = int(len(df[df['investment_grade'] == 'S급 (즉시 매수)']))
        a_grade = int(len(df[df['investment_grade'] == 'A급 (적극 매수)']))
        avg_score = float(df['final_investment_score'].mean())

        # 분석 날짜
        analysis_date = df['current_date'].iloc[0] if 'current_date' in df.columns else 'N/A'

        # 시장 상태
        market = get_market_status()

        return jsonify({
            'total_stocks': total,
            's_grade_count': s_grade,
            'a_grade_count': a_grade,
            'avg_score': round(avg_score, 1),
            'analysis_date': analysis_date,
            'market_status': market
        })
    except Exception as e:
        logger.error(f"Error in get_summary: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/kr/recommendations')
def get_recommendations():
    """추천 종목 목록"""
    try:
        df = load_analysis_results()

        if df.empty:
            return jsonify({'picks': [], 'message': 'No data available'})

        # 파라미터
        n = request.args.get('n', 20, type=int)
        grade = request.args.get('grade', None)
        market = request.args.get('market', None)

        # 필터링
        if grade:
            df = df[df['investment_grade'].str.contains(grade, case=False)]
        if market:
            df = df[df['market'] == market]

        # 상위 N개
        top = df.head(n)

        picks = []
        for _, row in top.iterrows():
            picks.append({
                'ticker': row['ticker'],
                'name': row.get('name', ''),
                'market': row.get('market', ''),
                'price': int(row.get('current_price', 0)),
                'score': round(row.get('final_investment_score', 0), 1),
                'grade': row.get('investment_grade', ''),
                'wave_stage': row.get('wave_stage', ''),
                'wave_score': row.get('wave_score', 0),
                'rsi': round(row.get('rsi', 50), 1),
                'volume_ratio': round(row.get('volume_ratio', 1), 2),
                'price_change_5d': round(row.get('price_change_5d', 0) * 100, 2) if pd.notna(row.get('price_change_5d')) else 0,
                'price_change_20d': round(row.get('price_change_20d', 0) * 100, 2) if pd.notna(row.get('price_change_20d')) else 0,
            })

        return jsonify({
            'picks': picks,
            'total': len(df)
        })
    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/kr/stock/<ticker>')
def get_stock_detail(ticker: str):
    """개별 종목 상세 정보"""
    try:
        df = load_analysis_results()
        ticker = ticker.zfill(6)

        stock = df[df['ticker'] == ticker]
        if stock.empty:
            return jsonify({'error': 'Stock not found'}), 404

        row = stock.iloc[0]

        # AI 리포트 로드 (있는 경우)
        ai_report = None
        ai_file = os.path.join(BASE_DIR, 'ai_reports', f'{ticker}.json')
        if os.path.exists(ai_file):
            with open(ai_file, 'r', encoding='utf-8') as f:
                ai_report = json.load(f)

        return jsonify({
            'ticker': row['ticker'],
            'name': row.get('name', ''),
            'market': row.get('market', ''),
            'price': int(row.get('current_price', 0)),
            'score': round(row.get('final_investment_score', 0), 1),
            'grade': row.get('investment_grade', ''),
            'wave_stage': row.get('wave_stage', ''),
            'wave_score': row.get('wave_score', 0),
            'supply_demand_stage': row.get('supply_demand_stage', 'N/A'),
            'supply_demand_score': row.get('supply_demand_score', 0),
            'fundamental_score': row.get('fundamental_score', 0),
            'rsi': round(row.get('rsi', 50), 1),
            'volume_ratio': round(row.get('volume_ratio', 1), 2),
            'position_52w': round(row.get('position_52w', 50), 1),
            'ma20': row.get('ma20', 0),
            'ma50': row.get('ma50', 0),
            'ma200': row.get('ma200', 0),
            'price_change_5d': round(row.get('price_change_5d', 0) * 100, 2) if pd.notna(row.get('price_change_5d')) else 0,
            'price_change_20d': round(row.get('price_change_20d', 0) * 100, 2) if pd.notna(row.get('price_change_20d')) else 0,
            'price_change_60d': round(row.get('price_change_60d', 0) * 100, 2) if pd.notna(row.get('price_change_60d')) else 0,
            'ai_report': ai_report
        })
    except Exception as e:
        logger.error(f"Error in get_stock_detail: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/kr/chart/<ticker>')
def get_stock_chart(ticker: str):
    """종목 차트 데이터"""
    try:
        period = request.args.get('period', '6mo')
        ticker = ticker.zfill(6)

        df = load_price_data(ticker)
        if df.empty:
            return jsonify({'error': 'No price data', 'candles': []})

        df = df.sort_values('date')

        # 기간 필터링
        period_days = {
            '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365
        }
        days = period_days.get(period, 180)
        df = df.tail(days)

        # 캔들 데이터 포맷
        candles = []
        for _, row in df.iterrows():
            candles.append({
                'time': row['date'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': int(row['volume'])
            })

        return jsonify({
            'ticker': ticker,
            'candles': candles
        })
    except Exception as e:
        logger.error(f"Error in get_stock_chart: {e}")
        return jsonify({'error': str(e), 'candles': []}), 500


@app.route('/api/kr/grade-distribution')
def get_grade_distribution():
    """등급별 분포"""
    try:
        df = load_analysis_results()
        if df.empty:
            return jsonify({'distribution': []})

        dist = df['investment_grade'].value_counts().to_dict()

        distribution = []
        grade_order = ['S급 (즉시 매수)', 'A급 (적극 매수)', 'B급 (매수 고려)',
                      'C급 (중립)', 'D급 (매도 고려)', 'F급 (회피)']

        for grade in grade_order:
            if grade in dist:
                distribution.append({
                    'grade': grade.split(' ')[0],
                    'full_name': grade,
                    'count': dist[grade]
                })

        return jsonify({'distribution': distribution})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/kr/wave-distribution')
def get_wave_distribution():
    """파동 단계별 분포"""
    try:
        df = load_analysis_results()
        if df.empty:
            return jsonify({'distribution': []})

        dist = df['wave_stage'].value_counts().to_dict()

        distribution = [{'stage': k, 'count': v} for k, v in dist.items()]
        return jsonify({'distribution': distribution})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/kr/performance')
def get_performance():
    """성과 추적 데이터"""
    try:
        from track_performance import PerformanceTracker

        tracker = PerformanceTracker()
        report = tracker.generate_report()

        if not report:
            return jsonify({
                'message': 'No performance data available',
                'stats': None
            })

        return jsonify({
            'stats': {
                'total_picks': report.get('total_picks', 0),
                'valid_picks': report.get('valid_picks', 0),
                'avg_return': round(report.get('avg_return', 0), 2),
                'win_rate': round(report.get('win_rate', 0), 1),
                'best_pick': report.get('best_pick'),
                'worst_pick': report.get('worst_pick')
            },
            'by_grade': report.get('by_grade', {})
        })
    except Exception as e:
        logger.error(f"Error in get_performance: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/kr/history-dates')
def get_history_dates():
    """히스토리 날짜 목록"""
    try:
        from track_performance import PerformanceTracker

        tracker = PerformanceTracker()
        dates = tracker.get_history_dates()

        return jsonify({'dates': dates[:30]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/kr/history/<date>')
def get_history(date: str):
    """특정 날짜 히스토리"""
    try:
        from track_performance import PerformanceTracker

        tracker = PerformanceTracker()
        picks = tracker.get_picks_by_date(date)

        if not picks:
            return jsonify({'error': 'No history for this date'}), 404

        # 현재 가격 추가
        for pick in picks:
            ticker = pick['ticker']
            current = tracker.get_current_price(ticker)
            if current and pick['price'] > 0:
                pick['current_price'] = current
                pick['return_pct'] = round(((current / pick['price']) - 1) * 100, 2)
            else:
                pick['current_price'] = None
                pick['return_pct'] = None

        return jsonify({
            'date': date,
            'picks': picks
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/kr/search')
def search_stocks():
    """종목 검색"""
    try:
        q = request.args.get('q', '')
        if len(q) < 2:
            return jsonify({'results': []})

        df = load_analysis_results()
        if df.empty:
            return jsonify({'results': []})

        # 티커 또는 이름으로 검색
        mask = (
            df['ticker'].str.contains(q, case=False) |
            df['name'].str.contains(q, case=False)
        )
        results = df[mask].head(10)

        return jsonify({
            'results': [
                {
                    'ticker': row['ticker'],
                    'name': row.get('name', ''),
                    'score': round(row.get('final_investment_score', 0), 1),
                    'grade': row.get('investment_grade', '')
                }
                for _, row in results.iterrows()
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=5003, debug=True, use_reloader=False)
