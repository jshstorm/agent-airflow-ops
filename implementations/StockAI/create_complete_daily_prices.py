#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_complete_daily_prices.py
네이버 금융에서 한국 주식 일별 시세 데이터를 수집합니다.

주요 기능:
- KOSPI/KOSDAQ 종목의 일별 OHLCV 데이터 수집
- 기존 데이터와 증분 병합 (중복 제거)
- 네이버 금융 크롤링 (API 키 불필요)
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Optional, Dict, List
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True,  # 기존 핸들러 덮어쓰기
    handlers=[logging.StreamHandler()]  # 표준 출력으로 즉시 내보내기
)
logger = logging.getLogger(__name__)

# 상수 정의
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOCKS_LIST_FILE = os.path.join(BASE_DIR, 'korean_stocks_list.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'daily_prices.csv')

# 네이버 금융 URL
NAVER_DAILY_URL = "https://finance.naver.com/item/sise_day.naver"

# 요청 헤더
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://finance.naver.com/'
}


def get_stock_daily_prices(ticker: str, pages: int = 10) -> pd.DataFrame:
    """
    네이버 금융에서 특정 종목의 일별 시세를 가져옵니다.
    
    Args:
        ticker: 종목 코드 (6자리)
        pages: 가져올 페이지 수 (1페이지 = 약 10일)
    
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    all_data = []
    
    for page in range(1, pages + 1):
        try:
            url = f"{NAVER_DAILY_URL}?code={ticker}&page={page}"
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.encoding = 'euc-kr'
            
            # HTML 테이블 파싱
            tables = pd.read_html(response.text, encoding='euc-kr')
            
            if not tables:
                break
                
            df = tables[0]
            df = df.dropna(subset=['날짜'])
            
            if df.empty:
                break
            
            all_data.append(df)
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            logger.warning(f"Error fetching {ticker} page {page}: {e}")
            break
    
    if not all_data:
        return pd.DataFrame()
    
    # 데이터 병합 및 정리
    result = pd.concat(all_data, ignore_index=True)
    
    # 컬럼명 변환
    result = result.rename(columns={
        '날짜': 'date',
        '시가': 'open',
        '고가': 'high',
        '저가': 'low',
        '종가': 'close',
        '거래량': 'volume'
    })
    
    # 필요한 컬럼만 선택
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    available_cols = [col for col in required_cols if col in result.columns]
    result = result[available_cols]
    
    # 날짜 형식 변환
    result['date'] = pd.to_datetime(result['date'], format='%Y.%m.%d', errors='coerce')
    result = result.dropna(subset=['date'])
    
    # 숫자 컬럼 변환
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col].astype(str).str.replace(',', ''), errors='coerce')
    
    # 중복 제거 및 정렬
    result = result.drop_duplicates(subset=['date'])
    result = result.sort_values('date', ascending=True).reset_index(drop=True)
    
    return result


def get_stock_info(ticker: str) -> Dict:
    """
    네이버 금융에서 종목 기본 정보를 가져옵니다.
    """
    try:
        url = f"https://finance.naver.com/item/main.naver?code={ticker}"
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.encoding = 'euc-kr'
        
        tables = pd.read_html(response.text, encoding='euc-kr')
        
        # 기본 정보 추출 (PER, PBR, ROE 등)
        info = {}
        
        for table in tables:
            if 'PER' in str(table.values):
                # PER, PBR 등 추출 시도
                pass
        
        return info
        
    except Exception as e:
        logger.warning(f"Error fetching info for {ticker}: {e}")
        return {}


def load_existing_data() -> pd.DataFrame:
    """기존 저장된 데이터를 로드합니다."""
    if os.path.exists(OUTPUT_FILE):
        try:
            df = pd.read_csv(OUTPUT_FILE, parse_dates=['date'])
            logger.info(f"Loaded existing data: {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"Error loading existing data: {e}")
    return pd.DataFrame()


def save_data(df: pd.DataFrame):
    """데이터를 CSV로 저장합니다."""
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved {len(df)} rows to {OUTPUT_FILE}")


def collect_all_prices(days_back: int = 365) -> pd.DataFrame:
    """
    모든 종목의 일별 시세를 수집합니다.
    
    Args:
        days_back: 수집할 과거 일수
    
    Returns:
        전체 종목의 일별 시세 DataFrame
    """
    # 종목 리스트 로드
    if not os.path.exists(STOCKS_LIST_FILE):
        logger.error(f"Stock list file not found: {STOCKS_LIST_FILE}")
        return pd.DataFrame()
    
    logger.info(f"Loading stock list from {STOCKS_LIST_FILE}...")
    stocks_df = pd.read_csv(STOCKS_LIST_FILE, dtype={'ticker': str})
    stocks_df['ticker'] = stocks_df['ticker'].str.zfill(6)
    
    logger.info(f"Loaded {len(stocks_df)} stocks. Starting collection for {days_back} days...")
    
    # 기존 데이터 로드
    logger.info("Loading existing price data...")
    existing_data = load_existing_data()
    if not existing_data.empty:
        logger.info(f"Existing data loaded: {len(existing_data)} rows.")
    else:
        logger.info("No existing data found. Starting fresh.")
    
    # 페이지 수 계산 (1페이지 ≈ 10일)
    pages = max(1, days_back // 10)
    logger.info(f"Fetching {pages} pages per stock...")
    
    all_data = []
    failed_tickers = []
    
    for _, row in tqdm(stocks_df.iterrows(), total=len(stocks_df), desc="Collecting prices"):
        ticker = row['ticker']
        name = row['name']
        market = row.get('market', 'KOSPI')
        
        try:
            df = get_stock_daily_prices(ticker, pages=pages)
            
            if not df.empty:
                df['ticker'] = ticker
                df['name'] = name
                df['market'] = market
                all_data.append(df)
            else:
                failed_tickers.append(ticker)
                
        except Exception as e:
            logger.warning(f"Failed to collect {ticker} ({name}): {e}")
            failed_tickers.append(ticker)
        
        # Rate limiting
        time.sleep(0.2)
    
    if not all_data:
        logger.error("No data collected!")
        return existing_data
    
    # 새 데이터 병합
    new_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Collected {len(new_data)} new rows from {len(all_data)} stocks")
    
    if failed_tickers:
        logger.warning(f"Failed tickers: {failed_tickers[:10]}...")
    
    # 기존 데이터와 병합
    if not existing_data.empty:
        combined = pd.concat([existing_data, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['ticker', 'date'], keep='last')
        combined = combined.sort_values(['ticker', 'date']).reset_index(drop=True)
    else:
        combined = new_data
    
    return combined


def update_today_prices() -> pd.DataFrame:
    """
    오늘 날짜의 시세만 업데이트합니다 (빠른 업데이트용).
    """
    stocks_df = pd.read_csv(STOCKS_LIST_FILE, dtype={'ticker': str})
    stocks_df['ticker'] = stocks_df['ticker'].str.zfill(6)
    
    existing_data = load_existing_data()
    
    all_data = []
    
    for _, row in tqdm(stocks_df.iterrows(), total=len(stocks_df), desc="Updating today's prices"):
        ticker = row['ticker']
        name = row['name']
        market = row.get('market', 'KOSPI')
        
        try:
            # 최근 1페이지만 가져오기
            df = get_stock_daily_prices(ticker, pages=1)
            
            if not df.empty:
                df['ticker'] = ticker
                df['name'] = name
                df['market'] = market
                all_data.append(df)
                
        except Exception as e:
            logger.warning(f"Failed to update {ticker}: {e}")
        
        time.sleep(0.1)
    
    if not all_data:
        return existing_data
    
    new_data = pd.concat(all_data, ignore_index=True)
    
    # 병합
    if not existing_data.empty:
        combined = pd.concat([existing_data, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['ticker', 'date'], keep='last')
        combined = combined.sort_values(['ticker', 'date']).reset_index(drop=True)
    else:
        combined = new_data
    
    return combined


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect Korean stock daily prices from Naver Finance')
    parser.add_argument('--days', type=int, default=365, help='Days of historical data to collect')
    parser.add_argument('--update', action='store_true', help='Only update recent data')
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("Starting Korean Stock Daily Price Collection")
    logger.info("=" * 50)
    
    start_time = time.time()
    
    if args.update:
        logger.info("Mode: Quick Update (today's prices only)")
        result = update_today_prices()
    else:
        logger.info(f"Mode: Full Collection ({args.days} days)")
        result = collect_all_prices(days_back=args.days)
    
    if not result.empty:
        save_data(result)
        
        # 요약 출력
        logger.info("\n" + "=" * 50)
        logger.info("Collection Summary:")
        logger.info(f"  - Total rows: {len(result):,}")
        logger.info(f"  - Unique tickers: {result['ticker'].nunique()}")
        logger.info(f"  - Date range: {result['date'].min()} ~ {result['date'].max()}")
        logger.info(f"  - Elapsed time: {time.time() - start_time:.1f} seconds")
        logger.info("=" * 50)
    else:
        logger.error("No data to save!")


if __name__ == "__main__":
    main()
