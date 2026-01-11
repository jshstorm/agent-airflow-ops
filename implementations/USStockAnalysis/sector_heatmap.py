#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sector Performance Heatmap Data Collector
Collects sector ETF performance data for heatmap visualization
"""

import os
import json
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict, List
import logging
from us_config import get_data_dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SectorHeatmapCollector:
    """Collect sector ETF performance data for heatmap visualization"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.output_file = os.path.join(data_dir, 'sector_heatmap.json')
        self.csv_output_file = os.path.join(data_dir, 'us_sector_heatmap.csv')
        
        # Sector ETFs with full names
        self.sector_etfs = {
            'XLK': {'name': 'Technology', 'color': '#4A90A4'},
            'XLF': {'name': 'Financials', 'color': '#6B8E23'},
            'XLV': {'name': 'Healthcare', 'color': '#FF69B4'},
            'XLE': {'name': 'Energy', 'color': '#FF6347'},
            'XLY': {'name': 'Consumer Disc.', 'color': '#FFD700'},
            'XLP': {'name': 'Consumer Staples', 'color': '#98D8C8'},
            'XLI': {'name': 'Industrials', 'color': '#DDA0DD'},
            'XLB': {'name': 'Materials', 'color': '#F0E68C'},
            'XLU': {'name': 'Utilities', 'color': '#87CEEB'},
            'XLRE': {'name': 'Real Estate', 'color': '#CD853F'},
            'XLC': {'name': 'Comm. Services', 'color': '#9370DB'},
        }
        
        # Sector stocks for detail map
        self.sector_stocks = {
            'Technology': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'CRM', 'AMD', 'ADBE', 'CSCO', 'INTC'],
            'Financials': ['BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'SCHW'],
            'Healthcare': ['UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PXD', 'VLO', 'PSX', 'OXY'],
            'Consumer Disc.': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG'],
            'Consumer Staples': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'CL', 'MDLZ', 'KMB'],
            'Industrials': ['CAT', 'HON', 'UNP', 'BA', 'RTX', 'DE', 'UPS', 'GE', 'LMT', 'MMM'],
            'Materials': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'ECL', 'NUE', 'DOW', 'DD', 'PPG'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'PEG', 'ED'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'WELL', 'SPG', 'DLR', 'AVB'],
            'Comm. Services': ['META', 'GOOGL', 'GOOG', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR'],
        }
    
    def get_sector_etf_performance(self, period: str = '5d') -> Dict:
        """Get sector ETF performance data"""
        logger.info(f"Fetching sector ETF performance ({period})...")
        
        etf_tickers = list(self.sector_etfs.keys())
        
        try:
            data = yf.download(etf_tickers, period=period, progress=False)
            
            if data.empty:
                return {'error': 'No data'}
            
            sectors = []
            for ticker, info in self.sector_etfs.items():
                try:
                    if ticker not in data['Close'].columns:
                        continue
                    
                    prices = data['Close'][ticker].dropna()
                    if len(prices) < 2:
                        continue
                    
                    current = prices.iloc[-1]
                    prev = prices.iloc[-2]
                    first = prices.iloc[0]
                    
                    change_1d = ((current / prev) - 1) * 100
                    change_period = ((current / first) - 1) * 100
                    
                    sectors.append({
                        'ticker': ticker,
                        'name': info['name'],
                        'price': round(current, 2),
                        'change_1d': round(change_1d, 2),
                        'change_period': round(change_period, 2),
                        'color': self._get_color(change_1d)
                    })
                except Exception as e:
                    logger.debug(f"Error processing {ticker}: {e}")
            
            # Sort by 1-day change
            sectors.sort(key=lambda x: x['change_1d'], reverse=True)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'period': period,
                'sectors': sectors
            }
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return {'error': str(e)}
    
    def get_full_market_map(self, period: str = '5d') -> Dict:
        """Get full market map data (Sectors -> Stocks) for Treemap"""
        logger.info(f"Fetching full market map data ({period})...")
        
        all_tickers = []
        ticker_to_sector = {}
        for sector, stocks in self.sector_stocks.items():
            all_tickers.extend(stocks)
            for stock in stocks:
                ticker_to_sector[stock] = sector
                
        try:
            data = yf.download(all_tickers, period=period, progress=False)
            
            if data.empty:
                return {'error': 'No data'}
            
            market_map = {name: [] for name in self.sector_stocks.keys()}
            
            for ticker in all_tickers:
                try:
                    if ticker not in data['Close'].columns:
                        continue
                    
                    prices = data['Close'][ticker].dropna()
                    if len(prices) < 2:
                        continue
                    
                    current = prices.iloc[-1]
                    prev = prices.iloc[-2]
                    change = ((current / prev) - 1) * 100
                    
                    # Weight by Volume * Price (Activity proxy)
                    vol = data['Volume'][ticker].iloc[-1] if 'Volume' in data.columns else 100000
                    weight = current * vol
                    
                    sector = ticker_to_sector.get(ticker, 'Unknown')
                    if sector in market_map:
                        market_map[sector].append({
                            'x': ticker,
                            'y': round(weight, 0),
                            'price': round(current, 2),
                            'change': round(change, 2),
                            'color': self._get_color(change)
                        })
                except Exception as e:
                    logger.debug(f"Error processing {ticker}: {e}")
            
            series = []
            for sector_name, stocks in market_map.items():
                if stocks:
                    stocks.sort(key=lambda x: x['y'], reverse=True)
                    series.append({'name': sector_name, 'data': stocks})
            
            series.sort(key=lambda s: sum(i['y'] for i in s['data']), reverse=True)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'period': period,
                'series': series
            }
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return {'error': str(e)}
            
    def _get_color(self, change: float) -> str:
        """Get color based on change percentage"""
        if change >= 3: return '#00C853'
        elif change >= 1: return '#4CAF50'
        elif change >= 0: return '#81C784'
        elif change >= -1: return '#EF9A9A'
        elif change >= -3: return '#F44336'
        else: return '#B71C1C'

    def run(self):
        """Run the collector and save data"""
        # Get sector ETF performance
        etf_data = self.get_sector_etf_performance('5d')
        
        # Get full market map
        market_map = self.get_full_market_map('5d')
        
        # Combine data
        combined = {
            'timestamp': datetime.now().isoformat(),
            'sector_etfs': etf_data,
            'market_map': market_map
        }
        
        # Save to file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        
        if isinstance(etf_data, dict) and etf_data.get('sectors'):
            df = pd.DataFrame(etf_data['sectors'])
            df.to_csv(self.csv_output_file, index=False)
            logger.info(f"Saved sector CSV: {self.csv_output_file}")

        logger.info(f"Saved to {self.output_file}")
        return combined


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sector Heatmap Data Collector')
    parser.add_argument('--dir', default=get_data_dir(), help='Data directory')
    args = parser.parse_args()
    
    os.makedirs(args.dir, exist_ok=True)
    collector = SectorHeatmapCollector(data_dir=args.dir)
    result = collector.run()
    
    # Print summary
    if 'sector_etfs' in result and 'sectors' in result['sector_etfs']:
        print("\nSector Performance Summary:")
        print("=" * 50)
        for sector in result['sector_etfs']['sectors']:
            emoji = "+" if sector['change_1d'] >= 0 else ""
            print(f"  {sector['name']}: {emoji}{sector['change_1d']:.2f}%")


if __name__ == "__main__":
    main()
