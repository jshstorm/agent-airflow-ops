#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Insider Trading Tracker
Tracks insider buying/selling activity from SEC filings via yfinance
"""

import os
import json
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List
from us_config import get_data_dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InsiderTracker:
    """Track and analyze insider trading activity"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.output_file = os.path.join(data_dir, 'insider_moves.json')
        
    def get_insider_activity(self, ticker: str) -> Dict:
        """Get insider trading activity for a single ticker"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get insider transactions
            transactions = stock.insider_transactions
            if transactions is None or transactions.empty:
                return {'ticker': ticker, 'transactions': [], 'summary': None}
            
            # Get insider purchases (last 6 months)
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=180)
            
            recent_buys = []
            recent_sells = []
            
            for date, row in transactions.iterrows():
                try:
                    if isinstance(date, pd.Timestamp) and date < cutoff:
                        continue
                    
                    text = str(row.get('Text', '')).lower()
                    transaction = str(row.get('Transaction', '')).lower()
                    
                    insider_name = row.get('Insider', 'Unknown')
                    value = float(row.get('Value', 0) or 0)
                    shares = int(row.get('Shares', 0) or 0)
                    
                    # Determine if buy or sell
                    is_buy = 'purchase' in text or 'buy' in text or 'acquisition' in transaction
                    is_sell = 'sale' in text or 'sell' in text or 'disposition' in transaction
                    
                    trans_data = {
                        'date': str(date.date()) if isinstance(date, pd.Timestamp) else str(date),
                        'insider': insider_name,
                        'value': value,
                        'shares': shares,
                        'transaction_type': row.get('Transaction', 'Unknown')
                    }
                    
                    if is_buy and not is_sell:
                        recent_buys.append(trans_data)
                    elif is_sell and not is_buy:
                        recent_sells.append(trans_data)
                        
                except Exception as e:
                    continue
            
            # Calculate summary
            total_buy_value = sum(t['value'] for t in recent_buys)
            total_sell_value = sum(t['value'] for t in recent_sells)
            
            # Insider score (0-100)
            score = 50
            
            # More buys than sells is positive
            if len(recent_buys) > len(recent_sells):
                score += 15
            elif len(recent_sells) > len(recent_buys):
                score -= 10
            
            # Large buy values are very positive
            if total_buy_value > 1_000_000:
                score += 20
            elif total_buy_value > 100_000:
                score += 10
            
            # Large sell values are negative
            if total_sell_value > 10_000_000:
                score -= 15
            elif total_sell_value > 1_000_000:
                score -= 5
            
            score = max(0, min(100, score))
            
            # Sentiment
            if score >= 70:
                sentiment = "Strong Insider Buying"
            elif score >= 55:
                sentiment = "Insider Buying"
            elif score >= 45:
                sentiment = "Neutral"
            elif score >= 30:
                sentiment = "Insider Selling"
            else:
                sentiment = "Heavy Insider Selling"
            
            return {
                'ticker': ticker,
                'transactions': {
                    'recent_buys': recent_buys[:10],  # Top 10
                    'recent_sells': recent_sells[:10]
                },
                'summary': {
                    'buy_count': len(recent_buys),
                    'sell_count': len(recent_sells),
                    'total_buy_value': total_buy_value,
                    'total_sell_value': total_sell_value,
                    'net_value': total_buy_value - total_sell_value,
                    'insider_score': score,
                    'sentiment': sentiment
                }
            }
            
        except Exception as e:
            logger.debug(f"Error analyzing {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}
    
    def analyze_tickers(self, tickers: List[str]) -> Dict:
        """Analyze insider activity for multiple tickers"""
        logger.info(f"Analyzing insider activity for {len(tickers)} tickers...")
        
        results = []
        top_buying = []
        top_selling = []
        
        for ticker in tickers:
            result = self.get_insider_activity(ticker)
            
            if 'error' not in result and result['summary'] is not None:
                results.append(result)
                
                summary = result['summary']
                if summary['buy_count'] > 0:
                    top_buying.append({
                        'ticker': ticker,
                        'buy_count': summary['buy_count'],
                        'total_value': summary['total_buy_value'],
                        'score': summary['insider_score']
                    })
                
                if summary['sell_count'] > 0:
                    top_selling.append({
                        'ticker': ticker,
                        'sell_count': summary['sell_count'],
                        'total_value': summary['total_sell_value'],
                        'score': summary['insider_score']
                    })
        
        # Sort by activity
        top_buying.sort(key=lambda x: x['total_value'], reverse=True)
        top_selling.sort(key=lambda x: x['total_value'], reverse=True)
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_analyzed': len(results),
                'with_buying': len([r for r in results if r['summary']['buy_count'] > 0]),
                'with_selling': len([r for r in results if r['summary']['sell_count'] > 0])
            },
            'top_buying': top_buying[:10],
            'top_selling': top_selling[:10],
            'details': {r['ticker']: r for r in results}
        }
        
        return output
    
    def run(self, tickers: List[str] = None):
        """Run analysis and save results"""
        if tickers is None:
            # Default to top S&P 500 stocks
            tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
                'UNH', 'JNJ', 'JPM', 'V', 'XOM', 'PG', 'MA', 'HD', 'CVX', 'MRK',
                'ABBV', 'LLY', 'PEP', 'KO', 'COST', 'AVGO', 'WMT', 'MCD', 'TMO',
                'CSCO', 'ABT', 'CRM'
            ]
        
        result = self.analyze_tickers(tickers)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved insider tracking data to {self.output_file}")
        return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Insider Trading Tracker')
    parser.add_argument('--dir', default=get_data_dir(), help='Data directory')
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to analyze')
    args = parser.parse_args()
    
    os.makedirs(args.dir, exist_ok=True)
    tracker = InsiderTracker(data_dir=args.dir)
    result = tracker.run(tickers=args.tickers)
    
    # Print summary
    print("\nInsider Activity Summary:")
    print("=" * 60)
    print(f"Analyzed: {result['summary']['total_analyzed']} stocks")
    print(f"With Insider Buying: {result['summary']['with_buying']}")
    print(f"With Insider Selling: {result['summary']['with_selling']}")
    
    if result['top_buying']:
        print("\nTop Insider Buying:")
        for item in result['top_buying'][:5]:
            print(f"  {item['ticker']}: {item['buy_count']} buys, ${item['total_value']:,.0f} value")
    
    if result['top_selling']:
        print("\nTop Insider Selling:")
        for item in result['top_selling'][:5]:
            print(f"  {item['ticker']}: {item['sell_count']} sells, ${item['total_value']:,.0f} value")


if __name__ == "__main__":
    main()
