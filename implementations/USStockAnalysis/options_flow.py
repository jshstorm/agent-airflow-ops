#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Options Flow Analyzer
Tracks options volume and unusual activity to detect large investor positioning
"""

import os
import json
import logging
from us_config import get_data_dir
import yfinance as yf
from datetime import datetime
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptionsFlowAnalyzer:
    """Analyze options flow to detect institutional positioning"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.output_file = os.path.join(data_dir, 'options_flow.json')
        
        # Default watchlist of major stocks
        self.watchlist = [
            'AAPL', 'NVDA', 'TSLA', 'MSFT', 'AMZN', 
            'META', 'GOOGL', 'SPY', 'QQQ', 'AMD',
            'NFLX', 'BA', 'DIS', 'JPM', 'V'
        ]
    
    def get_options_summary(self, ticker: str) -> Dict:
        """Get options summary for a single ticker"""
        try:
            stock = yf.Ticker(ticker)
            exps = stock.options
            
            if not exps:
                return {'ticker': ticker, 'error': 'No options available'}
            
            # Get nearest expiration
            opt = stock.option_chain(exps[0])
            calls, puts = opt.calls, opt.puts
            
            # Volume and Open Interest
            call_vol = calls['volume'].sum()
            put_vol = puts['volume'].sum()
            call_oi = calls['openInterest'].sum()
            put_oi = puts['openInterest'].sum()
            
            # Put/Call Ratio
            pc_ratio = put_vol / call_vol if call_vol > 0 else 0
            
            # Unusual activity detection
            avg_call_vol = calls['volume'].mean()
            avg_put_vol = puts['volume'].mean()
            
            unusual_calls = calls[calls['volume'] > avg_call_vol * 3]
            unusual_puts = puts[puts['volume'] > avg_put_vol * 3]
            
            # Get current stock price
            info = stock.info
            current_price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0) or 0
            
            # Find most active strikes
            top_call_strikes = calls.nlargest(3, 'volume')[['strike', 'volume', 'lastPrice']].to_dict('records')
            top_put_strikes = puts.nlargest(3, 'volume')[['strike', 'volume', 'lastPrice']].to_dict('records')
            
            # Signal interpretation
            if pc_ratio < 0.5:
                sentiment = "Bullish"
            elif pc_ratio > 1.0:
                sentiment = "Bearish"
            else:
                sentiment = "Neutral"
            
            return {
                'ticker': ticker,
                'current_price': round(current_price, 2),
                'expiration': exps[0],
                'metrics': {
                    'pc_ratio': round(pc_ratio, 2),
                    'call_volume': int(call_vol),
                    'put_volume': int(put_vol),
                    'call_oi': int(call_oi),
                    'put_oi': int(put_oi),
                    'total_volume': int(call_vol + put_vol),
                    'total_oi': int(call_oi + put_oi)
                },
                'unusual_activity': {
                    'unusual_calls': len(unusual_calls),
                    'unusual_puts': len(unusual_puts),
                    'has_unusual': len(unusual_calls) > 0 or len(unusual_puts) > 0
                },
                'top_strikes': {
                    'calls': top_call_strikes,
                    'puts': top_put_strikes
                },
                'sentiment': sentiment
            }
            
        except Exception as e:
            logger.debug(f"Error analyzing {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}
    
    def analyze_watchlist(self, tickers: List[str] = None) -> Dict:
        """Analyze options flow for watchlist"""
        if tickers is None:
            tickers = self.watchlist
        
        logger.info(f"Analyzing options flow for {len(tickers)} tickers...")
        
        results = []
        unusual_activity = []
        
        for ticker in tickers:
            result = self.get_options_summary(ticker)
            
            if 'error' not in result:
                results.append(result)
                
                # Track unusual activity
                if result['unusual_activity']['has_unusual']:
                    unusual_activity.append({
                        'ticker': ticker,
                        'unusual_calls': result['unusual_activity']['unusual_calls'],
                        'unusual_puts': result['unusual_activity']['unusual_puts'],
                        'sentiment': result['sentiment']
                    })
        
        # Sort by total volume
        results.sort(key=lambda x: x['metrics']['total_volume'], reverse=True)
        
        # Calculate market-wide metrics
        total_call_vol = sum(r['metrics']['call_volume'] for r in results)
        total_put_vol = sum(r['metrics']['put_volume'] for r in results)
        market_pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        
        # Market sentiment
        bullish_count = len([r for r in results if r['sentiment'] == 'Bullish'])
        bearish_count = len([r for r in results if r['sentiment'] == 'Bearish'])
        
        if bullish_count > bearish_count * 1.5:
            market_sentiment = "Bullish"
        elif bearish_count > bullish_count * 1.5:
            market_sentiment = "Bearish"
        else:
            market_sentiment = "Neutral"
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'market_summary': {
                'total_tickers': len(results),
                'market_pc_ratio': round(market_pc_ratio, 2),
                'market_sentiment': market_sentiment,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count
            },
            'unusual_activity': unusual_activity,
            'options_flow': results
        }
        
        return output
    
    def run(self, tickers: List[str] = None):
        """Run analysis and save results"""
        result = self.analyze_watchlist(tickers)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved options flow analysis to {self.output_file}")
        return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Options Flow Analyzer')
    parser.add_argument('--dir', default=get_data_dir(), help='Data directory')
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to analyze')
    args = parser.parse_args()
    
    os.makedirs(args.dir, exist_ok=True)
    analyzer = OptionsFlowAnalyzer(data_dir=args.dir)
    result = analyzer.run(tickers=args.tickers)
    
    # Print summary
    print("\nOptions Flow Summary:")
    print("=" * 60)
    print(f"Market P/C Ratio: {result['market_summary']['market_pc_ratio']}")
    print(f"Market Sentiment: {result['market_summary']['market_sentiment']}")
    print(f"Bullish: {result['market_summary']['bullish_count']} | Bearish: {result['market_summary']['bearish_count']}")
    
    if result['unusual_activity']:
        print("\nUnusual Activity Detected:")
        for item in result['unusual_activity'][:5]:
            print(f"  {item['ticker']}: Calls={item['unusual_calls']}, Puts={item['unusual_puts']} ({item['sentiment']})")
    
    print("\nTop 5 by Volume:")
    for item in result['options_flow'][:5]:
        print(f"  {item['ticker']}: Vol={item['metrics']['total_volume']:,} | P/C={item['metrics']['pc_ratio']:.2f} | {item['sentiment']}")


if __name__ == "__main__":
    main()
