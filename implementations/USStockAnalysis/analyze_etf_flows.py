#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US ETF Fund Flow Analysis
Analyzes major ETF fund flows and generates AI insights using Gemini
Tracks 24 major ETFs across different asset classes
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from tqdm import tqdm
from dotenv import load_dotenv
from us_config import get_data_dir

# Load environment variables
load_dotenv()

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ETFFlowAnalyzer:
    """
    Analyze ETF fund flows and generate AI insights
    Uses volume patterns and price momentum as flow proxies
    """
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.output_file = os.path.join(data_dir, 'us_etf_flows.csv')
        self.analysis_file = os.path.join(data_dir, 'etf_flow_analysis.json')
        
        # Major ETFs to track (24 ETFs across asset classes)
        self.tracked_etfs = {
            # Equity - US Broad Market
            'SPY': {'name': 'SPDR S&P 500', 'category': 'US Large Cap', 'asset_class': 'Equity'},
            'QQQ': {'name': 'Invesco QQQ Trust', 'category': 'US Tech/Growth', 'asset_class': 'Equity'},
            'IWM': {'name': 'iShares Russell 2000', 'category': 'US Small Cap', 'asset_class': 'Equity'},
            'DIA': {'name': 'SPDR Dow Jones', 'category': 'US Large Cap', 'asset_class': 'Equity'},
            'VTI': {'name': 'Vanguard Total Stock', 'category': 'US Total Market', 'asset_class': 'Equity'},
            
            # Equity - Sectors
            'XLF': {'name': 'Financial Select', 'category': 'US Financials', 'asset_class': 'Equity'},
            'XLK': {'name': 'Technology Select', 'category': 'US Technology', 'asset_class': 'Equity'},
            'XLE': {'name': 'Energy Select', 'category': 'US Energy', 'asset_class': 'Equity'},
            'XLV': {'name': 'Health Care Select', 'category': 'US Healthcare', 'asset_class': 'Equity'},
            'XLI': {'name': 'Industrial Select', 'category': 'US Industrials', 'asset_class': 'Equity'},
            
            # Equity - International
            'EFA': {'name': 'iShares MSCI EAFE', 'category': 'Developed Markets', 'asset_class': 'Equity'},
            'EEM': {'name': 'iShares MSCI Emerging', 'category': 'Emerging Markets', 'asset_class': 'Equity'},
            'VEA': {'name': 'Vanguard FTSE Developed', 'category': 'Developed ex-US', 'asset_class': 'Equity'},
            'VWO': {'name': 'Vanguard FTSE Emerging', 'category': 'Emerging Markets', 'asset_class': 'Equity'},
            
            # Fixed Income
            'TLT': {'name': 'iShares 20+ Year Treasury', 'category': 'Long-Term Bonds', 'asset_class': 'Fixed Income'},
            'IEF': {'name': 'iShares 7-10 Year Treasury', 'category': 'Intermediate Bonds', 'asset_class': 'Fixed Income'},
            'LQD': {'name': 'iShares Investment Grade', 'category': 'Corporate Bonds', 'asset_class': 'Fixed Income'},
            'HYG': {'name': 'iShares High Yield', 'category': 'High Yield Bonds', 'asset_class': 'Fixed Income'},
            
            # Commodities
            'GLD': {'name': 'SPDR Gold Shares', 'category': 'Gold', 'asset_class': 'Commodity'},
            'SLV': {'name': 'iShares Silver Trust', 'category': 'Silver', 'asset_class': 'Commodity'},
            'USO': {'name': 'United States Oil Fund', 'category': 'Oil', 'asset_class': 'Commodity'},
            
            # Real Estate
            'VNQ': {'name': 'Vanguard Real Estate', 'category': 'US REITs', 'asset_class': 'Real Estate'},
            
            # Volatility/Hedge
            'VIXY': {'name': 'ProShares VIX Short-Term', 'category': 'Volatility', 'asset_class': 'Volatility'},
            'SQQQ': {'name': 'ProShares UltraPro Short QQQ', 'category': 'Inverse Equity', 'asset_class': 'Inverse'},
        }
        
        # Gemini API setup
        self.gemini_api_key = os.getenv('GOOGLE_API_KEY')
        
    def fetch_etf_data(self, symbol: str, period: str = '3mo') -> pd.DataFrame:
        """Fetch ETF price and volume data"""
        try:
            etf = yf.Ticker(symbol)
            hist = etf.history(period=period)
            
            if hist.empty:
                return pd.DataFrame()
            
            hist = hist.reset_index()
            hist.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
            hist['ticker'] = symbol
            
            return hist[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.debug(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_flow_proxy(self, df: pd.DataFrame) -> Dict:
        """
        Calculate fund flow proxy using volume and price patterns
        Returns a flow score (0-100) based on:
        - OBV trend
        - Volume ratio (recent vs historical)
        - Price momentum
        """
        if len(df) < 20:
            return None
        
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate OBV
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        # OBV trend (20-day)
        obv_change = (obv[-1] - obv[-20]) / abs(obv[-20]) * 100 if obv[-20] != 0 else 0
        
        # Volume ratio (5-day vs 20-day)
        vol_5d = df['volume'].tail(5).mean()
        vol_20d = df['volume'].tail(20).mean()
        vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1
        
        # Price momentum (5-day and 20-day returns)
        latest_price = df['close'].iloc[-1]
        price_5d_ago = df['close'].iloc[-5] if len(df) >= 5 else df['close'].iloc[0]
        price_20d_ago = df['close'].iloc[-20] if len(df) >= 20 else df['close'].iloc[0]
        
        return_5d = (latest_price - price_5d_ago) / price_5d_ago * 100
        return_20d = (latest_price - price_20d_ago) / price_20d_ago * 100
        
        # Calculate flow score (0-100)
        score = 50
        
        # OBV contribution (max +/-20)
        if obv_change > 15:
            score += 20
        elif obv_change > 10:
            score += 15
        elif obv_change > 5:
            score += 10
        elif obv_change < -15:
            score -= 20
        elif obv_change < -10:
            score -= 15
        elif obv_change < -5:
            score -= 10
        
        # Volume contribution (max +/-15)
        if vol_ratio > 1.5:
            score += 15
        elif vol_ratio > 1.2:
            score += 10
        elif vol_ratio < 0.7:
            score -= 10
        
        # Momentum contribution (max +/-15)
        if return_5d > 3:
            score += 10
        elif return_5d > 1:
            score += 5
        elif return_5d < -3:
            score -= 10
        elif return_5d < -1:
            score -= 5
        
        if return_20d > 5:
            score += 5
        elif return_20d < -5:
            score -= 5
        
        score = max(0, min(100, score))
        
        # Determine flow category
        if score >= 70:
            flow_category = "Strong Inflow"
        elif score >= 55:
            flow_category = "Moderate Inflow"
        elif score >= 45:
            flow_category = "Neutral"
        elif score >= 30:
            flow_category = "Moderate Outflow"
        else:
            flow_category = "Strong Outflow"
        
        return {
            'date': df['date'].iloc[-1],
            'price': round(latest_price, 2),
            'return_5d': round(return_5d, 2),
            'return_20d': round(return_20d, 2),
            'volume_ratio': round(vol_ratio, 2),
            'obv_change_20d': round(obv_change, 2),
            'flow_score': round(score, 1),
            'flow_category': flow_category
        }
    
    def generate_ai_analysis(self, results_df: pd.DataFrame) -> Dict:
        """
        Generate AI-powered analysis of ETF fund flows using Gemini
        """
        if not self.gemini_api_key:
            logger.warning("GOOGLE_API_KEY not set. Skipping AI analysis.")
            return {"error": "API key not configured"}
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Prepare data summary for AI
            summary_data = []
            for _, row in results_df.iterrows():
                etf_info = self.tracked_etfs.get(row['ticker'], {})
                summary_data.append({
                    'ticker': row['ticker'],
                    'name': etf_info.get('name', row['ticker']),
                    'category': etf_info.get('category', 'Unknown'),
                    'asset_class': etf_info.get('asset_class', 'Unknown'),
                    'flow_score': row['flow_score'],
                    'flow_category': row['flow_category'],
                    'return_5d': row['return_5d'],
                    'return_20d': row['return_20d'],
                    'volume_ratio': row['volume_ratio']
                })
            
            # Group by asset class
            asset_class_summary = {}
            for item in summary_data:
                ac = item['asset_class']
                if ac not in asset_class_summary:
                    asset_class_summary[ac] = []
                asset_class_summary[ac].append(item)
            
            prompt = f"""
You are a professional fund flow analyst. Analyze the following ETF fund flow data and provide insights.

## ETF Fund Flow Data (by Asset Class):

{json.dumps(asset_class_summary, indent=2)}

## Analysis Guidelines:
1. Identify which asset classes are seeing inflows vs outflows
2. Note any sector rotation patterns
3. Identify risk-on vs risk-off signals
4. Highlight any unusual flow patterns
5. Provide a brief market sentiment assessment

Please provide a structured analysis in Korean with the following sections:
1. **자금 흐름 요약** (Fund Flow Summary)
2. **섹터 로테이션** (Sector Rotation)
3. **시장 심리** (Market Sentiment)
4. **주요 인사이트** (Key Insights)
5. **투자 시사점** (Investment Implications)

Keep the analysis concise but insightful.
"""
            
            response = model.generate_content(prompt)
            
            analysis_result = {
                'generated_at': datetime.now().isoformat(),
                'data_summary': asset_class_summary,
                'ai_analysis': response.text,
                'model': 'gemini-1.5-flash'
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {"error": str(e)}
    
    def run(self) -> pd.DataFrame:
        """Run ETF flow analysis for all tracked ETFs"""
        logger.info("Starting ETF Flow Analysis...")
        
        results = []
        
        for symbol, info in tqdm(self.tracked_etfs.items(), desc="Analyzing ETF flows"):
            # Fetch data
            df = self.fetch_etf_data(symbol)
            
            if df.empty:
                logger.warning(f"No data for {symbol}")
                continue
            
            # Calculate flow proxy
            flow_data = self.calculate_flow_proxy(df)
            
            if flow_data:
                result = {
                    'ticker': symbol,
                    'name': info['name'],
                    'category': info['category'],
                    'asset_class': info['asset_class'],
                    **flow_data
                }
                results.append(result)
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        if results_df.empty:
            logger.warning("No ETF data collected")
            return pd.DataFrame()
        
        # Save CSV results
        results_df.to_csv(self.output_file, index=False)
        logger.info(f"ETF flow data saved to {self.output_file}")
        
        # Generate AI analysis
        logger.info("Generating AI analysis...")
        ai_analysis = self.generate_ai_analysis(results_df)
        
        # Save AI analysis
        with open(self.analysis_file, 'w', encoding='utf-8') as f:
            json.dump(ai_analysis, f, ensure_ascii=False, indent=2)
        logger.info(f"AI analysis saved to {self.analysis_file}")
        
        # Print summary
        logger.info("\n--- ETF Flow Summary ---")
        for asset_class in results_df['asset_class'].unique():
            ac_data = results_df[results_df['asset_class'] == asset_class]
            avg_score = ac_data['flow_score'].mean()
            logger.info(f"{asset_class}: Avg Flow Score {avg_score:.1f}")
        
        return results_df


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ETF Fund Flow Analysis')
    parser.add_argument('--dir', default=get_data_dir(), help='Data directory')
    parser.add_argument('--no-ai', action='store_true', help='Skip AI analysis')
    args = parser.parse_args()
    
    os.makedirs(args.dir, exist_ok=True)
    analyzer = ETFFlowAnalyzer(data_dir=args.dir)
    
    if args.no_ai:
        analyzer.gemini_api_key = None
    
    results = analyzer.run()
    
    if not results.empty:
        # Show top inflows
        print("\nTop 5 Inflows:")
        top_5 = results.nlargest(5, 'flow_score')
        for _, row in top_5.iterrows():
            print(f"   {row['ticker']} ({row['category']}): Score {row['flow_score']} - {row['flow_category']}")
        
        # Show top outflows
        print("\nTop 5 Outflows:")
        bottom_5 = results.nsmallest(5, 'flow_score')
        for _, row in bottom_5.iterrows():
            print(f"   {row['ticker']} ({row['category']}): Score {row['flow_score']} - {row['flow_category']}")


if __name__ == "__main__":
    main()
