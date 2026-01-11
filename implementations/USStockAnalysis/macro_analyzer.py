#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Macro Market Analyzer
- Collects macro indicators (VIX, Yields, Commodities, etc.)
- Uses Gemini AI to generate investment strategy
"""

import os
import json
import requests
import yfinance as yf
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv
from us_config import get_data_dir

# Load .env
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MacroDataCollector:
    """Collect macro market data from various sources"""
    
    def __init__(self):
        self.macro_tickers = {
            'VIX': '^VIX',
            'DXY': 'DX-Y.NYB',
            '2Y_Yield': '^IRX',
            '10Y_Yield': '^TNX',
            'GOLD': 'GC=F',
            'OIL': 'CL=F',
            'BTC': 'BTC-USD',
            'SPY': 'SPY',
            'QQQ': 'QQQ'
        }
    
    def get_current_macro_data(self) -> Dict:
        """Fetch current macro indicator values"""
        logger.info("Fetching macro data...")
        macro_data = {}
        
        try:
            tickers = list(self.macro_tickers.values())
            data = yf.download(tickers, period='5d', progress=False)
            
            for name, ticker in self.macro_tickers.items():
                try:
                    if ticker not in data['Close'].columns:
                        continue
                    
                    hist = data['Close'][ticker].dropna()
                    if len(hist) < 2:
                        continue
                    
                    val = hist.iloc[-1]
                    prev = hist.iloc[-2]
                    change = ((val / prev) - 1) * 100
                    
                    # 52w High/Low
                    full_hist = yf.Ticker(ticker).history(period='1y')
                    high_52w = full_hist['High'].max() if not full_hist.empty else 0
                    low_52w = full_hist['Low'].min() if not full_hist.empty else 0
                    pct_from_high = ((val / high_52w) - 1) * 100 if high_52w > 0 else 0
                    
                    macro_data[name] = {
                        'value': round(val, 2),
                        'change_1d': round(change, 2),
                        'high_52w': round(high_52w, 2),
                        'low_52w': round(low_52w, 2),
                        'pct_from_high': round(pct_from_high, 1)
                    }
                except Exception as e:
                    logger.debug(f"Error processing {name}: {e}")
            
            # Calculate Yield Spread (10Y - 2Y)
            if '2Y_Yield' in macro_data and '10Y_Yield' in macro_data:
                spread = macro_data['10Y_Yield']['value'] - macro_data['2Y_Yield']['value']
                macro_data['YieldSpread'] = {
                    'value': round(spread, 2),
                    'change_1d': 0,
                    'interpretation': 'Inverted (Recession Signal)' if spread < 0 else 'Normal'
                }
            
            # Fear & Greed interpretation based on VIX
            if 'VIX' in macro_data:
                vix_val = macro_data['VIX']['value']
                if vix_val > 30:
                    fg_value = 20  # Extreme Fear
                    fg_label = "Extreme Fear"
                elif vix_val > 25:
                    fg_value = 35  # Fear
                    fg_label = "Fear"
                elif vix_val > 20:
                    fg_value = 50  # Neutral
                    fg_label = "Neutral"
                elif vix_val > 15:
                    fg_value = 65  # Greed
                    fg_label = "Greed"
                else:
                    fg_value = 80  # Extreme Greed
                    fg_label = "Extreme Greed"
                
                macro_data['FearGreed'] = {
                    'value': fg_value,
                    'label': fg_label,
                    'change_1d': 0
                }
            
        except Exception as e:
            logger.error(f"Error fetching macro data: {e}")
        
        return macro_data

    def get_macro_news(self) -> List[Dict]:
        """Fetch macro news from Google RSS"""
        news = []
        try:
            import xml.etree.ElementTree as ET
            
            url = "https://news.google.com/rss/search?q=Federal+Reserve+Economy+Markets&hl=en-US&gl=US&ceid=US:en"
            resp = requests.get(url, timeout=10)
            
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('.//item')[:5]:
                    title = item.find('title')
                    pub_date = item.find('pubDate')
                    news.append({
                        'title': title.text if title is not None else '',
                        'published': pub_date.text if pub_date is not None else '',
                        'source': 'Google News'
                    })
        except Exception as e:
            logger.debug(f"Error fetching news: {e}")
        
        return news
        
    def get_historical_patterns(self) -> List[Dict]:
        """Get historical market patterns for context"""
        return [
            {
                'event': 'Fed Pivot Signal (2023)',
                'conditions': 'VIX declining, Yields peaking, Dollar weakening',
                'outcome': {'SPY_3m': '+15%', 'best_sectors': ['Technology', 'Communications']}
            },
            {
                'event': 'Rate Hike Pause (2024)',
                'conditions': 'Inflation cooling, Employment stable',
                'outcome': {'SPY_3m': '+8%', 'best_sectors': ['Growth', 'Small Caps']}
            },
            {
                'event': 'VIX Spike Recovery',
                'conditions': 'VIX above 30, then declining',
                'outcome': {'SPY_1m': '+5%', 'strategy': 'Buy fear, sell complacency'}
            }
        ]


class MacroAIAnalyzer:
    """Gemini AI Analysis for Macro Data"""
    
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    
    def analyze(self, data: Dict, news: List, patterns: List, lang: str = 'ko') -> str:
        """Generate AI analysis of macro conditions"""
        if not self.api_key:
            return "API Key not configured. Set GOOGLE_API_KEY in .env file."
        
        prompt = self._build_prompt(data, news, patterns, lang)
        
        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 2000
                }
            }
            
            resp = requests.post(
                f"{self.url}?key={self.api_key}",
                json=payload,
                timeout=30
            )
            
            if resp.status_code == 200:
                result = resp.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return f"API Error: {resp.status_code}"
                
        except Exception as e:
            return f"Error generating analysis: {e}"
    
    def _build_prompt(self, data: Dict, news: List, patterns: List, lang: str) -> str:
        """Build the AI prompt"""
        metrics = "\n".join([f"- {k}: {v.get('value', 'N/A')} (1D: {v.get('change_1d', 0):+.2f}%)" 
                           for k, v in data.items()])
        headlines = "\n".join([f"- {n['title']}" for n in news[:5]])
        pattern_text = "\n".join([f"- {p['event']}: {p['outcome']}" for p in patterns])
        
        if lang == 'en':
            return f"""You are a professional macro strategist. Analyze current market conditions.

## Current Macro Indicators:
{metrics}

## Recent News Headlines:
{headlines}

## Historical Patterns:
{pattern_text}

## Provide Analysis:
1. **Market Summary**: Current state in 2-3 sentences
2. **Opportunities**: Which sectors/assets benefit from current conditions
3. **Risks**: Key risks to monitor
4. **Strategy**: Specific actionable recommendations

Be concise and professional. Focus on actionable insights."""
        else:
            return f"""당신은 전문 매크로 전략가입니다. 현재 시장 상황을 분석하세요.

## 현재 매크로 지표:
{metrics}

## 최근 뉴스 헤드라인:
{headlines}

## 역사적 패턴:
{pattern_text}

## 분석 요청:
1. **시장 요약**: 현재 상황 2-3문장 정리
2. **기회 영역**: 현재 조건에서 유리한 섹터/자산
3. **리스크 요인**: 주시해야 할 핵심 리스크
4. **투자 전략**: 구체적이고 실행 가능한 권고

간결하고 전문적으로 작성. 실행 가능한 인사이트에 집중."""


class MultiModelAnalyzer:
    """Main analyzer combining data collection and AI analysis"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.collector = MacroDataCollector()
        self.gemini = MacroAIAnalyzer()
    
    def run(self):
        """Run full macro analysis pipeline"""
        logger.info("Starting Macro Analysis...")
        
        # Collect data
        data = self.collector.get_current_macro_data()
        news = self.collector.get_macro_news()
        patterns = self.collector.get_historical_patterns()
        
        # Generate AI Analysis (Korean)
        logger.info("Generating AI analysis (Korean)...")
        analysis_ko = self.gemini.analyze(data, news, patterns, 'ko')
        
        # Generate AI Analysis (English)
        logger.info("Generating AI analysis (English)...")
        analysis_en = self.gemini.analyze(data, news, patterns, 'en')
        
        # Build output
        output = {
            'timestamp': datetime.now().isoformat(),
            'macro_indicators': data,
            'news': news,
            'historical_patterns': patterns,
            'ai_analysis': analysis_ko,
            'ai_analysis_en': analysis_en
        }
        
        # Save Korean version
        output_file = os.path.join(self.data_dir, 'macro_analysis.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {output_file}")
        
        # Save English version
        output_en = output.copy()
        output_en['ai_analysis'] = analysis_en
        output_file_en = os.path.join(self.data_dir, 'macro_analysis_en.json')
        with open(output_file_en, 'w', encoding='utf-8') as f:
            json.dump(output_en, f, indent=2)
        logger.info(f"Saved: {output_file_en}")
        
        return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Macro Market Analyzer')
    parser.add_argument('--dir', default=get_data_dir(), help='Data directory')
    args = parser.parse_args()
    
    os.makedirs(args.dir, exist_ok=True)
    analyzer = MultiModelAnalyzer(data_dir=args.dir)
    result = analyzer.run()
    
    # Print summary
    print("\nMacro Analysis Complete")
    print("=" * 60)
    
    if 'macro_indicators' in result:
        print("\nKey Indicators:")
        for name, data in result['macro_indicators'].items():
            if isinstance(data, dict) and 'value' in data:
                change = data.get('change_1d', 0)
                emoji = "+" if change >= 0 else ""
                print(f"  {name}: {data['value']} ({emoji}{change:.2f}%)")
    
    print("\nAI Analysis Preview:")
    print("-" * 40)
    if result.get('ai_analysis'):
        preview = result['ai_analysis'][:500]
        print(preview + "..." if len(result['ai_analysis']) > 500 else preview)


if __name__ == "__main__":
    main()
