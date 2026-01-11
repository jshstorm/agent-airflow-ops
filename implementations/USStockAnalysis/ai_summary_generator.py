#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Stock Summary Generator
Generates investment summaries for top picks using Gemini AI
"""

import os
import json
import logging
import time
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from us_config import get_data_dir

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NewsCollector:
    """Collect news for individual stocks"""
    
    def get_news(self, ticker: str) -> list:
        """Fetch recent news for a ticker from Google News RSS"""
        news = []
        try:
            import xml.etree.ElementTree as ET
            
            url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            resp = requests.get(url, timeout=5)
            
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('.//item')[:3]:
                    title = item.find('title')
                    pub_date = item.find('pubDate')
                    news.append({
                        'title': title.text if title is not None else '',
                        'published': pub_date.text if pub_date is not None else ''
                    })
        except Exception as e:
            logger.debug(f"Error fetching news for {ticker}: {e}")
        
        return news


class GeminiGenerator:
    """Generate AI summaries using Gemini"""
    
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        
    def generate(self, ticker: str, data: dict, news: list, lang: str = 'ko') -> str:
        """Generate investment summary for a stock"""
        if not self.api_key:
            return "API Key not configured"
        
        news_text = "\n".join([f"- {n['title']}" for n in news]) if news else "No recent news"
        
        # Build score info
        score_info = f"""
- Composite Score: {data.get('composite_score', 'N/A')}/100
- Grade: {data.get('grade', 'N/A')}
- S/D Score: {data.get('sd_score', 'N/A')}
- Technical Score: {data.get('tech_score', 'N/A')}
- Fundamental Score: {data.get('fund_score', 'N/A')}
- Price: ${data.get('current_price', 'N/A')}
- Target Upside: {data.get('target_upside', 'N/A')}%
"""
        
        if lang == 'ko':
            prompt = f"""종목: {ticker} - {data.get('name', ticker)}

퀀트 분석 결과:
{score_info}

최근 뉴스:
{news_text}

요청: 이 종목에 대한 3-4문장의 투자 의견을 작성하세요.
- 수급 동향과 기관 동향
- 기술적/펀더멘털 강점
- 투자 전략 제안
- 이모지 사용 X, 간결하게 작성"""
        else:
            prompt = f"""Stock: {ticker} - {data.get('name', ticker)}

Quantitative Analysis:
{score_info}

Recent News:
{news_text}

Request: Write a 3-4 sentence investment opinion.
- Supply/demand and institutional trends
- Technical/fundamental strengths
- Investment strategy suggestion
- No emojis, be concise"""

        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 500
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
            return f"Error: {e}"


class AIStockAnalyzer:
    """Main class for generating AI summaries"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.output_file = os.path.join(data_dir, 'ai_summaries.json')
        self.generator = GeminiGenerator()
        self.news_collector = NewsCollector()
        
    def run(self, top_n: int = 20, force_refresh: bool = False):
        """Generate AI summaries for top N stocks"""
        # Load smart money picks
        csv_path = os.path.join(self.data_dir, 'smart_money_picks_v2.csv')
        
        if not os.path.exists(csv_path):
            logger.error(f"Smart money picks not found: {csv_path}")
            return {}
        
        df = pd.read_csv(csv_path).head(top_n)
        logger.info(f"Generating AI summaries for {len(df)} stocks...")
        
        # Load existing summaries
        results = {}
        if os.path.exists(self.output_file) and not force_refresh:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Loaded {len(results)} existing summaries")
        
        # Generate summaries
        new_count = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating summaries"):
            ticker = row['ticker']
            
            # Skip if already exists and not forcing refresh
            if ticker in results and not force_refresh:
                continue
            
            # Get news
            news = self.news_collector.get_news(ticker)
            
            # Generate Korean summary
            summary_ko = self.generator.generate(ticker, row.to_dict(), news, 'ko')
            
            # Generate English summary
            summary_en = self.generator.generate(ticker, row.to_dict(), news, 'en')
            
            results[ticker] = {
                'name': row.get('name', ticker),
                'summary': summary_ko,
                'summary_ko': summary_ko,
                'summary_en': summary_en,
                'composite_score': row.get('composite_score'),
                'grade': row.get('grade'),
                'current_price': row.get('current_price'),
                'updated': datetime.now().isoformat()
            }
            
            new_count += 1
            time.sleep(1)  # Rate limiting
        
        # Save results
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(results)} summaries ({new_count} new)")
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='AI Stock Summary Generator')
    parser.add_argument('--dir', default=get_data_dir(), help='Data directory')
    parser.add_argument('--top', type=int, default=20, help='Number of top stocks')
    parser.add_argument('--refresh', action='store_true', help='Force refresh all summaries')
    args = parser.parse_args()
    
    os.makedirs(args.dir, exist_ok=True)
    analyzer = AIStockAnalyzer(data_dir=args.dir)
    results = analyzer.run(top_n=args.top, force_refresh=args.refresh)
    
    if results:
        print(f"\nGenerated AI Summaries for {len(results)} stocks")
        print("=" * 60)
        
        # Show sample
        for ticker, data in list(results.items())[:3]:
            print(f"\n{ticker} ({data.get('name', 'N/A')}):")
            print(f"  Score: {data.get('composite_score', 'N/A')}")
            print(f"  Summary: {data.get('summary_en', 'N/A')[:200]}...")


if __name__ == "__main__":
    main()
