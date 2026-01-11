#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Economic Calendar with AI Impact Analysis
Fetches economic events and provides AI-powered impact analysis
"""

import os
import json
import requests
import logging
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO
from dotenv import load_dotenv
from us_config import get_data_dir

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EconomicCalendar:
    """Economic event calendar with AI enrichment"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.output_file = os.path.join(data_dir, 'weekly_calendar.json')
        self.api_key = os.getenv('GOOGLE_API_KEY')
        
        # Major economic events (manual list as backup)
        self.major_events = [
            {
                'event': 'FOMC Interest Rate Decision',
                'impact': 'High',
                'category': 'Monetary Policy',
                'description': 'Federal Reserve interest rate decision and policy statement'
            },
            {
                'event': 'Non-Farm Payrolls',
                'impact': 'High',
                'category': 'Employment',
                'description': 'Monthly employment report showing job creation'
            },
            {
                'event': 'CPI (Consumer Price Index)',
                'impact': 'High',
                'category': 'Inflation',
                'description': 'Key inflation measure tracking consumer prices'
            },
            {
                'event': 'GDP Growth Rate',
                'impact': 'High',
                'category': 'Growth',
                'description': 'Quarterly economic growth measurement'
            },
            {
                'event': 'ISM Manufacturing PMI',
                'impact': 'Medium',
                'category': 'Manufacturing',
                'description': 'Manufacturing sector health indicator'
            },
            {
                'event': 'Retail Sales',
                'impact': 'Medium',
                'category': 'Consumer',
                'description': 'Monthly retail sales data'
            },
            {
                'event': 'Jobless Claims',
                'impact': 'Medium',
                'category': 'Employment',
                'description': 'Weekly unemployment claims'
            },
            {
                'event': 'Consumer Confidence',
                'impact': 'Medium',
                'category': 'Consumer',
                'description': 'Consumer sentiment indicator'
            }
        ]
        
    def get_events(self) -> list:
        """Fetch economic events"""
        events = []
        
        # Try to scrape from web sources
        try:
            events.extend(self._scrape_yahoo_calendar())
        except Exception as e:
            logger.debug(f"Yahoo scrape failed: {e}")
        
        # Add manual major events for coming week
        events.extend(self._get_weekly_events())
        
        # Remove duplicates
        seen = set()
        unique_events = []
        for ev in events:
            key = ev.get('event', '')
            if key not in seen:
                seen.add(key)
                unique_events.append(ev)
        
        return unique_events
    
    def _scrape_yahoo_calendar(self) -> list:
        """Scrape Yahoo Finance economic calendar"""
        events = []
        try:
            url = "https://finance.yahoo.com/calendar/economic"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            resp = requests.get(url, headers=headers, timeout=10)
            
            if resp.status_code == 200:
                dfs = pd.read_html(StringIO(resp.text))
                if dfs:
                    df = dfs[0]
                    
                    # Filter US events if country column exists
                    if 'Country' in df.columns:
                        us_events = df[df['Country'] == 'US']
                    else:
                        us_events = df
                    
                    for _, row in us_events.head(10).iterrows():
                        event_name = row.get('Event', row.get('event', ''))
                        if not event_name:
                            continue
                        
                        events.append({
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'event': str(event_name),
                            'impact': self._estimate_impact(str(event_name)),
                            'actual': str(row.get('Actual', '-')),
                            'estimate': str(row.get('Market Expectation', row.get('Estimate', '-'))),
                            'previous': str(row.get('Previous', '-')),
                            'category': self._categorize_event(str(event_name))
                        })
        except Exception as e:
            logger.debug(f"Yahoo calendar error: {e}")
        
        return events
    
    def _get_weekly_events(self) -> list:
        """Get scheduled events for the week"""
        events = []
        today = datetime.now()
        
        # Add placeholder events for the week
        for i in range(7):
            date = (today + timedelta(days=i)).strftime('%Y-%m-%d')
            day = (today + timedelta(days=i)).weekday()
            
            # Typical event schedule
            if day == 0:  # Monday
                events.append({
                    'date': date,
                    'event': 'Market Open',
                    'impact': 'Low',
                    'category': 'Market',
                    'description': 'Weekly market opening'
                })
            elif day == 3:  # Thursday
                events.append({
                    'date': date,
                    'event': 'Initial Jobless Claims',
                    'impact': 'Medium',
                    'category': 'Employment',
                    'description': 'Weekly unemployment claims data'
                })
        
        return events
    
    def _estimate_impact(self, event_name: str) -> str:
        """Estimate event impact based on name"""
        high_impact = ['FOMC', 'Fed', 'NFP', 'Payroll', 'CPI', 'GDP', 'Inflation', 'Interest Rate']
        medium_impact = ['PMI', 'ISM', 'Retail', 'Employment', 'Claims', 'Consumer']
        
        event_lower = event_name.lower()
        
        for keyword in high_impact:
            if keyword.lower() in event_lower:
                return 'High'
        
        for keyword in medium_impact:
            if keyword.lower() in event_lower:
                return 'Medium'
        
        return 'Low'
    
    def _categorize_event(self, event_name: str) -> str:
        """Categorize economic event"""
        categories = {
            'Employment': ['job', 'payroll', 'unemployment', 'claims', 'employment'],
            'Inflation': ['cpi', 'ppi', 'inflation', 'price'],
            'Monetary Policy': ['fomc', 'fed', 'rate', 'monetary'],
            'Growth': ['gdp', 'growth'],
            'Manufacturing': ['pmi', 'ism', 'manufacturing', 'industrial'],
            'Consumer': ['retail', 'consumer', 'spending', 'sales'],
            'Housing': ['housing', 'home', 'mortgage']
        }
        
        event_lower = event_name.lower()
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in event_lower:
                    return category
        
        return 'Other'
    
    def enrich_with_ai(self, events: list) -> list:
        """Add AI-powered impact analysis for high-impact events"""
        if not self.api_key:
            logger.warning("No API key for AI enrichment")
            return events
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        
        for event in events:
            if event.get('impact') == 'High':
                try:
                    prompt = f"""Briefly explain the market impact of this economic event in 2 sentences:
Event: {event['event']}
Category: {event.get('category', 'Economic')}
Focus on: Which sectors benefit, which may suffer, and typical market reaction."""
                    
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"maxOutputTokens": 200}
                    }
                    
                    resp = requests.post(
                        f"{url}?key={self.api_key}",
                        json=payload,
                        timeout=15
                    )
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        ai_text = result['candidates'][0]['content']['parts'][0]['text']
                        event['ai_analysis'] = ai_text
                        event['description'] = event.get('description', '') + f"\n\nAI Insight: {ai_text}"
                        
                except Exception as e:
                    logger.debug(f"AI enrichment failed for {event['event']}: {e}")
        
        return events

    def run(self):
        """Run calendar collection and save"""
        logger.info("Fetching economic calendar...")
        
        events = self.get_events()
        logger.info(f"Found {len(events)} events")
        
        # Enrich with AI
        if self.api_key:
            logger.info("Adding AI analysis...")
            events = self.enrich_with_ai(events)
        
        # Sort by date and impact
        impact_order = {'High': 0, 'Medium': 1, 'Low': 2}
        events.sort(key=lambda x: (x.get('date', ''), impact_order.get(x.get('impact', 'Low'), 2)))
        
        # Build output
        output = {
            'updated': datetime.now().isoformat(),
            'week_start': datetime.now().strftime('%Y-%m-%d'),
            'week_end': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
            'total_events': len(events),
            'high_impact_count': len([e for e in events if e.get('impact') == 'High']),
            'events': events,
            'reference_events': self.major_events
        }
        
        # Save
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved calendar to {self.output_file}")
        return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Economic Calendar')
    parser.add_argument('--dir', default=get_data_dir(), help='Data directory')
    args = parser.parse_args()
    
    os.makedirs(args.dir, exist_ok=True)
    calendar = EconomicCalendar(data_dir=args.dir)
    result = calendar.run()
    
    print("\nEconomic Calendar")
    print("=" * 60)
    print(f"Week: {result['week_start']} to {result['week_end']}")
    print(f"Total Events: {result['total_events']}")
    print(f"High Impact: {result['high_impact_count']}")
    
    print("\nUpcoming Events:")
    print("-" * 60)
    
    for event in result['events'][:10]:
        impact_emoji = {"High": "!!!", "Medium": "!!", "Low": "!"}
        print(f"  [{event.get('impact', 'Low'):6}] {event.get('event', 'Unknown')[:40]}")
        if event.get('ai_analysis'):
            print(f"          AI: {event['ai_analysis'][:80]}...")


if __name__ == "__main__":
    main()
