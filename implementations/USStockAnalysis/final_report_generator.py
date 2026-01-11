#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Report Generator
Generates the final Top 10 investment report combining quant scores and AI analysis
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime
from us_config import get_data_dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FinalReportGenerator:
    """Generate final investment report"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.output_file = os.path.join(data_dir, 'final_top10_report.json')
        self.dashboard_file = os.path.join(data_dir, 'smart_money_current.json')
        
    def run(self, top_n: int = 10):
        """Generate final report"""
        logger.info("Generating Final Report...")
        
        # Load Quant Data
        stats_path = os.path.join(self.data_dir, 'smart_money_picks_v2.csv')
        if not os.path.exists(stats_path):
            logger.error(f"Smart money picks not found: {stats_path}")
            return None
        
        df = pd.read_csv(stats_path)
        logger.info(f"Loaded {len(df)} stocks from quant analysis")
        
        # Load AI Summaries
        ai_path = os.path.join(self.data_dir, 'ai_summaries.json')
        ai_data = {}
        if os.path.exists(ai_path):
            with open(ai_path, 'r', encoding='utf-8') as f:
                ai_data = json.load(f)
            logger.info(f"Loaded {len(ai_data)} AI summaries")
        else:
            logger.warning("AI summaries not found, generating report without AI analysis")
        
        # Load Macro Analysis
        macro_path = os.path.join(self.data_dir, 'macro_analysis.json')
        macro_data = {}
        if os.path.exists(macro_path):
            with open(macro_path, 'r', encoding='utf-8') as f:
                macro_data = json.load(f)
        
        # Combine scores
        results = []
        for _, row in df.iterrows():
            ticker = row['ticker']
            
            # Get AI summary if available
            ai_info = ai_data.get(ticker, {})
            summary_ko = ai_info.get('summary_ko', '')
            summary_en = ai_info.get('summary_en', '')
            
            # Calculate AI bonus score based on summary content
            ai_score = 0
            rec = "Hold"
            
            # Korean keywords
            if "매수" in summary_ko or "긍정" in summary_ko:
                ai_score = 10
                rec = "Buy"
            if "적극" in summary_ko or "강력" in summary_ko:
                ai_score = 20
                rec = "Strong Buy"
            if "주의" in summary_ko or "리스크" in summary_ko:
                ai_score = max(ai_score - 5, 0)
                if ai_score < 5:
                    rec = "Caution"
            
            # English keywords
            if "buy" in summary_en.lower() or "positive" in summary_en.lower():
                ai_score = max(ai_score, 10)
                if rec == "Hold":
                    rec = "Buy"
            if "strong" in summary_en.lower() and "buy" in summary_en.lower():
                ai_score = 20
                rec = "Strong Buy"
            
            # Calculate final score (80% quant + 20% AI adjustment)
            quant_score = row.get('composite_score', 50)
            final_score = quant_score * 0.8 + ai_score
            
            # Determine final grade
            if final_score >= 80:
                final_grade = "S"
            elif final_score >= 70:
                final_grade = "A"
            elif final_score >= 60:
                final_grade = "B"
            elif final_score >= 50:
                final_grade = "C"
            else:
                final_grade = "D"
            
            results.append({
                'ticker': ticker,
                'name': row.get('name', ticker),
                'final_score': round(final_score, 1),
                'final_grade': final_grade,
                'quant_score': quant_score,
                'ai_adjustment': ai_score,
                'ai_recommendation': rec,
                'current_price': row.get('current_price', 0),
                'target_upside': row.get('target_upside', 0),
                'sd_score': row.get('sd_score', 50),
                'tech_score': row.get('tech_score', 50),
                'fund_score': row.get('fund_score', 50),
                'inst_score': row.get('inst_score', 50),
                'rs_score': row.get('rs_score', 50),
                'rsi': row.get('rsi', 50),
                'pe_ratio': row.get('pe_ratio', 'N/A'),
                'size': row.get('size', 'Unknown'),
                'ai_summary_ko': summary_ko,
                'ai_summary_en': summary_en
            })
        
        # Sort by final score and get top N
        results.sort(key=lambda x: x['final_score'], reverse=True)
        top_picks = results[:top_n]
        
        # Add rank
        for i, pick in enumerate(top_picks, 1):
            pick['rank'] = i
        
        # Calculate summary stats
        avg_score = sum(p['final_score'] for p in top_picks) / len(top_picks) if top_picks else 0
        strong_buys = len([p for p in top_picks if p['ai_recommendation'] == 'Strong Buy'])
        buys = len([p for p in top_picks if p['ai_recommendation'] == 'Buy'])
        
        # Build final report
        report = {
            'timestamp': datetime.now().isoformat(),
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'summary': {
                'total_analyzed': len(df),
                'top_picks_count': len(top_picks),
                'avg_score': round(avg_score, 1),
                'strong_buys': strong_buys,
                'buys': buys,
                'market_context': macro_data.get('ai_analysis', 'No macro analysis available')[:500] if macro_data else ''
            },
            'top_picks': top_picks,
            'methodology': {
                'quant_weight': 0.8,
                'ai_weight': 0.2,
                'factors': ['Supply/Demand', 'Technical', 'Fundamental', 'Institutional', 'Relative Strength', 'Analyst']
            }
        }
        
        # Save main report
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved report: {self.output_file}")
        
        # Save dashboard format
        dashboard = {
            'updated': datetime.now().isoformat(),
            'picks': top_picks
        }
        with open(self.dashboard_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved dashboard data: {self.dashboard_file}")
        
        return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Final Report Generator')
    parser.add_argument('--dir', default=get_data_dir(), help='Data directory')
    parser.add_argument('--top', type=int, default=10, help='Number of top picks')
    args = parser.parse_args()
    
    os.makedirs(args.dir, exist_ok=True)
    generator = FinalReportGenerator(data_dir=args.dir)
    report = generator.run(top_n=args.top)
    
    if report:
        print("\nFinal Investment Report")
        print("=" * 70)
        print(f"Generated: {report['report_date']}")
        print(f"Total Analyzed: {report['summary']['total_analyzed']}")
        print(f"Average Score: {report['summary']['avg_score']}")
        print(f"Strong Buys: {report['summary']['strong_buys']} | Buys: {report['summary']['buys']}")
        
        print("\nTop 10 Picks:")
        print("-" * 70)
        print(f"{'Rank':<5} {'Ticker':<8} {'Name':<20} {'Score':<8} {'Grade':<6} {'Rec':<12} {'Price':<10}")
        print("-" * 70)
        
        for pick in report['top_picks']:
            print(f"{pick['rank']:<5} {pick['ticker']:<8} {pick['name'][:18]:<20} "
                  f"{pick['final_score']:<8.1f} {pick['final_grade']:<6} "
                  f"{pick['ai_recommendation']:<12} ${pick['current_price']:<10.2f}")


if __name__ == "__main__":
    main()
