#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Risk Analyzer
Analyzes portfolio risk metrics including correlation, volatility, and diversification
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, List
from us_config import get_data_dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PortfolioRiskAnalyzer:
    """Analyze portfolio risk and correlation"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.output_file = os.path.join(data_dir, 'portfolio_risk.json')
    
    def analyze_portfolio(self, tickers: List[str], weights: List[float] = None) -> Dict:
        """
        Analyze portfolio risk metrics
        
        Args:
            tickers: List of stock tickers
            weights: Optional portfolio weights (equal weight if not provided)
        """
        if len(tickers) < 2:
            return {'error': 'Need at least 2 tickers for correlation analysis'}
        
        if weights is None:
            weights = [1/len(tickers)] * len(tickers)
        
        weights = np.array(weights)
        
        try:
            logger.info(f"Analyzing portfolio of {len(tickers)} stocks...")
            
            # Download historical data
            data = yf.download(tickers, period='6mo', progress=False)['Close']
            
            if data.empty:
                return {'error': 'No data available'}
            
            # Calculate daily returns
            returns = data.pct_change().dropna()
            
            if len(returns) < 20:
                return {'error': 'Insufficient data for analysis'}
            
            # Correlation Matrix
            corr_matrix = returns.corr()
            
            # Find highly correlated pairs
            high_correlations = []
            cols = corr_matrix.columns
            for i in range(len(cols)):
                for j in range(i+1, len(cols)):
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val > 0.7:
                        high_correlations.append({
                            'pair': [cols[i], cols[j]],
                            'correlation': round(corr_val, 3),
                            'risk': 'High' if corr_val > 0.85 else 'Moderate'
                        })
            
            # Sort by correlation
            high_correlations.sort(key=lambda x: x['correlation'], reverse=True)
            
            # Covariance Matrix (annualized)
            cov_matrix = returns.cov() * 252
            
            # Portfolio Variance and Volatility
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Individual stock volatilities
            individual_vols = {}
            for ticker in tickers:
                if ticker in returns.columns:
                    vol = returns[ticker].std() * np.sqrt(252) * 100
                    individual_vols[ticker] = round(vol, 2)
            
            # Beta calculation (vs SPY)
            try:
                spy_data = yf.download('SPY', period='6mo', progress=False)['Close']
                spy_returns = spy_data.pct_change().dropna()
                
                # Align dates
                common_dates = returns.index.intersection(spy_returns.index)
                aligned_returns = returns.loc[common_dates]
                aligned_spy = spy_returns.loc[common_dates]
                
                betas = {}
                for ticker in tickers:
                    if ticker in aligned_returns.columns:
                        cov_with_market = aligned_returns[ticker].cov(aligned_spy)
                        market_var = aligned_spy.var()
                        beta = cov_with_market / market_var if market_var > 0 else 1
                        betas[ticker] = round(beta, 2)
                
                portfolio_beta = sum(betas.get(t, 1) * w for t, w in zip(tickers, weights))
            except:
                betas = {t: 1.0 for t in tickers}
                portfolio_beta = 1.0
            
            # Diversification ratio
            weighted_avg_vol = sum(individual_vols.get(t, 20) * w for t, w in zip(tickers, weights))
            diversification_ratio = weighted_avg_vol / (portfolio_volatility * 100) if portfolio_volatility > 0 else 1
            
            # Risk assessment
            if portfolio_volatility * 100 > 30:
                risk_level = "Very High"
            elif portfolio_volatility * 100 > 20:
                risk_level = "High"
            elif portfolio_volatility * 100 > 15:
                risk_level = "Moderate"
            elif portfolio_volatility * 100 > 10:
                risk_level = "Low"
            else:
                risk_level = "Very Low"
            
            # Diversification assessment
            if diversification_ratio > 1.3:
                diversification_grade = "Excellent"
            elif diversification_ratio > 1.15:
                diversification_grade = "Good"
            elif diversification_ratio > 1.0:
                diversification_grade = "Fair"
            else:
                diversification_grade = "Poor"
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'portfolio': {
                    'tickers': tickers,
                    'weights': [round(w, 4) for w in weights],
                    'num_stocks': len(tickers)
                },
                'risk_metrics': {
                    'portfolio_volatility': round(portfolio_volatility * 100, 2),
                    'portfolio_beta': round(portfolio_beta, 2),
                    'risk_level': risk_level,
                    'diversification_ratio': round(diversification_ratio, 2),
                    'diversification_grade': diversification_grade
                },
                'individual_metrics': {
                    'volatilities': individual_vols,
                    'betas': betas
                },
                'correlation_analysis': {
                    'high_correlations': high_correlations[:10],
                    'avg_correlation': round(corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean(), 3),
                    'matrix': corr_matrix.round(3).to_dict()
                },
                'recommendations': self._generate_recommendations(
                    portfolio_volatility * 100, 
                    diversification_ratio, 
                    high_correlations,
                    portfolio_beta
                )
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, volatility: float, div_ratio: float, 
                                   high_corrs: List, beta: float) -> List[str]:
        """Generate risk recommendations"""
        recommendations = []
        
        if volatility > 25:
            recommendations.append("Portfolio volatility is high. Consider adding defensive stocks or bonds.")
        
        if div_ratio < 1.1:
            recommendations.append("Low diversification benefit. Consider adding uncorrelated assets.")
        
        if len(high_corrs) > 3:
            recommendations.append(f"Multiple highly correlated pairs detected. Review concentration risk.")
        
        if beta > 1.3:
            recommendations.append("Portfolio beta is high. Will amplify market moves. Consider reducing cyclical exposure.")
        elif beta < 0.7:
            recommendations.append("Portfolio beta is low. May underperform in bull markets.")
        
        if not recommendations:
            recommendations.append("Portfolio risk metrics appear balanced.")
        
        return recommendations
    
    def run(self, tickers: List[str] = None, weights: List[float] = None):
        """Run analysis and save results"""
        if tickers is None:
            # Default example portfolio
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'JNJ', 'PG']
        
        result = self.analyze_portfolio(tickers, weights)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved portfolio risk analysis to {self.output_file}")
        return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Portfolio Risk Analyzer')
    parser.add_argument('--dir', default=get_data_dir(), help='Data directory')
    parser.add_argument('--tickers', nargs='+', help='Portfolio tickers')
    parser.add_argument('--weights', nargs='+', type=float, help='Portfolio weights')
    args = parser.parse_args()
    
    os.makedirs(args.dir, exist_ok=True)
    analyzer = PortfolioRiskAnalyzer(data_dir=args.dir)
    result = analyzer.run(tickers=args.tickers, weights=args.weights)
    
    if 'error' not in result:
        print("\nPortfolio Risk Analysis:")
        print("=" * 60)
        print(f"Stocks: {', '.join(result['portfolio']['tickers'])}")
        print(f"\nRisk Metrics:")
        print(f"  Volatility: {result['risk_metrics']['portfolio_volatility']:.1f}%")
        print(f"  Beta: {result['risk_metrics']['portfolio_beta']:.2f}")
        print(f"  Risk Level: {result['risk_metrics']['risk_level']}")
        print(f"  Diversification: {result['risk_metrics']['diversification_grade']}")
        
        if result['correlation_analysis']['high_correlations']:
            print(f"\nHigh Correlations:")
            for item in result['correlation_analysis']['high_correlations'][:3]:
                print(f"  {item['pair'][0]}-{item['pair'][1]}: {item['correlation']:.2f}")
        
        print(f"\nRecommendations:")
        for rec in result['recommendations']:
            print(f"  - {rec}")
    else:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
