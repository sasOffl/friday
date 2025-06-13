"""
Comparative Analysis Agents for McKinsey Stock Performance Monitor
Handles multi-stock comparison, correlation analysis, and peer benchmarking
"""

from crewai import Agent
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import yfinance as yf
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ComparativeAgent(Agent):
    """Agent for comparing multiple stocks across various metrics"""
    
    def __init__(self):
        super().__init__(
            role="Stock Comparison Analyst",
            goal="Compare multiple stocks across key performance metrics",
            backstory="Expert in comparative financial analysis with deep understanding of relative valuation metrics",
            verbose=True
        )
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-stock comparison task"""
        try:
            symbols = task.get('symbols', [])
            period = task.get('period', '1y')
            
            if len(symbols) < 2:
                return {"error": "Need at least 2 symbols for comparison"}
            
            # Fetch data for all symbols
            comparison_data = {}
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    if not hist.empty:
                        # Calculate key metrics
                        current_price = hist['Close'].iloc[-1]
                        start_price = hist['Close'].iloc[0]
                        total_return = ((current_price - start_price) / start_price) * 100
                        volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100
                        
                        # Get additional info
                        info = ticker.info
                        
                        comparison_data[symbol] = {
                            'current_price': current_price,
                            'total_return': total_return,
                            'volatility': volatility,
                            'market_cap': info.get('marketCap', 0),
                            'pe_ratio': info.get('trailingPE', 0),
                            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                            'beta': info.get('beta', 1),
                            'sector': info.get('sector', 'Unknown'),
                            'industry': info.get('industry', 'Unknown')
                        }
                        
                        logger.info(f"Processed comparison data for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    continue
            
            # Create comparison rankings
            rankings = self._create_rankings(comparison_data)
            
            return {
                'comparison_data': comparison_data,
                'rankings': rankings,
                'analysis_summary': self._generate_comparison_summary(comparison_data, rankings)
            }
            
        except Exception as e:
            logger.error(f"Error in ComparativeAgent: {str(e)}")
            return {"error": f"Comparison analysis failed: {str(e)}"}
    
    def _create_rankings(self, data: Dict[str, Dict]) -> Dict[str, List]:
        """Create rankings for different metrics"""
        rankings = {}
        
        # Return ranking (descending)
        returns = [(symbol, info['total_return']) for symbol, info in data.items()]
        rankings['returns'] = sorted(returns, key=lambda x: x[1], reverse=True)
        
        # Volatility ranking (ascending - lower is better)
        volatility = [(symbol, info['volatility']) for symbol, info in data.items()]
        rankings['volatility'] = sorted(volatility, key=lambda x: x[1])
        
        # Market cap ranking (descending)
        market_cap = [(symbol, info['market_cap']) for symbol, info in data.items() if info['market_cap'] > 0]
        rankings['market_cap'] = sorted(market_cap, key=lambda x: x[1], reverse=True)
        
        # PE ratio ranking (ascending - lower is better)
        pe_ratios = [(symbol, info['pe_ratio']) for symbol, info in data.items() if info['pe_ratio'] > 0]
        rankings['pe_ratio'] = sorted(pe_ratios, key=lambda x: x[1])
        
        return rankings
    
    def _generate_comparison_summary(self, data: Dict, rankings: Dict) -> str:
        """Generate natural language summary of comparison"""
        summary = []
        
        if 'returns' in rankings and rankings['returns']:
            best_return = rankings['returns'][0]
            worst_return = rankings['returns'][-1]
            summary.append(f"Best performer: {best_return[0]} with {best_return[1]:.2f}% return")
            summary.append(f"Worst performer: {worst_return[0]} with {worst_return[1]:.2f}% return")
        
        if 'volatility' in rankings and rankings['volatility']:
            least_volatile = rankings['volatility'][0]
            most_volatile = rankings['volatility'][-1]
            summary.append(f"Most stable: {least_volatile[0]} with {least_volatile[1]:.2f}% volatility")
            summary.append(f"Most volatile: {most_volatile[0]} with {most_volatile[1]:.2f}% volatility")
        
        return " | ".join(summary)


class CorrelationInsightAgent(Agent):
    """Agent for analyzing correlations between stocks and features"""
    
    def __init__(self):
        super().__init__(
            role="Correlation Analysis Specialist",
            goal="Identify and analyze correlations between stocks and market features",
            backstory="Statistical analyst specializing in financial correlation patterns and market relationships",
            verbose=True
        )
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute correlation analysis task"""
        try:
            symbols = task.get('symbols', [])
            period = task.get('period', '1y')
            
            if len(symbols) < 2:
                return {"error": "Need at least 2 symbols for correlation analysis"}
            
            # Collect price data for all symbols
            price_data = pd.DataFrame()
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    if not hist.empty:
                        price_data[symbol] = hist['Close']
                        logger.info(f"Collected price data for {symbol}")
                except Exception as e:
                    logger.error(f"Error collecting data for {symbol}: {str(e)}")
                    continue
            
            if price_data.empty:
                return {"error": "No valid price data collected"}
            
            # Calculate returns
            returns_data = price_data.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns_data.corr()
            
            # Find strongest correlations
            strong_correlations = self._find_strong_correlations(correlation_matrix)
            
            # Calculate rolling correlations for trend analysis
            rolling_correlations = self._calculate_rolling_correlations(returns_data)
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'strong_correlations': strong_correlations,
                'rolling_correlations': rolling_correlations,
                'insights': self._generate_correlation_insights(correlation_matrix, strong_correlations)
            }
            
        except Exception as e:
            logger.error(f"Error in CorrelationInsightAgent: {str(e)}")
            return {"error": f"Correlation analysis failed: {str(e)}"}
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame) -> List[Dict]:
        """Find correlations above threshold"""
        strong_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                symbol1 = corr_matrix.columns[i]
                symbol2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                
                if abs(correlation) > 0.7:  # Strong correlation threshold
                    strong_corr.append({
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'correlation': correlation,
                        'strength': 'Strong' if abs(correlation) > 0.8 else 'Moderate'
                    })
        
        return sorted(strong_corr, key=lambda x: abs(x['correlation']), reverse=True)
    
    def _calculate_rolling_correlations(self, returns_data: pd.DataFrame) -> Dict:
        """Calculate 30-day rolling correlations"""
        rolling_corr = {}
        
        if len(returns_data.columns) >= 2:
            # Take first two symbols for rolling correlation example
            symbol1, symbol2 = returns_data.columns[0], returns_data.columns[1]
            rolling_30d = returns_data[symbol1].rolling(window=30).corr(returns_data[symbol2])
            
            rolling_corr[f"{symbol1}_{symbol2}"] = {
                'current': rolling_30d.iloc[-1] if not rolling_30d.empty else 0,
                'average': rolling_30d.mean() if not rolling_30d.empty else 0,
                'volatility': rolling_30d.std() if not rolling_30d.empty else 0
            }
        
        return rolling_corr
    
    def _generate_correlation_insights(self, corr_matrix: pd.DataFrame, strong_corr: List) -> str:
        """Generate insights from correlation analysis"""
        insights = []
        
        if strong_corr:
            highest_corr = strong_corr[0]
            insights.append(f"Strongest correlation: {highest_corr['symbol1']} and {highest_corr['symbol2']} ({highest_corr['correlation']:.3f})")
            
            positive_corr = [c for c in strong_corr if c['correlation'] > 0]
            negative_corr = [c for c in strong_corr if c['correlation'] < 0]
            
            if positive_corr:
                insights.append(f"Found {len(positive_corr)} strong positive correlations")
            if negative_corr:
                insights.append(f"Found {len(negative_corr)} strong negative correlations")
        else:
            insights.append("No strong correlations found among selected stocks")
        
        return " | ".join(insights)


class PeerComparisonAgent(Agent):
    """Agent for comparing stocks within industry sectors"""
    
    def __init__(self):
        super().__init__(
            role="Peer Comparison Analyst",
            goal="Compare stocks against industry peers and sector benchmarks",
            backstory="Industry specialist with expertise in sector-specific valuation metrics and peer analysis",
            verbose=True
        )
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute peer comparison task"""
        try:
            symbols = task.get('symbols', [])
            period = task.get('period', '1y')
            
            # Group stocks by sector
            sector_groups = {}
            stock_info = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period=period)
                    
                    if not hist.empty:
                        sector = info.get('sector', 'Unknown')
                        
                        if sector not in sector_groups:
                            sector_groups[sector] = []
                        
                        # Calculate performance metrics
                        current_price = hist['Close'].iloc[-1]
                        start_price = hist['Close'].iloc[0]
                        total_return = ((current_price - start_price) / start_price) * 100
                        
                        stock_data = {
                            'symbol': symbol,
                            'sector': sector,
                            'industry': info.get('industry', 'Unknown'),
                            'total_return': total_return,
                            'market_cap': info.get('marketCap', 0),
                            'pe_ratio': info.get('trailingPE', 0),
                            'pb_ratio': info.get('priceToBook', 0),
                            'roe': info.get('returnOnEquity', 0),
                            'debt_to_equity': info.get('debtToEquity', 0),
                            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
                        }
                        
                        sector_groups[sector].append(stock_data)
                        stock_info[symbol] = stock_data
                        
                        logger.info(f"Processed peer data for {symbol} in {sector}")
                        
                except Exception as e:
                    logger.error(f"Error processing peer data for {symbol}: {str(e)}")
                    continue
            
            # Calculate sector benchmarks
            sector_benchmarks = self._calculate_sector_benchmarks(sector_groups)
            
            # Generate peer rankings
            peer_rankings = self._generate_peer_rankings(sector_groups)
            
            return {
                'sector_groups': sector_groups,
                'sector_benchmarks': sector_benchmarks,
                'peer_rankings': peer_rankings,
                'peer_insights': self._generate_peer_insights(stock_info, sector_benchmarks)
            }
            
        except Exception as e:
            logger.error(f"Error in PeerComparisonAgent: {str(e)}")
            return {"error": f"Peer comparison failed: {str(e)}"}
    
    def _calculate_sector_benchmarks(self, sector_groups: Dict) -> Dict:
        """Calculate average metrics for each sector"""
        benchmarks = {}
        
        for sector, stocks in sector_groups.items():
            if len(stocks) > 1:  # Only calculate if multiple stocks in sector
                metrics = {}
                
                # Calculate averages for numerical metrics
                numerical_fields = ['total_return', 'pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity', 'dividend_yield']
                
                for field in numerical_fields:
                    values = [stock[field] for stock in stocks if stock[field] and stock[field] > 0]
                    if values:
                        metrics[f"avg_{field}"] = np.mean(values)
                        metrics[f"median_{field}"] = np.median(values)
                
                benchmarks[sector] = metrics
        
        return benchmarks
    
    def _generate_peer_rankings(self, sector_groups: Dict) -> Dict:
        """Generate rankings within each sector"""
        rankings = {}
        
        for sector, stocks in sector_groups.items():
            if len(stocks) > 1:
                sector_rankings = {}
                
                # Rank by return (descending)
                by_return = sorted(stocks, key=lambda x: x['total_return'], reverse=True)
                sector_rankings['by_return'] = [(stock['symbol'], stock['total_return']) for stock in by_return]
                
                # Rank by PE ratio (ascending - lower is better)
                valid_pe = [stock for stock in stocks if stock['pe_ratio'] > 0]
                if valid_pe:
                    by_pe = sorted(valid_pe, key=lambda x: x['pe_ratio'])
                    sector_rankings['by_pe'] = [(stock['symbol'], stock['pe_ratio']) for stock in by_pe]
                
                rankings[sector] = sector_rankings
        
        return rankings
    
    def _generate_peer_insights(self, stock_info: Dict, benchmarks: Dict) -> Dict:
        """Generate insights comparing each stock to its sector"""
        insights = {}
        
        for symbol, info in stock_info.items():
            sector = info['sector']
            stock_insights = []
            
            if sector in benchmarks:
                benchmark = benchmarks[sector]
                
                # Compare return to sector average
                if 'avg_total_return' in benchmark:
                    sector_avg_return = benchmark['avg_total_return']
                    if info['total_return'] > sector_avg_return:
                        stock_insights.append(f"Outperforming sector by {info['total_return'] - sector_avg_return:.2f}%")
                    else:
                        stock_insights.append(f"Underperforming sector by {sector_avg_return - info['total_return']:.2f}%")
                
                # Compare PE ratio
                if 'avg_pe_ratio' in benchmark and info['pe_ratio'] > 0:
                    sector_avg_pe = benchmark['avg_pe_ratio']
                    if info['pe_ratio'] < sector_avg_pe:
                        stock_insights.append(f"Trading at discount to sector (PE: {info['pe_ratio']:.2f} vs {sector_avg_pe:.2f})")
                    else:
                        stock_insights.append(f"Trading at premium to sector (PE: {info['pe_ratio']:.2f} vs {sector_avg_pe:.2f})")
            
            insights[symbol] = stock_insights
        
        return insights