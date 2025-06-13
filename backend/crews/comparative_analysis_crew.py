"""
Comparative Analysis Crew - McKinsey Stock Performance Monitor
Handles multi-stock comparisons, correlation analysis, and peer benchmarking
"""

from crewai import Crew, Task
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from agents.comparative_agents import (
    ComparativeAgent,
    CorrelationInsightAgent,
    PeerComparisonAgent
)
from tools.visualization_tools import PlotlyToolKit
from tools.technical_tools import VolatilityScanner
from utils.logger import AnalysisLogger


class ComparativeAnalysisCrew:
    """
    Crew responsible for comparative analysis across multiple stocks
    Performs cross-stock comparisons, correlation analysis, and peer benchmarking
    """
    
    def __init__(self, session_id: str, shared_memory: Dict[str, Any]):
        self.session_id = session_id
        self.shared_memory = shared_memory
        self.logger = AnalysisLogger()
        self.plot_toolkit = PlotlyToolKit()
        self.volatility_scanner = VolatilityScanner()
        
        # Initialize agents
        self.comparative_agent = ComparativeAgent()
        self.correlation_agent = CorrelationInsightAgent()
        self.peer_comparison_agent = PeerComparisonAgent()
    
    def create_crew(self) -> Crew:
        """
        Create and configure the comparative analysis crew
        Returns: CrewAI Crew object with agents and tasks
        """
        
        # Define tasks for comparative analysis
        tasks = [
            self._create_compare_stock_metrics_task(),
            self._create_analyze_correlations_task(),
            self._create_benchmark_against_peers_task()
        ]
        
        # Create crew with agents and tasks
        crew = Crew(
            agents=[
                self.comparative_agent,
                self.correlation_agent,
                self.peer_comparison_agent
            ],
            tasks=tasks,
            verbose=True,
            memory=True,
            max_execution_time=300  # 5 minutes timeout
        )
        
        self.logger.log_crew_activity(
            "ComparativeAnalysisCrew",
            f"Crew created for session {self.session_id}",
            "INFO"
        )
        
        return crew
    
    def _create_compare_stock_metrics_task(self) -> Task:
        """
        Task: Compare multiple stocks across key metrics
        Creates comparative visualizations and rankings
        """
        return Task(
            description="""
            Compare multiple stocks across key performance metrics including:
            - Returns (1d, 7d, 30d, 90d, YTD)
            - Volatility measures
            - Technical indicator values
            - Volume patterns
            - Price momentum
            
            Create comparative charts and ranking tables.
            Generate insights about relative performance.
            """,
            agent=self.comparative_agent,
            expected_output="""
            Dictionary containing:
            - comparative_charts: Interactive comparison visualizations
            - performance_rankings: Stocks ranked by various metrics
            - relative_insights: Key comparative insights
            - metric_summary: Summary statistics for all stocks
            """,
            callback=self._handle_compare_metrics_result
        )
    
    def _create_analyze_correlations_task(self) -> Task:
        """
        Task: Analyze correlations between stocks and features
        Identifies relationships and dependencies
        """
        return Task(
            description="""
            Analyze correlations between:
            - Stock price movements
            - Technical indicators across stocks
            - Sentiment scores and price changes
            - Volume patterns
            - Market timing relationships
            
            Create correlation matrices and network graphs.
            Identify strongest relationships and anomalies.
            """,
            agent=self.correlation_agent,
            expected_output="""
            Dictionary containing:
            - correlation_matrix: Price correlation matrix
            - feature_correlations: Technical indicator correlations
            - sentiment_correlations: Sentiment-price relationships
            - correlation_insights: Key correlation findings
            - network_graph: Relationship network visualization
            """,
            callback=self._handle_correlation_result
        )
    
    def _create_benchmark_against_peers_task(self) -> Task:
        """
        Task: Compare stocks against sector/industry peers
        Provides context for relative performance
        """
        return Task(
            description="""
            Benchmark selected stocks against:
            - Sector indices (if applicable)
            - Industry peers
            - Market benchmarks (S&P 500, etc.)
            - Historical performance ranges
            
            Calculate relative strength and positioning.
            Identify outperformers and underperformers.
            """,
            agent=self.peer_comparison_agent,
            expected_output="""
            Dictionary containing:
            - benchmark_comparisons: Performance vs benchmarks
            - sector_analysis: Sector-relative positioning
            - peer_rankings: Rankings within peer groups
            - relative_strength: Strength indicators
            - benchmark_insights: Key benchmarking insights
            """,
            callback=self._handle_benchmark_result
        )
    
    def _handle_compare_metrics_result(self, result: Dict[str, Any]) -> None:
        """Handle results from stock metrics comparison task"""
        try:
            # Store comparative analysis results
            if 'comparative_analysis' not in self.shared_memory:
                self.shared_memory['comparative_analysis'] = {}
            
            self.shared_memory['comparative_analysis']['metrics_comparison'] = result
            
            # Create additional comparative visualizations
            stock_data = self.shared_memory.get('processed_data', {})
            if stock_data:
                result['performance_chart'] = self._create_performance_comparison_chart(stock_data)
                result['volatility_comparison'] = self._create_volatility_comparison(stock_data)
            
            self.logger.log_crew_activity(
                "ComparativeAnalysisCrew",
                "Stock metrics comparison completed successfully",
                "INFO"
            )
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ComparativeAnalysisCrew",
                f"Error in metrics comparison: {str(e)}",
                "ERROR"
            )
    
    def _handle_correlation_result(self, result: Dict[str, Any]) -> None:
        """Handle results from correlation analysis task"""
        try:
            # Store correlation analysis results
            self.shared_memory['comparative_analysis']['correlation_analysis'] = result
            
            # Generate correlation visualizations
            if 'processed_data' in self.shared_memory:
                result['correlation_heatmap'] = self._create_correlation_heatmap(
                    self.shared_memory['processed_data']
                )
            
            self.logger.log_crew_activity(
                "ComparativeAnalysisCrew",
                "Correlation analysis completed successfully",
                "INFO"
            )
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ComparativeAnalysisCrew",
                f"Error in correlation analysis: {str(e)}",
                "ERROR"
            )
    
    def _handle_benchmark_result(self, result: Dict[str, Any]) -> None:
        """Handle results from peer benchmarking task"""
        try:
            # Store benchmarking results
            self.shared_memory['comparative_analysis']['benchmark_analysis'] = result
            
            # Create benchmark visualization
            if 'processed_data' in self.shared_memory:
                result['benchmark_chart'] = self._create_benchmark_chart(
                    self.shared_memory['processed_data']
                )
            
            self.logger.log_crew_activity(
                "ComparativeAnalysisCrew",
                "Peer benchmarking completed successfully",
                "INFO"
            )
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ComparativeAnalysisCrew",
                f"Error in peer benchmarking: {str(e)}",
                "ERROR"
            )
    
    def _create_performance_comparison_chart(self, stock_data: Dict[str, pd.DataFrame]) -> Dict:
        """Create performance comparison chart for multiple stocks"""
        try:
            # Calculate normalized returns for comparison
            normalized_data = {}
            for symbol, df in stock_data.items():
                if not df.empty and 'Close' in df.columns:
                    # Normalize to starting value of 100
                    normalized_data[symbol] = (df['Close'] / df['Close'].iloc[0] * 100).tolist()
            
            if not normalized_data:
                return {}
            
            # Create line chart comparing normalized performance
            dates = stock_data[list(stock_data.keys())[0]].index.strftime('%Y-%m-%d').tolist()
            
            chart_data = {
                'data': [],
                'layout': {
                    'title': 'Normalized Stock Performance Comparison',
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': 'Normalized Price (Base = 100)'},
                    'hovermode': 'x unified'
                }
            }
            
            for symbol, values in normalized_data.items():
                chart_data['data'].append({
                    'x': dates,
                    'y': values,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': symbol,
                    'line': {'width': 2}
                })
            
            return chart_data
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ComparativeAnalysisCrew",
                f"Error creating performance comparison chart: {str(e)}",
                "ERROR"
            )
            return {}
    
    def _create_volatility_comparison(self, stock_data: Dict[str, pd.DataFrame]) -> Dict:
        """Create volatility comparison visualization"""
        try:
            volatility_data = {}
            
            for symbol, df in stock_data.items():
                if not df.empty and 'Close' in df.columns:
                    # Calculate rolling volatility
                    returns = df['Close'].pct_change().dropna()
                    volatility = returns.rolling(window=20).std() * np.sqrt(252) * 100  # Annualized %
                    volatility_data[symbol] = volatility.dropna().tolist()
            
            if not volatility_data:
                return {}
            
            # Create volatility comparison chart
            dates = stock_data[list(stock_data.keys())[0]].index[-len(list(volatility_data.values())[0]):].strftime('%Y-%m-%d').tolist()
            
            chart_data = {
                'data': [],
                'layout': {
                    'title': 'Rolling Volatility Comparison (20-day)',
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': 'Annualized Volatility (%)'},
                    'hovermode': 'x unified'
                }
            }
            
            for symbol, values in volatility_data.items():
                chart_data['data'].append({
                    'x': dates,
                    'y': values,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': f'{symbol} Volatility',
                    'line': {'width': 2}
                })
            
            return chart_data
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ComparativeAnalysisCrew",
                f"Error creating volatility comparison: {str(e)}",
                "ERROR"
            )
            return {}
    
    def _create_correlation_heatmap(self, stock_data: Dict[str, pd.DataFrame]) -> Dict:
        """Create correlation heatmap for stock returns"""
        try:
            # Calculate returns for each stock
            returns_data = {}
            for symbol, df in stock_data.items():
                if not df.empty and 'Close' in df.columns:
                    returns = df['Close'].pct_change().dropna()
                    returns_data[symbol] = returns
            
            if len(returns_data) < 2:
                return {}
            
            # Create correlation matrix
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            # Create heatmap
            chart_data = {
                'data': [{
                    'z': correlation_matrix.values.tolist(),
                    'x': correlation_matrix.columns.tolist(),
                    'y': correlation_matrix.columns.tolist(),
                    'type': 'heatmap',
                    'colorscale': 'RdBu',
                    'zmid': 0,
                    'text': correlation_matrix.round(3).values.tolist(),
                    'texttemplate': '%{text}',
                    'textfont': {'size': 12},
                    'hoverongaps': False
                }],
                'layout': {
                    'title': 'Stock Returns Correlation Matrix',
                    'xaxis': {'title': 'Stocks'},
                    'yaxis': {'title': 'Stocks'},
                    'width': 600,
                    'height': 600
                }
            }
            
            return chart_data
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ComparativeAnalysisCrew",
                f"Error creating correlation heatmap: {str(e)}",
                "ERROR"
            )
            return {}
    
    def _create_benchmark_chart(self, stock_data: Dict[str, pd.DataFrame]) -> Dict:
        """Create benchmark comparison chart"""
        try:
            # This is a simplified benchmark - in a real implementation,
            # you would fetch actual benchmark data (S&P 500, sector indices, etc.)
            
            if not stock_data:
                return {}
            
            # Calculate equal-weighted portfolio as simple benchmark
            portfolio_values = []
            dates = None
            
            for symbol, df in stock_data.items():
                if not df.empty and 'Close' in df.columns:
                    normalized = df['Close'] / df['Close'].iloc[0]
                    if portfolio_values:
                        portfolio_values = [p + n for p, n in zip(portfolio_values, normalized.tolist())]
                    else:
                        portfolio_values = normalized.tolist()
                        dates = df.index.strftime('%Y-%m-%d').tolist()
            
            if not portfolio_values:
                return {}
            
            # Average the portfolio
            portfolio_values = [p / len(stock_data) for p in portfolio_values]
            
            chart_data = {
                'data': [{
                    'x': dates,
                    'y': portfolio_values,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Equal-Weight Portfolio',
                    'line': {'width': 3, 'color': 'orange'}
                }],
                'layout': {
                    'title': 'Portfolio vs Individual Stocks',
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': 'Normalized Performance'},
                    'hovermode': 'x unified'
                }
            }
            
            # Add individual stocks to the chart
            for symbol, df in stock_data.items():
                if not df.empty and 'Close' in df.columns:
                    normalized = (df['Close'] / df['Close'].iloc[0]).tolist()
                    chart_data['data'].append({
                        'x': dates,
                        'y': normalized,
                        'type': 'scatter',
                        'mode': 'lines',
                        'name': symbol,
                        'line': {'width': 2},
                        'opacity': 0.7
                    })
            
            return chart_data
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ComparativeAnalysisCrew",
                f"Error creating benchmark chart: {str(e)}",
                "ERROR"
            )
            return {}


def create_comparative_analysis_crew(session_id: str, shared_memory: Dict[str, Any]) -> Crew:
    """
    Factory function to create a comparative analysis crew instance
    
    Args:
        session_id: Unique session identifier
        shared_memory: Shared memory dictionary for crew communication
    
    Returns:
        Configured CrewAI Crew object for comparative analysis
    """
    crew_manager = ComparativeAnalysisCrew(session_id, shared_memory)
    return crew_manager.create_crew()