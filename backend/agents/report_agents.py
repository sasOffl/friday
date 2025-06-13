"""
McKinsey Stock Performance Monitor - Report Generation Agents

This module contains agents responsible for dynamic report generation, visualization,
and strategic recommendations with real-time updates and interactive dashboards.
"""

from crewai import Agent
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging

from backend.tools.visualization_tools import PlotlyToolKit
from backend.utils.logger import AnalysisLogger

logger = AnalysisLogger(session_id="temp-agent-session")


class ReportComposerAgent:
    """
    Agent responsible for composing comprehensive analysis reports with dynamic insights.
    
    Capabilities:
    - Generate executive summaries
    - Create detailed technical analysis sections
    - Compose risk assessments
    - Provide investment thesis narratives
    """
    
    def __init__(self):
        self.agent = Agent(
            role="Senior Investment Analyst & Report Writer",
            goal="Compose comprehensive, McKinsey-style investment reports with actionable insights",
            backstory="""You are a senior investment analyst with 15+ years at top-tier 
            consulting firms. You excel at synthesizing complex financial data into clear, 
            actionable insights for C-level executives and institutional investors.""",
            verbose=True,
            allow_delegation=False
        )
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dynamic report composition task."""
        try:
            all_results = task.get('all_analysis_results', {})
            session_id = task.get('session_id')
            
            logger.log_crew_activity(
                'ReportGeneration',
                f"Starting dynamic report composition for {len(all_results)} symbols",
                'INFO'
            )
            
            reports = {}
            portfolio_summary = self._generate_portfolio_summary(all_results)
            
            for symbol in all_results.keys():
                try:
                    # Extract all data for this symbol
                    symbol_data = self._extract_symbol_data(symbol, all_results)
                    
                    # Generate dynamic report sections
                    executive_summary = self._create_executive_summary(symbol, symbol_data)
                    technical_analysis = self._create_technical_section(symbol, symbol_data)
                    health_assessment = self._create_health_section(symbol, symbol_data)
                    sentiment_analysis = self._create_sentiment_section(symbol, symbol_data)
                    risk_assessment = self._create_risk_section(symbol, symbol_data)
                    investment_thesis = self._create_investment_thesis(symbol, symbol_data)
                    
                    reports[symbol] = {
                        'symbol': symbol,
                        'report_timestamp': datetime.now().isoformat(),
                        'executive_summary': executive_summary,
                        'technical_analysis': technical_analysis,
                        'health_assessment': health_assessment,
                        'sentiment_analysis': sentiment_analysis,
                        'risk_assessment': risk_assessment,
                        'investment_thesis': investment_thesis,
                        'key_metrics': self._extract_key_metrics(symbol_data),
                        'real_time_alerts': self._generate_real_time_alerts(symbol, symbol_data)
                    }
                    
                    logger.log_crew_activity(
                        'ReportGeneration',
                        f"Report composed for {symbol} - Recommendation: {investment_thesis.get('recommendation', 'HOLD')}",
                        'INFO'
                    )
                    
                except Exception as e:
                    logger.log_crew_activity(
                        'ReportGeneration',
                        f"Error composing report for {symbol}: {str(e)}",
                        'ERROR'
                    )
            
            return {
                'status': 'completed',
                'data': {
                    'individual_reports': reports,
                    'portfolio_summary': portfolio_summary,
                    'generation_metadata': {
                        'total_symbols': len(reports),
                        'generation_time': datetime.now().isoformat(),
                        'report_version': '2.0-dynamic'
                    }
                }
            }
            
        except Exception as e:
            logger.log_crew_activity(
                'ReportGeneration',
                f"Report composition failed: {str(e)}",
                'ERROR'
            )
            return {'status': 'failed', 'error': str(e)}
    
    def _extract_symbol_data(self, symbol: str, all_results: Dict) -> Dict:
        """Extract and organize all data for a specific symbol."""
        return {
            'processed_data': all_results.get('data_ingestion', {}).get('data', {}).get(symbol, {}),
            'predictions': all_results.get('model_prediction', {}).get('data', {}).get(symbol, {}),
            'health_data': all_results.get('health_analytics', {}).get('data', {}).get(symbol, {}),
            'comparative_data': all_results.get('comparative_analysis', {}).get('data', {}).get(symbol, {})
        }
    
    def _create_executive_summary(self, symbol: str, data: Dict) -> Dict:
        """Generate dynamic executive summary with key insights."""
        health_score = data.get('health_data', {}).get('composite_score', 50)
        prediction = data.get('predictions', {})
        current_price = self._get_current_price(data)
        
        # Dynamic recommendation logic
        if health_score >= 75 and prediction.get('trend', '') == 'BULLISH':
            recommendation = 'STRONG BUY'
            confidence = 'HIGH'
        elif health_score >= 60 and prediction.get('trend', '') in ['BULLISH', 'NEUTRAL']:
            recommendation = 'BUY'
            confidence = 'MODERATE'
        elif health_score <= 35 or prediction.get('trend', '') == 'BEARISH':
            recommendation = 'SELL'
            confidence = 'HIGH'
        else:
            recommendation = 'HOLD'
            confidence = 'MODERATE'
        
        return {
            'recommendation': recommendation,
            'confidence_level': confidence,
            'health_score': health_score,
            'current_price': current_price,
            'target_price': prediction.get('forecasted_price', current_price),
            'key_highlights': self._generate_key_highlights(symbol, data),
            'risk_level': self._assess_risk_level(data),
            'time_horizon': '3-6 months'
        }
    
    def _create_technical_section(self, symbol: str, data: Dict) -> Dict:
        """Create dynamic technical analysis section."""
        indicators = data.get('health_data', {}).get('component_scores', {}).get('technical', 50)
        
        return {
            'technical_score': indicators,
            'key_indicators': {
                'rsi': self._get_indicator_status(data, 'rsi'),
                'macd': self._get_indicator_status(data, 'macd'),
                'bollinger_bands': self._get_indicator_status(data, 'bb'),
                'moving_averages': self._get_indicator_status(data, 'trend')
            },
            'support_resistance': self._calculate_support_resistance(data),
            'chart_patterns': self._identify_chart_patterns(data),
            'momentum_analysis': self._analyze_momentum(data)
        }
    
    def _create_health_section(self, symbol: str, data: Dict) -> Dict:
        """Create comprehensive health assessment section."""
        health_data = data.get('health_data', {})
        
        return {
            'overall_grade': health_data.get('health_grade', 'C'),
            'composite_score': health_data.get('composite_score', 50),
            'component_breakdown': health_data.get('component_scores', {}),
            'health_trends': self._analyze_health_trends(data),
            'peer_comparison': self._get_peer_health_comparison(symbol, data),
            'improvement_areas': self._identify_improvement_areas(health_data)
        }
    
    def _create_investment_thesis(self, symbol: str, data: Dict) -> Dict:
        """Generate dynamic investment thesis based on all available data."""
        health_score = data.get('health_data', {}).get('composite_score', 50)
        prediction = data.get('predictions', {})
        sentiment = data.get('health_data', {}).get('sentiment_analysis', {})
        
        # Dynamic thesis generation
        bullish_factors = []
        bearish_factors = []
        
        if health_score > 70:
            bullish_factors.append("Strong fundamental health metrics")
        if prediction.get('trend') == 'BULLISH':
            bullish_factors.append("Positive price momentum predicted")
        if sentiment.get('overall_trend') == 'POSITIVE':
            bullish_factors.append("Improving market sentiment")
        
        if health_score < 40:
            bearish_factors.append("Weak fundamental health indicators")
        if prediction.get('trend') == 'BEARISH':
            bearish_factors.append("Negative price momentum expected")
        if sentiment.get('overall_trend') == 'NEGATIVE':
            bearish_factors.append("Deteriorating market sentiment")
        
        return {
            'recommendation': self._determine_final_recommendation(bullish_factors, bearish_factors),
            'bullish_factors': bullish_factors,
            'bearish_factors': bearish_factors,
            'catalyst_events': self._identify_catalysts(data),
            'price_targets': self._calculate_price_targets(data),
            'investment_horizon': self._recommend_time_horizon(data)
        }
    
    def _generate_real_time_alerts(self, symbol: str, data: Dict) -> List[Dict]:
        """Generate real-time trading alerts and notifications."""
        alerts = []
        current_time = datetime.now().isoformat()
        
        # Technical alerts
        indicators = data.get('health_data', {})
        if indicators.get('rsi_current', 50) > 70:
            alerts.append({
                'type': 'TECHNICAL',
                'severity': 'MEDIUM',
                'message': f"RSI indicates overbought conditions at {indicators.get('rsi_current', 0):.1f}",
                'timestamp': current_time
            })
        
        # Health alerts
        health_score = data.get('health_data', {}).get('composite_score', 50)
        if health_score < 30:
            alerts.append({
                'type': 'HEALTH',
                'severity': 'HIGH',
                'message': f"Critical health score: {health_score}/100 - Immediate attention required",
                'timestamp': current_time
            })
        
        return alerts
    
    def _generate_portfolio_summary(self, all_results: Dict) -> Dict:
        """Generate dynamic portfolio-level summary."""
        symbols = list(all_results.get('data_ingestion', {}).get('data', {}).keys())
        
        portfolio_health = []
        recommendations = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        for symbol in symbols:
            health_data = all_results.get('health_analytics', {}).get('data', {}).get(symbol, {})
            health_score = health_data.get('composite_score', 50)
            portfolio_health.append(health_score)
        
        return {
            'total_symbols': len(symbols),
            'average_health_score': np.mean(portfolio_health) if portfolio_health else 0,
            'portfolio_grade': self._calculate_portfolio_grade(portfolio_health),
            'diversification_score': self._calculate_diversification_score(all_results),
            'risk_metrics': self._calculate_portfolio_risk(all_results),
            'top_performers': self._identify_top_performers(all_results),
            'underperformers': self._identify_underperformers(all_results)
        }


class VisualizationAgent:
    """
    Agent responsible for creating dynamic, interactive visualizations and dashboards.
    """
    
    def __init__(self):
        self.plotly_toolkit = PlotlyToolKit()
        self.agent = Agent(
            role="Data Visualization Specialist",
            goal="Create compelling, interactive visualizations that tell the investment story",
            backstory="""You are a data visualization expert who creates stunning, 
            interactive dashboards for financial institutions. Your visualizations 
            help investors make quick, informed decisions.""",
            verbose=True,
            allow_delegation=False
        )
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dynamic visualization creation task."""
        try:
            all_results = task.get('all_analysis_results', {})
            session_id = task.get('session_id')
            
            logger.log_crew_activity(
                'ReportGeneration',
                f"Creating dynamic visualizations for {len(all_results)} symbols",
                'INFO'
            )
            
            visualizations = {}
            
            # Create individual stock charts
            for symbol in all_results.get('data_ingestion', {}).get('data', {}).keys():
                try:
                    symbol_data = self._extract_visualization_data(symbol, all_results)
                    
                    visualizations[symbol] = {
                        'price_chart': self._create_dynamic_price_chart(symbol, symbol_data),
                        'health_gauge': self._create_health_gauge(symbol, symbol_data),
                        'technical_indicators': self._create_indicators_chart(symbol, symbol_data),
                        'sentiment_timeline': self._create_sentiment_chart(symbol, symbol_data),
                        'volume_analysis': self._create_volume_chart(symbol, symbol_data),
                        'prediction_chart': self._create_prediction_chart(symbol, symbol_data)
                    }
                    
                except Exception as e:
                    logger.log_crew_activity(
                        'ReportGeneration',
                        f"Error creating visualizations for {symbol}: {str(e)}",
                        'ERROR'
                    )
            
            # Create portfolio-level visualizations
            portfolio_charts = {
                'portfolio_overview': self._create_portfolio_dashboard(all_results),
                'comparative_analysis': self._create_comparative_charts(all_results),
                'correlation_matrix': self._create_correlation_heatmap(all_results),
                'risk_return_scatter': self._create_risk_return_plot(all_results)
            }
            
            return {
                'status': 'completed',
                'data': {
                    'individual_charts': visualizations,
                    'portfolio_charts': portfolio_charts,
                    'real_time_config': self._generate_real_time_config()
                }
            }
            
        except Exception as e:
            logger.log_crew_activity(
                'ReportGeneration',
                f"Visualization creation failed: {str(e)}",
                'ERROR'
            )
            return {'status': 'failed', 'error': str(e)}
    
    def _create_dynamic_price_chart(self, symbol: str, data: Dict) -> Dict:
        """Create interactive price chart with predictions and alerts."""
        stock_data = data.get('stock_data')
        predictions = data.get('predictions', {})
        
        if stock_data is None or stock_data.empty:
            return {'error': 'No stock data available'}
        
        # Create candlestick chart with volume
        chart_config = {
            'type': 'candlestick_with_volume',
            'data': {
                'dates': stock_data.index.strftime('%Y-%m-%d').tolist(),
                'open': stock_data['Open'].tolist(),
                'high': stock_data['High'].tolist(),
                'low': stock_data['Low'].tolist(),
                'close': stock_data['Close'].tolist(),
                'volume': stock_data['Volume'].tolist()
            },
            'predictions': predictions.get('forecasted_prices', []),
            'technical_overlays': self._get_technical_overlays(data),
            'alerts': self._get_chart_alerts(data),
            'real_time': True,
            'auto_refresh': 300  # 5 minutes
        }
        
        return chart_config
    
    def _create_health_gauge(self, symbol: str, data: Dict) -> Dict:
        """Create dynamic health score gauge with real-time updates."""
        health_data = data.get('health_data', {})
        
        return {
            'type': 'gauge',
            'value': health_data.get('composite_score', 50),
            'max_value': 100,
            'ranges': [
                {'range': [0, 30], 'color': '#ff4444', 'label': 'Poor'},
                {'range': [30, 50], 'color': '#ff8800', 'label': 'Fair'},
                {'range': [50, 70], 'color': '#ffdd00', 'label': 'Good'},
                {'range': [70, 85], 'color': '#88dd00', 'label': 'Very Good'},
                {'range': [85, 100], 'color': '#00dd44', 'label': 'Excellent'}
            ],
            'components': health_data.get('component_scores', {}),
            'trend': self._calculate_health_trend(data),
            'real_time': True
        }
    
    def _create_portfolio_dashboard(self, all_results: Dict) -> Dict:
        """Create comprehensive portfolio dashboard."""
        symbols = list(all_results.get('data_ingestion', {}).get('data', {}).keys())
        
        dashboard_config = {
            'type': 'multi_panel_dashboard',
            'panels': [
                {
                    'title': 'Portfolio Health Overview',
                    'type': 'health_summary',
                    'data': self._get_portfolio_health_data(all_results)
                },
                {
                    'title': 'Performance Comparison',
                    'type': 'performance_bars',
                    'data': self._get_performance_comparison_data(all_results)
                },
                {
                    'title': 'Risk vs Return',
                    'type': 'scatter_plot',
                    'data': self._get_risk_return_data(all_results)
                },
                {
                    'title': 'Sector Allocation',
                    'type': 'pie_chart',
                    'data': self._get_sector_allocation_data(all_results)
                }
            ],
            'real_time_updates': True,
            'refresh_interval': 60
        }
        
        return dashboard_config
    
    def _generate_real_time_config(self) -> Dict:
        """Generate configuration for real-time updates."""
        return {
            'websocket_endpoints': {
                'price_updates': '/ws/prices',
                'health_updates': '/ws/health',
                'alert_updates': '/ws/alerts'
            },
            'update_intervals': {
                'price_data': 60,      # 1 minute
                'health_scores': 300,   # 5 minutes
                'sentiment_data': 900,  # 15 minutes
                'predictions': 3600     # 1 hour
            },
            'auto_refresh': True,
            'notification_settings': {
                'price_alerts': True,
                'health_warnings': True,
                'breakout_signals': True
            }
        }


class StrategyAdvisorAgent:
    """
    Agent responsible for generating dynamic investment strategies and recommendations.
    """
    
    def __init__(self):
        self.agent = Agent(
            role="Senior Investment Strategist",
            goal="Provide dynamic, adaptive investment strategies based on real-time market conditions",
            backstory="""You are a senior portfolio strategist with expertise in 
            quantitative analysis and risk management. You create adaptive strategies 
            that respond to changing market conditions in real-time.""",
            verbose=True,
            allow_delegation=False
        )
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dynamic strategy generation task."""
        try:
            all_results = task.get('all_analysis_results', {})
            reports = task.get('reports', {})
            session_id = task.get('session_id')
            
            logger.log_crew_activity(
                'ReportGeneration',
                f"Generating dynamic investment strategies",
                'INFO'
            )
            
            # Generate individual stock strategies
            stock_strategies = {}
            for symbol in all_results.get('data_ingestion', {}).get('data', {}).keys():
                strategy = self._generate_stock_strategy(symbol, all_results, reports)
                stock_strategies[symbol] = strategy
            
            # Generate portfolio-level strategy
            portfolio_strategy = self._generate_portfolio_strategy(all_results, stock_strategies)
            
            # Create dynamic trading signals
            trading_signals = self._generate_trading_signals(all_results)
            
            # Risk management recommendations
            risk_management = self._generate_risk_management_plan(all_results)
            
            return {
                'status': 'completed',
                'data': {
                    'stock_strategies': stock_strategies,
                    'portfolio_strategy': portfolio_strategy,
                    'trading_signals': trading_signals,
                    'risk_management': risk_management,
                    'dynamic_rules': self._create_dynamic_rules(),
                    'market_regime_analysis': self._analyze_market_regime(all_results)
                }
            }
            
        except Exception as e:
            logger.log_crew_activity(
                'ReportGeneration',
                f"Strategy generation failed: {str(e)}",
                'ERROR'
            )
            return {'status': 'failed', 'error': str(e)}
    
    def _generate_stock_strategy(self, symbol: str, all_results: Dict, reports: Dict) -> Dict:
        """Generate adaptive strategy for individual stock."""
        health_data = all_results.get('health_analytics', {}).get('data', {}).get(symbol, {})
        prediction_data = all_results.get('model_prediction', {}).get('data', {}).get(symbol, {})
        
        health_score = health_data.get('composite_score', 50)
        predicted_return = prediction_data.get('expected_return', 0)
        
        # Dynamic strategy logic
        if health_score >= 80 and predicted_return > 0.1:
            strategy_type = 'AGGRESSIVE_GROWTH'
            allocation = 0.25  # 25% of portfolio
            holding_period = '6-12 months'
        elif health_score >= 60 and predicted_return > 0.05:
            strategy_type = 'MODERATE_GROWTH'
            allocation = 0.15
            holding_period = '3-6 months'
        elif health_score >= 40:
            strategy_type = 'CONSERVATIVE_HOLD'
            allocation = 0.10
            holding_period = '1-3 months'
        else:
            strategy_type = 'DEFENSIVE_EXIT'
            allocation = 0.00
            holding_period = 'Immediate'
        
        return {
            'strategy_type': strategy_type,
            'recommended_allocation': allocation,
            'holding_period': holding_period,
            'entry_price': self._calculate_entry_price(all_results, symbol),
            'stop_loss': self._calculate_stop_loss(all_results, symbol),
            'take_profit': self._calculate_take_profit(all_results, symbol),
            'dynamic_adjustments': self._create_dynamic_adjustments(symbol, all_results)
        }
    
    def _create_dynamic_rules(self) -> List[Dict]:
        """Create dynamic rules that adapt to market conditions."""
        return [
            {
                'rule_id': 'volatility_adjustment',
                'condition': 'market_volatility > 0.25',
                'action': 'reduce_position_sizes_by_20_percent',
                'priority': 'HIGH'
            },
            {
                'rule_id': 'health_deterioration',
                'condition': 'health_score_drops_below_30',
                'action': 'trigger_stop_loss',
                'priority': 'CRITICAL'
            },
            {
                'rule_id': 'momentum_reversal',
                'condition': 'rsi_crosses_70_from_below',
                'action': 'consider_profit_taking',
                'priority': 'MEDIUM'
            },
            {
                'rule_id': 'sentiment_shift',
                'condition': 'sentiment_trend_reverses',
                'action': 'reassess_position_size',
                'priority': 'MEDIUM'
            }
        ]