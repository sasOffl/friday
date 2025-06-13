"""
Report Generation Crew - McKinsey Stock Performance Monitor
Handles comprehensive report generation, visualization creation, and investment recommendations
"""

from crewai import Crew, Task
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from agents.report_agents import (
    ReportComposerAgent,
    VisualizationAgent,
    StrategyAdvisorAgent
)
from tools.visualization_tools import PlotlyToolKit
from utils.logger import AnalysisLogger


class ReportGenerationCrew:
    """
    Crew responsible for generating comprehensive analysis reports
    Creates final visualizations, composes insights, and provides investment recommendations
    """
    
    def __init__(self, session_id: str, shared_memory: Dict[str, Any]):
        self.session_id = session_id
        self.shared_memory = shared_memory
        self.logger = AnalysisLogger(session_id="temp-agent-session")
        self.plot_toolkit = PlotlyToolKit()
        
        # Initialize agents
        self.report_composer = ReportComposerAgent()
        self.visualization_agent = VisualizationAgent()
        self.strategy_advisor = StrategyAdvisorAgent()
    
    def create_crew(self) -> Crew:
        """
        Create and configure the report generation crew
        Returns: CrewAI Crew object with agents and tasks
        """
        
        # Define tasks for report generation
        tasks = [
            self._create_compose_report_task(),
            self._create_visualizations_task(),
            self._create_formulate_recommendations_task(),
            self._create_format_output_task()
        ]
        
        # Create crew with agents and tasks
        crew = Crew(
            agents=[
                self.report_composer,
                self.visualization_agent,
                self.strategy_advisor
            ],
            tasks=tasks,
            verbose=True,
            memory=True,
            max_execution_time=300  # 5 minutes timeout
        )
        
        self.logger.log_crew_activity(
            "ReportGenerationCrew",
            f"Crew created for session {self.session_id}",
            "INFO"
        )
        
        return crew
    
    def _create_compose_report_task(self) -> Task:
        """
        Task: Compose comprehensive analysis report
        Generates natural language insights and summaries
        """
        return Task(
            description="""
            Compose a comprehensive stock analysis report including:
            - Executive summary with key findings
            - Individual stock analysis and insights
            - Technical indicator interpretations
            - Sentiment analysis summary
            - Prediction model results and confidence levels
            - Risk assessment and volatility analysis
            - Comparative analysis insights
            - Market context and implications
            
            Use clear, professional language suitable for investment decision-making.
            Structure the report with clear sections and actionable insights.
            """,
            agent=self.report_composer,
            expected_output="""
            Dictionary containing:
            - executive_summary: High-level findings and recommendations
            - stock_analyses: Detailed analysis for each stock
            - technical_insights: Technical indicator interpretations
            - sentiment_summary: Sentiment analysis conclusions
            - prediction_analysis: Forecast results and reliability
            - risk_assessment: Risk factors and volatility analysis
            - market_context: Broader market implications
            - key_recommendations: Actionable investment guidance
            """,
            callback=self._handle_report_composition_result
        )
    
    def _create_visualizations_task(self) -> Task:
        """
        Task: Create all interactive visualizations
        Produces charts and graphs for the dashboard
        """
        return Task(
            description="""
            Create comprehensive interactive visualizations including:
            - Stock price charts with technical indicators
            - Prediction forecasts with confidence intervals
            - Sentiment timeline charts
            - Performance comparison charts
            - Correlation heatmaps
            - Volatility analysis charts
            - Health score dashboards
            - Risk-return scatter plots
            
            Ensure all charts are interactive, professional, and optimized for web display.
            Include proper legends, tooltips, and responsive design.
            """,
            agent=self.visualization_agent,
            expected_output="""
            Dictionary containing:
            - main_dashboard: Primary dashboard with key charts
            - individual_charts: Detailed charts for each stock
            - comparison_charts: Multi-stock comparison visualizations
            - technical_charts: Technical analysis visualizations
            - prediction_charts: Forecast and confidence interval charts
            - sentiment_charts: Sentiment analysis visualizations
            - risk_charts: Risk and volatility analysis charts
            """,
            callback=self._handle_visualizations_result
        )
    
    def _create_formulate_recommendations_task(self) -> Task:
        """
        Task: Formulate investment strategy recommendations
        Provides buy/hold/sell recommendations with rationale
        """
        return Task(
            description="""
            Formulate investment strategy recommendations based on:
            - Technical analysis results
            - Prediction model forecasts
            - Sentiment analysis trends
            - Risk-adjusted returns
            - Comparative performance
            - Market conditions
            
            For each stock, provide:
            - Clear recommendation (Buy/Hold/Sell)
            - Confidence level (1-10)
            - Key supporting factors
            - Risk considerations
            - Time horizon guidance
            - Entry/exit price targets (if applicable)
            
            Consider portfolio diversification and risk management principles.
            """,
            agent=self.strategy_advisor,
            expected_output="""
            Dictionary containing:
            - stock_recommendations: Individual stock recommendations
            - portfolio_guidance: Overall portfolio strategy
            - risk_management: Risk mitigation recommendations
            - timing_guidance: Entry/exit timing suggestions
            - diversification_advice: Portfolio balance recommendations
            - market_outlook: Overall market perspective
            """,
            callback=self._handle_recommendations_result
        )
    
    def _create_format_output_task(self) -> Task:
        """
        Task: Format final output for dashboard
        Structures all results for frontend consumption
        """
        return Task(
            description="""
            Format and structure all analysis results for the frontend dashboard:
            - Organize data in dashboard-friendly format
            - Ensure all visualizations are properly formatted
            - Create summary statistics and key metrics
            - Structure recommendations for easy display
            - Validate data integrity and completeness
            - Create responsive layout data
            
            Output should be ready for direct consumption by the web interface.
            """,
            agent=self.report_composer,
            expected_output="""
            Dictionary containing:
            - dashboard_data: Complete dashboard data structure
            - summary_metrics: Key performance indicators
            - chart_configurations: All chart data and configurations
            - recommendations_summary: Formatted recommendations
            - metadata: Analysis metadata and timestamps
            """,
            callback=self._handle_format_output_result
        )
    
    def _handle_report_composition_result(self, result: Dict[str, Any]) -> None:
        """Handle results from report composition task"""
        try:
            # Store report composition results
            if 'final_report' not in self.shared_memory:
                self.shared_memory['final_report'] = {}
            
            self.shared_memory['final_report']['narrative_report'] = result
            
            # Generate executive summary statistics
            result['summary_stats'] = self._generate_summary_statistics()
            
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                "Report composition completed successfully",
                "INFO"
            )
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                f"Error in report composition: {str(e)}",
                "ERROR"
            )
    
    def _handle_visualizations_result(self, result: Dict[str, Any]) -> None:
        """Handle results from visualizations creation task"""
        try:
            # Store visualization results
            self.shared_memory['final_report']['visualizations'] = result
            
            # Create additional dashboard-specific charts
            result['dashboard_summary'] = self._create_dashboard_summary_charts()
            
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                "Visualizations creation completed successfully",
                "INFO"
            )
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                f"Error in visualizations creation: {str(e)}",
                "ERROR"
            )
    
    def _handle_recommendations_result(self, result: Dict[str, Any]) -> None:
        """Handle results from strategy recommendations task"""
        try:
            # Store recommendations results
            self.shared_memory['final_report']['recommendations'] = result
            
            # Generate recommendation summary
            result['recommendation_summary'] = self._create_recommendation_summary(result)
            
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                "Strategy recommendations completed successfully",
                "INFO"
            )
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                f"Error in strategy recommendations: {str(e)}",
                "ERROR"
            )
    
    def _handle_format_output_result(self, result: Dict[str, Any]) -> None:
        """Handle results from output formatting task"""
        try:
            # Store final formatted output
            self.shared_memory['final_report']['formatted_output'] = result
            
            # Mark analysis as complete
            self.shared_memory['analysis_status'] = 'completed'
            self.shared_memory['completion_time'] = datetime.now().isoformat()
            
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                "Final output formatting completed - Analysis complete",
                "INFO"
            )
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                f"Error in output formatting: {str(e)}",
                "ERROR"
            )
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate high-level summary statistics"""
        try:
            stats = {
                'total_stocks_analyzed': 0,
                'analysis_period_days': 0,
                'prediction_horizon_days': 0,
                'average_health_score': 0,
                'total_news_articles': 0,
                'average_sentiment': 0,
                'high_confidence_predictions': 0
            }
            
            # Calculate stats from shared memory
            if 'processed_data' in self.shared_memory:
                stats['total_stocks_analyzed'] = len(self.shared_memory['processed_data'])
            
            if 'health_analysis' in self.shared_memory:
                health_scores = []
                for stock_health in self.shared_memory['health_analysis'].values():
                    if isinstance(stock_health, dict) and 'health_score' in stock_health:
                        health_scores.append(stock_health['health_score'])
                
                if health_scores:
                    stats['average_health_score'] = round(np.mean(health_scores), 1)
            
            if 'sentiment_data' in self.shared_memory:
                sentiment_scores = []
                total_articles = 0
                for stock_sentiment in self.shared_memory['sentiment_data'].values():
                    if isinstance(stock_sentiment, dict):
                        if 'articles' in stock_sentiment:
                            total_articles += len(stock_sentiment['articles'])
                        if 'sentiment_score' in stock_sentiment:
                            sentiment_scores.append(stock_sentiment['sentiment_score'])
                
                stats['total_news_articles'] = total_articles
                if sentiment_scores:
                    stats['average_sentiment'] = round(np.mean(sentiment_scores), 3)
            
            if 'predictions' in self.shared_memory:
                high_confidence = 0
                for stock_pred in self.shared_memory['predictions'].values():
                    if isinstance(stock_pred, dict) and 'confidence' in stock_pred:
                        if stock_pred['confidence'] > 0.7:  # High confidence threshold
                            high_confidence += 1
                
                stats['high_confidence_predictions'] = high_confidence
            
            return stats
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                f"Error generating summary statistics: {str(e)}",
                "ERROR"
            )
            return {}
    
    def _create_dashboard_summary_charts(self) -> Dict[str, Any]:
        """Create summary charts for the main dashboard"""
        try:
            charts = {}
            
            # Health Score Summary Chart
            charts['health_overview'] = self._create_health_overview_chart()
            
            # Performance Summary Chart
            charts['performance_overview'] = self._create_performance_overview_chart()
            
            # Sentiment Overview Chart
            charts['sentiment_overview'] = self._create_sentiment_overview_chart()
            
            return charts
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                f"Error creating dashboard summary charts: {str(e)}",
                "ERROR"
            )
            return {}
    
    def _create_health_overview_chart(self) -> Dict:
        """Create health score overview chart"""
        try:
            if 'health_analysis' not in self.shared_memory:
                return {}
            
            stocks = []
            health_scores = []
            
            for symbol, health_data in self.shared_memory['health_analysis'].items():
                if isinstance(health_data, dict) and 'health_score' in health_data:
                    stocks.append(symbol)
                    health_scores.append(health_data['health_score'])
            
            if not stocks:
                return {}
            
            # Create bar chart
            chart_data = {
                'data': [{
                    'x': stocks,
                    'y': health_scores,
                    'type': 'bar',
                    'name': 'Health Score',
                    'marker': {
                        'color': ['green' if score >= 70 else 'orange' if score >= 50 else 'red' 
                                for score in health_scores]
                    }
                }],
                'layout': {
                    'title': 'Stock Health Scores Overview',
                    'xaxis': {'title': 'Stocks'},
                    'yaxis': {'title': 'Health Score (0-100)', 'range': [0, 100]},
                    'showlegend': False
                }
            }
            
            return chart_data
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                f"Error creating health overview chart: {str(e)}",
                "ERROR"
            )
            return {}
    
    def _create_performance_overview_chart(self) -> Dict:
        """Create performance overview chart"""
        try:
            if 'processed_data' not in self.shared_memory:
                return {}
            
            stocks = []
            returns = []
            
            for symbol, stock_data in self.shared_memory['processed_data'].items():
                if isinstance(stock_data, pd.DataFrame) and not stock_data.empty:
                    # Calculate total return
                    if 'Close' in stock_data.columns and len(stock_data) > 1:
                        start_price = stock_data['Close'].iloc[0]
                        end_price = stock_data['Close'].iloc[-1]
                        total_return = ((end_price - start_price) / start_price) * 100
                        
                        stocks.append(symbol)
                        returns.append(total_return)
            
            if not stocks:
                return {}
            
            # Create bar chart
            chart_data = {
                'data': [{
                    'x': stocks,
                    'y': returns,
                    'type': 'bar',
                    'name': 'Total Return (%)',
                    'marker': {
                        'color': ['green' if ret > 0 else 'red' for ret in returns]
                    }
                }],
                'layout': {
                    'title': 'Stock Performance Overview',
                    'xaxis': {'title': 'Stocks'},
                    'yaxis': {'title': 'Total Return (%)'},
                    'showlegend': False
                }
            }
            
            return chart_data
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                f"Error creating performance overview chart: {str(e)}",
                "ERROR"
            )
            return {}
    
    def _create_sentiment_overview_chart(self) -> Dict:
        """Create sentiment overview chart"""
        try:
            if 'sentiment_data' not in self.shared_memory:
                return {}
            
            stocks = []
            sentiment_scores = []
            
            for symbol, sentiment_data in self.shared_memory['sentiment_data'].items():
                if isinstance(sentiment_data, dict) and 'sentiment_score' in sentiment_data:
                    stocks.append(symbol)
                    sentiment_scores.append(sentiment_data['sentiment_score'])
            
            if not stocks:
                return {}
            
            # Create bar chart
            chart_data = {
                'data': [{
                    'x': stocks,
                    'y': sentiment_scores,
                    'type': 'bar',
                    'name': 'Sentiment Score',
                    'marker': {
                        'color': ['green' if score > 0.1 else 'orange' if score > -0.1 else 'red' 
                                for score in sentiment_scores]
                    }
                }],
                'layout': {
                    'title': 'Sentiment Analysis Overview',
                    'xaxis': {'title': 'Stocks'},
                    'yaxis': {'title': 'Sentiment Score (-1 to 1)', 'range': [-1, 1]},
                    'showlegend': False
                }
            }
            
            return chart_data
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                f"Error creating sentiment overview chart: {str(e)}",
                "ERROR"
            )
            return {}
    
    def _create_recommendation_summary(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Create recommendation summary"""
        try:
            summary = {
                'total_stocks': 0,
                'buy_recommendations': 0,
                'hold_recommendations': 0,
                'sell_recommendations': 0,
                'average_confidence': 0,
                'high_confidence_count': 0,
                'risk_distribution': {'low': 0, 'medium': 0, 'high': 0}
            }
            
            if 'stock_recommendations' in recommendations:
                stock_recs = recommendations['stock_recommendations']
                summary['total_stocks'] = len(stock_recs)
                
                confidences = []
                for symbol, rec in stock_recs.items():
                    if isinstance(rec, dict):
                        # Count recommendations
                        if 'recommendation' in rec:
                            rec_type = rec['recommendation'].lower()
                            if 'buy' in rec_type:
                                summary['buy_recommendations'] += 1
                            elif 'sell' in rec_type:
                                summary['sell_recommendations'] += 1
                            else:
                                summary['hold_recommendations'] += 1
                        
                        # Track confidence
                        if 'confidence' in rec:
                            conf = rec['confidence']
                            confidences.append(conf)
                            if conf > 7:  # High confidence threshold
                                summary['high_confidence_count'] += 1
                        
                        # Track risk distribution
                        if 'risk_level' in rec:
                            risk = rec['risk_level'].lower()
                            if risk in summary['risk_distribution']:
                                summary['risk_distribution'][risk] += 1
                
                if confidences:
                    summary['average_confidence'] = round(np.mean(confidences), 1)
            
            return summary
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                f"Error creating recommendation summary: {str(e)}",
                "ERROR"
            )
            return {}
    
    def get_crew_status(self) -> Dict[str, Any]:
        """Get current crew execution status"""
        try:
            status = {
                'crew_name': 'ReportGenerationCrew',
                'session_id': self.session_id,
                'status': self.shared_memory.get('analysis_status', 'unknown'),
                'completion_time': self.shared_memory.get('completion_time'),
                'has_final_report': 'final_report' in self.shared_memory,
                'report_sections': []
            }
            
            if 'final_report' in self.shared_memory:
                final_report = self.shared_memory['final_report']
                if 'narrative_report' in final_report:
                    status['report_sections'].append('narrative_report')
                if 'visualizations' in final_report:
                    status['report_sections'].append('visualizations')
                if 'recommendations' in final_report:
                    status['report_sections'].append('recommendations')
                if 'formatted_output' in final_report:
                    status['report_sections'].append('formatted_output')
            
            return status
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                f"Error getting crew status: {str(e)}",
                "ERROR"
            )
            return {'crew_name': 'ReportGenerationCrew', 'status': 'error'}
    
    def get_final_report(self) -> Optional[Dict[str, Any]]:
        """Get the final generated report"""
        try:
            if 'final_report' not in self.shared_memory:
                return None
            
            final_report = self.shared_memory['final_report']
            
            # Ensure all required sections are present
            required_sections = ['narrative_report', 'visualizations', 'recommendations', 'formatted_output']
            for section in required_sections:
                if section not in final_report:
                    self.logger.log_crew_activity(
                        "ReportGenerationCrew",
                        f"Missing required section: {section}",
                        "WARNING"
                    )
                    return None
            
            # Add metadata
            final_report['metadata'] = {
                'session_id': self.session_id,
                'generation_time': datetime.now().isoformat(),
                'crew_name': 'ReportGenerationCrew',
                'report_version': '1.0',
                'total_sections': len(final_report),
                'summary_stats': final_report.get('narrative_report', {}).get('summary_stats', {})
            }
            
            return final_report
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                f"Error retrieving final report: {str(e)}",
                "ERROR"
            )
            return None
    
    def cleanup_resources(self) -> None:
        """Clean up crew resources"""
        try:
            # Clean up temporary data
            if hasattr(self, 'plot_toolkit'):
                del self.plot_toolkit
            
            # Clear references to agents
            if hasattr(self, 'report_composer'):
                del self.report_composer
            if hasattr(self, 'visualization_agent'):
                del self.visualization_agent
            if hasattr(self, 'strategy_advisor'):
                del self.strategy_advisor
            
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                f"Resources cleaned up for session {self.session_id}",
                "INFO"
            )
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                f"Error during cleanup: {str(e)}",
                "ERROR"
            )
    
    def validate_shared_memory(self) -> bool:
        """Validate that shared memory contains required data for report generation"""
        try:
            required_keys = [
                'processed_data',
                'health_analysis',
                'sentiment_data',
                'predictions'
            ]
            
            missing_keys = []
            for key in required_keys:
                if key not in self.shared_memory:
                    missing_keys.append(key)
            
            if missing_keys:
                self.logger.log_crew_activity(
                    "ReportGenerationCrew",
                    f"Missing required data in shared memory: {missing_keys}",
                    "WARNING"
                )
                return False
            
            # Check if data is not empty
            for key in required_keys:
                if not self.shared_memory[key]:
                    self.logger.log_crew_activity(
                        "ReportGenerationCrew",
                        f"Empty data found for key: {key}",
                        "WARNING"
                    )
                    return False
            
            return True
            
        except Exception as e:
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                f"Error validating shared memory: {str(e)}",
                "ERROR"
            )
            return False
    
    def execute_crew(self) -> Dict[str, Any]:
        """Execute the report generation crew"""
        try:
            # Validate shared memory first
            if not self.validate_shared_memory():
                return {
                    'status': 'failed',
                    'error': 'Invalid or missing data in shared memory',
                    'session_id': self.session_id
                }
            
            # Create and execute crew
            crew = self.create_crew()
            
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                f"Starting crew execution for session {self.session_id}",
                "INFO"
            )
            
            # Execute crew tasks
            results = crew.kickoff()
            
            # Check if execution was successful
            if 'final_report' in self.shared_memory and self.shared_memory.get('analysis_status') == 'completed':
                self.logger.log_crew_activity(
                    "ReportGenerationCrew",
                    f"Crew execution completed successfully for session {self.session_id}",
                    "INFO"
                )
                
                return {
                    'status': 'completed',
                    'session_id': self.session_id,
                    'results': results,
                    'final_report': self.get_final_report()
                }
            else:
                self.logger.log_crew_activity(
                    "ReportGenerationCrew",
                    f"Crew execution incomplete for session {self.session_id}",
                    "WARNING"
                )
                
                return {
                    'status': 'incomplete',
                    'session_id': self.session_id,
                    'results': results
                }
                
        except Exception as e:
            self.logger.log_crew_activity(
                "ReportGenerationCrew",
                f"Error during crew execution: {str(e)}",
                "ERROR"
            )
            
            return {
                'status': 'failed',
                'error': str(e),
                'session_id': self.session_id
            }
        finally:
            # Always cleanup resources
            self.cleanup_resources()


# Factory function to create ReportGenerationCrew instances
def create_report_generation_crew(session_id: str, shared_memory: Dict[str, Any]) -> ReportGenerationCrew:
    """
    Factory function to create ReportGenerationCrew instances
    
    Args:
        session_id: Unique session identifier
        shared_memory: Shared memory dictionary containing analysis data
    
    Returns:
        ReportGenerationCrew instance
    """
    return ReportGenerationCrew(session_id, shared_memory)


# Utility function to validate crew execution results
def validate_crew_results(results: Dict[str, Any]) -> bool:
    """
    Validate that crew execution results are complete and valid
    
    Args:
        results: Crew execution results dictionary
    
    Returns:
        Boolean indicating if results are valid
    """
    try:
        if not isinstance(results, dict):
            return False
        
        required_keys = ['status', 'session_id']
        for key in required_keys:
            if key not in results:
                return False
        
        if results['status'] == 'completed':
            return 'final_report' in results and results['final_report'] is not None
        
        return results['status'] in ['completed', 'incomplete', 'failed']
        
    except Exception:
        return False