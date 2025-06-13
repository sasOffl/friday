"""
McKinsey Stock Performance Monitor - Agents Package

This package contains all specialized agents for the stock analysis system.
Agents are organized by functional domains:
- Data Agents: Market data loading, news sentiment, preprocessing
- Prediction Agents: Forecasting, model evaluation
- Health Agents: Technical analysis, health scoring, sentiment trends
- Comparative Agents: Multi-stock comparison, correlation analysis
- Report Agents: Report composition, visualization, strategy advice
"""

from .data_agents import (
    MarketDataLoaderAgent,
    NewsSentimentAgent, 
    DataPreprocessingAgent
)

from .prediction_agents import (
    ForecastAgent,
    EvaluationAgent
)

from .health_agents import (
    IndicatorAnalysisAgent,
    StockHealthAgent,
    SentimentTrendAgent
)

from .comparative_agents import (
    ComparativeAgent,
    CorrelationInsightAgent,
    PeerComparisonAgent
)

from .report_agents import (
    ReportComposerAgent,
    VisualizationAgent,
    StrategyAdvisorAgent
)

__all__ = [
    # Data Agents
    'MarketDataLoaderAgent',
    'NewsSentimentAgent',
    'DataPreprocessingAgent',
    
    # Prediction Agents
    'ForecastAgent',
    'EvaluationAgent',
    
    # Health Agents
    'IndicatorAnalysisAgent',
    'StockHealthAgent',
    'SentimentTrendAgent',
    
    # Comparative Agents
    'ComparativeAgent',
    'CorrelationInsightAgent',
    'PeerComparisonAgent',
    
    # Report Agents
    'ReportComposerAgent',
    'VisualizationAgent',
    'StrategyAdvisorAgent'
]