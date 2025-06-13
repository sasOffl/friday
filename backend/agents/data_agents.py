"""
McKinsey Stock Performance Monitor - Data Agents

This module contains agents responsible for data collection, processing, and validation.
Agents handle market data loading, news sentiment analysis, and data preprocessing.
"""

from crewai import Agent
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging

from backend.tools.market_data_tools import YahooFinanceTool
from backend.tools.sentiment_tools import FirecrawlScraper, FinBERTWrapper, ChromaDBTool
from backend.utils.logger import AnalysisLogger

logger = AnalysisLogger(session_id="temp-agent-session")


class MarketDataLoaderAgent:
    """
    Agent responsible for loading and validating stock market data.
    
    Capabilities:
    - Fetch OHLCV data from Yahoo Finance
    - Calculate technical indicators
    - Validate data quality and completeness
    - Store processed data to database
    """
    
    def __init__(self):
        self.yahoo_tool = YahooFinanceTool()
        self.agent = Agent(
            role="Market Data Specialist",
            goal="Collect comprehensive and accurate stock market data",
            backstory="""You are a seasoned financial data analyst with expertise in 
            market data collection and quality assurance. You ensure data integrity 
            and completeness for accurate financial analysis.""",
            verbose=True,
            allow_delegation=False
        )
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute market data loading task.
        
        Args:
            task: Dictionary containing symbols, period, and parameters
            
        Returns:
            Dictionary with loaded data and validation results
        """
        try:
            symbols = task.get('symbols', [])
            period = task.get('period', '1y')
            session_id = task.get('session_id')
            
            logger.log_crew_activity(
                'DataIngestion', 
                f"Starting market data collection for {len(symbols)} symbols", 
                'INFO'
            )
            
            results = {}
            
            for symbol in symbols:
                try:
                    # Fetch stock data
                    stock_data = self.yahoo_tool.fetch_stock_data(symbol, period)
                    
                    if stock_data is not None and not stock_data.empty:
                        # Calculate technical indicators
                        technical_data = self.yahoo_tool.get_technical_indicators(symbol, stock_data)
                        
                        # Validate data quality
                        validation_result = self._validate_data_quality(stock_data)
                        
                        results[symbol] = {
                            'stock_data': stock_data,
                            'technical_indicators': technical_data,
                            'validation': validation_result,
                            'data_points': len(stock_data),
                            'date_range': {
                                'start': stock_data.index.min().strftime('%Y-%m-%d'),
                                'end': stock_data.index.max().strftime('%Y-%m-%d')
                            }
                        }
                        
                        logger.log_crew_activity(
                            'DataIngestion',
                            f"Successfully loaded {len(stock_data)} data points for {symbol}",
                            'INFO'
                        )
                    else:
                        logger.log_crew_activity(
                            'DataIngestion',
                            f"Failed to load data for {symbol}",
                            'WARNING'
                        )
                        
                except Exception as e:
                    logger.log_crew_activity(
                        'DataIngestion',
                        f"Error loading data for {symbol}: {str(e)}",
                        'ERROR'
                    )
                    
            return {
                'status': 'completed',
                'data': results,
                'summary': {
                    'symbols_processed': len(results),
                    'symbols_requested': len(symbols),
                    'success_rate': len(results) / len(symbols) if symbols else 0
                }
            }
            
        except Exception as e:
            logger.log_crew_activity(
                'DataIngestion',
                f"Market data loading failed: {str(e)}",
                'ERROR'
            )
            return {'status': 'failed', 'error': str(e)}
    
    def _validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and completeness."""
        validation = {
            'is_valid': True,
            'issues': [],
            'data_quality_score': 100
        }
        
        # Check for missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_pct > 0.05:  # More than 5% missing
            validation['issues'].append(f"High missing data: {missing_pct:.2%}")
            validation['data_quality_score'] -= 20
        
        # Check for zero volume days
        zero_volume_pct = (data['Volume'] == 0).sum() / len(data)
        if zero_volume_pct > 0.1:  # More than 10% zero volume
            validation['issues'].append(f"High zero volume days: {zero_volume_pct:.2%}")
            validation['data_quality_score'] -= 15
        
        # Check for price anomalies
        price_changes = data['Close'].pct_change().abs()
        extreme_changes = (price_changes > 0.2).sum()  # >20% daily change
        if extreme_changes > len(data) * 0.02:  # More than 2% of days
            validation['issues'].append(f"Unusual price volatility detected")
            validation['data_quality_score'] -= 10
        
        validation['is_valid'] = validation['data_quality_score'] >= 70
        return validation


class NewsSentimentAgent:
    """
    Agent responsible for news scraping and sentiment analysis.
    
    Capabilities:
    - Scrape financial news from multiple sources
    - Perform sentiment analysis using FinBERT
    - Store news embeddings in ChromaDB
    - Track sentiment trends over time
    """
    
    def __init__(self):
        self.scraper = FirecrawlScraper()
        self.sentiment_analyzer = FinBERTWrapper()
        self.chroma_tool = ChromaDBTool()
        self.agent = Agent(
            role="Financial News Analyst",
            goal="Gather and analyze financial news sentiment for accurate market insights",
            backstory="""You are an expert financial journalist and sentiment analyst 
            who specializes in interpreting market news and public sentiment. You have 
            deep understanding of how news impacts stock performance.""",
            verbose=True,
            allow_delegation=False
        )
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute news sentiment analysis task.
        
        Args:
            task: Dictionary containing symbols and analysis parameters
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            symbols = task.get('symbols', [])
            days_back = task.get('days_back', 30)
            session_id = task.get('session_id')
            
            logger.log_crew_activity(
                'DataIngestion',
                f"Starting news sentiment analysis for {len(symbols)} symbols",
                'INFO'
            )
            
            results = {}
            
            for symbol in symbols:
                try:
                    # Scrape news articles
                    articles = self.scraper.scrape_stock_news(symbol, days_back)
                    
                    if articles:
                        # Analyze sentiment for each article
                        sentiment_scores = []
                        processed_articles = []
                        
                        for article in articles:
                            sentiment_score = self.sentiment_analyzer.analyze_sentiment(
                                article.get('content', '')
                            )
                            
                            article_data = {
                                **article,
                                'sentiment_score': sentiment_score,
                                'symbol': symbol
                            }
                            
                            sentiment_scores.append(sentiment_score)
                            processed_articles.append(article_data)
                        
                        # Store embeddings in ChromaDB
                        self.chroma_tool.store_embeddings(processed_articles, symbol)
                        
                        # Calculate aggregate sentiment metrics
                        sentiment_summary = self._calculate_sentiment_summary(sentiment_scores)
                        
                        results[symbol] = {
                            'articles': processed_articles,
                            'sentiment_summary': sentiment_summary,
                            'article_count': len(articles),
                            'date_range': {
                                'start': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                                'end': datetime.now().strftime('%Y-%m-%d')
                            }
                        }
                        
                        logger.log_crew_activity(
                            'DataIngestion',
                            f"Processed {len(articles)} articles for {symbol}, avg sentiment: {sentiment_summary['mean']:.3f}",
                            'INFO'
                        )
                    else:
                        logger.log_crew_activity(
                            'DataIngestion',
                            f"No news articles found for {symbol}",
                            'WARNING'
                        )
                        
                except Exception as e:
                    logger.log_crew_activity(
                        'DataIngestion',
                        f"Error processing news for {symbol}: {str(e)}",
                        'ERROR'
                    )
            
            return {
                'status': 'completed',
                'data': results,
                'summary': {
                    'symbols_processed': len(results),
                    'total_articles': sum(r.get('article_count', 0) for r in results.values()),
                    'avg_sentiment': self._calculate_overall_sentiment(results)
                }
            }
            
        except Exception as e:
            logger.log_crew_activity(
                'DataIngestion',
                f"News sentiment analysis failed: {str(e)}",
                'ERROR'
            )
            return {'status': 'failed', 'error': str(e)}
    
    def _calculate_sentiment_summary(self, scores: List[float]) -> Dict[str, float]:
        """Calculate sentiment summary statistics."""
        if not scores:
            return {'mean': 0, 'std': 0, 'positive_ratio': 0, 'negative_ratio': 0}
        
        import numpy as np
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'positive_ratio': sum(1 for s in scores if s > 0.1) / len(scores),
            'negative_ratio': sum(1 for s in scores if s < -0.1) / len(scores)
        }
    
    def _calculate_overall_sentiment(self, results: Dict) -> float:
        """Calculate overall sentiment across all symbols."""
        all_scores = []
        for symbol_data in results.values():
            if 'sentiment_summary' in symbol_data:
                all_scores.append(symbol_data['sentiment_summary']['mean'])
        
        return sum(all_scores) / len(all_scores) if all_scores else 0


class DataPreprocessingAgent:
    """
    Agent responsible for data cleaning and preprocessing.
    
    Capabilities:
    - Clean and normalize datasets
    - Handle missing values and outliers
    - Feature engineering for ML models
    - Data validation and quality checks
    """
    
    def __init__(self):
        self.agent = Agent(
            role="Data Engineer",
            goal="Ensure data quality and prepare datasets for analysis",
            backstory="""You are a meticulous data engineer with expertise in 
            financial data preprocessing. You ensure data consistency, handle 
            missing values, and prepare clean datasets for accurate analysis.""",
            verbose=True,
            allow_delegation=False
        )
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute data preprocessing task.
        
        Args:
            task: Dictionary containing raw data and preprocessing parameters
            
        Returns:
            Dictionary with cleaned and processed data
        """
        try:
            raw_data = task.get('raw_data', {})
            session_id = task.get('session_id')
            
            logger.log_crew_activity(
                'DataIngestion',
                f"Starting data preprocessing for {len(raw_data)} symbols",
                'INFO'
            )
            
            processed_data = {}
            
            for symbol, symbol_data in raw_data.items():
                try:
                    # Process stock data
                    if 'stock_data' in symbol_data:
                        cleaned_stock_data = self._clean_stock_data(symbol_data['stock_data'])
                        
                        # Engineer features
                        feature_data = self._engineer_features(cleaned_stock_data)
                        
                        processed_data[symbol] = {
                            'cleaned_stock_data': cleaned_stock_data,
                            'features': feature_data,
                            'technical_indicators': symbol_data.get('technical_indicators'),
                            'processing_summary': self._get_processing_summary(
                                symbol_data['stock_data'], 
                                cleaned_stock_data
                            )
                        }
                        
                        logger.log_crew_activity(
                            'DataIngestion',
                            f"Successfully preprocessed data for {symbol}",
                            'INFO'
                        )
                        
                except Exception as e:
                    logger.log_crew_activity(
                        'DataIngestion',
                        f"Error preprocessing data for {symbol}: {str(e)}",
                        'ERROR'
                    )
            
            return {
                'status': 'completed',
                'data': processed_data,
                'summary': {
                    'symbols_processed': len(processed_data),
                    'total_features_created': sum(
                        len(d.get('features', {}).columns) 
                        for d in processed_data.values() 
                        if 'features' in d and hasattr(d['features'], 'columns')
                    )
                }
            }
            
        except Exception as e:
            logger.log_crew_activity(
                'DataIngestion',
                f"Data preprocessing failed: {str(e)}",
                'ERROR'
            )
            return {'status': 'failed', 'error': str(e)}
    
    def _clean_stock_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean stock data by handling missing values and outliers."""
        cleaned_data = data.copy()
        
        # Forward fill missing values
        cleaned_data = cleaned_data.fillna(method='ffill')
        
        # Remove extreme outliers (beyond 5 standard deviations)
        for column in ['Open', 'High', 'Low', 'Close']:
            if column in cleaned_data.columns:
                mean_val = cleaned_data[column].mean()
                std_val = cleaned_data[column].std()
                
                # Cap extreme values
                upper_bound = mean_val + 5 * std_val
                lower_bound = mean_val - 5 * std_val
                
                cleaned_data[column] = cleaned_data[column].clip(lower_bound, upper_bound)
        
        # Ensure Volume is non-negative
        if 'Volume' in cleaned_data.columns:
            cleaned_data['Volume'] = cleaned_data['Volume'].clip(lower=0)
        
        return cleaned_data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for machine learning models."""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        features['volatility_5d'] = features['returns'].rolling(5).std()
        features['volatility_20d'] = features['returns'].rolling(20).std()
        
        # Moving averages
        features['sma_5'] = data['Close'].rolling(5).mean()
        features['sma_20'] = data['Close'].rolling(20).mean()
        features['sma_50'] = data['Close'].rolling(50).mean()
        
        # Price position features
        features['price_position_20d'] = (data['Close'] - data['Close'].rolling(20).min()) / (
            data['Close'].rolling(20).max() - data['Close'].rolling(20).min()
        )
        
        # Volume features
        if 'Volume' in data.columns:
            features['volume_sma_20'] = data['Volume'].rolling(20).mean()
            features['volume_ratio'] = data['Volume'] / features['volume_sma_20']
        
        # Trend features
        features['trend_5d'] = (data['Close'] / data['Close'].shift(5) - 1)
        features['trend_20d'] = (data['Close'] / data['Close'].shift(20) - 1)
        
        return features.fillna(method='ffill').fillna(0)
    
    def _get_processing_summary(self, original: pd.DataFrame, processed: pd.DataFrame) -> Dict[str, Any]:
        """Generate processing summary statistics."""
        return {
            'original_rows': len(original),
            'processed_rows': len(processed),
            'rows_removed': len(original) - len(processed),
            'missing_values_filled': original.isnull().sum().sum(),
            'processing_date': datetime.now().isoformat()
        }