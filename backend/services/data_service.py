"""
Data Service - Handles data storage and retrieval operations
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from sqlalchemy.orm import Session
from ..config.database import get_database_session
from ..models.database_models import (
    StockData, NewsArticle, TechnicalIndicator, 
    Prediction, AnalysisSession
)
from backend.utils.logger import AnalysisLogger
from backend.utils.exceptions import DataFetchException
from backend.utils.helpers import validate_stock_symbols, calculate_date_range


class DataService:
    """Service for data storage and retrieval operations"""
    
    def __init__(self):
        self.logger = AnalysisLogger(session_id="temp-agent-session")
    
    def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1y"
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve cached historical data from database
        
        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with OHLCV data or None if not found
        """
        try:
            # Validate symbol
            validated_symbols = validate_stock_symbols([symbol])
            if not validated_symbols:
                raise DataFetchException(f"Invalid stock symbol: {symbol}")
            
            symbol = validated_symbols[0]
            
            # Calculate date range
            end_date = datetime.utcnow().date()
            start_date, _ = calculate_date_range(period)
            
            # Query database
            db_session = next(get_database_session())
            stock_data = db_session.query(StockData).filter(
                StockData.symbol == symbol,
                StockData.date >= start_date,
                StockData.date <= end_date
            ).order_by(StockData.date).all()
            
            if not stock_data:
                self.logger.log_crew_activity(
                    "DataService", 
                    f"No cached data found for {symbol}", 
                    "info"
                )
                return None
            
            # Convert to DataFrame
            data = []
            for record in stock_data:
                data.append({
                    'Date': record.date,
                    'Open': record.open_price,
                    'High': record.high_price,
                    'Low': record.low_price,
                    'Close': record.close_price,
                    'Volume': record.volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('Date', inplace=True)
            
            self.logger.log_crew_activity(
                "DataService", 
                f"Retrieved {len(df)} records for {symbol}", 
                "info"
            )
            
            return df
            
        except Exception as e:
            self.logger.log_crew_activity(
                "DataService", 
                f"Failed to get historical data for {symbol}: {str(e)}", 
                "error"
            )
            raise DataFetchException(f"Failed to retrieve historical data: {str(e)}")
    
    def store_stock_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Store stock data in database
        
        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            db_session = next(get_database_session())
            
            # Clear existing data for symbol
            db_session.query(StockData).filter(StockData.symbol == symbol).delete()
            
            # Insert new data
            for date, row in data.iterrows():
                stock_record = StockData(
                    symbol=symbol,
                    date=date.date() if hasattr(date, 'date') else date,
                    open_price=float(row['Open']),
                    high_price=float(row['High']),
                    low_price=float(row['Low']),
                    close_price=float(row['Close']),
                    volume=int(row['Volume']) if pd.notna(row['Volume']) else 0
                )
                db_session.add(stock_record)
            
            db_session.commit()
            
            self.logger.log_crew_activity(
                "DataService", 
                f"Stored {len(data)} records for {symbol}", 
                "info"
            )
            
            return True
            
        except Exception as e:
            db_session.rollback()
            self.logger.log_crew_activity(
                "DataService", 
                f"Failed to store stock data for {symbol}: {str(e)}", 
                "error"
            )
            return False
    
    def store_technical_indicators(
        self, 
        symbol: str, 
        indicators: pd.DataFrame
    ) -> bool:
        """
        Store technical indicators in database
        
        Args:
            symbol: Stock symbol
            indicators: DataFrame with technical indicators
            
        Returns:
            True if successful, False otherwise
        """
        try:
            db_session = next(get_database_session())
            
            # Clear existing indicators for symbol
            db_session.query(TechnicalIndicator).filter(
                TechnicalIndicator.symbol == symbol
            ).delete()
            
            # Insert new indicators
            for date, row in indicators.iterrows():
                indicator_record = TechnicalIndicator(
                    symbol=symbol,
                    date=date.date() if hasattr(date, 'date') else date,
                    rsi=float(row.get('RSI', 0)) if pd.notna(row.get('RSI')) else None,
                    macd=float(row.get('MACD', 0)) if pd.notna(row.get('MACD')) else None,
                    macd_signal=float(row.get('MACD_Signal', 0)) if pd.notna(row.get('MACD_Signal')) else None,
                    macd_histogram=float(row.get('MACD_Histogram', 0)) if pd.notna(row.get('MACD_Histogram')) else None,
                    bb_upper=float(row.get('BB_Upper', 0)) if pd.notna(row.get('BB_Upper')) else None,
                    bb_middle=float(row.get('BB_Middle', 0)) if pd.notna(row.get('BB_Middle')) else None,
                    bb_lower=float(row.get('BB_Lower', 0)) if pd.notna(row.get('BB_Lower')) else None
                )
                db_session.add(indicator_record)
            
            db_session.commit()
            
            self.logger.log_crew_activity(
                "DataService", 
                f"Stored technical indicators for {symbol}", 
                "info"
            )
            
            return True
            
        except Exception as e:
            db_session.rollback()
            self.logger.log_crew_activity(
                "DataService", 
                f"Failed to store technical indicators for {symbol}: {str(e)}", 
                "error"
            )
            return False
    
    def store_predictions(
        self, 
        symbol: str, 
        predictions: pd.DataFrame
    ) -> bool:
        """
        Store price predictions in database
        
        Args:
            symbol: Stock symbol
            predictions: DataFrame with predictions and confidence intervals
            
        Returns:
            True if successful, False otherwise
        """
        try:
            db_session = next(get_database_session())
            
            # Clear existing predictions for symbol
            db_session.query(Prediction).filter(
                Prediction.symbol == symbol
            ).delete()
            
            # Insert new predictions
            for date, row in predictions.iterrows():
                prediction_record = Prediction(
                    symbol=symbol,
                    prediction_date=date.date() if hasattr(date, 'date') else date,
                    predicted_price=float(row.get('yhat', 0)),
                    lower_bound=float(row.get('yhat_lower', 0)) if 'yhat_lower' in row else None,
                    upper_bound=float(row.get('yhat_upper', 0)) if 'yhat_upper' in row else None,
                    confidence_interval=float(row.get('confidence', 0.95)) if 'confidence' in row else 0.95,
                    created_at=datetime.utcnow()
                )
                db_session.add(prediction_record)
            
            db_session.commit()
            
            self.logger.log_crew_activity(
                "DataService", 
                f"Stored {len(predictions)} predictions for {symbol}", 
                "info"
            )
            
            return True
            
        except Exception as e:
            db_session.rollback()
            self.logger.log_crew_activity(
                "DataService", 
                f"Failed to store predictions for {symbol}: {str(e)}", 
                "error"
            )
            return False
    
    def store_news_articles(self, symbol: str, articles: List[Dict[str, Any]]) -> bool:
        """
        Store news articles with sentiment scores
        
        Args:
            symbol: Stock symbol
            articles: List of article dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            db_session = next(get_database_session())
            
            for article in articles:
                news_record = NewsArticle(
                    symbol=symbol,
                    url=article.get('url', ''),
                    title=article.get('title', ''),
                    content=article.get('content', ''),
                    sentiment_score=float(article.get('sentiment_score', 0.0)),
                    published_date=article.get('published_date', datetime.utcnow()),
                    scraped_at=datetime.utcnow()
                )
                db_session.add(news_record)
            
            db_session.commit()
            
            self.logger.log_crew_activity(
                "DataService", 
                f"Stored {len(articles)} news articles for {symbol}", 
                "info"
            )
            
            return True
            
        except Exception as e:
            db_session.rollback()
            self.logger.log_crew_activity(
                "DataService", 
                f"Failed to store news articles for {symbol}: {str(e)}", 
                "error"
            )
            return False
    
    def store_analysis_results(self, session_id: str, results: Dict[str, Any]) -> bool:
        """
        Persist analysis results to database
        
        Args:
            session_id: Analysis session ID
            results: Complete analysis results dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            db_session = next(get_database_session())
            
            # Update analysis session with results
            analysis_session = db_session.query(AnalysisSession).filter(
                AnalysisSession.session_id == session_id
            ).first()
            
            if analysis_session:
                analysis_session.results = json.dumps(results)
                analysis_session.completed_at = datetime.utcnow()
                analysis_session.updated_at = datetime.utcnow()
                db_session.commit()
                
                self.logger.log_crew_activity(
                    "DataService", 
                    f"Stored analysis results for session {session_id}", 
                    "info"
                )
                
                return True
            else:
                self.logger.log_crew_activity(
                    "DataService", 
                    f"Analysis session {session_id} not found", 
                    "error"
                )
                return False
            
        except Exception as e:
            db_session.rollback()
            self.logger.log_crew_activity(
                "DataService", 
                f"Failed to store analysis results: {str(e)}", 
                "error"
            )
            return False
    
    def get_cached_news(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get cached news articles for symbol
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            List of news articles with sentiment scores
        """
        try:
            db_session = next(get_database_session())
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            news_articles = db_session.query(NewsArticle).filter(
                NewsArticle.symbol == symbol,
                NewsArticle.published_date >= cutoff_date
            ).order_by(NewsArticle.published_date.desc()).all()
            
            articles = []
            for article in news_articles:
                articles.append({
                    'url': article.url,
                    'title': article.title,
                    'content': article.content,
                    'sentiment_score': article.sentiment_score,
                    'published_date': article.published_date
                })
            
            return articles
            
        except Exception as e:
            self.logger.log_crew_activity(
                "DataService", 
                f"Failed to get cached news for {symbol}: {str(e)}", 
                "error"
            )
            return []
    
    def cleanup_old_data(self, days: int = 30) -> bool:
        """
        Clean up old data from database
        
        Args:
            days: Number of days to keep
            
        Returns:
            True if successful, False otherwise
        """
        try:
            db_session = next(get_database_session())
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Clean old analysis sessions
            deleted_sessions = db_session.query(AnalysisSession).filter(
                AnalysisSession.created_at < cutoff_date
            ).delete()
            
            # Clean old news articles
            deleted_news = db_session.query(NewsArticle).filter(
                NewsArticle.scraped_at < cutoff_date
            ).delete()
            
            db_session.commit()
            
            self.logger.log_crew_activity(
                "DataService", 
                f"Cleaned up {deleted_sessions} sessions and {deleted_news} news articles", 
                "info"
            )
            
            return True
            
        except Exception as e:
            db_session.rollback()
            self.logger.log_crew_activity(
                "DataService", 
                f"Failed to cleanup old data: {str(e)}", 
                "error"
            )
            return False