# ===== models/database_models.py =====
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.config.base import Base

class StockData(Base):
    """Historical stock price data"""
    __tablename__ = "stock_data"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    date = Column(DateTime, index=True, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=func.now())

class NewsArticle(Base):
    """News articles with sentiment analysis"""
    __tablename__ = "news_articles"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    url = Column(String(500), unique=True, nullable=False)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    published_date = Column(DateTime, nullable=False)
    sentiment_score = Column(Float, nullable=True)
    sentiment_label = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=func.now())

class TechnicalIndicator(Base):
    """Technical indicators data"""
    __tablename__ = "technical_indicators"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    date = Column(DateTime, index=True, nullable=False)
    rsi = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    macd_histogram = Column(Float, nullable=True)
    bb_upper = Column(Float, nullable=True)
    bb_middle = Column(Float, nullable=True)
    bb_lower = Column(Float, nullable=True)
    volume_sma = Column(Float, nullable=True)
    created_at = Column(DateTime, default=func.now())

class Prediction(Base):
    """Stock price predictions"""
    __tablename__ = "predictions"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    prediction_date = Column(DateTime, nullable=False)
    target_date = Column(DateTime, nullable=False)
    predicted_price = Column(Float, nullable=False)
    confidence_lower = Column(Float, nullable=True)
    confidence_upper = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=False)
    accuracy_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=func.now())

class AnalysisSession(Base):
    """Analysis session tracking"""
    __tablename__ = "analysis_sessions"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), unique=True, index=True, nullable=False)
    symbols = Column(JSON, nullable=False)
    status = Column(String(20), default="started")
    progress = Column(Integer, default=0)
    parameters = Column(JSON, nullable=True)
    results = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)

class SessionLog(Base):
    """Session activity logs"""
    __tablename__ = "session_logs"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), ForeignKey("analysis_sessions.session_id"), nullable=False)
    crew_name = Column(String(50), nullable=False)
    agent_name = Column(String(50), nullable=True)
    task_name = Column(String(100), nullable=True)
    message = Column(Text, nullable=False)
    level = Column(String(10), default="INFO")
    timestamp = Column(DateTime, default=func.now())
    
    # Relationship
    session = relationship("AnalysisSession", backref="logs")
