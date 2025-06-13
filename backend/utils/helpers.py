# ===== utils/helpers.py =====
import re
from datetime import datetime, timedelta
from typing import List, Tuple
import pandas as pd

def validate_stock_symbols(symbols: List[str]) -> List[str]:
    """Validate and clean stock symbols"""
    cleaned_symbols = []
    pattern = re.compile(r'^[A-Z]{1,5}$')
    
    for symbol in symbols:
        cleaned = symbol.upper().strip()
        if pattern.match(cleaned):
            cleaned_symbols.append(cleaned)
    
    return cleaned_symbols

def calculate_date_range(period: str) -> Tuple[datetime, datetime]:
    """Calculate date range for data fetching"""
    end_date = datetime.now()
    
    period_map = {
        "1d": timedelta(days=1),
        "5d": timedelta(days=5),
        "1mo": timedelta(days=30),
        "3mo": timedelta(days=90),
        "6mo": timedelta(days=180),
        "1y": timedelta(days=365),
        "2y": timedelta(days=730),
        "5y": timedelta(days=1825),
        "10y": timedelta(days=3650),
        "ytd": timedelta(days=(end_date - datetime(end_date.year, 1, 1)).days),
    }
    
    if period == "max":
        start_date = datetime(1970, 1, 1)
    else:
        start_date = end_date - period_map.get(period, timedelta(days=365))
    
    return start_date, end_date

def calculate_health_score(
    technical_indicators: dict,
    sentiment_score: float,
    volatility: float,
    volume_trend: float
) -> int:
    """Calculate composite health score (0-100)"""
    
    score = 50  # Base score
    
    # Technical indicators contribution (40%)
    if 'rsi' in technical_indicators:
        rsi = technical_indicators['rsi']
        if 30 <= rsi <= 70:
            score += 10
        elif rsi < 30:
            score += 5  # Oversold - potential upside
        else:
            score -= 10  # Overbought
    
    if 'macd' in technical_indicators:
        macd = technical_indicators['macd']
        macd_signal = technical_indicators.get('macd_signal', 0)
        if macd > macd_signal:
            score += 10
        else:
            score -= 5
    
    # Sentiment contribution (30%)
    if sentiment_score > 0.1:
        score += 15
    elif sentiment_score < -0.1:
        score -= 15
    
    # Volatility contribution (20%)
    if volatility < 0.02:  # Low volatility
        score += 10
    elif volatility > 0.05:  # High volatility
        score -= 10
    
    # Volume trend contribution (10%)
    if volume_trend > 1.1:  # Increasing volume
        score += 5
    elif volume_trend < 0.9:  # Decreasing volume
        score -= 5
    
    return max(0, min(100, score))