# tools/market_data_tools.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class YahooFinanceTool:
    """Tool for fetching stock data from Yahoo Finance"""
    
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
    
    def fetch_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch historical stock data"""
        try:
            cache_key = f"{symbol}_{period}"
            
            # Check cache
            if self._is_cache_valid(cache_key):
                logger.info(f"Using cached data for {symbol}")
                return self.cache[cache_key]
            
            logger.info(f"Fetching fresh data for {symbol}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Clean column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            data.reset_index(inplace=True)
            
            # Cache the data
            self.cache[cache_key] = data
            self.cache_expiry[cache_key] = datetime.now() + timedelta(hours=1)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """Get current stock price and basic info"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'open': info.get('open', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {str(e)}")
            return {}
    
    def get_technical_indicators(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            df = data.copy()
            
            # RSI (Relative Strength Index)
            df['rsi'] = self._calculate_rsi(df['close'], period=14)
            
            # MACD
            macd_data = self._calculate_macd(df['close'])
            df['macd'] = macd_data['macd']
            df['macd_signal'] = macd_data['signal']
            df['macd_histogram'] = macd_data['histogram']
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(df['close'])
            df['bb_upper'] = bb_data['upper']
            df['bb_middle'] = bb_data['middle']
            df['bb_lower'] = bb_data['lower']
            
            # Volume SMA
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: int = 2) -> Dict:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache is valid"""
        if cache_key not in self.cache:
            return False
        return datetime.now() < self.cache_expiry.get(cache_key, datetime.min)