"""
Technical Analysis Tools for Stock Performance Monitoring
Provides RSI, MACD, and volatility calculation capabilities
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RSIResult:
    """RSI calculation result container"""
    values: pd.Series
    current_value: float
    signal: str  # 'oversold', 'overbought', 'neutral'


@dataclass
class MACDResult:
    """MACD calculation result container"""
    macd_line: pd.Series
    signal_line: pd.Series
    histogram: pd.Series
    current_signal: str  # 'bullish', 'bearish', 'neutral'


@dataclass
class VolatilityResult:
    """Volatility analysis result container"""
    rolling_volatility: pd.Series
    annualized_volatility: float
    volatility_percentile: float
    risk_level: str  # 'low', 'medium', 'high'


class RSI_MACD_Tool:
    """
    Technical analysis tool for RSI and MACD indicators
    Provides comprehensive momentum analysis capabilities
    """
    
    def __init__(self, rsi_period: int = 14, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
    
    def calculate_rsi(self, prices: pd.Series, period: Optional[int] = None) -> RSIResult:
        """
        Calculate Relative Strength Index
        
        Args:
            prices: Price series (typically closing prices)
            period: RSI calculation period (default: class init value)
            
        Returns:
            RSIResult object with values and signals
        """
        if period is None:
            period = self.rsi_period
            
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses using exponential moving average
        avg_gains = gains.ewm(span=period).mean()
        avg_losses = losses.ewm(span=period).mean()
        
        # Calculate RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Determine current signal
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50
        if current_rsi > 70:
            signal = 'overbought'
        elif current_rsi < 30:
            signal = 'oversold'
        else:
            signal = 'neutral'
        
        return RSIResult(
            values=rsi,
            current_value=current_rsi,
            signal=signal
        )
    
    def calculate_macd(self, prices: pd.Series) -> MACDResult:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Price series (typically closing prices)
            
        Returns:
            MACDResult object with MACD components and signals
        """
        # Calculate exponential moving averages
        ema_fast = prices.ewm(span=self.macd_fast).mean()
        ema_slow = prices.ewm(span=self.macd_slow).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=self.macd_signal).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Determine current signal
        current_histogram = histogram.iloc[-1] if not histogram.empty else 0
        prev_histogram = histogram.iloc[-2] if len(histogram) > 1 else 0
        
        if current_histogram > 0 and prev_histogram <= 0:
            current_signal = 'bullish'
        elif current_histogram < 0 and prev_histogram >= 0:
            current_signal = 'bearish'
        else:
            current_signal = 'neutral'
        
        return MACDResult(
            macd_line=macd_line,
            signal_line=signal_line,
            histogram=histogram,
            current_signal=current_signal
        )
    
    def get_momentum_signals(self, prices: pd.Series) -> Dict[str, str]:
        """
        Get combined momentum signals from RSI and MACD
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary with combined signals
        """
        rsi_result = self.calculate_rsi(prices)
        macd_result = self.calculate_macd(prices)
        
        # Combine signals for overall momentum assessment
        if rsi_result.signal == 'overbought' and macd_result.current_signal == 'bearish':
            overall_signal = 'strong_sell'
        elif rsi_result.signal == 'oversold' and macd_result.current_signal == 'bullish':
            overall_signal = 'strong_buy'
        elif rsi_result.signal == 'overbought' or macd_result.current_signal == 'bearish':
            overall_signal = 'sell'
        elif rsi_result.signal == 'oversold' or macd_result.current_signal == 'bullish':
            overall_signal = 'buy'
        else:
            overall_signal = 'hold'
        
        return {
            'rsi_signal': rsi_result.signal,
            'macd_signal': macd_result.current_signal,
            'overall_signal': overall_signal,
            'rsi_value': rsi_result.current_value,
            'macd_histogram': macd_result.histogram.iloc[-1] if not macd_result.histogram.empty else 0
        }


class VolatilityScanner:
    """
    Volatility analysis tool for risk assessment
    Calculates various volatility metrics and risk levels
    """
    
    def __init__(self, window: int = 20, trading_days: int = 252):
        self.window = window
        self.trading_days = trading_days
    
    def calculate_volatility(self, prices: pd.Series, window: Optional[int] = None) -> VolatilityResult:
        """
        Calculate comprehensive volatility metrics
        
        Args:
            prices: Price series
            window: Rolling window for volatility calculation
            
        Returns:
            VolatilityResult object with volatility metrics
        """
        if window is None:
            window = self.window
        
        # Calculate daily returns
        returns = prices.pct_change().dropna()
        
        # Calculate rolling volatility (standard deviation of returns)
        rolling_vol = returns.rolling(window=window).std()
        
        # Annualize volatility
        current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else 0
        annualized_vol = current_vol * np.sqrt(self.trading_days)
        
        # Calculate volatility percentile (current vs historical)
        vol_percentile = (rolling_vol <= current_vol).sum() / len(rolling_vol) * 100
        
        # Determine risk level
        if annualized_vol < 0.15:  # Less than 15% annualized
            risk_level = 'low'
        elif annualized_vol < 0.30:  # Less than 30% annualized
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return VolatilityResult(
            rolling_volatility=rolling_vol,
            annualized_volatility=annualized_vol,
            volatility_percentile=vol_percentile,
            risk_level=risk_level
        )
    
    def calculate_beta(self, stock_prices: pd.Series, market_prices: pd.Series) -> float:
        """
        Calculate stock beta relative to market
        
        Args:
            stock_prices: Individual stock price series
            market_prices: Market index price series
            
        Returns:
            Beta coefficient
        """
        # Calculate returns
        stock_returns = stock_prices.pct_change().dropna()
        market_returns = market_prices.pct_change().dropna()
        
        # Align data
        aligned_data = pd.concat([stock_returns, market_returns], axis=1, join='inner')
        aligned_data.columns = ['stock', 'market']
        
        if len(aligned_data) < 2:
            return 1.0  # Default beta
        
        # Calculate beta using covariance and variance
        covariance = aligned_data['stock'].cov(aligned_data['market'])
        market_variance = aligned_data['market'].var()
        
        beta = covariance / market_variance if market_variance != 0 else 1.0
        return beta
    
    def get_risk_metrics(self, prices: pd.Series, market_prices: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Get comprehensive risk metrics
        
        Args:
            prices: Stock price series
            market_prices: Optional market index for beta calculation
            
        Returns:
            Dictionary with risk metrics
        """
        vol_result = self.calculate_volatility(prices)
        returns = prices.pct_change().dropna()
        
        # Calculate Value at Risk (VaR) at 95% confidence
        var_95 = returns.quantile(0.05)
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        metrics = {
            'annualized_volatility': vol_result.annualized_volatility,
            'volatility_percentile': vol_result.volatility_percentile,
            'value_at_risk_95': var_95,
            'max_drawdown': max_drawdown,
            'risk_level': vol_result.risk_level
        }
        
        # Add beta if market data provided
        if market_prices is not None:
            metrics['beta'] = self.calculate_beta(prices, market_prices)
        
        return metrics


class BollingerBandsTool:
    """
    Bollinger Bands technical indicator tool
    Provides price envelope analysis for volatility-based trading signals
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
    
    def calculate_bollinger_bands(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        # Calculate middle band (simple moving average)
        middle_band = prices.rolling(window=self.period).mean()
        
        # Calculate standard deviation
        std = prices.rolling(window=self.period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * self.std_dev)
        lower_band = middle_band - (std * self.std_dev)
        
        return {
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band,
            'bandwidth': (upper_band - lower_band) / middle_band * 100
        }
    
    def get_bollinger_signals(self, prices: pd.Series) -> Dict[str, str]:
        """
        Generate trading signals based on Bollinger Bands
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary with band-based signals
        """
        bands = self.calculate_bollinger_bands(prices)
        current_price = prices.iloc[-1]
        
        upper_band_current = bands['upper_band'].iloc[-1]
        lower_band_current = bands['lower_band'].iloc[-1]
        bandwidth_current = bands['bandwidth'].iloc[-1]
        
        # Generate signals based on price position relative to bands
        if current_price >= upper_band_current:
            position_signal = 'overbought'
        elif current_price <= lower_band_current:
            position_signal = 'oversold'
        else:
            position_signal = 'neutral'
        
        # Determine volatility condition
        avg_bandwidth = bands['bandwidth'].rolling(50).mean().iloc[-1]
        if bandwidth_current < avg_bandwidth * 0.8:
            volatility_signal = 'squeeze'  # Low volatility, potential breakout
        elif bandwidth_current > avg_bandwidth * 1.2:
            volatility_signal = 'expansion'  # High volatility
        else:
            volatility_signal = 'normal'
        
        return {
            'position_signal': position_signal,
            'volatility_signal': volatility_signal,
            'bandwidth_percentile': bandwidth_current,
            'price_position': (current_price - lower_band_current) / (upper_band_current - lower_band_current) * 100
        }


def calculate_all_indicators(prices: pd.Series, market_prices: Optional[pd.Series] = None) -> Dict:
    """
    Calculate all technical indicators for a given stock
    
    Args:
        prices: Stock price series
        market_prices: Optional market index for beta calculation
        
    Returns:
        Comprehensive dictionary with all technical indicators
    """
    rsi_macd_tool = RSI_MACD_Tool()
    volatility_scanner = VolatilityScanner()
    bollinger_tool = BollingerBandsTool()
    
    # Calculate all indicators
    rsi_result = rsi_macd_tool.calculate_rsi(prices)
    macd_result = rsi_macd_tool.calculate_macd(prices)
    momentum_signals = rsi_macd_tool.get_momentum_signals(prices)
    volatility_result = volatility_scanner.calculate_volatility(prices)
    risk_metrics = volatility_scanner.get_risk_metrics(prices, market_prices)
    bollinger_bands = bollinger_tool.calculate_bollinger_bands(prices)
    bollinger_signals = bollinger_tool.get_bollinger_signals(prices)
    
    return {
        'rsi': {
            'values': rsi_result.values,
            'current_value': rsi_result.current_value,
            'signal': rsi_result.signal
        },
        'macd': {
            'macd_line': macd_result.macd_line,
            'signal_line': macd_result.signal_line,
            'histogram': macd_result.histogram,
            'current_signal': macd_result.current_signal
        },
        'momentum_signals': momentum_signals,
        'volatility': {
            'rolling_volatility': volatility_result.rolling_volatility,
            'annualized_volatility': volatility_result.annualized_volatility,
            'risk_level': volatility_result.risk_level
        },
        'risk_metrics': risk_metrics,
        'bollinger_bands': bollinger_bands,
        'bollinger_signals': bollinger_signals
    }