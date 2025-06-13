"""
McKinsey Stock Performance Monitor - Health Agents

This module contains agents responsible for stock health analysis and diagnostic assessment.
Agents handle technical indicator analysis, health scoring, and sentiment trend tracking.
"""

from crewai import Agent
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from ..tools.technical_tools import RSI_MACD_Tool, VolatilityScanner
from ..utils.logger import AnalysisLogger

logger = AnalysisLogger()


class IndicatorAnalysisAgent:
    """
    Agent responsible for technical indicator analysis and interpretation.
    
    Capabilities:
    - Calculate RSI, MACD, Bollinger Bands
    - Interpret technical signals
    - Identify buy/sell signals
    - Generate technical analysis insights
    """
    
    def __init__(self):
        self.rsi_macd_tool = RSI_MACD_Tool()
        self.volatility_scanner = VolatilityScanner()
        self.agent = Agent(
            role="Technical Analysis Specialist",
            goal="Analyze technical indicators to assess stock momentum and trend strength",
            backstory="""You are a seasoned technical analyst with deep expertise in 
            chart patterns and technical indicators. You excel at interpreting RSI, MACD, 
            and Bollinger Bands to identify trading opportunities and market trends.""",
            verbose=True,
            allow_delegation=False
        )
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute technical indicator analysis task.
        
        Args:
            task: Dictionary containing processed data and analysis parameters
            
        Returns:
            Dictionary with technical analysis results and interpretations
        """
        try:
            processed_data = task.get('processed_data', {})
            session_id = task.get('session_id')
            
            logger.log_crew_activity(
                'HealthAnalytics',
                f"Starting technical indicator analysis for {len(processed_data)} symbols",
                'INFO'
            )
            
            indicator_results = {}
            
            for symbol, symbol_data in processed_data.items():
                try:
                    if 'cleaned_stock_data' not in symbol_data:
                        continue
                    
                    stock_data = symbol_data['cleaned_stock_data']
                    
                    # Calculate technical indicators
                    indicators = self._calculate_all_indicators(stock_data)
                    
                    # Interpret indicators
                    interpretations = self._interpret_indicators(indicators, stock_data)
                    
                    # Generate trading signals
                    signals = self._generate_trading_signals(indicators, interpretations)
                    
                    # Calculate indicator strength scores
                    strength_scores = self._calculate_indicator_strengths(indicators)
                    
                    indicator_results[symbol] = {
                        'indicators': indicators,
                        'interpretations': interpretations,
                        'signals': signals,
                        'strength_scores': strength_scores,
                        'analysis_date': datetime.now().isoformat(),
                        'data_points_analyzed': len(stock_data)
                    }
                    
                    logger.log_crew_activity(
                        'HealthAnalytics',
                        f"Completed technical analysis for {symbol} - Signal: {signals.get('overall_signal', 'NEUTRAL')}",
                        'INFO'
                    )
                    
                except Exception as e:
                    logger.log_crew_activity(
                        'HealthAnalytics',
                        f"Error analyzing indicators for {symbol}: {str(e)}",
                        'ERROR'
                    )
            
            return {
                'status': 'completed',
                'data': indicator_results,
                'summary': {
                    'symbols_analyzed': len(indicator_results),
                    'bullish_signals': len([r for r in indicator_results.values() 
                                          if r.get('signals', {}).get('overall_signal') == 'BUY']),
                    'bearish_signals': len([r for r in indicator_results.values() 
                                          if r.get('signals', {}).get('overall_signal') == 'SELL']),
                    'neutral_signals': len([r for r in indicator_results.values() 
                                          if r.get('signals', {}).get('overall_signal') == 'NEUTRAL'])
                }
            }
            
        except Exception as e:
            logger.log_crew_activity(
                'HealthAnalytics',
                f"Technical indicator analysis failed: {str(e)}",
                'ERROR'
            )
            return {'status': 'failed', 'error': str(e)}
    
    def _calculate_all_indicators(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators."""
        indicators = {}
        
        try:
            prices = stock_data['Close']
            
            # RSI
            indicators['rsi'] = self.rsi_macd_tool.calculate_rsi(prices, 14)
            indicators['rsi_current'] = float(indicators['rsi'].iloc[-1])
            
            # MACD
            macd_data = self.rsi_macd_tool.calculate_macd(prices)
            indicators['macd'] = macd_data['macd']
            indicators['macd_signal'] = macd_data['signal']
            indicators['macd_histogram'] = macd_data['histogram']
            indicators['macd_current'] = float(macd_data['macd'].iloc[-1])
            indicators['macd_signal_current'] = float(macd_data['signal'].iloc[-1])
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(prices)
            indicators.update(bb_data)
            
            # Volatility
            volatility_data = self.volatility_scanner.calculate_volatility(prices, 20)
            indicators['volatility'] = volatility_data
            indicators['volatility_percentile'] = self._calculate_volatility_percentile(volatility_data)
            
            # Moving Averages
            indicators['sma_20'] = prices.rolling(20).mean()
            indicators['sma_50'] = prices.rolling(50).mean()
            indicators['ema_12'] = prices.ewm(span=12).mean()
            indicators['ema_26'] = prices.ewm(span=26).mean()
            
            # Current price position
            indicators['price_vs_sma20'] = (prices.iloc[-1] - indicators['sma_20'].iloc[-1]) / indicators['sma_20'].iloc[-1]
            indicators['price_vs_sma50'] = (prices.iloc[-1] - indicators['sma_50'].iloc[-1]) / indicators['sma_50'].iloc[-1]
            
        except Exception as e:
            logger.log_crew_activity(
                'HealthAnalytics',
                f"Error calculating indicators: {str(e)}",
                'ERROR'
            )
        
        return indicators
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, Any]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        # Calculate current position
        current_price = prices.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        bb_position = (current_price - current_lower) / (current_upper - current_lower)
        
        return {
            'bb_upper': upper_band,
            'bb_middle': sma,
            'bb_lower': lower_band,
            'bb_position': float(bb_position),
            'bb_width': float((current_upper - current_lower) / sma.iloc[-1])
        }
    
    def _calculate_volatility_percentile(self, volatility: pd.Series) -> float:
        """Calculate current volatility percentile."""
        if len(volatility) < 50:
            return 0.5
        
        current_vol = volatility.iloc[-1]
        historical_vol = volatility.iloc[-252:]  # Last year
        
        return (historical_vol < current_vol).sum() / len(historical_vol)
    
    def _interpret_indicators(self, indicators: Dict[str, Any], stock_data: pd.DataFrame) -> Dict[str, str]:
        """Interpret technical indicators."""
        interpretations = {}
        
        try:
            # RSI interpretation
            rsi_current = indicators.get('rsi_current', 50)
            if rsi_current > 70:
                interpretations['rsi'] = 'Overbought - potential sell signal'
            elif rsi_current < 30:
                interpretations['rsi'] = 'Oversold - potential buy signal'
            elif rsi_current > 50:
                interpretations['rsi'] = 'Bullish momentum'
            else:
                interpretations['rsi'] = 'Bearish momentum'
            
            # MACD interpretation
            macd_current = indicators.get('macd_current', 0)
            macd_signal_current = indicators.get('macd_signal_current', 0)
            
            if macd_current > macd_signal_current:
                if macd_current > 0:
                    interpretations['macd'] = 'Strong bullish signal'
                else:
                    interpretations['macd'] = 'Weak bullish signal'
            else:
                if macd_current < 0:
                    interpretations['macd'] = 'Strong bearish signal'
                else:
                    interpretations['macd'] = 'Weak bearish signal'
            
            # Bollinger Bands interpretation
            bb_position = indicators.get('bb_position', 0.5)
            if bb_position > 0.8:
                interpretations['bb'] = 'Price near upper band - potential resistance'
            elif bb_position < 0.2:
                interpretations['bb'] = 'Price near lower band - potential support'
            else:
                interpretations['bb'] = 'Price within normal range'
            
            # Moving Average interpretation
            price_vs_sma20 = indicators.get('price_vs_sma20', 0)
            price_vs_sma50 = indicators.get('price_vs_sma50', 0)
            
            if price_vs_sma20 > 0.02 and price_vs_sma50 > 0.02:
                interpretations['trend'] = 'Strong uptrend'
            elif price_vs_sma20 < -0.02 and price_vs_sma50 < -0.02:
                interpretations['trend'] = 'Strong downtrend'
            elif price_vs_sma20 > 0:
                interpretations['trend'] = 'Short-term uptrend'
            else:
                interpretations['trend'] = 'Sideways or weak trend'
            
            # Volatility interpretation
            vol_percentile = indicators.get('volatility_percentile', 0.5)
            if vol_percentile > 0.8:
                interpretations['volatility'] = 'Very high volatility - increased risk'
            elif vol_percentile > 0.6:
                interpretations['volatility'] = 'Above average volatility'
            elif vol_percentile < 0.2:
                interpretations['volatility'] = 'Low volatility - potential breakout ahead'
            else:
                interpretations['volatility'] = 'Normal volatility levels'
            
        except Exception as e:
            interpretations['error'] = f'Interpretation failed: {str(e)}'
        
        return interpretations
    
    def _generate_trading_signals(self, indicators: Dict[str, Any], 
                                interpretations: Dict[str, str]) -> Dict[str, Any]:
        """Generate trading signals based on indicators."""
        signals = {
            'individual_signals': {},
            'signal_strength': 0,
            'overall_signal': 'NEUTRAL'
        }
        
        try:
            signal_score = 0
            signal_count = 0
            
            # RSI signal
            rsi_current = indicators.get('rsi_current', 50)
            if rsi_current < 30:
                signals['individual_signals']['rsi'] = 'BUY'
                signal_score += 2
            elif rsi_current > 70:
                signals['individual_signals']['rsi'] = 'SELL'
                signal_score -= 2
            else:
                signals['individual_signals']['rsi'] = 'NEUTRAL'
            signal_count += 1
            
            # MACD signal
            macd_current = indicators.get('macd_current', 0)
            macd_signal_current = indicators.get('macd_signal_current', 0)
            
            if macd_current > macd_signal_current:
                signals['individual_signals']['macd'] = 'BUY'
                signal_score += 1 if macd_current > 0 else 0.5
            else:
                signals['individual_signals']['macd'] = 'SELL'
                signal_score -= 1 if macd_current < 0 else 0.5
            signal_count += 1
            
            # Bollinger Bands signal
            bb_position = indicators.get('bb_position', 0.5)
            if bb_position < 0.1:
                signals['individual_signals']['bb'] = 'BUY'
                signal_score += 1
            elif bb_position > 0.9:
                signals['individual_signals']['bb'] = 'SELL'
                signal_score -= 1
            else:
                signals['individual_signals']['bb'] = 'NEUTRAL'
            signal_count += 1
            
            # Trend signal
            price_vs_sma20 = indicators.get('price_vs_sma20', 0)
            price_vs_sma50 = indicators.get('price_vs_sma50', 0)
            
            if price_vs_sma20 > 0.02 and price_vs_sma50 > 0.02:
                signals['individual_signals']['trend'] = 'BUY'
                signal_score += 1.5
            elif price_vs_sma20 < -0.02 and price_vs_sma50 < -0.02:
                signals['individual_signals']['trend'] = 'SELL'
                signal_score -= 1.5
            else:
                signals['individual_signals']['trend'] = 'NEUTRAL'
            signal_count += 1
            
            # Calculate overall signal
            if signal_count > 0:
                avg_signal = signal_score / signal_count
                signals['signal_strength'] = abs(avg_signal)
                
                if avg_signal > 0.5:
                    signals['overall_signal'] = 'BUY'
                elif avg_signal < -0.5:
                    signals['overall_signal'] = 'SELL'
                else:
                    signals['overall_signal'] = 'NEUTRAL'
            
            signals['confidence'] = min(1.0, signals['signal_strength'] / 2.0)
            
        except Exception as e:
            signals['error'] = f'Signal generation failed: {str(e)}'
        
        return signals
    
    def _calculate_indicator_strengths(self, indicators: Dict[str, Any]) -> Dict[str, float]:
        """Calculate strength scores for each indicator."""
        strengths = {}
        
        try:
            # RSI strength (distance from neutral 50)
            rsi_current = indicators.get('rsi_current', 50)
            strengths['rsi'] = abs(rsi_current - 50) / 50
            
            # MACD strength (histogram magnitude)
            if 'macd_histogram' in indicators and len(indicators['macd_histogram']) > 0:
                macd_hist = abs(indicators['macd_histogram'].iloc[-1])
                strengths['macd'] = min(1.0, macd_hist / 0.5)  # Normalize to 0-1
            else:
                strengths['macd'] = 0.5
            
            # Trend strength (deviation from moving averages)
            price_vs_sma20 = abs(indicators.get('price_vs_sma20', 0))
            strengths['trend'] = min(1.0, price_vs_sma20 / 0.1)  # 10% deviation = max strength
            
            # Volatility strength
            vol_percentile = indicators.get('volatility_percentile', 0.5)
            strengths['volatility'] = abs(vol_percentile - 0.5) * 2  # Distance from median
            
        except Exception as e:
            logger.log_crew_activity(
                'HealthAnalytics',
                f"Error calculating indicator strengths: {str(e)}",
                'ERROR'
            )
        
        return strengths


class StockHealthAgent:
    """
    Agent responsible for calculating comprehensive stock health scores.
    
    Capabilities:
    - Calculate composite health score (0-100)
    - Assess financial health indicators
    - Evaluate technical health metrics
    - Generate health trend analysis
    """
    
    def __init__(self):
        self.agent = Agent(
            role="Stock Health Diagnostician",
            goal="Assess overall stock health using comprehensive financial and technical metrics",
            backstory="""You are a financial health specialist who evaluates stocks 
            across multiple dimensions. You combine technical indicators, fundamental 
            metrics, and market sentiment to provide holistic health assessments.""",
            verbose=True,
            allow_delegation=False
        )
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute stock health assessment task.
        
        Args:
            task: Dictionary containing indicator results and health parameters
            
        Returns:
            Dictionary with comprehensive health scores and assessments
        """
        try:
            indicator_results = task.get('indicator_results', {})
            processed_data = task.get('processed_data', {})
            session_id = task.get('session_id')
            
            logger.log_crew_activity(
                'HealthAnalytics',
                f"Starting stock health assessment for {len(indicator_results)} symbols",
                'INFO'
            )
            
            health_results = {}
            
            for symbol in indicator_results.keys():
                try:
                    # Get all required data
                    technical_data = indicator_results[symbol]
                    stock_data = processed_data.get(symbol, {}).get('cleaned_stock_data')
                    
                    if stock_data is None:
                        continue
                    
                    # Calculate health components
                    technical_health = self._calculate_technical_health(technical_data)
                    price_health = self._calculate_price_health(stock_data)
                    volume_health = self._calculate_volume_health(stock_data)
                    momentum_health = self._calculate_momentum_health(stock_data, technical_data)
                    
                    # Calculate composite health score
                    composite_score = self._calculate_composite_health_score(
                        technical_health, price_health, volume_health, momentum_health
                    )
                    
                    # Generate health assessment
                    health_assessment = self._generate_health_assessment(
                        composite_score, technical_health, price_health, 
                        volume_health, momentum_health
                    )
                    
                    health_results[symbol] = {
                        'composite_score': composite_score,
                        'component_scores': {
                            'technical': technical_health,
                            'price': price_health,
                            'volume': volume_health,
                            'momentum': momentum_health
                        },
                        'health_assessment': health_assessment,
                        'health_grade': self._get_health_grade(composite_score),
                        'analysis_date': datetime.now().isoformat()
                    }
                    
                    logger.log_crew_activity(
                        'HealthAnalytics',
                        f"Health assessment for {symbol}: {composite_score:.0f}/100 ({self._get_health_grade(composite_score)})",
                        'INFO'
                    )
                    
                except Exception as e:
                    logger.log_crew_activity(
                        'HealthAnalytics',
                        f"Error assessing health for {symbol}: {str(e)}",
                        'ERROR'
                    )
            
            return {
                'status': 'completed',
                'data': health_results,
                'summary': {
                    'symbols_assessed': len(health_results),
                    'average_health_score': np.mean([
                        r['composite_score'] for r in health_results.values()
                    ]) if health_results else 0,
                    'excellent_health': len([r for r in health_results.values() 
                                           if r['composite_score'] >= 80]),
                    'good_health': len([r for r in health_results.values() 
                                      if 60 <= r['composite_score'] < 80]),
                    'fair_health': len([r for r in health_results.values() 
                                      if 40 <= r['composite_score'] < 60]),
                    'poor_health': len([r for r in health_results.values() 
                                      if r['composite_score'] < 40])
                }
            }
            
        except Exception as e:
            logger.log_crew_activity(
                'HealthAnalytics',
                f"Stock health assessment failed: {str(e)}",
                'ERROR'
            )
            return {'status': 'failed', 'error': str(e)}
    
    def _calculate_technical_health(self, technical_data: Dict[str, Any]) -> float:
        """Calculate technical health score (0-100)."""
        try:
            signals = technical_data.get('signals', {})
            strength_scores = technical_data.get('strength_scores', {})
            
            # Base score from overall signal
            base_score = 50  # Neutral
            if signals.get('overall_signal') == 'BUY':
                base_score = 70
            elif signals.get('overall_signal') == 'SELL':
                base_score = 30
            
            # Adjust based on signal confidence
            confidence = signals.get('confidence', 0.5)
            confidence_adjustment = (confidence - 0.5) * 20
            
            # Adjust based on indicator strengths
            avg_strength = np.mean(list(strength_scores.values())) if strength_scores else 0.5
            strength_adjustment = (avg_strength - 0.5) * 20
            
            technical_score = base_score + confidence_adjustment + strength_adjustment
            return max(0, min(100, technical_score))
            
        except Exception as e:
            logger.log_crew_activity(
                'HealthAnalytics',
                f"Error calculating technical health: {str(e)}",
                'ERROR'
            )
            return 50.0
    
    def _calculate_price_health(self, stock_data: pd.DataFrame) -> float:
        """Calculate price action health score (0-100)."""
        try:
            prices = stock_data['Close']
            
            # Price trend over different periods
            returns_1w = (prices.iloc[-1] / prices.iloc[-5] - 1) if len(prices) >= 5 else 0
            returns_1m = (prices.iloc[-1] / prices.iloc[-20] - 1) if len(prices) >= 20 else 0
            returns_3m = (prices.iloc[-1] / prices.iloc[-60] - 1) if len(prices) >= 60 else 0
            
            # Score based on consistent positive returns
            trend_score = 50
            if returns_1w > 0 and returns_1m > 0 and returns_3m > 0:
                trend_score = 80
            elif returns_1w > 0 and returns_1m > 0:
                trend_score = 70
            elif returns_1w > 0:
                trend_score = 60
            elif returns_1w < 0 and returns_1m < 0 and returns_3m < 0:
                trend_score = 20
            elif returns_1w < 0 and returns_1m < 0:
                trend_score = 30
            elif returns_1w < 0:
                trend_score = 40
            
            # Adjust for magnitude of returns
            avg_return = np.mean([returns_1w, returns_1m, returns_3m])
            magnitude_adjustment = min(20, abs(avg_return) * 100)
            
            if avg_return > 0:
                trend_score += magnitude_adjustment
            else:
                trend_score -= magnitude_adjustment
            
            return max(0, min(100, trend_score))
            
        except Exception as e:
            logger.log_crew_activity(
                'HealthAnalytics',
                f"Error calculating price health: {str(e)}",
                'ERROR'
            )
            return 50.0
    
    def _calculate_volume_health(self, stock_data: pd.DataFrame) -> float:
        """Calculate volume health score (0-100)."""
        try:
            volumes = stock_data['Volume']
            prices = stock_data['Close']
            
            # Volume trend analysis
            recent_volume = volumes.iloc[-5:].mean() if len(volumes) >= 5 else volumes.iloc[-1]
            historical_volume = volumes.iloc[-60:-5].mean() if len(volumes) >= 60 else volumes.mean()
            
            volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1
            
            # Price-volume confirmation
            price_change = (prices.iloc[-1] / prices.iloc[-5] - 1) if len(prices) >= 5 else 0
            
            # Base score
            volume_score = 50
            
            # Healthy volume patterns
            if volume_ratio > 1.2 and price_change > 0:  # High volume with price increase
                volume_score = 80
            elif volume_ratio > 1.0 and price_change > 0:  # Normal volume with price increase
                volume_score = 70
            elif volume_ratio < 0.8 and price_change < 0:  # Low volume with price decrease
                volume_score = 60
            elif volume_ratio > 1.2 and price_change < 0:  # High volume with price decrease
                volume_score = 30
            
            return max(0, min(100, volume_score))
            
        except Exception as e:
            logger.log_crew_activity(
                'HealthAnalytics',
                f"Error calculating volume health: {str(e)}",
                'ERROR'
            )
            return 50.0
    
    def _calculate_momentum_health(self, stock_data: pd.DataFrame, technical_data: Dict[str, Any]) -> float:
        """Calculate momentum health score (0-100)."""
        try:
            indicators = technical_data.get('indicators', {})
            
            # RSI momentum
            rsi_current = indicators.get('rsi_current', 50)
            rsi_score = 50
            if 30 < rsi_current < 70:
                rsi_score = 70  # Healthy range
            elif rsi_current < 30:
                rsi_score = 80  # Oversold - potential bounce
            elif rsi_current > 70:
                rsi_score = 30  # Overbought - potential decline
            
            # MACD momentum
            macd_current = indicators.get('macd_current', 0)
            macd_signal = indicators.get('macd_signal_current', 0)
            
            macd_score = 50
            if macd_current > macd_signal and macd_current > 0:
                macd_score = 80
            elif macd_current > macd_signal:
                macd_score = 65
            elif macd_current < macd_signal and macd_current < 0:
                macd_score = 20
            else:
                macd_score = 35
            
            # Price vs moving averages
            price_vs_sma20 = indicators.get('price_vs_sma20', 0)
            trend_score = 50 + (price_vs_sma20 * 200)  # Scale to 0-100
            trend_score = max(0, min(100, trend_score))
            
            # Composite momentum score
            momentum_score = (rsi_score * 0.3 + macd_score * 0.4 + trend_score * 0.3)
            
            return max(0, min(100, momentum_score))
            
        except Exception as e:
            logger.log_crew_activity(
                'HealthAnalytics',
                f"Error calculating momentum health: {str(e)}",
                'ERROR'
            )
            return 50.0
    
    def _calculate_composite_health_score(self, technical: float, price: float, 
                                        volume: float, momentum: float) -> float:
        """Calculate weighted composite health score."""
        # Weights for different components
        weights = {
            'technical': 0.30,
            'price': 0.25,
            'volume': 0.20,
            'momentum': 0.25
        }
        
        composite = (
            technical * weights['technical'] +
            price * weights['price'] +
            volume * weights['volume'] +
            momentum * weights['momentum']
        )
        
        return round(composite, 1)
    
    def _generate_health_assessment(self, composite_score: float, technical: float, 
                                  price: float, volume: float, momentum: float) -> str:
        """Generate narrative health assessment."""
        grade = self._get_health_grade(composite_score)
        
        assessment_parts = [f"Overall health grade: {grade} ({composite_score:.1f}/100)."]
        
        # Component analysis
        if technical >= 70:
            assessment_parts.append("Technical indicators show bullish signals.")
        elif technical <= 30:
            assessment_parts.append("Technical indicators show bearish signals.")
        else:
            assessment_parts.append("Technical indicators are mixed.")
        
        if price >= 70:
            assessment_parts.append("Price action demonstrates strong upward momentum.")
        elif price <= 30:
            assessment_parts.append("Price action shows concerning downward trend.")
        else:
            assessment_parts.append("Price action is sideways or mixed.")
        
        if volume >= 70:
            assessment_parts.append("Volume patterns support current price movement.")
        elif volume <= 30:
            assessment_parts.append("Volume patterns raise concerns about sustainability.")
        else:
            assessment_parts.append("Volume patterns are neutral.")
        
        if momentum >= 70:
            assessment_parts.append("Momentum indicators suggest continued strength.")
        elif momentum <= 30:
            assessment_parts.append("Momentum indicators suggest potential weakness.")
        else:
            assessment_parts.append("Momentum indicators are neutral.")
        
        return " ".join(assessment_parts)
    
    def _get_health_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B+"
        elif score >= 60:
            return "B"
        elif score >= 50:
            return "C+"
        elif score >= 40:
            return "C"
        elif score >= 30:
            return "D"
        else:
            return "F"


class SentimentTrendAgent:
    """
    Agent responsible for analyzing sentiment trends and patterns.
    
    Capabilities:
    - Track sentiment evolution over time
    - Identify sentiment momentum shifts
    - Correlate sentiment with price movements
    - Generate sentiment-based insights
    """
    
    def __init__(self):
        self.agent = Agent(
            role="Sentiment Trend Analyst",
            goal="Analyze sentiment patterns and their correlation with stock performance",
            backstory="""You are a sentiment analysis expert who specializes in 
            tracking market sentiment trends and their impact on stock prices. 
            You excel at identifying sentiment momentum shifts and market psychology.""",
            verbose=True,
            allow_delegation=False
        )
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute sentiment trend analysis task.
        
        Args:
            task: Dictionary containing sentiment data and analysis parameters
            
        Returns:
            Dictionary with sentiment trend analysis results
        """
        try:
            processed_data = task.get('processed_data', {})
            session_id = task.get('session_id')
            
            logger.log_crew_activity(
                'HealthAnalytics',
                f"Starting sentiment trend analysis for {len(processed_data)} symbols",
                'INFO'
            )
            
            sentiment_results = {}
            
            for symbol, symbol_data in processed_data.items():
                try:
                    sentiment_data = symbol_data.get('sentiment_data', [])
                    if not sentiment_data:
                        continue
                    
                    # Analyze sentiment trends
                    trend_analysis = self._analyze_sentiment_trends(sentiment_data)
                    
                    # Calculate sentiment momentum
                    momentum_analysis = self._calculate_sentiment_momentum(sentiment_data)
                    
                    # Correlate with price movements
                    price_correlation = self._correlate_with_prices(
                        sentiment_data, symbol_data.get('cleaned_stock_data')
                    )
                    
                    # Generate insights
                    insights = self._generate_sentiment_insights(
                        trend_analysis, momentum_analysis, price_correlation
                    )
                    
                    sentiment_results[symbol] = {
                        'trend_analysis': trend_analysis,
                        'momentum_analysis': momentum_analysis,
                        'price_correlation': price_correlation,
                        'insights': insights,
                        'analysis_date': datetime.now().isoformat()
                    }
                    
                    logger.log_crew_activity(
                        'HealthAnalytics',
                        f"Sentiment analysis completed for {symbol} - Trend: {trend_analysis.get('overall_trend', 'NEUTRAL')}",
                        'INFO'
                    )
                    
                except Exception as e:
                    logger.log_crew_activity(
                        'HealthAnalytics',
                        f"Error analyzing sentiment for {symbol}: {str(e)}",
                        'ERROR'
                    )
            
            return {
                'status': 'completed',
                'data': sentiment_results,
                'summary': {
                    'symbols_analyzed': len(sentiment_results),
                    'positive_sentiment_trend': len([r for r in sentiment_results.values() 
                                                   if r.get('trend_analysis', {}).get('overall_trend') == 'POSITIVE']),
                    'negative_sentiment_trend': len([r for r in sentiment_results.values() 
                                                   if r.get('trend_analysis', {}).get('overall_trend') == 'NEGATIVE']),
                    'neutral_sentiment_trend': len([r for r in sentiment_results.values() 
                                                  if r.get('trend_analysis', {}).get('overall_trend') == 'NEUTRAL'])
                }
            }
            
        except Exception as e:
            logger.log_crew_activity(
                'HealthAnalytics',
                f"Sentiment trend analysis failed: {str(e)}",
                'ERROR'
            )
            return {'status': 'failed', 'error': str(e)}
    
    def _analyze_sentiment_trends(self, sentiment_data: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment trends over time."""
        if not sentiment_data:
            return {'overall_trend': 'NEUTRAL', 'trend_strength': 0.0}
        
        try:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(sentiment_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Calculate rolling average sentiment
            df['sentiment_ma'] = df['sentiment'].rolling(window=min(7, len(df))).mean()
            
            # Calculate trend
            recent_sentiment = df['sentiment_ma'].iloc[-3:].mean()
            historical_sentiment = df['sentiment_ma'].iloc[:-3].mean() if len(df) > 3 else 0
            
            trend_change = recent_sentiment - historical_sentiment
            
            # Determine overall trend
            if trend_change > 0.1:
                overall_trend = 'POSITIVE'
            elif trend_change < -0.1:
                overall_trend = 'NEGATIVE'
            else:
                overall_trend = 'NEUTRAL'
            
            return {
                'overall_trend': overall_trend,
                'trend_strength': abs(trend_change),
                'current_sentiment': float(recent_sentiment),
                'sentiment_volatility': float(df['sentiment'].std()),
                'data_points': len(df)
            }
            
        except Exception as e:
            logger.log_crew_activity(
                'HealthAnalytics',
                f"Error analyzing sentiment trends: {str(e)}",
                'ERROR'
            )
            return {'overall_trend': 'NEUTRAL', 'trend_strength': 0.0}
    
    def _calculate_sentiment_momentum(self, sentiment_data: List[Dict]) -> Dict[str, Any]:
        """Calculate sentiment momentum indicators."""
        if not sentiment_data:
            return {'momentum': 'NEUTRAL', 'momentum_score': 0.0}
        
        try:
            df = pd.DataFrame(sentiment_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Calculate momentum as rate of change
            if len(df) >= 5:
                recent_avg = df['sentiment'].iloc[-3:].mean()
                previous_avg = df['sentiment'].iloc[-6:-3].mean()
                momentum_score = (recent_avg - previous_avg) / abs(previous_avg) if previous_avg != 0 else 0
            else:
                momentum_score = 0
            
            # Determine momentum direction
            if momentum_score > 0.2:
                momentum = 'ACCELERATING_POSITIVE'
            elif momentum_score > 0.05:
                momentum = 'POSITIVE'
            elif momentum_score < -0.2:
                momentum = 'ACCELERATING_NEGATIVE'
            elif momentum_score < -0.05:
                momentum = 'NEGATIVE'
            else:
                momentum = 'NEUTRAL'
            
            return {
                'momentum': momentum,
                'momentum_score': float(momentum_score),
                'momentum_strength': abs(momentum_score)
            }
            
        except Exception as e:
            return {'momentum': 'NEUTRAL', 'momentum_score': 0.0}
    
    def _correlate_with_prices(self, sentiment_data: List[Dict], stock_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Correlate sentiment with price movements."""
        if not sentiment_data or stock_data is None:
            return {'correlation': 0.0, 'correlation_strength': 'WEAK'}
        
        try:
            # Prepare sentiment data
            sentiment_df = pd.DataFrame(sentiment_data)
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            sentiment_df = sentiment_df.sort_values('date')
            
            # Prepare price data
            price_df = stock_data.copy()
            price_df['date'] = pd.to_datetime(price_df.index)
            price_df['price_change'] = price_df['Close'].pct_change()
            
            # Merge on date
            merged = pd.merge(sentiment_df, price_df, on='date', how='inner')
            
            if len(merged) < 5:
                return {'correlation': 0.0, 'correlation_strength': 'INSUFFICIENT_DATA'}
            
            # Calculate correlation
            correlation = merged['sentiment'].corr(merged['price_change'])
            
            # Determine correlation strength
            if abs(correlation) > 0.7:
                strength = 'STRONG'
            elif abs(correlation) > 0.4:
                strength = 'MODERATE'
            elif abs(correlation) > 0.2:
                strength = 'WEAK'
            else:
                strength = 'VERY_WEAK'
            
            return {
                'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                'correlation_strength': strength,
                'data_points': len(merged)
            }
            
        except Exception as e:
            return {'correlation': 0.0, 'correlation_strength': 'ERROR'}
    
    def _generate_sentiment_insights(self, trend_analysis: Dict, momentum_analysis: Dict, 
                                   price_correlation: Dict) -> List[str]:
        """Generate actionable sentiment insights."""
        insights = []
        
        try:
            # Trend insights
            trend = trend_analysis.get('overall_trend', 'NEUTRAL')
            strength = trend_analysis.get('trend_strength', 0)
            
            if trend == 'POSITIVE' and strength > 0.2:
                insights.append("Strong positive sentiment trend detected - market optimism increasing")
            elif trend == 'NEGATIVE' and strength > 0.2:
                insights.append("Strong negative sentiment trend detected - market pessimism growing")
            
            # Momentum insights
            momentum = momentum_analysis.get('momentum', 'NEUTRAL')
            if momentum == 'ACCELERATING_POSITIVE':
                insights.append("Sentiment momentum is accelerating positively - potential breakout signal")
            elif momentum == 'ACCELERATING_NEGATIVE':
                insights.append("Sentiment momentum is accelerating negatively - caution advised")
            
            # Correlation insights
            correlation = price_correlation.get('correlation', 0)
            corr_strength = price_correlation.get('correlation_strength', 'WEAK')
            
            if corr_strength in ['STRONG', 'MODERATE'] and correlation > 0.4:
                insights.append("Strong positive correlation between sentiment and price - sentiment is a reliable indicator")
            elif corr_strength in ['STRONG', 'MODERATE'] and correlation < -0.4:
                insights.append("Strong negative correlation between sentiment and price - contrarian signal detected")
            
            # Volatility insights
            volatility = trend_analysis.get('sentiment_volatility', 0)
            if volatility > 0.3:
                insights.append("High sentiment volatility detected - expect increased price volatility")
            
            if not insights:
                insights.append("Sentiment patterns are neutral - no strong directional signals")
            
        except Exception as e:
            insights.append(f"Error generating insights: {str(e)}")
        
        return insights