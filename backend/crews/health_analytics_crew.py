"""
Health Analytics Crew for McKinsey Stock Performance Monitor
Handles technical analysis, health scoring, and sentiment trend analysis
"""

from crewai import Crew, Task
from agents.health_agents import IndicatorAnalysisAgent, StockHealthAgent, SentimentTrendAgent
from tools.technical_tools import RSI_MACD_Tool, VolatilityScanner
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class HealthAnalyticsCrew:
    """Crew responsible for stock health analysis and diagnostics"""
    
    def __init__(self):
        self.indicator_agent = IndicatorAnalysisAgent()
        self.health_agent = StockHealthAgent()
        self.sentiment_agent = SentimentTrendAgent()
        
        # Initialize tools
        self.rsi_macd_tool = RSI_MACD_Tool()
        self.volatility_scanner = VolatilityScanner()
    
    def create_crew(self) -> Crew:
        """Create and configure the health analytics crew"""
        
        # Define tasks
        analyze_technical_indicators_task = Task(
            description="Calculate and interpret technical indicators for all stocks",
            agent=self.indicator_agent,
            expected_output="Technical analysis results including RSI, MACD, and volatility metrics"
        )
        
        assess_stock_health_task = Task(
            description="Compute comprehensive health scores for each stock",
            agent=self.health_agent,
            expected_output="Health scores (0-100) with breakdown by category",
            dependencies=[analyze_technical_indicators_task]
        )
        
        track_sentiment_trends_task = Task(
            description="Analyze sentiment evolution patterns over time",
            agent=self.sentiment_agent,
            expected_output="Sentiment trend analysis with momentum indicators"
        )
        
        # Create crew
        crew = Crew(
            agents=[
                self.indicator_agent,
                self.health_agent,
                self.sentiment_agent
            ],
            tasks=[
                analyze_technical_indicators_task,
                assess_stock_health_task,
                track_sentiment_trends_task
            ],
            verbose=True,
            process="sequential"
        )
        
        return crew
    
    def execute_health_analysis(self, market_data: Dict[str, Any], 
                              sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete health analysis process"""
        try:
            logger.info(f"Starting health analysis for {len(market_data)} stocks")
            
            # Prepare shared context
            shared_context = {
                'market_data': market_data,
                'sentiment_data': sentiment_data,
                'technical_indicators': {},
                'health_scores': {},
                'sentiment_trends': {}
            }
            
            # Analyze technical indicators
            logger.info("Analyzing technical indicators...")
            indicators_task = {
                'market_data': market_data,
                'context': shared_context
            }
            indicators_results = self.indicator_agent.execute_task(indicators_task)
            shared_context['technical_indicators'] = indicators_results.get('technical_indicators', {})
            
            # Assess stock health
            logger.info("Assessing stock health...")
            health_task = {
                'context': shared_context
            }
            health_results = self.health_agent.execute_task(health_task)
            shared_context['health_scores'] = health_results.get('health_scores', {})
            
            # Track sentiment trends
            logger.info("Tracking sentiment trends...")
            sentiment_task = {
                'sentiment_data': sentiment_data,
                'context': shared_context
            }
            sentiment_results = self.sentiment_agent.execute_task(sentiment_task)
            shared_context['sentiment_trends'] = sentiment_results.get('sentiment_trends', {})
            
            # Combine all results
            final_results = {
                'status': 'completed',
                'technical_indicators': shared_context['technical_indicators'],
                'health_scores': shared_context['health_scores'],
                'sentiment_trends': shared_context['sentiment_trends'],
                'health_summary': self._generate_health_summary(shared_context),
                'analysis_metadata': {
                    'stocks_analyzed': len([s for s in market_data.keys() if market_data[s].get('status') == 'success']),
                    'indicators_calculated': len(shared_context['technical_indicators']),
                    'health_scores_generated': len(shared_context['health_scores']),
                    'analysis_date': datetime.now().isoformat()
                }
            }
            
            logger.info("Health analysis completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in health analytics crew: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'partial_results': shared_context if 'shared_context' in locals() else {}
            }
    
    def _generate_health_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall health summary"""
        health_scores = context.get('health_scores', {})
        
        if not health_scores:
            return {'summary': 'No health data available'}
        
        # Calculate aggregate metrics
        all_scores = [score['overall_score'] for score in health_scores.values() 
                     if isinstance(score, dict) and 'overall_score' in score]
        
        if not all_scores:
            return {'summary': 'No valid health scores calculated'}
        
        summary = {
            'average_health_score': np.mean(all_scores),
            'highest_score': max(all_scores),
            'lowest_score': min(all_scores),
            'healthy_stocks': len([s for s in all_scores if s >= 70]),
            'neutral_stocks': len([s for s in all_scores if 50 <= s < 70]),
            'unhealthy_stocks': len([s for s in all_scores if s < 50]),
            'total_analyzed': len(all_scores)
        }
        
        # Add qualitative assessment
        avg_score = summary['average_health_score']
        if avg_score >= 70:
            summary['portfolio_health'] = 'Strong'
        elif avg_score >= 50:
            summary['portfolio_health'] = 'Moderate'
        else:
            summary['portfolio_health'] = 'Weak'
        
        return summary


class AnalyzeTechnicalIndicatorsTask:
    """Task for calculating technical indicators"""
    
    @staticmethod
    def execute(market_data: Dict[str, Any], rsi_macd_tool: RSI_MACD_Tool, 
                volatility_scanner: VolatilityScanner) -> Dict[str, Any]:
        """Execute technical indicators analysis"""
        technical_indicators = {}
        
        for symbol, data in market_data.items():
            if data.get('status') != 'success' or 'data' not in data:
                continue
            
            try:
                stock_data = data['data']
                prices = stock_data['Close']
                
                # Calculate RSI
                rsi_values = rsi_macd_tool.calculate_rsi(prices, period=14)
                current_rsi = rsi_values.iloc[-1] if not rsi_values.empty else 50
                
                # Calculate MACD
                macd_line, signal_line, histogram = rsi_macd_tool.calculate_macd(prices)
                current_macd = macd_line.iloc[-1] if not macd_line.empty else 0
                current_signal = signal_line.iloc[-1] if not signal_line.empty else 0
                current_histogram = histogram.iloc[-1] if not histogram.empty else 0
                
                # Calculate volatility
                volatility_metrics = volatility_scanner.calculate_volatility(prices, window=30)
                
                # Determine signal strengths
                rsi_signal = 'oversold' if current_rsi < 30 else 'overbought' if current_rsi > 70 else 'neutral'
                macd_signal = 'bullish' if current_macd > current_signal else 'bearish'
                
                technical_indicators[symbol] = {
                    'rsi': {
                        'current': current_rsi,
                        'signal': rsi_signal,
                        'strength': abs(current_rsi - 50) / 50  # 0-1 scale
                    },
                    'macd': {
                        'macd_line': current_macd,
                        'signal_line': current_signal,
                        'histogram': current_histogram,
                        'signal': macd_signal,
                        'crossover': 'bullish' if current_histogram > 0 else 'bearish'
                    },
                    'volatility': volatility_metrics,
                    'overall_signal': 'bullish' if (rsi_signal == 'oversold' or macd_signal == 'bullish') else 'bearish',
                    'status': 'calculated'
                }
                
                logger.info(f"Calculated technical indicators for {symbol}")
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {str(e)}")
                technical_indicators[symbol] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return {'technical_indicators': technical_indicators}


class AssessStockHealthTask:
    """Task for computing stock health scores"""
    
    @staticmethod
    def execute(context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute stock health assessment"""
        health_scores = {}
        technical_indicators = context.get('technical_indicators', {})
        market_data = context.get('market_data', {})
        
        for symbol in market_data.keys():
            if market_data[symbol].get('status') != 'success':
                continue
            
            try:
                # Initialize score components
                price_score = 0
                technical_score = 0
                momentum_score = 0
                volatility_score = 0
                
                # Calculate price performance score (0-25 points)
                stock_data = market_data[symbol]['data']
                current_price = stock_data['Close'].iloc[-1]
                start_price = stock_data['Close'].iloc[0]
                price_return = ((current_price - start_price) / start_price) * 100
                
                # Price score based on return
                if price_return > 20:
                    price_score = 25
                elif price_return > 10:
                    price_score = 20
                elif price_return > 0:
                    price_score = 15
                elif price_return > -10:
                    price_score = 10
                else:
                    price_score = 5
                
                # Technical indicators score (0-25 points)
                if symbol in technical_indicators:
                    indicators = technical_indicators[symbol]
                    
                    if indicators.get('status') == 'calculated':
                        rsi = indicators['rsi']['current']
                        macd_signal = indicators['macd']['signal']
                        
                        # RSI scoring
                        if 40 <= rsi <= 60:  # Neutral zone
                            rsi_score = 15
                        elif 30 <= rsi <= 70:  # Acceptable range
                            rsi_score = 10
                        else:  # Extreme levels
                            rsi_score = 5
                        
                        # MACD scoring
                        macd_score = 10 if macd_signal == 'bullish' else 5
                        
                        technical_score = rsi_score + macd_score
                
                # Momentum score (0-25 points)
                # Calculate short-term vs long-term moving averages
                ma_5 = stock_data['Close'].rolling(window=5).mean().iloc[-1]
                ma_20 = stock_data['Close'].rolling(window=20).mean().iloc[-1]
                
                if ma_5 > ma_20:
                    momentum_score = 20
                elif abs(ma_5 - ma_20) / ma_20 < 0.02:  # Within 2%
                    momentum_score = 15
                else:
                    momentum_score = 10
                
                # Volatility score (0-25 points)
                price_volatility = stock_data['Close'].pct_change().std() * 100
                if price_volatility < 2:
                    volatility_score = 25
                elif price_volatility < 4:
                    volatility_score = 20
                elif price_volatility < 6:
                    volatility_score = 15
                else:
                    volatility_score = 10
                
                # Calculate overall health score
                overall_score = price_score + technical_score + momentum_score + volatility_score
                
                # Determine health category
                if overall_score >= 80:
                    health_category = 'Excellent'
                elif overall_score >= 70:
                    health_category = 'Good'
                elif overall_score >= 60:
                    health_category = 'Fair'
                elif overall_score >= 50:
                    health_category = 'Poor'
                else:
                    health_category = 'Critical'
                
                health_scores[symbol] = {
                    'overall_score': overall_score,
                    'health_category': health_category,
                    'component_scores': {
                        'price_performance': price_score,
                        'technical_indicators': technical_score,
                        'momentum': momentum_score,
                        'volatility': volatility_score
                    },
                    'metrics': {
                        'return_pct': price_return,
                        'current_price': current_price,
                        'volatility_pct': price_volatility
                    },
                    'status': 'calculated'
                }
                
                logger.info(f"Calculated health score for {symbol}: {overall_score}")
                
            except Exception as e:
                logger.error(f"Error calculating health score for {symbol}: {str(e)}")
                health_scores[symbol] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return {'health_scores': health_scores}


class TrackSentimentTrendsTask:
    """Task for analyzing sentiment trends"""
    
    @staticmethod
    def execute(sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sentiment trend analysis"""
        sentiment_trends = {}
        
        for symbol, data in sentiment_data.items():
            if data.get('status') != 'success' or 'articles' not in data:
                continue
            
            try:
                articles = data['articles']
                
                if not articles:
                    sentiment_trends[symbol] = {
                        'status': 'no_data',
                        'message': 'No articles available for sentiment analysis'
                    }
                    continue
                
                # Extract sentiment scores and dates
                sentiments = []
                for article in articles:
                    if 'sentiment_score' in article:
                        sentiments.append(article['sentiment_score'])
                
                if not sentiments:
                    sentiment_trends[symbol] = {
                        'status': 'no_sentiment',
                        'message': 'No sentiment scores available'
                    }
                    continue
                
                # Calculate sentiment metrics
                avg_sentiment = np.mean(sentiments)
                sentiment_volatility = np.std(sentiments)
                positive_ratio = len([s for s in sentiments if s > 0.1]) / len(sentiments)
                negative_ratio = len([s for s in sentiments if s < -0.1]) / len(sentiments)
                
                # Determine sentiment trend
                if avg_sentiment > 0.2:
                    trend = 'Very Positive'
                elif avg_sentiment > 0.05:
                    trend = 'Positive'
                elif avg_sentiment > -0.05:
                    trend = 'Neutral'
                elif avg_sentiment > -0.2:
                    trend = 'Negative'
                else:
                    trend = 'Very Negative'
                
                # Calculate sentiment momentum (recent vs older articles)
                if len(sentiments) >= 4:
                    recent_sentiment = np.mean(sentiments[-len(sentiments)//2:])
                    older_sentiment = np.mean(sentiments[:len(sentiments)//2])
                    momentum = recent_sentiment - older_sentiment
                else:
                    momentum = 0
                
                sentiment_trends[symbol] = {
                    'current_sentiment': avg_sentiment,
                    'sentiment_trend': trend,
                    'sentiment_momentum': momentum,
                    'sentiment_volatility': sentiment_volatility,
                    'positive_ratio': positive_ratio,
                    'negative_ratio': negative_ratio,
                    'neutral_ratio': 1 - positive_ratio - negative_ratio,
                    'total_articles': len(articles),
                    'sentiment_strength': abs(avg_sentiment),
                    'status': 'analyzed'
                }
                
                logger.info(f"Analyzed sentiment trends for {symbol}")
                
            except Exception as e:
                logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
                sentiment_trends[symbol] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return {'sentiment_trends': sentiment_trends}