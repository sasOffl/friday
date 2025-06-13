"""
McKinsey Stock Performance Monitor - Prediction Agents

This module contains agents responsible for stock price forecasting and model evaluation.
Agents handle NeuralProphet model training, prediction generation, and accuracy assessment.
"""

from crewai import Agent
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from ..tools.prediction_tools import NeuralProphetWrapper
from ..utils.logger import AnalysisLogger

logger = AnalysisLogger()


class ForecastAgent:
    """
    Agent responsible for stock price forecasting using NeuralProphet.
    
    Capabilities:
    - Train NeuralProphet models on historical data
    - Generate price forecasts with confidence intervals
    - Handle multiple stocks simultaneously
    - Optimize model parameters for each stock
    """
    
    def __init__(self):
        self.neural_prophet = NeuralProphetWrapper()
        self.agent = Agent(
            role="Quantitative Forecast Analyst",
            goal="Generate accurate stock price predictions using advanced time series models",
            backstory="""You are a quantitative analyst with deep expertise in time series 
            forecasting and machine learning. You specialize in NeuralProphet models and 
            have a track record of producing reliable financial forecasts.""",
            verbose=True,
            allow_delegation=False
        )
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute stock price forecasting task.
        
        Args:
            task: Dictionary containing processed data and forecasting parameters
            
        Returns:
            Dictionary with forecast results and model performance
        """
        try:
            processed_data = task.get('processed_data', {})
            horizon_days = task.get('horizon_days', 30)
            session_id = task.get('session_id')
            
            logger.log_crew_activity(
                'ModelPrediction',
                f"Starting forecasting for {len(processed_data)} symbols with {horizon_days} day horizon",
                'INFO'
            )
            
            forecast_results = {}
            
            for symbol, symbol_data in processed_data.items():
                try:
                    if 'cleaned_stock_data' not in symbol_data:
                        logger.log_crew_activity(
                            'ModelPrediction',
                            f"No cleaned data available for {symbol}",
                            'WARNING'
                        )
                        continue
                    
                    stock_data = symbol_data['cleaned_stock_data']
                    
                    # Prepare data for NeuralProphet
                    prophet_data = self._prepare_prophet_data(stock_data)
                    
                    # Train model
                    logger.log_crew_activity(
                        'ModelPrediction',
                        f"Training NeuralProphet model for {symbol}",
                        'INFO'
                    )
                    
                    model = self.neural_prophet.train_model(prophet_data)
                    
                    # Generate predictions
                    predictions = self.neural_prophet.predict_prices(model, horizon_days)
                    
                    # Calculate prediction confidence metrics
                    confidence_metrics = self._calculate_confidence_metrics(predictions)
                    
                    # Generate prediction summary
                    prediction_summary = self._generate_prediction_summary(
                        stock_data, predictions, horizon_days
                    )
                    
                    forecast_results[symbol] = {
                        'predictions': predictions,
                        'confidence_metrics': confidence_metrics,
                        'prediction_summary': prediction_summary,
                        'model_trained': True,
                        'forecast_horizon': horizon_days,
                        'training_data_points': len(prophet_data)
                    }
                    
                    logger.log_crew_activity(
                        'ModelPrediction',
                        f"Successfully generated {horizon_days}-day forecast for {symbol}",
                        'INFO'
                    )
                    
                except Exception as e:
                    logger.log_crew_activity(
                        'ModelPrediction',
                        f"Error forecasting {symbol}: {str(e)}",
                        'ERROR'
                    )
                    
                    forecast_results[symbol] = {
                        'predictions': None,
                        'error': str(e),
                        'model_trained': False
                    }
            
            return {
                'status': 'completed',
                'data': forecast_results,
                'summary': {
                    'symbols_forecasted': len([r for r in forecast_results.values() if r.get('model_trained')]),
                    'symbols_failed': len([r for r in forecast_results.values() if not r.get('model_trained')]),
                    'forecast_horizon': horizon_days,
                    'total_predictions': sum(
                        len(r.get('predictions', [])) for r in forecast_results.values()
                        if r.get('predictions') is not None
                    )
                }
            }
            
        except Exception as e:
            logger.log_crew_activity(
                'ModelPrediction',
                f"Forecasting task failed: {str(e)}",
                'ERROR'
            )
            return {'status': 'failed', 'error': str(e)}
    
    def _prepare_prophet_data(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare stock data for NeuralProphet model."""
        prophet_data = pd.DataFrame()
        prophet_data['ds'] = stock_data.index
        prophet_data['y'] = stock_data['Close'].values
        
        # Add additional regressors
        prophet_data['volume'] = stock_data['Volume'].values if 'Volume' in stock_data.columns else 0
        prophet_data['high_low_ratio'] = (
            stock_data['High'] / stock_data['Low'] - 1
        ).values if all(col in stock_data.columns for col in ['High', 'Low']) else 0
        
        return prophet_data.reset_index(drop=True)
    
    def _calculate_confidence_metrics(self, predictions: pd.DataFrame) -> Dict[str, float]:
        """Calculate confidence metrics for predictions."""
        if predictions is None or predictions.empty:
            return {'confidence_score': 0, 'uncertainty_range': 0}
        
        # Calculate confidence based on prediction intervals
        if 'yhat_lower' in predictions.columns and 'yhat_upper' in predictions.columns:
            uncertainty_range = (
                predictions['yhat_upper'] - predictions['yhat_lower']
            ).mean() / predictions['yhat'].mean()
            
            confidence_score = max(0, 1 - uncertainty_range)
        else:
            confidence_score = 0.5  # Default moderate confidence
            uncertainty_range = 0.1
        
        return {
            'confidence_score': confidence_score,
            'uncertainty_range': uncertainty_range,
            'prediction_variance': predictions['yhat'].var() if 'yhat' in predictions.columns else 0
        }
    
    def _generate_prediction_summary(self, historical_data: pd.DataFrame, 
                                   predictions: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Generate comprehensive prediction summary."""
        if predictions is None or predictions.empty:
            return {'error': 'No predictions available'}
        
        current_price = historical_data['Close'].iloc[-1]
        predicted_price = predictions['yhat'].iloc[-1]
        
        # Calculate expected returns
        expected_return = (predicted_price - current_price) / current_price
        
        # Calculate volatility forecast
        price_changes = predictions['yhat'].diff().dropna()
        predicted_volatility = price_changes.std() / predictions['yhat'].mean()
        
        return {
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'expected_return': float(expected_return),
            'expected_return_pct': f"{expected_return * 100:.2f}%",
            'predicted_volatility': float(predicted_volatility),
            'forecast_trend': 'bullish' if expected_return > 0.02 else 'bearish' if expected_return < -0.02 else 'neutral',
            'prediction_dates': {
                'start': predictions['ds'].iloc[0].strftime('%Y-%m-%d'),
                'end': predictions['ds'].iloc[-1].strftime('%Y-%m-%d')
            }
        }


class EvaluationAgent:
    """
    Agent responsible for evaluating prediction model performance.
    
    Capabilities:
    - Calculate accuracy metrics (RMSE, MAE, MAPE)
    - Perform backtesting analysis
    - Generate model performance reports
    - Compare different model configurations
    """
    
    def __init__(self):
        self.agent = Agent(
            role="Model Performance Evaluator",
            goal="Assess and validate the accuracy of prediction models",
            backstory="""You are a model validation specialist with expertise in 
            statistical analysis and backtesting. You ensure model reliability and 
            provide comprehensive performance assessments.""",
            verbose=True,
            allow_delegation=False
        )
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute model evaluation task.
        
        Args:
            task: Dictionary containing forecast results and evaluation parameters
            
        Returns:
            Dictionary with evaluation metrics and performance analysis
        """
        try:
            forecast_results = task.get('forecast_results', {})
            processed_data = task.get('processed_data', {})
            session_id = task.get('session_id')
            
            logger.log_crew_activity(
                'ModelPrediction',
                f"Starting model evaluation for {len(forecast_results)} symbols",
                'INFO'
            )
            
            evaluation_results = {}
            
            for symbol, forecast_data in forecast_results.items():
                try:
                    if not forecast_data.get('model_trained'):
                        continue
                    
                    # Get historical data for backtesting
                    if symbol not in processed_data:
                        continue
                    
                    historical_data = processed_data[symbol]['cleaned_stock_data']
                    predictions = forecast_data['predictions']
                    
                    # Perform backtesting evaluation
                    backtest_results = self._perform_backtesting(historical_data, predictions)
                    
                    # Calculate accuracy metrics
                    accuracy_metrics = self._calculate_accuracy_metrics(
                        historical_data, predictions
                    )
                    
                    # Generate performance analysis
                    performance_analysis = self._analyze_model_performance(
                        accuracy_metrics, backtest_results
                    )
                    
                    evaluation_results[symbol] = {
                        'accuracy_metrics': accuracy_metrics,
                        'backtest_results': backtest_results,
                        'performance_analysis': performance_analysis,
                        'evaluation_date': datetime.now().isoformat(),
                        'model_quality_score': self._calculate_model_quality_score(accuracy_metrics)
                    }
                    
                    logger.log_crew_activity(
                        'ModelPrediction',
                        f"Completed evaluation for {symbol} - Quality Score: {evaluation_results[symbol]['model_quality_score']:.0f}/100",
                        'INFO'
                    )
                    
                except Exception as e:
                    logger.log_crew_activity(
                        'ModelPrediction',
                        f"Error evaluating model for {symbol}: {str(e)}",
                        'ERROR'
                    )
            
            return {
                'status': 'completed',
                'data': evaluation_results,
                'summary': {
                    'models_evaluated': len(evaluation_results),
                    'average_quality_score': np.mean([
                        r['model_quality_score'] for r in evaluation_results.values()
                    ]) if evaluation_results else 0,
                    'high_quality_models': len([
                        r for r in evaluation_results.values() 
                        if r['model_quality_score'] >= 80
                    ])
                }
            }
            
        except Exception as e:
            logger.log_crew_activity(
                'ModelPrediction',
                f"Model evaluation failed: {str(e)}",
                'ERROR'
            )
            return {'status': 'failed', 'error': str(e)}
    
    def _perform_backtesting(self, historical_data: pd.DataFrame, 
                           predictions: pd.DataFrame) -> Dict[str, Any]:
        """Perform backtesting analysis on the model."""
        try:
            # Use last 30 days of historical data for backtesting
            backtest_period = min(30, len(historical_data) // 4)
            
            if len(historical_data) < backtest_period * 2:
                return {'error': 'Insufficient data for backtesting'}
            
            # Split data
            train_data = historical_data.iloc[:-backtest_period]
            test_data = historical_data.iloc[-backtest_period:]
            
            # Simulate predictions on test period
            test_predictions = self._simulate_predictions(train_data, test_data)
            
            # Calculate backtest metrics
            actual_prices = test_data['Close'].values
            predicted_prices = test_predictions
            
            rmse = np.sqrt(np.mean((actual_prices - predicted_prices) ** 2))
            mae = np.mean(np.abs(actual_prices - predicted_prices))
            mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
            
            # Direction accuracy
            actual_direction = np.sign(np.diff(actual_prices))
            predicted_direction = np.sign(np.diff(predicted_prices))
            direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
            
            return {
                'backtest_period': backtest_period,
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'direction_accuracy': float(direction_accuracy),
                'mean_actual_price': float(np.mean(actual_prices)),
                'mean_predicted_price': float(np.mean(predicted_prices))
            }
            
        except Exception as e:
            return {'error': f'Backtesting failed: {str(e)}'}
    
    def _simulate_predictions(self, train_data: pd.DataFrame, 
                            test_data: pd.DataFrame) -> np.ndarray:
        """Simulate predictions for backtesting."""
        # Simple linear trend extrapolation for backtesting
        train_prices = train_data['Close'].values
        
        # Calculate trend from last 10 days
        recent_prices = train_prices[-10:]
        trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        
        # Generate predictions
        predictions = []
        last_price = train_prices[-1]
        
        for i in range(len(test_data)):
            predicted_price = last_price + trend * (i + 1)
            predictions.append(predicted_price)
        
        return np.array(predictions)
    
    def _calculate_accuracy_metrics(self, historical_data: pd.DataFrame,
                                  predictions: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive accuracy metrics."""
        if predictions is None or predictions.empty:
            return {'error': 'No predictions to evaluate'}
        
        try:
            # Get recent historical data for comparison
            recent_data = historical_data.iloc[-min(30, len(historical_data)):]
            
            # Calculate various metrics based on available data
            metrics = {
                'data_coverage': len(predictions) / 30,  # Assuming 30-day forecast
                'prediction_range': float(predictions['yhat'].max() - predictions['yhat'].min()) if 'yhat' in predictions.columns else 0,
                'trend_consistency': self._calculate_trend_consistency(predictions),
                'volatility_realism': self._assess_volatility_realism(historical_data, predictions)
            }
            
            return metrics
            
        except Exception as e:
            return {'error': f'Metrics calculation failed: {str(e)}'}
    
    def _calculate_trend_consistency(self, predictions: pd.DataFrame) -> float:
        """Calculate how consistent the predicted trend is."""
        if 'yhat' not in predictions.columns or len(predictions) < 5:
            return 0.5
        
        # Calculate day-to-day changes
        changes = predictions['yhat'].diff().dropna()
        
        # Measure trend consistency (less volatility in changes = more consistent)
        if len(changes) > 0 and changes.std() > 0:
            consistency = 1 / (1 + changes.std() / changes.mean() if changes.mean() != 0 else 1)
            return max(0, min(1, consistency))
        
        return 0.5
    
    def _assess_volatility_realism(self, historical_data: pd.DataFrame,
                                 predictions: pd.DataFrame) -> float:
        """Assess if predicted volatility is realistic compared to historical."""
        try:
            # Historical volatility
            hist_returns = historical_data['Close'].pct_change().dropna()
            hist_volatility = hist_returns.std()
            
            # Predicted volatility
            if 'yhat' in predictions.columns and len(predictions) > 1:
                pred_changes = predictions['yhat'].pct_change().dropna()
                pred_volatility = pred_changes.std()
                
                # Compare volatilities (closer to historical = more realistic)
                volatility_ratio = min(pred_volatility, hist_volatility) / max(pred_volatility, hist_volatility)
                return volatility_ratio
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _analyze_model_performance(self, accuracy_metrics: Dict[str, Any],
                                 backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance analysis."""
        analysis = {
            'overall_quality': 'unknown',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        try:
            # Analyze backtest results
            if 'error' not in backtest_results:
                mape = backtest_results.get('mape', 100)
                direction_accuracy = backtest_results.get('direction_accuracy', 50)
                
                if mape < 5:
                    analysis['strengths'].append('Excellent price accuracy')
                elif mape < 10:
                    analysis['strengths'].append('Good price accuracy')
                else:
                    analysis['weaknesses'].append('High prediction error')
                
                if direction_accuracy > 70:
                    analysis['strengths'].append('Strong directional accuracy')
                elif direction_accuracy < 55:
                    analysis['weaknesses'].append('Poor directional prediction')
            
            # Analyze accuracy metrics
            if 'error' not in accuracy_metrics:
                trend_consistency = accuracy_metrics.get('trend_consistency', 0.5)
                volatility_realism = accuracy_metrics.get('volatility_realism', 0.5)
                
                if trend_consistency > 0.7:
                    analysis['strengths'].append('Consistent trend prediction')
                if volatility_realism > 0.7:
                    analysis['strengths'].append('Realistic volatility modeling')
            
            # Determine overall quality
            if len(analysis['strengths']) >= 2 and len(analysis['weaknesses']) == 0:
                analysis['overall_quality'] = 'high'
            elif len(analysis['strengths']) >= 1 and len(analysis['weaknesses']) <= 1:
                analysis['overall_quality'] = 'medium'
            else:
                analysis['overall_quality'] = 'low'
            
            # Generate recommendations
            if analysis['overall_quality'] == 'low':
                analysis['recommendations'].append('Consider increasing training data')
                analysis['recommendations'].append('Review feature engineering')
            elif analysis['overall_quality'] == 'medium':
                analysis['recommendations'].append('Monitor model performance closely')
            else:
                analysis['recommendations'].append('Model suitable for production use')
            
        except Exception as e:
            analysis['error'] = f'Analysis failed: {str(e)}'
        
        return analysis
    
    def _calculate_model_quality_score(self, accuracy_metrics: Dict[str, Any]) -> float:
        """Calculate overall model quality score (0-100)."""
        if 'error' in accuracy_metrics:
            return 30  # Low score for failed models
        
        try:
            score = 50  # Base score
            
            # Add points for good metrics
            trend_consistency = accuracy_metrics.get('trend_consistency', 0.5)
            volatility_realism = accuracy_metrics.get('volatility_realism', 0.5)
            data_coverage = accuracy_metrics.get('data_coverage', 0.5)
            
            score += trend_consistency * 20
            score += volatility_realism * 20
            score += data_coverage * 10
            
            return min(100, max(0, score))
            
        except Exception:
            return 40  # Default moderate score