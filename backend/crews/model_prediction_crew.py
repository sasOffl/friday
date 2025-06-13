"""
Model Prediction Crew for McKinsey Stock Performance Monitor
Handles training, prediction, and model evaluation tasks
"""

from crewai import Crew, Task
from agents.prediction_agents import ForecastAgent, EvaluationAgent
from tools.prediction_tools import NeuralProphetWrapper
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ModelPredictionCrew:
    """Crew responsible for model training and price prediction"""
    
    def __init__(self):
        self.forecast_agent = ForecastAgent()
        self.evaluation_agent = EvaluationAgent()
        self.neural_prophet = NeuralProphetWrapper()
    
    def create_crew(self) -> Crew:
        """Create and configure the model prediction crew"""
        
        # Define tasks
        train_forecast_model_task = Task(
            description="Train NeuralProphet forecasting models for each stock",
            agent=self.forecast_agent,
            expected_output="Trained models for all stocks with training metrics"
        )
        
        generate_predictions_task = Task(
            description="Generate price forecasts with confidence intervals",
            agent=self.forecast_agent,
            expected_output="Price predictions for specified forecast horizon",
            dependencies=[train_forecast_model_task]
        )
        
        evaluate_model_task = Task(
            description="Evaluate model performance using validation metrics",
            agent=self.evaluation_agent,
            expected_output="Model accuracy metrics including RMSE, MAE, and directional accuracy",
            dependencies=[train_forecast_model_task, generate_predictions_task]
        )
        
        # Create crew
        crew = Crew(
            agents=[
                self.forecast_agent,
                self.evaluation_agent
            ],
            tasks=[
                train_forecast_model_task,
                generate_predictions_task,
                evaluate_model_task
            ],
            verbose=True,
            process="sequential"
        )
        
        return crew
    
    def execute_prediction(self, market_data: Dict[str, Any], forecast_horizon: int = 30) -> Dict[str, Any]:
        """Execute the complete prediction process"""
        try:
            logger.info(f"Starting prediction process for {len(market_data)} stocks")
            
            # Prepare shared context
            shared_context = {
                'market_data': market_data,
                'forecast_horizon': forecast_horizon,
                'trained_models': {},
                'predictions': {},
                'evaluation_metrics': {}
            }
            
            # Train models
            logger.info("Training forecasting models...")
            train_task = {
                'market_data': market_data,
                'forecast_horizon': forecast_horizon,
                'context': shared_context
            }
            train_results = self.forecast_agent.execute_task(train_task)
            shared_context['trained_models'] = train_results.get('trained_models', {})
            
            # Generate predictions
            logger.info("Generating price predictions...")
            predict_task = {
                'trained_models': shared_context['trained_models'],
                'forecast_horizon': forecast_horizon,
                'context': shared_context
            }
            prediction_results = self.forecast_agent.execute_task(predict_task)
            shared_context['predictions'] = prediction_results.get('predictions', {})
            
            # Evaluate models
            logger.info("Evaluating model performance...")
            eval_task = {
                'context': shared_context
            }
            evaluation_results = self.evaluation_agent.execute_task(eval_task)
            
            # Combine all results
            final_results = {
                'status': 'completed',
                'predictions': shared_context['predictions'],
                'model_performance': evaluation_results.get('metrics', {}),
                'training_summary': train_results.get('training_summary', {}),
                'forecast_metadata': {
                    'models_trained': len(shared_context['trained_models']),
                    'predictions_generated': len(shared_context['predictions']),
                    'forecast_horizon': forecast_horizon,
                    'training_date': datetime.now().isoformat()
                }
            }
            
            logger.info("Prediction process completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in prediction crew: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'partial_results': shared_context if 'shared_context' in locals() else {}
            }


class TrainForecastModelTask:
    """Task for training NeuralProphet models"""
    
    @staticmethod
    def execute(market_data: Dict[str, Any], neural_prophet: NeuralProphetWrapper) -> Dict[str, Any]:
        """Execute model training task"""
        trained_models = {}
        training_summary = {}
        
        for symbol, data in market_data.items():
            if data.get('status') != 'success' or 'data' not in data:
                logger.warning(f"Skipping {symbol} - no valid data available")
                continue
            
            try:
                stock_data = data['data']
                
                # Prepare data for NeuralProphet (requires 'ds' and 'y' columns)
                prophet_data = pd.DataFrame({
                    'ds': stock_data.index,
                    'y': stock_data['Close']
                }).reset_index(drop=True)
                
                # Train the model
                model = neural_prophet.train_model(prophet_data)
                
                if model is not None:
                    trained_models[symbol] = {
                        'model': model,
                        'training_data': prophet_data,
                        'data_points': len(prophet_data),
                        'status': 'success'
                    }
                    
                    training_summary[symbol] = {
                        'training_samples': len(prophet_data),
                        'date_range': f"{prophet_data['ds'].min()} to {prophet_data['ds'].max()}",
                        'price_range': f"${prophet_data['y'].min():.2f} - ${prophet_data['y'].max():.2f}",
                        'status': 'trained'
                    }
                    
                    logger.info(f"Successfully trained model for {symbol}")
                else:
                    training_summary[symbol] = {
                        'status': 'failed',
                        'error': 'Model training failed'
                    }
                    
            except Exception as e:
                logger.error(f"Error training model for {symbol}: {str(e)}")
                training_summary[symbol] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return {
            'trained_models': trained_models,
            'training_summary': training_summary
        }


class GeneratePredictionsTask:
    """Task for generating price forecasts"""
    
    @staticmethod
    def execute(trained_models: Dict[str, Any], forecast_horizon: int, 
                neural_prophet: NeuralProphetWrapper) -> Dict[str, Any]:
        """Execute prediction generation task"""
        predictions = {}
        
        for symbol, model_data in trained_models.items():
            if model_data.get('status') != 'success':
                continue
            
            try:
                model = model_data['model']
                training_data = model_data['training_data']
                
                # Generate predictions
                forecast_results = neural_prophet.predict_prices(model, forecast_horizon)
                
                if forecast_results is not None and not forecast_results.empty:
                    # Extract prediction components
                    last_actual_price = training_data['y'].iloc[-1]
                    
                    predictions[symbol] = {
                        'forecasts': forecast_results.to_dict('records'),
                        'last_actual_price': last_actual_price,
                        'forecast_horizon': forecast_horizon,
                        'prediction_summary': {
                            'predicted_price_end': forecast_results['yhat1'].iloc[-1],
                            'price_change': forecast_results['yhat1'].iloc[-1] - last_actual_price,
                            'price_change_pct': ((forecast_results['yhat1'].iloc[-1] - last_actual_price) / last_actual_price) * 100,
                            'confidence_upper': forecast_results['yhat1_upper'].iloc[-1],
                            'confidence_lower': forecast_results['yhat1_lower'].iloc[-1],
                            'trend_direction': 'up' if forecast_results['yhat1'].iloc[-1] > last_actual_price else 'down'
                        },
                        'status': 'success'
                    }
                    
                    logger.info(f"Generated predictions for {symbol}")
                else:
                    predictions[symbol] = {
                        'status': 'failed',
                        'error': 'Prediction generation failed'
                    }
                    
            except Exception as e:
                logger.error(f"Error generating predictions for {symbol}: {str(e)}")
                predictions[symbol] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return {'predictions': predictions}


class EvaluateModelTask:
    """Task for evaluating model performance"""
    
    @staticmethod
    def execute(context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model evaluation task"""
        try:
            trained_models = context.get('trained_models', {})
            predictions = context.get('predictions', {})
            
            evaluation_metrics = {}
            overall_performance = {
                'models_evaluated': 0,
                'avg_accuracy': 0,
                'successful_predictions': 0
            }
            
            successful_evaluations = []
            
            for symbol in trained_models.keys():
                if symbol not in predictions or predictions[symbol].get('status') != 'success':
                    continue
                
                try:
                    model_data = trained_models[symbol]
                    prediction_data = predictions[symbol]
                    
                    # Calculate basic metrics using training data
                    training_data = model_data['training_data']
                    
                    # Simple evaluation metrics
                    data_points = len(training_data)
                    price_volatility = training_data['y'].std()
                    price_trend = 'increasing' if training_data['y'].iloc[-1] > training_data['y'].iloc[0] else 'decreasing'
                    
                    # Calculate prediction confidence
                    pred_summary = prediction_data['prediction_summary']
                    confidence_range = pred_summary['confidence_upper'] - pred_summary['confidence_lower']
                    relative_confidence = (confidence_range / pred_summary['predicted_price_end']) * 100
                    
                    evaluation_metrics[symbol] = {
                        'data_quality_score': min(100, (data_points / 252) * 100),  # Based on trading days in a year
                        'price_volatility': price_volatility,
                        'historical_trend': price_trend,
                        'prediction_confidence': max(0, 100 - relative_confidence),  # Higher is better
                        'forecast_direction': pred_summary['trend_direction'],
                        'expected_return': pred_summary['price_change_pct'],
                        'confidence_interval_width': relative_confidence,
                        'model_status': 'evaluated'
                    }
                    
                    successful_evaluations.append(evaluation_metrics[symbol]['prediction_confidence'])
                    overall_performance['models_evaluated'] += 1
                    overall_performance['successful_predictions'] += 1
                    
                    logger.info(f"Evaluated model for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating model for {symbol}: {str(e)}")
                    evaluation_metrics[symbol] = {
                        'model_status': 'evaluation_failed',
                        'error': str(e)
                    }
            
            # Calculate overall performance
            if successful_evaluations:
                overall_performance['avg_accuracy'] = np.mean(successful_evaluations)
            
            return {
                'status': 'completed',
                'metrics': evaluation_metrics,
                'overall_performance': overall_performance
            }
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }