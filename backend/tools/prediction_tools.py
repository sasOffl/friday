"""
Prediction Tools - McKinsey Stock Performance Monitor
Advanced forecasting tools using NeuralProphet and ensemble methods
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import pickle
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from neuralprophet import NeuralProphet
from utils.logger import AnalysisLogger
from utils.exceptions import ModelTrainingException

logger = AnalysisLogger()


class NeuralProphetWrapper:
    """
    Wrapper for NeuralProphet model with enhanced features for stock price prediction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize NeuralProphet wrapper
        
        Args:
            config: Configuration dictionary for model parameters
        """
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history = {}
        self.feature_importance = {}
        
        # Default model configuration
        self.default_config = {
            'n_forecasts': 1,
            'n_lags': 14,
            'n_changepoints': 10,
            'changepoints_range': 0.8,
            'trend_reg': 0.01,
            'seasonality_reg': 0.01,
            'ar_reg': 0.01,
            'learning_rate': 0.01,
            'epochs': 100,
            'batch_size': 32,
            'loss_func': 'Huber',
            'normalize': 'soft',
            'impute_missing': True
        }
        
        # Update with provided config
        self.model_config = {**self.default_config, **self.config}
        
        logger.log_crew_activity(
            "NeuralProphetWrapper",
            "NeuralProphet wrapper initialized",
            "INFO"
        )
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'Close') -> pd.DataFrame:
        """
        Prepare data for NeuralProphet training
        
        Args:
            df: Input DataFrame with stock data
            target_column: Target column name
            
        Returns:
            Prepared DataFrame for NeuralProphet
        """
        try:
            # Create a copy to avoid modifying original
            data = df.copy()
            
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Prepare NeuralProphet format (ds, y)
            prophet_data = pd.DataFrame({
                'ds': data.index,
                'y': data[target_column]
            })
            
            # Add additional features as regressors
            feature_columns = [col for col in data.columns if col != target_column]
            for col in feature_columns:
                prophet_data[col] = data[col].values
            
            # Remove any rows with NaN values
            prophet_data = prophet_data.dropna()
            
            # Sort by date
            prophet_data = prophet_data.sort_values('ds')
            
            logger.log_crew_activity(
                "NeuralProphetWrapper",
                f"Data prepared: {len(prophet_data)} rows, {len(feature_columns)} features",
                "INFO"
            )
            
            return prophet_data
            
        except Exception as e:
            logger.log_crew_activity(
                "NeuralProphetWrapper",
                f"Error preparing data: {str(e)}",
                "ERROR"
            )
            raise ModelTrainingException(f"Data preparation failed: {str(e)}")
    
    def train_model(self, historical_data: pd.DataFrame, target_column: str = 'Close') -> Dict[str, Any]:
        """
        Train NeuralProphet model on historical data
        
        Args:
            historical_data: Historical stock data
            target_column: Target column to predict
            
        Returns:
            Training results and metrics
        """
        try:
            # Prepare data
            train_data = self.prepare_data(historical_data, target_column)
            
            # Initialize NeuralProphet model
            self.model = NeuralProphet(
                n_forecasts=self.model_config['n_forecasts'],
                n_lags=self.model_config['n_lags'],
                n_changepoints=self.model_config['n_changepoints'],
                changepoints_range=self.model_config['changepoints_range'],
                trend_reg=self.model_config['trend_reg'],
                seasonality_reg=self.model_config['seasonality_reg'],
                ar_reg=self.model_config['ar_reg'],
                learning_rate=self.model_config['learning_rate'],
                epochs=self.model_config['epochs'],
                batch_size=self.model_config['batch_size'],
                loss_func=self.model_config['loss_func'],
                normalize=self.model_config['normalize'],
                impute_missing=self.model_config['impute_missing']
            )
            
            # Add regressors for additional features
            feature_columns = [col for col in train_data.columns if col not in ['ds', 'y']]
            for col in feature_columns:
                self.model.add_lagged_regressor(col)
            
            # Train the model
            logger.log_crew_activity(
                "NeuralProphetWrapper",
                "Starting model training...",
                "INFO"
            )
            
            metrics = self.model.fit(train_data, freq='D')
            
            # Store training history
            self.training_history = {
                'training_data_shape': train_data.shape,
                'features_used': feature_columns,
                'training_period': (train_data['ds'].min(), train_data['ds'].max()),
                'model_config': self.model_config,
                'training_metrics': metrics.to_dict() if hasattr(metrics, 'to_dict') else str(metrics)
            }
            
            self.is_trained = True
            
            logger.log_crew_activity(
                "NeuralProphetWrapper",
                f"Model training completed successfully",
                "INFO"
            )
            
            return {
                'status': 'success',
                'training_history': self.training_history,
                'model_ready': True
            }
            
        except Exception as e:
            logger.log_crew_activity(
                "NeuralProphetWrapper",
                f"Model training failed: {str(e)}",
                "ERROR"
            )
            raise ModelTrainingException(f"NeuralProphet training failed: {str(e)}")
    
    def predict_prices(self, horizon_days: int, future_regressors: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate price predictions with confidence intervals
        
        Args:
            horizon_days: Number of days to forecast
            future_regressors: Future values of regressor variables
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        try:
            if not self.is_trained or self.model is None:
                raise ModelTrainingException("Model must be trained before making predictions")
            
            # Create future dataframe
            future = self.model.make_future_dataframe(
                periods=horizon_days,
                n_historic_predictions=True
            )
            
            # Add future regressors if provided
            if future_regressors is not None:
                for col in future_regressors.columns:
                    if col in self.training_history.get('features_used', []):
                        # Extend future regressors to match future dataframe length
                        future[col] = np.concatenate([
                            future_regressors[col].iloc[-len(future):].values,
                            [future_regressors[col].iloc[-1]] * max(0, len(future) - len(future_regressors))
                        ])
            
            # Generate predictions
            forecast = self.model.predict(future)
            
            # Extract prediction columns
            predictions = forecast[['ds', 'yhat1']].copy()
            predictions.columns = ['Date', 'Predicted_Price']
            
            # Add confidence intervals if available
            if 'yhat1_lower' in forecast.columns and 'yhat1_upper' in forecast.columns:
                predictions['Lower_CI'] = forecast['yhat1_lower']
                predictions['Upper_CI'] = forecast['yhat1_upper']
            else:
                # Calculate approximate confidence intervals
                std_dev = predictions['Predicted_Price'].std()
                predictions['Lower_CI'] = predictions['Predicted_Price'] - 1.96 * std_dev
                predictions['Upper_CI'] = predictions['Predicted_Price'] + 1.96 * std_dev
            
            # Filter to future predictions only
            last_date = future['ds'].iloc[-horizon_days-1]
            future_predictions = predictions[predictions['Date'] > last_date].copy()
            
            logger.log_crew_activity(
                "NeuralProphetWrapper",
                f"Generated {len(future_predictions)} price predictions",
                "INFO"
            )
            
            return future_predictions
            
        except Exception as e:
            logger.log_crew_activity(
                "NeuralProphetWrapper",
                f"Prediction generation failed: {str(e)}",
                "ERROR"
            )
            raise ModelTrainingException(f"Prediction failed: {str(e)}")
    
    def evaluate_model(self, test_data: pd.DataFrame, target_column: str = 'Close') -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: Test dataset
            target_column: Target column name
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if not self.is_trained:
                raise ModelTrainingException("Model must be trained before evaluation")
            
            # Prepare test data
            test_prepared = self.prepare_data(test_data, target_column)
            
            # Generate predictions for test period
            predictions = self.model.predict(test_prepared)
            
            # Calculate metrics
            actual = test_prepared['y'].values
            predicted = predictions['yhat1'].values
            
            # Ensure same length
            min_len = min(len(actual), len(predicted))
            actual = actual[:min_len]
            predicted = predicted[:min_len]
            
            metrics = {
                'MAE': mean_absolute_error(actual, predicted),
                'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
                'MAPE': np.mean(np.abs((actual - predicted) / actual)) * 100,
                'R2': r2_score(actual, predicted),
                'Accuracy': 100 - np.mean(np.abs((actual - predicted) / actual)) * 100
            }
            
            logger.log_crew_activity(
                "NeuralProphetWrapper",
                f"Model evaluation completed - RMSE: {metrics['RMSE']:.4f}",
                "INFO"
            )
            
            return metrics
            
        except Exception as e:
            logger.log_crew_activity(
                "NeuralProphetWrapper",
                f"Model evaluation failed: {str(e)}",
                "ERROR"
            )
            return {'error': str(e)}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Extract feature importance from trained model
        
        Returns:
            Dictionary of feature importance scores
        """
        try:
            if not self.is_trained:
                return {}
            
            # NeuralProphet doesn't provide direct feature importance
            # We'll use a proxy method based on model components
            importance = {}
            
            # Get model components
            if hasattr(self.model, 'model'):
                # Extract importance from model parameters
                for name, param in self.model.model.named_parameters():
                    if 'lagged_regressor' in name:
                        importance[name] = float(param.abs().mean().detach().numpy())
            
            return importance
            
        except Exception as e:
            logger.log_crew_activity(
                "NeuralProphetWrapper",
                f"Feature importance extraction failed: {str(e)}",
                "ERROR"
            )
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """
        Save trained model to file
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_trained:
                return False
            
            model_data = {
                'model': self.model,
                'config': self.model_config,
                'training_history': self.training_history,
                'is_trained': self.is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.log_crew_activity(
                "NeuralProphetWrapper",
                f"Model saved to {filepath}",
                "INFO"
            )
            
            return True
            
        except Exception as e:
            logger.log_crew_activity(
                "NeuralProphetWrapper",
                f"Model save failed: {str(e)}",
                "ERROR"
            )
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load trained model from file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.model_config = model_data['config']
            self.training_history = model_data['training_history']
            self.is_trained = model_data['is_trained']
            
            logger.log_crew_activity(
                "NeuralProphetWrapper",
                f"Model loaded from {filepath}",
                "INFO"
            )
            
            return True
            
        except Exception as e:
            logger.log_crew_activity(
                "NeuralProphetWrapper",
                f"Model load failed: {str(e)}",
                "ERROR"
            )
            return False


class EnsemblePredictionTool:
    """
    Ensemble prediction tool combining multiple forecasting models
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ensemble prediction tool
        
        Args:
            config: Configuration dictionary for ensemble parameters
        """
        self.config = config or {}
        self.models = {}
        self.weights = {}
        self.is_trained = False
        self.ensemble_history = {}
        
        # Default ensemble configuration
        self.default_config = {
            'use_neural_prophet': True,
            'use_random_forest': True,
            'use_gradient_boosting': True,
            'use_linear_regression': True,
            'weight_method': 'performance',  # 'equal', 'performance', 'adaptive'
            'cv_folds': 5,
            'lookback_window': 30,
            'feature_lag_periods': [1, 3, 5, 10, 20]
        }
        
        # Update with provided config
        self.ensemble_config = {**self.default_config, **self.config}
        
        logger.log_crew_activity(
            "EnsemblePredictionTool",
            "Ensemble prediction tool initialized",
            "INFO"
        )
    
    def create_features(self, df: pd.DataFrame, target_column: str = 'Close') -> pd.DataFrame:
        """
        Create feature matrix for ensemble models
        
        Args:
            df: Input DataFrame with stock data
            target_column: Target column name
            
        Returns:
            Feature matrix DataFrame
        """
        try:
            features = df.copy()
            
            # Add lagged features
            for lag in self.ensemble_config['feature_lag_periods']:
                features[f'{target_column}_lag_{lag}'] = features[target_column].shift(lag)
            
            # Add rolling statistics
            for window in [5, 10, 20, 50]:
                features[f'{target_column}_mean_{window}'] = features[target_column].rolling(window).mean()
                features[f'{target_column}_std_{window}'] = features[target_column].rolling(window).std()
                features[f'{target_column}_min_{window}'] = features[target_column].rolling(window).min()
                features[f'{target_column}_max_{window}'] = features[target_column].rolling(window).max()
            
            # Add technical indicators if available
            if 'Volume' in features.columns:
                features['Volume_lag_1'] = features['Volume'].shift(1)
                features['Volume_mean_10'] = features['Volume'].rolling(10).mean()
            
            # Add returns
            features['Returns'] = features[target_column].pct_change()
            features['Returns_lag_1'] = features['Returns'].shift(1)
            
            # Add volatility
            features['Volatility_10'] = features['Returns'].rolling(10).std()
            features['Volatility_20'] = features['Returns'].rolling(20).std()
            
            # Drop rows with NaN values
            features = features.dropna()
            
            logger.log_crew_activity(
                "EnsemblePredictionTool",
                f"Created feature matrix with {features.shape[1]} features",
                "INFO"
            )
            
            return features
            
        except Exception as e:
            logger.log_crew_activity(
                "EnsemblePredictionTool",
                f"Feature creation failed: {str(e)}",
                "ERROR"
            )
            raise ModelTrainingException(f"Feature creation failed: {str(e)}")
    
    def train_ensemble(self, historical_data: pd.DataFrame, target_column: str = 'Close') -> Dict[str, Any]:
        """
        Train ensemble of prediction models
        
        Args:
            historical_data: Historical stock data
            target_column: Target column to predict
            
        Returns:
            Training results and metrics
        """
        try:
            # Create features
            features_df = self.create_features(historical_data, target_column)
            
            # Prepare target and feature matrices
            y = features_df[target_column].values
            X = features_df.drop(columns=[target_column]).values
            
            # Split into train/validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Initialize models
            model_configs = {}
            
            if self.ensemble_config['use_neural_prophet']:
                # NeuralProphet requires different data format
                np_wrapper = NeuralProphetWrapper()
                np_result = np_wrapper.train_model(historical_data, target_column)
                self.models['neural_prophet'] = np_wrapper
                model_configs['neural_prophet'] = np_result
            
            if self.ensemble_config['use_random_forest']:
                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                rf_model.fit(X_train, y_train)
                self.models['random_forest'] = rf_model
                model_configs['random_forest'] = {'trained': True}
            
            if self.ensemble_config['use_gradient_boosting']:
                gb_model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                gb_model.fit(X_train, y_train)
                self.models['gradient_boosting'] = gb_model
                model_configs['gradient_boosting'] = {'trained': True}
            
            if self.ensemble_config['use_linear_regression']:
                lr_model = LinearRegression()
                lr_model.fit(X_train, y_train)
                self.models['linear_regression'] = lr_model
                model_configs['linear_regression'] = {'trained': True}
            
            # Calculate model weights based on validation performance
            self.weights = self._calculate_model_weights(X_val, y_val, features_df.index[split_idx:])
            
            # Store ensemble history
            self.ensemble_history = {
                'training_data_shape': features_df.shape,
                'models_trained': list(self.models.keys()),
                'model_weights': self.weights,
                'training_period': (features_df.index.min(), features_df.index.max()),
                'ensemble_config': self.ensemble_config,
                'model_configs': model_configs
            }
            
            self.is_trained = True
            
            logger.log_crew_activity(
                "EnsemblePredictionTool",
                f"Ensemble training completed with {len(self.models)} models",
                "INFO"
            )
            
            return {
                'status': 'success',
                'ensemble_history': self.ensemble_history,
                'models_trained': len(self.models),
                'model_weights': self.weights
            }
            
        except Exception as e:
            logger.log_crew_activity(
                "EnsemblePredictionTool",
                f"Ensemble training failed: {str(e)}",
                "ERROR"
            )
            raise ModelTrainingException(f"Ensemble training failed: {str(e)}")
    
    def _calculate_model_weights(self, X_val: np.ndarray, y_val: np.ndarray, val_dates: pd.DatetimeIndex) -> Dict[str, float]:
        """
        Calculate model weights based on validation performance
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            val_dates: Validation dates for NeuralProphet
            
        Returns:
            Dictionary of model weights
        """
        try:
            weights = {}
            performance_scores = {}
            
            for model_name, model in self.models.items():
                try:
                    if model_name == 'neural_prophet':
                        # NeuralProphet predictions require different approach
                        val_df = pd.DataFrame(index=val_dates, data={'Close': y_val})
                        predictions = model.predict_prices(len(y_val))
                        y_pred = predictions['Predicted_Price'].values[:len(y_val)]
                    else:
                        # Scikit-learn models
                        y_pred = model.predict(X_val)
                    
                    # Calculate performance metrics
                    mae = mean_absolute_error(y_val, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                    
                    # Use inverse RMSE as performance score (higher is better)
                    performance_scores[model_name] = 1.0 / (rmse + 1e-8)
                    
                except Exception as e:
                    logger.log_crew_activity(
                        "EnsemblePredictionTool",
                        f"Performance calculation failed for {model_name}: {str(e)}",
                        "WARNING"
                    )
                    performance_scores[model_name] = 0.1  # Low weight for failed models
            
            # Normalize weights
            if self.ensemble_config['weight_method'] == 'equal':
                # Equal weights
                weight_value = 1.0 / len(performance_scores)
                weights = {name: weight_value for name in performance_scores.keys()}
            
            elif self.ensemble_config['weight_method'] == 'performance':
                # Performance-based weights
                total_score = sum(performance_scores.values())
                weights = {name: score / total_score for name, score in performance_scores.items()}
            
            else:  # adaptive
                # Adaptive weights with minimum threshold
                min_weight = 0.05
                adjusted_scores = {name: max(score, min_weight) for name, score in performance_scores.items()}
                total_score = sum(adjusted_scores.values())
                weights = {name: score / total_score for name, score in adjusted_scores.items()}
            
            return weights
            
        except Exception as e:
            logger.log_crew_activity(
                "EnsemblePredictionTool",
                f"Weight calculation failed: {str(e)}",
                "ERROR"
            )
            # Return equal weights as fallback
            return {name: 1.0 / len(self.models) for name in self.models.keys()}
    
    def predict_ensemble(self, horizon_days: int, last_data: pd.DataFrame, target_column: str = 'Close') -> pd.DataFrame:
        """
        Generate ensemble predictions
        
        Args:
            horizon_days: Number of days to forecast
            last_data: Recent data for feature generation
            target_column: Target column name
            
        Returns:
            DataFrame with ensemble predictions
        """
        try:
            if not self.is_trained:
                raise ModelTrainingException("Ensemble must be trained before making predictions")
            
            predictions = {}
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                try:
                    if model_name == 'neural_prophet':
                        # NeuralProphet predictions
                        model_pred = model.predict_prices(horizon_days)
                        predictions[model_name] = model_pred['Predicted_Price'].values
                    
                    else:
                        # Scikit-learn models need feature matrix
                        features_df = self.create_features(last_data, target_column)
                        X_last = features_df.drop(columns=[target_column]).iloc[-1:].values
                        
                        # Generate multi-step predictions
                        model_predictions = []
                        current_features = X_last.copy()
                        
                        for _ in range(horizon_days):
                            pred = model.predict(current_features)[0]
                            model_predictions.append(pred)
                            
                            # Update features for next prediction (simplified approach)
                            # In practice, you'd want more sophisticated feature updating
                            current_features = np.roll(current_features, -1)
                            current_features[0, -1] = pred
                        
                        predictions[model_name] = np.array(model_predictions)
                
                except Exception as e:
                    logger.log_crew_activity(
                        "EnsemblePredictionTool",
                        f"Prediction failed for {model_name}: {str(e)}",
                        "WARNING"
                    )
                    # Use fallback prediction
                    last_price = last_data[target_column].iloc[-1]
                    predictions[model_name] = np.full(horizon_days, last_price)
            
            # Combine predictions using weights
            ensemble_pred = np.zeros(horizon_days)
            for model_name, pred in predictions.items():
                weight = self.weights.get(model_name, 0.0)
                ensemble_pred += weight * pred[:horizon_days]
            
            # Create result DataFrame
            future_dates = pd.date_range(
                start=last_data.index[-1] + timedelta(days=1),
                periods=horizon_days,
                freq='D'
            )
            
            result_df = pd.DataFrame({
                'Date': future_dates,
                'Ensemble_Prediction': ensemble_pred
            })
            
            # Add individual model predictions
            for model_name, pred in predictions.items():
                result_df[f'{model_name}_prediction'] = pred[:horizon_days]
            
            # Add confidence intervals (simplified approach)
            pred_std = np.std([pred[:horizon_days] for pred in predictions.values()], axis=0)
            result_df['Lower_CI'] = ensemble_pred - 1.96 * pred_std
            result_df['Upper_CI'] = ensemble_pred + 1.96 * pred_std
            
            logger.log_crew_activity(
                "EnsemblePredictionTool",
                f"Generated ensemble predictions for {horizon_days} days",
                "INFO"
            )
            
            return result_df
            
        except Exception as e:
            logger.log_crew_activity(
                "EnsemblePredictionTool",
                f"Ensemble prediction failed: {str(e)}",
                "ERROR"
            )
            raise ModelTrainingException(f"Ensemble prediction failed: {str(e)}")
    
    def evaluate_ensemble(self, test_data: pd.DataFrame, target_column: str = 'Close') -> Dict[str, Any]:
        """
        Evaluate ensemble performance
        
        Args:
            test_data: Test dataset
            target_column: Target column name
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if not self.is_trained:
                raise ModelTrainingException("Ensemble must be trained before evaluation")
            
            # Generate predictions for test period
            test_predictions = self.predict_ensemble(
                horizon_days=len(test_data),
                last_data=test_data.iloc[:-len(test_data)],  # Use data before test period
                target_column=target_column
            )
            
            # Calculate metrics
            actual = test_data[target_column].values
            predicted = test_predictions['Ensemble_Prediction'].values
            
            # Ensure same length
            min_len = min(len(actual), len(predicted))
            actual = actual[:min_len]
            predicted = predicted[:min_len]
            
            # Calculate ensemble metrics
            ensemble_metrics = {
                'MAE': mean_absolute_error(actual, predicted),
                'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
                'MAPE': np.mean(np.abs((actual - predicted) / actual)) * 100,
                'R2': r2_score(actual, predicted),
                'Accuracy': 100 - np.mean(np.abs((actual - predicted) / actual)) * 100
            }
            
            # Calculate individual model metrics
            individual_metrics = {}
            for model_name in self.models.keys():
                if f'{model_name}_prediction' in test_predictions.columns:
                    model_pred = test_predictions[f'{model_name}_prediction'].values[:min_len]
                    individual_metrics[model_name] = {
                        'MAE': mean_absolute_error(actual, model_pred),
                        'RMSE': np.sqrt(mean_squared_error(actual, model_pred)),
                        'MAPE': np.mean(np.abs((actual - model_pred) / actual)) * 100,
                        'R2': r2_score(actual, model_pred)
                    }
            
            logger.log_crew_activity(
                "EnsemblePredictionTool", f"Ensemble evaluation completed - RMSE: {ensemble_metrics['RMSE']:.4f}",
                "INFO"
            )
            
            return {
                'ensemble_metrics': ensemble_metrics,
                'individual_metrics': individual_metrics,
                'model_weights': self.weights,
                'evaluation_summary': {
                    'best_individual_model': min(individual_metrics.keys(), 
                                               key=lambda x: individual_metrics[x]['RMSE']),
                    'ensemble_improvement': self._calculate_improvement(ensemble_metrics, individual_metrics)
                }
            }
            
        except Exception as e:
            logger.log_crew_activity(
                "EnsemblePredictionTool",
                f"Ensemble evaluation failed: {str(e)}",
                "ERROR"
            )
            return {'error': str(e)}
    
    def _calculate_improvement(self, ensemble_metrics: Dict[str, float], 
                             individual_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate improvement of ensemble over individual models
        
        Args:
            ensemble_metrics: Ensemble performance metrics
            individual_metrics: Individual model metrics
            
        Returns:
            Dictionary of improvement percentages
        """
        try:
            improvements = {}
            
            # Find best individual model RMSE
            best_individual_rmse = min(metrics['RMSE'] for metrics in individual_metrics.values())
            ensemble_rmse = ensemble_metrics['RMSE']
            
            improvements['RMSE_improvement'] = ((best_individual_rmse - ensemble_rmse) / best_individual_rmse) * 100
            
            # Find best individual model MAE
            best_individual_mae = min(metrics['MAE'] for metrics in individual_metrics.values())
            ensemble_mae = ensemble_metrics['MAE']
            
            improvements['MAE_improvement'] = ((best_individual_mae - ensemble_mae) / best_individual_mae) * 100
            
            # Find best individual model R2
            best_individual_r2 = max(metrics['R2'] for metrics in individual_metrics.values())
            ensemble_r2 = ensemble_metrics['R2']
            
            improvements['R2_improvement'] = ((ensemble_r2 - best_individual_r2) / abs(best_individual_r2)) * 100
            
            return improvements
            
        except Exception as e:
            logger.log_crew_activity(
                "EnsemblePredictionTool",
                f"Improvement calculation failed: {str(e)}",
                "WARNING"
            )
            return {}
    
    def get_model_insights(self) -> Dict[str, Any]:
        """
        Get insights about the ensemble model
        
        Returns:
            Dictionary of model insights
        """
        try:
            if not self.is_trained:
                return {'error': 'Ensemble not trained'}
            
            insights = {
                'model_weights': self.weights,
                'dominant_model': max(self.weights.keys(), key=lambda x: self.weights[x]),
                'model_diversity': self._calculate_model_diversity(),
                'ensemble_stability': self._calculate_ensemble_stability(),
                'feature_importance': self._get_ensemble_feature_importance()
            }
            
            return insights
            
        except Exception as e:
            logger.log_crew_activity(
                "EnsemblePredictionTool",
                f"Model insights generation failed: {str(e)}",
                "ERROR"
            )
            return {'error': str(e)}
    
    def _calculate_model_diversity(self) -> float:
        """
        Calculate diversity among ensemble models
        
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        try:
            # Calculate entropy of weights as diversity measure
            weights_array = np.array(list(self.weights.values()))
            weights_array = weights_array / np.sum(weights_array)  # Normalize
            
            # Calculate entropy
            entropy = -np.sum(weights_array * np.log(weights_array + 1e-8))
            max_entropy = np.log(len(weights_array))
            
            diversity = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return float(diversity)
            
        except Exception as e:
            return 0.0
    
    def _calculate_ensemble_stability(self) -> float:
        """
        Calculate stability of ensemble predictions
        
        Returns:
            Stability score (0-1, higher is more stable)
        """
        try:
            # Use weight distribution as proxy for stability
            weights_array = np.array(list(self.weights.values()))
            
            # Calculate coefficient of variation (lower is more stable)
            cv = np.std(weights_array) / (np.mean(weights_array) + 1e-8)
            
            # Convert to stability score (higher is better)
            stability = 1.0 / (1.0 + cv)
            
            return float(stability)
            
        except Exception as e:
            return 0.0
    
    def _get_ensemble_feature_importance(self) -> Dict[str, float]:
        """
        Get combined feature importance from ensemble models
        
        Returns:
            Dictionary of feature importance scores
        """
        try:
            combined_importance = {}
            
            for model_name, model in self.models.items():
                weight = self.weights.get(model_name, 0.0)
                
                if model_name == 'neural_prophet':
                    # Get NeuralProphet feature importance
                    np_importance = model.get_feature_importance()
                    for feature, importance in np_importance.items():
                        combined_importance[feature] = combined_importance.get(feature, 0.0) + weight * importance
                
                elif hasattr(model, 'feature_importances_'):
                    # Scikit-learn models with feature importance
                    importances = model.feature_importances_
                    for i, importance in enumerate(importances):
                        feature_name = f'feature_{i}'
                        combined_importance[feature_name] = combined_importance.get(feature_name, 0.0) + weight * importance
            
            return combined_importance
            
        except Exception as e:
            logger.log_crew_activity(
                "EnsemblePredictionTool",
                f"Feature importance calculation failed: {str(e)}",
                "WARNING"
            )
            return {}


class ModelValidationTool:
    """
    Tool for comprehensive model validation and backtesting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model validation tool
        
        Args:
            config: Configuration dictionary for validation parameters
        """
        self.config = config or {}
        self.validation_results = {}
        
        # Default validation configuration
        self.default_config = {
            'cv_folds': 5,
            'test_size': 0.2,
            'rolling_window_size': 252,  # Trading days in a year
            'walk_forward_steps': 30,
            'min_train_size': 100,
            'metrics': ['MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']
        }
        
        # Update with provided config
        self.validation_config = {**self.default_config, **self.config}
        
        logger.log_crew_activity(
            "ModelValidationTool",
            "Model validation tool initialized",
            "INFO"
        )
    
    def time_series_cross_validation(self, model, data: pd.DataFrame, 
                                   target_column: str = 'Close') -> Dict[str, Any]:
        """
        Perform time series cross-validation
        
        Args:
            model: Model object to validate
            data: Historical data
            target_column: Target column name
            
        Returns:
            Cross-validation results
        """
        try:
            cv_results = {
                'fold_metrics': [],
                'overall_metrics': {},
                'fold_predictions': []
            }
            
            # Prepare data
            n_samples = len(data)
            fold_size = n_samples // self.validation_config['cv_folds']
            min_train = self.validation_config['min_train_size']
            
            for fold in range(self.validation_config['cv_folds']):
                try:
                    # Define train/test split for this fold
                    test_start = min_train + fold * fold_size
                    test_end = min(test_start + fold_size, n_samples)
                    
                    if test_end - test_start < 10:  # Skip if test set too small
                        continue
                    
                    train_data = data.iloc[:test_start]
                    test_data = data.iloc[test_start:test_end]
                    
                    # Train model on fold data
                    if hasattr(model, 'train_model'):
                        model.train_model(train_data, target_column)
                    elif hasattr(model, 'train_ensemble'):
                        model.train_ensemble(train_data, target_column)
                    else:
                        # Assume scikit-learn style model
                        features_df = self._create_basic_features(train_data, target_column)
                        X_train = features_df.drop(columns=[target_column]).values
                        y_train = features_df[target_column].values
                        model.fit(X_train, y_train)
                    
                    # Generate predictions
                    if hasattr(model, 'predict_prices'):
                        predictions = model.predict_prices(len(test_data))
                        y_pred = predictions['Predicted_Price'].values
                    elif hasattr(model, 'predict_ensemble'):
                        predictions = model.predict_ensemble(len(test_data), train_data, target_column)
                        y_pred = predictions['Ensemble_Prediction'].values
                    else:
                        # Scikit-learn style prediction
                        test_features = self._create_basic_features(test_data, target_column)
                        X_test = test_features.drop(columns=[target_column]).values
                        y_pred = model.predict(X_test)
                    
                    y_actual = test_data[target_column].values
                    
                    # Ensure same length
                    min_len = min(len(y_actual), len(y_pred))
                    y_actual = y_actual[:min_len]
                    y_pred = y_pred[:min_len]
                    
                    # Calculate metrics for this fold
                    fold_metrics = self._calculate_metrics(y_actual, y_pred)
                    fold_metrics['fold'] = fold
                    fold_metrics['train_size'] = len(train_data)
                    fold_metrics['test_size'] = min_len
                    
                    cv_results['fold_metrics'].append(fold_metrics)
                    cv_results['fold_predictions'].append({
                        'fold': fold,
                        'actual': y_actual,
                        'predicted': y_pred,
                        'dates': test_data.index[:min_len]
                    })
                    
                except Exception as e:
                    logger.log_crew_activity(
                        "ModelValidationTool",
                        f"Fold {fold} validation failed: {str(e)}",
                        "WARNING"
                    )
                    continue
            
            # Calculate overall metrics
            if cv_results['fold_metrics']:
                cv_results['overall_metrics'] = self._aggregate_cv_metrics(cv_results['fold_metrics'])
            
            logger.log_crew_activity(
                "ModelValidationTool",
                f"Time series CV completed with {len(cv_results['fold_metrics'])} folds",
                "INFO"
            )
            
            return cv_results
            
        except Exception as e:
            logger.log_crew_activity(
                "ModelValidationTool",
                f"Time series cross-validation failed: {str(e)}",
                "ERROR"
            )
            return {'error': str(e)}
    
    def walk_forward_validation(self, model, data: pd.DataFrame, 
                               target_column: str = 'Close') -> Dict[str, Any]:
        """
        Perform walk-forward validation
        
        Args:
            model: Model object to validate
            data: Historical data
            target_column: Target column name
            
        Returns:
            Walk-forward validation results
        """
        try:
            wf_results = {
                'step_metrics': [],
                'overall_metrics': {},
                'predictions_timeline': []
            }
            
            window_size = self.validation_config['rolling_window_size']
            step_size = self.validation_config['walk_forward_steps']
            
            # Start with minimum required training data
            start_idx = window_size
            
            while start_idx + step_size < len(data):
                try:
                    # Define training and testing windows
                    train_end = start_idx
                    test_start = start_idx
                    test_end = min(start_idx + step_size, len(data))
                    
                    train_data = data.iloc[train_end - window_size:train_end]
                    test_data = data.iloc[test_start:test_end]
                    
                    # Train model on current window
                    if hasattr(model, 'train_model'):
                        model.train_model(train_data, target_column)
                    elif hasattr(model, 'train_ensemble'):
                        model.train_ensemble(train_data, target_column)
                    
                    # Generate predictions for next period
                    if hasattr(model, 'predict_prices'):
                        predictions = model.predict_prices(len(test_data))
                        y_pred = predictions['Predicted_Price'].values
                    elif hasattr(model, 'predict_ensemble'):
                        predictions = model.predict_ensemble(len(test_data), train_data, target_column)
                        y_pred = predictions['Ensemble_Prediction'].values
                    
                    y_actual = test_data[target_column].values
                    
                    # Calculate metrics for this step
                    step_metrics = self._calculate_metrics(y_actual, y_pred)
                    step_metrics['step'] = len(wf_results['step_metrics'])
                    step_metrics['train_period'] = (train_data.index[0], train_data.index[-1])
                    step_metrics['test_period'] = (test_data.index[0], test_data.index[-1])
                    
                    wf_results['step_metrics'].append(step_metrics)
                    wf_results['predictions_timeline'].append({
                        'step': step_metrics['step'],
                        'dates': test_data.index,
                        'actual': y_actual,
                        'predicted': y_pred
                    })
                    
                    # Move to next step
                    start_idx += step_size
                    
                except Exception as e:
                    logger.log_crew_activity(
                        "ModelValidationTool",
                        f"Walk-forward step failed: {str(e)}",
                        "WARNING"
                    )
                    start_idx += step_size
                    continue
            
            # Calculate overall metrics
            if wf_results['step_metrics']:
                wf_results['overall_metrics'] = self._aggregate_wf_metrics(wf_results['step_metrics'])
            
            logger.log_crew_activity(
                "ModelValidationTool",
                f"Walk-forward validation completed with {len(wf_results['step_metrics'])} steps",
                "INFO"
            )
            
            return wf_results
            
        except Exception as e:
            logger.log_crew_activity(
                "ModelValidationTool",
                f"Walk-forward validation failed: {str(e)}",
                "ERROR"
            )
            return {'error': str(e)}
    
    def _calculate_metrics(self, y_actual: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Args:
            y_actual: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        try:
            metrics = {}
            
            # Basic regression metrics
            metrics['MAE'] = mean_absolute_error(y_actual, y_pred)
            metrics['RMSE'] = np.sqrt(mean_squared_error(y_actual, y_pred))
            metrics['MAPE'] = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
            metrics['R2'] = r2_score(y_actual, y_pred)
            
            # Directional accuracy
            actual_direction = np.diff(y_actual) > 0
            pred_direction = np.diff(y_pred) > 0
            if len(actual_direction) > 0:
                metrics['Directional_Accuracy'] = np.mean(actual_direction == pred_direction) * 100
            else:
                metrics['Directional_Accuracy'] = 0.0
            
            # Additional financial metrics
            returns_actual = np.diff(y_actual) / y_actual[:-1]
            returns_pred = np.diff(y_pred) / y_actual[:-1]  # Use actual prices for realistic returns
            
            if len(returns_actual) > 0:
                metrics['Return_Correlation'] = np.corrcoef(returns_actual, returns_pred)[0, 1]
                metrics['Volatility_Ratio'] = np.std(returns_pred) / np.std(returns_actual)
            else:
                metrics['Return_Correlation'] = 0.0
                metrics['Volatility_Ratio'] = 1.0
            
            return metrics
            
        except Exception as e:
            logger.log_crew_activity(
                "ModelValidationTool",
                f"Metrics calculation failed: {str(e)}",
                "WARNING"
            )
            return {'error': str(e)}
    
    def _aggregate_cv_metrics(self, fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate cross-validation metrics
        
        Args:
            fold_metrics: List of metrics from each fold
            
        Returns:
            Aggregated metrics
        """
        try:
            aggregated = {}
            
            # Get all metric names (excluding non-numeric fields)
            metric_names = [key for key in fold_metrics[0].keys() 
                          if key not in ['fold', 'train_size', 'test_size', 'train_period', 'test_period']]
            
            for metric in metric_names:
                values = [fold[metric] for fold in fold_metrics if metric in fold and not isinstance(fold[metric], str)]
                if values:
                    aggregated[f'{metric}_mean'] = np.mean(values)
                    aggregated[f'{metric}_std'] = np.std(values)
                    aggregated[f'{metric}_min'] = np.min(values)
                    aggregated[f'{metric}_max'] = np.max(values)
            
            aggregated['n_folds'] = len(fold_metrics)
            
            return aggregated
            
        except Exception as e:
            logger.log_crew_activity(
                "ModelValidationTool",
                f"CV metrics aggregation failed: {str(e)}",
                "WARNING"
            )
            return {}
    
    def _aggregate_wf_metrics(self, step_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate walk-forward metrics
        
        Args:
            step_metrics: List of metrics from each step
            
        Returns:
            Aggregated metrics
        """
        try:
            aggregated = {}
            
            # Get all metric names (excluding non-numeric fields)
            metric_names = [key for key in step_metrics[0].keys() 
                          if key not in ['step', 'train_period', 'test_period']]
            
            for metric in metric_names:
                values = [step[metric] for step in step_metrics if metric in step and not isinstance(step[metric], str)]
                if values:
                    aggregated[f'{metric}_mean'] = np.mean(values)
                    aggregated[f'{metric}_std'] = np.std(values)
                    aggregated[f'{metric}_trend'] = self._calculate_trend(values)
            
            aggregated['n_steps'] = len(step_metrics)
            
            return aggregated
            
        except Exception as e:
            logger.log_crew_activity(
                "ModelValidationTool",
                f"Walk-forward metrics aggregation failed: {str(e)}",
                "WARNING"
            )
            return {}
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend in metric values over time
        
        Args:
            values: List of metric values
            
        Returns:
            Trend coefficient
        """
        try:
            if len(values) < 2:
                return 0.0
            
            x = np.arange(len(values))
            y = np.array(values)
            
            # Calculate linear regression slope
            slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
            
            return float(slope)
            
        except Exception as e:
            return 0.0
    
    def _create_basic_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Create basic features for scikit-learn models
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            
        Returns:
            DataFrame with basic features
        """
        try:
            features = df.copy()
            
            # Add simple lagged features
            for lag in [1, 2, 3, 5]:
                features[f'{target_column}_lag_{lag}'] = features[target_column].shift(lag)
            
            # Add moving averages
            for window in [5, 10, 20]:
                features[f'{target_column}_ma_{window}'] = features[target_column].rolling(window).mean()
            
            # Add returns
            features['returns'] = features[target_column].pct_change()
            
            # Drop rows with NaN
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.log_crew_activity(
                "ModelValidationTool",
                f"Basic feature creation failed: {str(e)}",
                "WARNING"
            )
            return df


# Export main classes
__all__ = [
    'NeuralProphetWrapper',
    'EnsemblePredictionTool', 
    'ModelValidationTool'
]
