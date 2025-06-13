"""
Visualization Tools for Stock Performance Monitoring
Provides comprehensive charting and visualization capabilities using Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import json


class PlotlyToolKit:
    """
    Comprehensive visualization toolkit using Plotly
    Creates interactive charts for stock analysis dashboard
    """
    
    def __init__(self, theme: str = "plotly_dark"):
        self.theme = theme
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'bullish': '#00ff88',
            'bearish': '#ff4444',
            'neutral': '#888888'
        }
    
    def create_price_chart(self, data: pd.DataFrame, predictions: Optional[pd.DataFrame] = None, 
                          technical_indicators: Optional[Dict] = None, symbol: str = "Stock") -> str:
        """
        Create comprehensive price chart with predictions and technical indicators
        
        Args:
            data: Historical OHLCV data
            predictions: NeuralProphet prediction data
            technical_indicators: Technical analysis data
            symbol: Stock symbol for chart title
            
        Returns:
            Plotly chart JSON string
        """
        # Create subplot with secondary y-axis for volume
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxis=True,
            vertical_spacing=0.05,
            subplot_titles=[f'{symbol} Price & Predictions', 'Volume', 'RSI', 'MACD'],
            row_heights=[0.5, 0.2, 0.15, 0.15],
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Main price candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=f'{symbol} Price',
                increasing_line_color=self.colors['bullish'],
                decreasing_line_color=self.colors['bearish']
            ),
            row=1, col=1
        )
        
        # Add Bollinger Bands if available
        if technical_indicators and 'bollinger_bands' in technical_indicators:
            bb = technical_indicators['bollinger_bands']
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=bb['upper_band'],
                    name='BB Upper',
                    line=dict(color='rgba(173, 204, 255, 0.5)', width=1),
                    fill=None
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=bb['lower_band'],
                    name='BB Lower',
                    line=dict(color='rgba(173, 204, 255, 0.5)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(173, 204, 255, 0.1)'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=bb['middle_band'],
                    name='BB Middle',
                    line=dict(color='rgba(173, 204, 255, 0.8)', width=1, dash='dash')
                ),
                row=1, col=1
            )
        
        # Add predictions if available
        if predictions is not None:
            fig.add_trace(
                go.Scatter(
                    x=predictions.index,
                    y=predictions['yhat'],
                    name='Prediction',
                    line=dict(color=self.colors['warning'], width=2, dash='dash'),
                    mode='lines'
                ),
                row=1, col=1
            )
            
            # Add confidence intervals
            if 'yhat_lower' in predictions.columns and 'yhat_upper' in predictions.columns:
                fig.add_trace(
                    go.Scatter(
                        x=predictions.index,
                        y=predictions['yhat_upper'],
                        name='Upper Confidence',
                        line=dict(color='rgba(255, 127, 14, 0.3)', width=0),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=predictions.index,
                        y=predictions['yhat_lower'],
                        name='Confidence Interval',
                        line=dict(color='rgba(255, 127, 14, 0.3)', width=0),
                        fill='tonexty',
                        fillcolor='rgba(255, 127, 14, 0.2)'
                    ),
                    row=1, col=1
                )
        
        # Volume chart
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(data['Close'], data['Open'])]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # RSI chart
        if technical_indicators and 'rsi' in technical_indicators:
            rsi_data = technical_indicators['rsi']['values']
            fig.add_trace(
                go.Scatter(
                    x=rsi_data.index,
                    y=rsi_data,
                    name='RSI',
                    line=dict(color=self.colors['primary'], width=2)
                ),
                row=3, col=1
            )
            # Add RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)
        
        # MACD chart
        if technical_indicators and 'macd' in technical_indicators:
            macd_data = technical_indicators['macd']
            fig.add_trace(
                go.Scatter(
                    x=macd_data['macd_line'].index,
                    y=macd_data['macd_line'],
                    name='MACD',
                    line=dict(color=self.colors['primary'], width=2)
                ),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=macd_data['signal_line'].index,
                    y=macd_data['signal_line'],
                    name='Signal',
                    line=dict(color=self.colors['secondary'], width=2)
                ),
                row=4, col=1
            )
            # MACD histogram
            colors = ['green' if val >= 0 else 'red' for val in macd_data['histogram']]
            fig.add_trace(
                go.Bar(
                    x=macd_data['histogram'].index,
                    y=macd_data['histogram'],
                    name='Histogram',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Technical Analysis Dashboard',
            template=self.theme,
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=4, col=1)
        
        return fig.to_json()
    
    def create_sentiment_timeline(self, sentiment_data: pd.DataFrame, symbol: str = "Stock") -> str:
        """
        Create sentiment analysis timeline visualization
        
        Args:
            sentiment_data: DataFrame with sentiment scores over time
            symbol: Stock symbol for chart title
            
        Returns:
            Plotly chart JSON string
        """
        fig = go.Figure()
        
        # Sentiment line
        fig.add_trace(
            go.Scatter(
                x=sentiment_data.index,
                y=sentiment_data['sentiment_score'],
                mode='lines+markers',
                name='Sentiment Score',
                line=dict(color=self.colors['primary'], width=2),
                marker=dict(size=6),
                fill='tonexty'
            )
        )
        
        # Add sentiment zones
        fig.add_hline(y=0.5, line_dash="dash", line_color="green", opacity=0.5, 
                     annotation_text="Positive Threshold")
        fig.add_hline(y=-0.5, line_dash="dash", line_color="red", opacity=0.5,
                     annotation_text="Negative Threshold")
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3,
                     annotation_text="Neutral")
        
        # Add volume of news if available
        if 'news_volume' in sentiment_data.columns:
            fig.add_trace(
                go.Bar(
                    x=sentiment_data.index,
                    y=sentiment_data['news_volume'],
                    name='News Volume',
                    yaxis='y2',
                    opacity=0.3,
                    marker_color=self.colors['info']
                )
            )
        
        fig.update_layout(
            title=f'{symbol} Sentiment Analysis Timeline',
            template=self.theme,
            height=400,
            yaxis=dict(title='Sentiment Score', range=[-1, 1]),
            yaxis2=dict(title='News Volume', overlaying='y', side='right'),
            hovermode='x unified'
        )
        
        return fig.to_json()
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame, 
                                 title: str = "Stock Correlation Matrix") -> str:
        """
        Create correlation heatmap for multiple stocks
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            title: Chart title
            
        Returns:
            Plotly chart JSON string
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=500,
            width=600
        )
        
        return fig.to_json()
    
    def create_performance_comparison(self, performance_data: Dict[str, pd.Series], 
                                   title: str = "Stock Performance Comparison") -> str:
        """
        Create multi-stock performance comparison chart
        
        Args:
            performance_data: Dictionary of stock symbols and their return series
            title: Chart title
            
        Returns:
            Plotly chart JSON string
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, (symbol, returns) in enumerate(performance_data.items()):
            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod() - 1
            
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns * 100,
                    name=symbol,
                    line=dict(color=colors[i % len(colors)], width=2),
                    mode='lines'
                )
            )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=500,
            yaxis_title="Cumulative Returns (%)",
            xaxis_title="Date",
            hovermode='x unified'
        )
        
        return fig.to_json()
    
    def create_risk_return_scatter(self, risk_return_data: pd.DataFrame,
                                 title: str = "Risk-Return Analysis") -> str:
        """
        Create risk-return scatter plot
        
        Args:
            risk_return_data: DataFrame with columns: symbol, return, risk, market_cap
            title: Chart title
            
        Returns:
            Plotly chart JSON string
        """
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=risk_return_data['risk'] * 100,
                y=risk_return_data['return'] * 100,
                mode='markers+text',
                text=risk_return_data['symbol'],
                textposition="top center",
                marker=dict(
                    size=risk_return_data.get('market_cap', [50] * len(risk_return_data)),
                    sizemode='diameter',
                    sizeref=2. * max(risk_return_data.get('market_cap', [1])) / (40.**2),
                    sizemin=4,
                    color=risk_return_data['return'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Returns")
                ),
                hovertemplate="<b>%{text}</b><br>" +
                            "Risk: %{x:.2f}%<br>" +
                            "Return: %{y:.2f}%<br>" +
                            "<extra></extra>"
            )
        )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=500,
            xaxis_title="Risk (Volatility %)",
            yaxis_title="Returns (%)"
        )
        
        return fig.to_json()
    
    def create_health_score_gauge(self, health_score: float, symbol: str) -> str:
        """
        Create health score gauge chart
        
        Args:
            health_score: Health score (0-100)
            symbol: Stock symbol
            
        Returns:
            Plotly chart JSON string
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{symbol} Health Score"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "gray"},
                    {'range': [50, 75], 'color': "lightgreen"},
                    {'range': [75, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            template=self.theme,
            height=400,
            width=400
        )
        
        return fig.to_json()
    
    def create_volume_profile(self, data: pd.DataFrame, symbol: str) -> str:
        """
        Create volume profile chart
        
        Args:
            data: OHLCV data
            symbol: Stock symbol
            
        Returns:
            Plotly chart JSON string
        """
        # Calculate price bins and volume distribution
        price_range = np.linspace(data['Low'].min(), data['High'].max(), 50)
        volume_profile = []
        
        for i in range(len(price_range) - 1):
            mask = (data['Close'] >= price_range[i]) & (data['Close'] < price_range[i + 1])
            volume_at_price = data[mask]['Volume'].sum()
            volume_profile.append(volume_at_price)
        
        price_centers = (price_range[:-1] + price_range[1:]) / 2
        
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.7, 0.3],
            shared_yaxes=True,
            subplot_titles=[f'{symbol} Price Chart', 'Volume Profile']
        )
        
        # Price candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=f'{symbol} Price'
            ),
            row=1, col=1
        )
        
        # Volume profile
        fig.add_trace(
            go.Bar(
                y=price_centers,
                x=volume_profile,
                orientation='h',
                name='Volume Profile',
                marker_color=self.colors['info'],
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'{symbol} Volume Profile Analysis',
            template=self.theme,
            height=600,
            showlegend=False
        )
        
        return fig.to_json()
    
    def create_earnings_impact_chart(self, price_data: pd.DataFrame, 
                                   earnings_dates: List[datetime],
                                   symbol: str) -> str:
        """
        Create earnings impact visualization
        
        Args:
            price_data: Stock price data
            earnings_dates: List of earnings announcement dates
            symbol: Stock symbol
            
        Returns:
            Plotly chart JSON string
        """
        fig = go.Figure()
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['Close'],
                mode='lines',
                name=f'{symbol} Price',
                line=dict(color=self.colors['primary'], width=2)
            )
        )
        
        # Add earnings date markers
        for date in earnings_dates:
            if date in price_data.index:
                fig.add_vline(
                    x=date,
                    line_dash="dash",
                    line_color="red",
                    opacity=0.7,
                    annotation_text="Earnings",
                    annotation_position="top"
                )
        
        fig.update_layout(
            title=f'{symbol} Price Movement Around Earnings',
            template=self.theme,
            height=400,
            yaxis_title="Price ($)",
            hovermode='x unified'
        )
        
        return fig.to_json()


class DashboardComposer:
    """
    Compose multiple charts into comprehensive dashboard layout
    """
    
    def __init__(self):
        self.toolkit = PlotlyToolKit()
    
    def create_comprehensive_dashboard(self, stock_data: Dict[str, Any], 
                                     symbol: str) -> Dict[str, str]:
        """
        Create complete dashboard with all relevant charts
        
        Args:
            stock_data: Complete stock analysis data
            symbol: Stock symbol
            
        Returns:
            Dictionary of chart names and their JSON representations
        """
        dashboard_charts = {}
        
        # Main price chart
        if 'price_data' in stock_data:
            dashboard_charts['price_chart'] = self.toolkit.create_price_chart(
                data=stock_data['price_data'],
                predictions=stock_data.get('predictions'),
                technical_indicators=stock_data.get('technical_indicators'),
                symbol=symbol
            )
        
        # Sentiment timeline
        if 'sentiment_data' in stock_data:
            dashboard_charts['sentiment_chart'] = self.toolkit.create_sentiment_timeline(
                sentiment_data=stock_data['sentiment_data'],
                symbol=symbol
            )
        
        # Health score gauge
        if 'health_score' in stock_data:
            dashboard_charts['health_gauge'] = self.toolkit.create_health_score_gauge(
                health_score=stock_data['health_score'],
                symbol=symbol
            )
        
        # Volume profile
        if 'price_data' in stock_data:
            dashboard_charts['volume_profile'] = self.toolkit.create_volume_profile(
                data=stock_data['price_data'],
                symbol=symbol
            )
        
        return dashboard_charts
    
    def create_multi_stock_dashboard(self, multi_stock_data: Dict[str, Dict]) -> Dict[str, str]:
        """
        Create dashboard for multiple stock comparison
        
        Args:
            multi_stock_data: Dictionary of stock symbols and their data
            
        Returns:
            Dictionary of comparative charts
        """
        dashboard_charts = {}
        
        # Extract performance data
        performance_data = {}
        risk_return_data = []
        
        for symbol, data in multi_stock_data.items():
            if 'returns' in data:
                performance_data[symbol] = data['returns']
            
            if 'risk' in data and 'return' in data:
                risk_return_data.append({
                    'symbol': symbol,
                    'risk': data['risk'],
                    'return': data['return'],
                    'market_cap': data.get('market_cap', 1000)
                })
        
        # Performance comparison
        if performance_data:
            dashboard_charts['performance_comparison'] = self.toolkit.create_performance_comparison(
                performance_data=performance_data
            )
        
        # Risk-return scatter
        if risk_return_data:
            risk_return_df = pd.DataFrame(risk_return_data)
            dashboard_charts['risk_return_scatter'] = self.toolkit.create_risk_return_scatter(
                risk_return_data=risk_return_df
            )
        
        # Correlation heatmap
        if len(multi_stock_data) > 1:
            # Create correlation matrix from returns
            returns_df = pd.DataFrame(performance_data)
            correlation_matrix = returns_df.corr()
            dashboard_charts['correlation_heatmap'] = self.toolkit.create_correlation_heatmap(
                correlation_matrix=correlation_matrix
            )
        
        return dashboard_charts


def create_all_visualizations(analysis_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Create all visualizations for the stock analysis dashboard
    
    Args:
        analysis_data: Complete analysis results from crews
        
    Returns:
        Dictionary of all chart JSON strings for frontend
    """
    composer = DashboardComposer()
    all_charts = {}
    
    # Single stock dashboards
    for symbol, stock_data in analysis_data.get('individual_stocks', {}).items():
        stock_charts = composer.create_comprehensive_dashboard(stock_data, symbol)
        all_charts.update({f"{symbol}_{chart_name}": chart_json 
                          for chart_name, chart_json in stock_charts.items()})
    
    # Multi-stock comparative dashboard
    if len(analysis_data.get('individual_stocks', {})) > 1:
        comparative_charts = composer.create_multi_stock_dashboard(
            analysis_data['individual_stocks']
        )
        all_charts.update(comparative_charts)
    
    return all_charts