# backend_complete.py - Complete McKinsey-Style Stock Analyzer with Missing Components
import asyncio
import sqlite3
import logging
import json
import os
import redis
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import yfinance as yf
from transformers import pipeline
import chromadb
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from neuralprophet import NeuralProphet
import plotly.graph_objects as go
import plotly.express as px
from firecrawl import FirecrawlApp
import openai
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from flask import Flask, jsonify, request, websocket
from flask_socketio import SocketIO, emit
from celery import Celery
import time
import threading
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Configuration for stock analysis"""
    stocks: List[str]
    period_days: int = 90
    prediction_horizon: int = 30
    openai_api_key: str = ""
    firecrawl_api_key: str = ""
    redis_url: str = "redis://localhost:6379"
    enable_realtime: bool = True

# =============================================================================
# MISSING COMPONENTS IMPLEMENTATION
# =============================================================================

class RateLimiter:
    """Rate limiting for API calls"""
    
    def __init__(self, redis_client, max_requests: int = 100, window_seconds: int = 3600):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        try:
            current_time = int(time.time())
            window_start = current_time - self.window_seconds
            
            # Clean old entries
            self.redis.zremrangebyscore(f"rate_limit:{identifier}", 0, window_start)
            
            # Count current requests
            current_count = self.redis.zcard(f"rate_limit:{identifier}")
            
            if current_count < self.max_requests:
                # Add current request
                self.redis.zadd(f"rate_limit:{identifier}", {current_time: current_time})
                self.redis.expire(f"rate_limit:{identifier}", self.window_seconds)
                return True
            
            return False
        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            return True  # Allow on error

class SharedMemory:
    """Session-wide shared memory for crew communication"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def set_data(self, session_id: str, key: str, data: Any):
        """Store data in shared memory"""
        try:
            full_key = f"session:{session_id}:{key}"
            self.redis.setex(full_key, 3600, json.dumps(data, default=str))
        except Exception as e:
            logger.error(f"Error setting shared memory: {e}")
    
    def get_data(self, session_id: str, key: str) -> Any:
        """Retrieve data from shared memory"""
        try:
            full_key = f"session:{session_id}:{key}"
            data = self.redis.get(full_key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error getting shared memory: {e}")
            return None
    
    def get_all_session_data(self, session_id: str) -> Dict[str, Any]:
        """Get all data for a session"""
        try:
            pattern = f"session:{session_id}:*"
            keys = self.redis.keys(pattern)
            data = {}
            for key in keys:
                key_name = key.decode().split(':')[-1]
                value = self.redis.get(key)
                data[key_name] = json.loads(value) if value else None
            return data
        except Exception as e:
            logger.error(f"Error getting session data: {e}")
            return {}

class WebSocketHandler:
    """Real-time WebSocket communication"""
    
    def __init__(self, socketio):
        self.socketio = socketio
        self.active_sessions = {}
    
    def emit_update(self, session_id: str, event: str, data: Dict[str, Any]):
        """Emit update to specific session"""
        try:
            self.socketio.emit(event, data, room=session_id)
            logger.info(f"Emitted {event} to session {session_id}")
        except Exception as e:
            logger.error(f"Error emitting update: {e}")
    
    def broadcast_market_update(self, data: Dict[str, Any]):
        """Broadcast market updates to all sessions"""
        try:
            self.socketio.emit('market_update', data, broadcast=True)
        except Exception as e:
            logger.error(f"Error broadcasting market update: {e}")

class HealthScoreSynthesizer(BaseTool):
    """Calculate comprehensive health score"""
    name: str = "Health Score Synthesizer"
    description: str = "Calculates composite health score from multiple indicators"
    
    def _run(self, technical_data: Dict, sentiment_data: Dict, price_data: Dict) -> Dict[str, Any]:
        try:
            # Technical indicators score (0-40 points)
            tech_score = 0
            if technical_data.get('rsi'):
                rsi = technical_data['rsi']
                if 30 <= rsi <= 70:  # Neutral zone
                    tech_score += 10
                elif rsi < 30:  # Oversold (potentially good buy)
                    tech_score += 15
                else:  # Overbought
                    tech_score += 5
            
            if technical_data.get('macd') and technical_data.get('macd') > 0:
                tech_score += 10
            
            if technical_data.get('sma_20') and technical_data.get('sma_50'):
                if technical_data['sma_20'] > technical_data['sma_50']:
                    tech_score += 10
            
            # Volume analysis
            if price_data.get('volume_trend') == 'increasing':
                tech_score += 10
            
            # Sentiment score (0-30 points)
            sentiment_score = 0
            if sentiment_data.get('combined_sentiment'):
                sentiment = sentiment_data['combined_sentiment']
                if sentiment > 0.1:
                    sentiment_score = min(30, int(sentiment * 30))
                elif sentiment < -0.1:
                    sentiment_score = max(0, int((1 + sentiment) * 30))
                else:
                    sentiment_score = 15  # Neutral
            
            # Price momentum score (0-30 points)
            momentum_score = 0
            if price_data.get('price_change_7d'):
                change_7d = price_data['price_change_7d']
                if change_7d > 0.05:  # > 5% gain
                    momentum_score += 15
                elif change_7d > 0:
                    momentum_score += 10
                elif change_7d > -0.05:
                    momentum_score += 5
            
            if price_data.get('volatility') and price_data['volatility'] < 0.02:  # Low volatility
                momentum_score += 15
            
            # Calculate total health score
            total_score = tech_score + sentiment_score + momentum_score
            
            # Determine health status
            if total_score >= 80:
                health_status = "Excellent"
            elif total_score >= 60:
                health_status = "Good"
            elif total_score >= 40:
                health_status = "Fair"
            elif total_score >= 20:
                health_status = "Poor"
            else:
                health_status = "Critical"
            
            return {
                "success": True,
                "health_score": total_score,
                "health_status": health_status,
                "breakdown": {
                    "technical_score": tech_score,
                    "sentiment_score": sentiment_score,
                    "momentum_score": momentum_score
                },
                "recommendations": self._generate_recommendations(total_score, technical_data, sentiment_data)
            }
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_recommendations(self, score: int, tech_data: Dict, sentiment_data: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if score >= 80:
            recommendations.append("Strong buy signal - all indicators positive")
            recommendations.append("Consider increasing position size")
        elif score >= 60:
            recommendations.append("Good buying opportunity")
            recommendations.append("Monitor for continued strength")
        elif score >= 40:
            recommendations.append("Hold position if owned")
            recommendations.append("Wait for better entry point if buying")
        elif score >= 20:
            recommendations.append("Consider reducing position")
            recommendations.append("Set stop-loss orders")
        else:
            recommendations.append("Strong sell signal")
            recommendations.append("Exit positions immediately")
        
        # Add specific recommendations based on indicators
        if tech_data.get('rsi') and tech_data['rsi'] > 70:
            recommendations.append("RSI indicates overbought conditions")
        
        if sentiment_data.get('combined_sentiment', 0) < -0.2:
            recommendations.append("Negative sentiment may impact price")
        
        return recommendations

class RealTimeDataStreamer:
    """Real-time market data streaming"""
    
    def __init__(self, symbols: List[str], websocket_handler: WebSocketHandler, redis_client):
        self.symbols = symbols
        self.websocket_handler = websocket_handler
        self.redis = redis_client
        self.running = False
        self.update_interval = 30  # seconds
    
    def start_streaming(self):
        """Start real-time data streaming"""
        self.running = True
        threading.Thread(target=self._stream_loop, daemon=True).start()
        logger.info("Real-time data streaming started")
    
    def stop_streaming(self):
        """Stop real-time data streaming"""
        self.running = False
        logger.info("Real-time data streaming stopped")
    
    def _stream_loop(self):
        """Main streaming loop"""
        while self.running:
            try:
                for symbol in self.symbols:
                    data = self._fetch_real_time_data(symbol)
                    if data:
                        self._cache_data(symbol, data)
                        self.websocket_handler.broadcast_market_update({
                            'symbol': symbol,
                            'data': data,
                            'timestamp': datetime.now().isoformat()
                        })
                
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                time.sleep(5)
    
    def _fetch_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time data for symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                return None
            
            latest = hist.iloc[-1]
            return {
                'symbol': symbol,
                'price': float(latest['Close']),
                'change': float(latest['Close'] - hist.iloc[-2]['Close']) if len(hist) > 1 else 0,
                'change_percent': float((latest['Close'] - hist.iloc[-2]['Close']) / hist.iloc[-2]['Close'] * 100) if len(hist) > 1 else 0,
                'volume': int(latest['Volume']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'timestamp': latest.name.isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {e}")
            return None
    
    def _cache_data(self, symbol: str, data: Dict[str, Any]):
        """Cache real-time data"""
        try:
            key = f"realtime:{symbol}"
            self.redis.setex(key, 300, json.dumps(data, default=str))
        except Exception as e:
            logger.error(f"Error caching data: {e}")

class TaskQueue:
    """Celery task queue for async processing"""
    
    def __init__(self, redis_url: str):
        self.celery_app = Celery('stock_analyzer', broker=redis_url, backend=redis_url)
        self.setup_tasks()
    
    def setup_tasks(self):
        """Setup Celery tasks"""
        
        @self.celery_app.task
        def analyze_stock_async(symbol: str, config_dict: Dict):
            """Async stock analysis task"""
            try:
                config = AnalysisConfig(**config_dict)
                workflow = StockAnalyzerWorkflow(config)
                result = asyncio.run(workflow.analyze_stock(symbol))
                return result
            except Exception as e:
                logger.error(f"Error in async analysis: {e}")
                return {"error": str(e), "status": "failed"}
        
        @self.celery_app.task
        def update_real_time_data(symbols: List[str]):
            """Update real-time data for symbols"""
            try:
                results = {}
                for symbol in symbols:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    results[symbol] = {
                        'price': info.get('currentPrice', 0),
                        'change': info.get('regularMarketChange', 0),
                        'change_percent': info.get('regularMarketChangePercent', 0),
                        'volume': info.get('volume', 0)
                    }
                return results
            except Exception as e:
                logger.error(f"Error updating real-time data: {e}")
                return {}
    
    def submit_analysis(self, symbol: str, config: AnalysisConfig) -> str:
        """Submit analysis task"""
        task = self.celery_app.send_task(
            'analyze_stock_async',
            args=[symbol, config.__dict__]
        )
        return task.id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status"""
        task = self.celery_app.AsyncResult(task_id)
        return {
            'task_id': task_id,
            'status': task.status,
            'result': task.result if task.ready() else None
        }

# =============================================================================
# ENHANCED DATABASE MANAGER
# =============================================================================

class EnhancedDatabaseManager:
    """Enhanced database manager with all required tables"""
    
    def __init__(self, db_path: str = "stock_analyzer.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize comprehensive database schema"""
        with sqlite3.connect(self.db_path) as conn:
            # Existing tables from original code
            self._create_core_tables(conn)
            
            # Additional tables for missing components
            self._create_enhanced_tables(conn)
            
            conn.commit()
            logger.info("Enhanced database initialized successfully")
    
    def _create_core_tables(self, conn):
        """Create core tables from original implementation"""
        # Stock prices table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            ds DATE NOT NULL,
            y REAL NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL,
            volume INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, ds)
        )
        """)
        
        # Create indexes for performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_ds ON stock_prices(symbol, ds)")
        
    def _create_enhanced_tables(self, conn):
        """Create additional tables for missing components"""
        
        # User sessions table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            user_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'active'
        )
        """)
        
        # Task queue status
        conn.execute("""
        CREATE TABLE IF NOT EXISTS task_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT UNIQUE NOT NULL,
            session_id TEXT,
            task_type TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at DATETIME,
            result TEXT,
            error_message TEXT
        )
        """)
        
        # Real-time price cache
        conn.execute("""
        CREATE TABLE IF NOT EXISTS realtime_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            change_amount REAL,
            change_percent REAL,
            volume INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp)
        )
        """)
        
        # Health scores history
        conn.execute("""
        CREATE TABLE IF NOT EXISTS health_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            session_id TEXT,
            health_score INTEGER,
            health_status TEXT,
            technical_score INTEGER,
            sentiment_score INTEGER,
            momentum_score INTEGER,
            recommendations TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Performance metrics
        conn.execute("""
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            model_type TEXT NOT NULL,
            mae REAL,
            rmse REAL,
            mape REAL,
            accuracy_score REAL,
            training_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

# =============================================================================
# FLASK API SERVER
# =============================================================================

class StockAnalyzerAPI:
    """Flask API server for the stock analyzer"""
    
    def __init__(self, config: AnalysisConfig):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'your-secret-key-here'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize components
        self.config = config
        self.redis_client = redis.from_url(config.redis_url)
        self.rate_limiter = RateLimiter(self.redis_client)
        self.shared_memory = SharedMemory(self.redis_client)
        self.websocket_handler = WebSocketHandler(self.socketio)
        self.task_queue = TaskQueue(config.redis_url)
        self.db_manager = EnhancedDatabaseManager()
        
        # Real-time streaming
        if config.enable_realtime:
            self.streamer = RealTimeDataStreamer(
                config.stocks, 
                self.websocket_handler, 
                self.redis_client
            )
            self.streamer.start_streaming()
        
        self.setup_routes()
        self.setup_websocket_handlers()
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/api/analyze', methods=['POST'])
        def analyze_stock():
            try:
                data = request.json
                symbol = data.get('symbol')
                client_id = data.get('client_id', 'anonymous')
                
                # Rate limiting
                if not self.rate_limiter.is_allowed(client_id):
                    return jsonify({'error': 'Rate limit exceeded'}), 429
                
                # Submit analysis task
                task_id = self.task_queue.submit_analysis(symbol, self.config)
                
                return jsonify({
                    'task_id': task_id,
                    'symbol': symbol,
                    'status': 'submitted'
                })
            except Exception as e:
                logger.error(f"Error in analyze endpoint: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/task/<task_id>', methods=['GET'])
        def get_task_status(task_id):
            try:
                status = self.task_queue.get_task_status(task_id)
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/realtime/<symbol>', methods=['GET'])
        def get_realtime_data(symbol):
            try:
                data = self.redis_client.get(f"realtime:{symbol}")
                if data:
                    return jsonify(json.loads(data))
                return jsonify({'error': 'No data available'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            })
    
    def setup_websocket_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")
            emit('connected', {'session_id': request.sid})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            symbols = data.get('symbols', [])
            logger.info(f"Client {request.sid} subscribed to {symbols}")
            # Add client to symbol-specific rooms
            for symbol in symbols:
                self.socketio.join_room(f"symbol_{symbol}")
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        logger.info(f"Starting Stock Analyzer API on {host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug)


# =============================================================================
# UPDATED MAIN WORKFLOW WITH MISSING COMPONENTS
# =============================================================================

class EnhancedStockAnalyzerWorkflow:
    """Enhanced workflow with all missing components"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.db_manager = EnhancedDatabaseManager()
        self.chroma_manager = ChromaDBManager()
        self.redis_client = redis.from_url(config.redis_url)
        self.shared_memory = SharedMemory(self.redis_client)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize enhanced tools
        self.health_synthesizer = HealthScoreSynthesizer()
        
        logger.info(f"Enhanced workflow initialized with session ID: {self.session_id}")
    
    async def analyze_stock_enhanced(self, symbol: str) -> Dict[str, Any]:
        """Enhanced stock analysis with all components"""
        try:
            logger.info(f"Starting enhanced analysis for {symbol}")
            
            # Store initial status in shared memory
            self.shared_memory.set_data(self.session_id, 'status', {
                'symbol': symbol,
                'stage': 'initializing',
                'progress': 0
            })
            
            # Run original analysis
            result = await self.analyze_stock(symbol)
            
            # Calculate health score
            if result.get('status') == 'completed':
                health_data = self._calculate_health_score(symbol, result)
                result['health_analysis'] = health_data
                
                # Store health score in database
                self._store_health_score(symbol, health_data)
            
            # Update shared memory with final result
            self.shared_memory.set_data(self.session_id, 'final_result', result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced analysis for {symbol}: {e}")
            return {
                'symbol': symbol,
                'session_id': self.session_id,
                'error': str(e),
                'status': 'failed'
            }
    
    def _calculate_health_score(self, symbol: str, analysis_result: Dict) -> Dict[str, Any]:
        """Calculate comprehensive health score"""
        try:
            # Extract data for health calculation
            technical_data = analysis_result.get('technical_indicators', {})
            sentiment_data = analysis_result.get('sentiment_analysis', {})
            price_data = analysis_result.get('price_data', {})
            
            # Calculate health score
            health_result = self.health_synthesizer._run(
                technical_data, sentiment_data, price_data
            )
            
            return health_result
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return {'error': str(e)}
    
    def _store_health_score(self, symbol: str, health_data: Dict):
        """Store health score in database"""
        try:
            if health_data.get('success'):
                with sqlite3.connect(self.db_manager.db_path) as conn:
                    conn.execute("""
                    INSERT INTO health_scores 
                    (symbol, session_id, health_score, health_status, 
                     technical_score, sentiment_score, momentum_score, recommendations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        self.session_id,
                        health_data['health_score'],
                        health_data['health_status'],
                        health_data['breakdown']['technical_score'],
                        health_data['breakdown']['sentiment_score'],
                        health_data['breakdown']['momentum_score'],
                        json.dumps(health_data.get('recommendations', []))
                    ))
                    conn.commit()
        except Exception as e:
            logger.error(f"Error storing health score: {e}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def main():
    """Example usage of the enhanced stock analyzer"""
    
    # Configuration
    config = AnalysisConfig(
        stocks=["AAPL", "GOOGL", "MSFT", "TSLA"],
        period_days=90,
        prediction_horizon=30,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY", ""),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        enable_realtime=True
    )
    
    # Initialize API server
    api_server = StockAnalyzerAPI(config)
    
    # Run the server
    logger.info("Starting enhanced stock analyzer API...")
    api_server.run(debug=True)

if __name__ == "__main__":
    
    # Run the application
    asyncio.run(main())


    