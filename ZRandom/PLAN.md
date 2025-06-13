# McKinsey Stock Performance Monitor - System Architecture

## 🏗️ Project Structure
```
mckinsey_stock_analyzer/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py         # Configuration management
│   │   └── database.py         # Database connection setup
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database_models.py  # SQLAlchemy models
│   │   └── pydantic_models.py  # Request/Response models
│   ├── crews/
│   │   ├── __init__.py
│   │   ├── data_ingestion_crew.py
│   │   ├── model_prediction_crew.py
│   │   ├── health_analytics_crew.py
│   │   ├── comparative_analysis_crew.py
│   │   └── report_generation_crew.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── data_agents.py
│   │   ├── prediction_agents.py
│   │   ├── health_agents.py
│   │   ├── comparative_agents.py
│   │   └── report_agents.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── market_data_tools.py
│   │   ├── sentiment_tools.py
│   │   ├── prediction_tools.py
│   │   ├── technical_tools.py
│   │   └── visualization_tools.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── analysis_service.py
│   │   ├── data_service.py
│   │   └── websocket_service.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── exceptions.py
│   │   └── helpers.py
│   └── routers/
│       ├── __init__.py
│       ├── analysis.py
│       └── websocket.py
├── frontend/
│   └── index.html              # Enhanced dashboard from provided file
├── data/
│   ├── stock_data.db          # SQLite database
│   └── chroma_db/             # ChromaDB vector store
├── requirements.txt
└── README.md
```

---

## 🔧 Core Configuration & Setup

### **settings.py**
- **Class**: `Settings`
  - **Input**: Environment variables
  - **Output**: Configuration object
  - **Purpose**: Centralized configuration management for API keys, database URLs, model parameters

### **database.py**
- **Function**: `get_database_session()`
  - **Input**: None
  - **Output**: SQLAlchemy session
  - **Purpose**: Database connection factory
- **Function**: `init_database()`
  - **Input**: None
  - **Output**: None
  - **Purpose**: Initialize database tables and ChromaDB collections

---

## 📊 Database Models

### **database_models.py**
- **Class**: `StockData`
  - **Input**: Symbol, date, OHLCV data
  - **Output**: Database record
  - **Purpose**: Store historical stock prices and volume
- **Class**: `NewsArticle`
  - **Input**: URL, content, sentiment score
  - **Output**: Database record
  - **Purpose**: Store scraped news with sentiment analysis
- **Class**: `TechnicalIndicator`
  - **Input**: Symbol, date, indicator values
  - **Output**: Database record
  - **Purpose**: Store RSI, MACD, Bollinger bands data
- **Class**: `Prediction`
  - **Input**: Symbol, prediction date, forecasted values
  - **Output**: Database record
  - **Purpose**: Store NeuralProphet model predictions
- **Class**: `AnalysisSession`
  - **Input**: Session ID, parameters, status
  - **Output**: Database record
  - **Purpose**: Track analysis sessions and logs

### **pydantic_models.py**
- **Class**: `AnalysisRequest`
  - **Input**: Stock symbols list, time period, prediction horizon
  - **Output**: Validated request object
  - **Purpose**: API request validation
- **Class**: `AnalysisResponse`
  - **Input**: Analysis results, charts data, insights
  - **Output**: JSON response
  - **Purpose**: API response structure
- **Class**: `StockInsight`
  - **Input**: Symbol, health score, predictions, sentiment
  - **Output**: Structured insight object
  - **Purpose**: Individual stock analysis results

---

## 🛠️ Tools Layer

### **market_data_tools.py**
- **Class**: `YahooFinanceTool`
  - **Method**: `fetch_stock_data(symbol, period)`
    - **Input**: Stock symbol, time period
    - **Output**: OHLCV DataFrame
    - **Purpose**: Fetch historical stock data
  - **Method**: `get_technical_indicators(symbol, data)`
    - **Input**: Symbol, price data
    - **Output**: Technical indicators DataFrame
    - **Purpose**: Calculate RSI, MACD, Bollinger bands

### **sentiment_tools.py**
- **Class**: `FirecrawlScraper`
  - **Method**: `scrape_stock_news(symbol, days)`
    - **Input**: Stock symbol, lookback days
    - **Output**: List of news articles
    - **Purpose**: Scrape financial news from multiple sources
- **Class**: `FinBERTWrapper`
  - **Method**: `analyze_sentiment(text)`
    - **Input**: News article text
    - **Output**: Sentiment score (-1 to 1)
    - **Purpose**: Financial sentiment analysis using FinBERT
- **Class**: `ChromaDBTool`
  - **Method**: `store_embeddings(articles, embeddings)`
    - **Input**: Articles list, vector embeddings
    - **Output**: Storage confirmation
    - **Purpose**: Store news embeddings for similarity search

### **prediction_tools.py**
- **Class**: `NeuralProphetWrapper`
  - **Method**: `train_model(historical_data)`
    - **Input**: Historical price DataFrame
    - **Output**: Trained model object
    - **Purpose**: Train NeuralProphet forecasting model
  - **Method**: `predict_prices(model, horizon_days)`
    - **Input**: Trained model, prediction days
    - **Output**: Predictions DataFrame
    - **Purpose**: Generate price forecasts with confidence intervals

### **technical_tools.py**
- **Class**: `RSI_MACD_Tool`
  - **Method**: `calculate_rsi(prices, period)`
    - **Input**: Price series, RSI period
    - **Output**: RSI values
    - **Purpose**: Calculate Relative Strength Index
  - **Method**: `calculate_macd(prices)`
    - **Input**: Price series
    - **Output**: MACD line, signal line, histogram
    - **Purpose**: Calculate MACD indicator
- **Class**: `VolatilityScanner`
  - **Method**: `calculate_volatility(prices, window)`
    - **Input**: Price series, rolling window
    - **Output**: Volatility measures
    - **Purpose**: Calculate price volatility metrics

### **visualization_tools.py**
- **Class**: `PlotlyToolKit`
  - **Method**: `create_price_chart(data, predictions)`
    - **Input**: Historical data, prediction data
    - **Output**: Plotly chart JSON
    - **Purpose**: Generate interactive price charts
  - **Method**: `create_sentiment_timeline(sentiment_data)`
    - **Input**: Sentiment time series
    - **Output**: Plotly chart JSON
    - **Purpose**: Visualize sentiment trends over time

---

## 🤖 Agents Layer

### **data_agents.py**
- **Class**: `MarketDataLoaderAgent`
  - **Method**: `execute_task(task)`
    - **Input**: Data loading task
    - **Output**: Market data results
    - **Purpose**: Load and validate stock market data
- **Class**: `NewsSentimentAgent`
  - **Method**: `execute_task(task)`
    - **Input**: News scraping task
    - **Output**: Sentiment analysis results
    - **Purpose**: Scrape news and analyze sentiment
- **Class**: `DataPreprocessingAgent`
  - **Method**: `execute_task(task)`
    - **Input**: Raw data preprocessing task
    - **Output**: Cleaned datasets
    - **Purpose**: Clean and normalize data for analysis

### **prediction_agents.py**
- **Class**: `ForecastAgent`
  - **Method**: `execute_task(task)`
    - **Input**: Forecasting task with parameters
    - **Output**: Price predictions with confidence
    - **Purpose**: Generate stock price forecasts using NeuralProphet
- **Class**: `EvaluationAgent`
  - **Method**: `execute_task(task)`
    - **Input**: Model evaluation task
    - **Output**: Accuracy metrics (RMSE, MAE)
    - **Purpose**: Evaluate prediction model performance

### **health_agents.py**
- **Class**: `IndicatorAnalysisAgent`
  - **Method**: `execute_task(task)`
    - **Input**: Technical analysis task
    - **Output**: Technical indicator insights
    - **Purpose**: Analyze RSI, MACD, Bollinger bands
- **Class**: `StockHealthAgent`
  - **Method**: `execute_task(task)`
    - **Input**: Health assessment task
    - **Output**: Health score (0-100)
    - **Purpose**: Calculate composite stock health score
- **Class**: `SentimentTrendAgent`
  - **Method**: `execute_task(task)`
    - **Input**: Sentiment trend analysis task
    - **Output**: Sentiment trajectory insights
    - **Purpose**: Analyze sentiment patterns over time

### **comparative_agents.py**
- **Class**: `ComparativeAgent`
  - **Method**: `execute_task(task)`
    - **Input**: Multi-stock comparison task
    - **Output**: Comparative analysis charts
    - **Purpose**: Compare multiple stocks across metrics
- **Class**: `CorrelationInsightAgent`
  - **Method**: `execute_task(task)`
    - **Input**: Correlation analysis task
    - **Output**: Feature correlation matrix
    - **Purpose**: Identify relationships between features
- **Class**: `PeerComparisonAgent`
  - **Method**: `execute_task(task)`
    - **Input**: Peer comparison task
    - **Output**: Sector comparison results
    - **Purpose**: Compare stocks within industry sectors

### **report_agents.py**
- **Class**: `ReportComposerAgent`
  - **Method**: `execute_task(task)`
    - **Input**: Report generation task
    - **Output**: Structured analysis report
    - **Purpose**: Compose comprehensive stock analysis report
- **Class**: `VisualizationAgent`
  - **Method**: `execute_task(task)`
    - **Input**: Visualization task
    - **Output**: Interactive charts and graphs
    - **Purpose**: Create all visualization components
- **Class**: `StrategyAdvisorAgent`
  - **Method**: `execute_task(task)`
    - **Input**: Strategy recommendation task
    - **Output**: Buy/Hold/Sell recommendations
    - **Purpose**: Provide investment strategy recommendations

---

## 👥 Crews Layer

### **data_ingestion_crew.py**
- **Class**: `DataIngestionCrew`
  - **Method**: `create_crew()`
    - **Input**: None
    - **Output**: CrewAI Crew object
    - **Purpose**: Initialize data ingestion crew with agents and tasks
  - **Tasks**:
    - `LoadMarketDataTask`: Fetch OHLCV data for selected stocks
    - `ScrapeNewsTask`: Gather recent news articles for sentiment analysis
    - `PreprocessDataTask`: Clean and normalize all collected data

### **model_prediction_crew.py**
- **Class**: `ModelPredictionCrew`
  - **Method**: `create_crew()`
    - **Input**: None
    - **Output**: CrewAI Crew object
    - **Purpose**: Initialize prediction crew with forecasting agents
  - **Tasks**:
    - `TrainForecastModelTask`: Train NeuralProphet models for each stock
    - `GeneratePredictionsTask`: Create price forecasts with confidence intervals
    - `EvaluateModelTask`: Calculate prediction accuracy metrics

### **health_analytics_crew.py**
- **Class**: `HealthAnalyticsCrew`
  - **Method**: `create_crew()`
    - **Input**: None
    - **Output**: CrewAI Crew object
    - **Purpose**: Initialize health analysis crew with diagnostic agents
  - **Tasks**:
    - `AnalyzeTechnicalIndicatorsTask`: Calculate and interpret technical indicators
    - `AssessStockHealthTask`: Compute comprehensive health scores
    - `TrackSentimentTrendsTask`: Analyze sentiment evolution patterns

### **comparative_analysis_crew.py**
- **Class**: `ComparativeAnalysisCrew`
  - **Method**: `create_crew()`
    - **Input**: None
    - **Output**: CrewAI Crew object
    - **Purpose**: Initialize comparative analysis crew
  - **Tasks**:
    - `CompareStockMetricsTask`: Create comparative visualizations
    - `AnalyzeCorrelationsTask`: Identify feature relationships
    - `BenchmarkAgainstPeersTask`: Compare stocks within sectors

### **report_generation_crew.py**
- **Class**: `ReportGenerationCrew`
  - **Method**: `create_crew()`
    - **Input**: None
    - **Output**: CrewAI Crew object
    - **Purpose**: Initialize report generation crew
  - **Tasks**:
    - `ComposeAnalysisReportTask`: Generate comprehensive analysis reports
    - `CreateVisualizationsTask`: Produce all charts and graphs
    - `FormulateRecommendationsTask`: Provide investment strategy advice

---

## 🚀 Services Layer

### **analysis_service.py**
- **Class**: `AnalysisService`
  - **Method**: `run_full_analysis(symbols, period, horizon)`
    - **Input**: Stock symbols, analysis period, prediction horizon
    - **Output**: Complete analysis results
    - **Purpose**: Orchestrate all crews for comprehensive analysis
  - **Method**: `get_analysis_status(session_id)`
    - **Input**: Analysis session ID
    - **Output**: Current analysis progress
    - **Purpose**: Track analysis progress for real-time updates

### **data_service.py**
- **Class**: `DataService`
  - **Method**: `get_historical_data(symbol, period)`
    - **Input**: Stock symbol, time period
    - **Output**: Historical data from database
    - **Purpose**: Retrieve cached historical data
  - **Method**: `store_analysis_results(session_id, results)`
    - **Input**: Session ID, analysis results
    - **Output**: Storage confirmation
    - **Purpose**: Persist analysis results to database

### **websocket_service.py**
- **Class**: `WebSocketService`
  - **Method**: `broadcast_log_update(session_id, log_entry)`
    - **Input**: Session ID, log message
    - **Output**: None
    - **Purpose**: Send real-time log updates to frontend
  - **Method**: `broadcast_progress_update(session_id, progress)`
    - **Input**: Session ID, progress percentage
    - **Output**: None
    - **Purpose**: Send analysis progress updates

---

## 🌐 API Layer (FastAPI)

### **main.py**
- **Function**: `startup_event()`
  - **Input**: None
  - **Output**: None
  - **Purpose**: Initialize database and ChromaDB on startup
- **Function**: `shutdown_event()`
  - **Input**: None
  - **Output**: None
  - **Purpose**: Clean up resources on shutdown

### **routers/analysis.py**
- **Endpoint**: `POST /api/analysis/start`
  - **Input**: AnalysisRequest (symbols, period, horizon)
  - **Output**: Session ID and initial status
  - **Purpose**: Start new stock analysis session
- **Endpoint**: `GET /api/analysis/{session_id}/status`
  - **Input**: Session ID
  - **Output**: Analysis progress and current status
  - **Purpose**: Get real-time analysis progress
- **Endpoint**: `GET /api/analysis/{session_id}/results`
  - **Input**: Session ID
  - **Output**: Complete analysis results
  - **Purpose**: Retrieve finished analysis results

### **routers/websocket.py**
- **Endpoint**: `WebSocket /ws/{session_id}`
  - **Input**: Session ID, WebSocket connection
  - **Output**: Real-time log and progress updates
  - **Purpose**: Provide real-time updates during analysis

---

## 🛡️ Utilities

### **logger.py**
- **Class**: `AnalysisLogger`
  - **Method**: `log_crew_activity(crew_name, message, level)`
    - **Input**: Crew name, log message, severity level
    - **Output**: None
    - **Purpose**: Log crew activities to database and console
  - **Method**: `get_session_logs(session_id)`
    - **Input**: Session ID
    - **Output**: List of log entries
    - **Purpose**: Retrieve logs for specific analysis session

### **exceptions.py**
- **Class**: `AnalysisException`
  - **Purpose**: Custom exception for analysis errors
- **Class**: `DataFetchException`
  - **Purpose**: Exception for data retrieval failures
- **Class**: `ModelTrainingException`
  - **Purpose**: Exception for prediction model failures

### **helpers.py**
- **Function**: `validate_stock_symbols(symbols)`
  - **Input**: List of stock symbols
  - **Output**: Validated symbols list
  - **Purpose**: Validate and clean stock symbol inputs
- **Function**: `calculate_date_range(period)`
  - **Input**: Period in days
  - **Output**: Start and end dates
  - **Purpose**: Calculate date ranges for data fetching

---

## 📋 Task Definitions

### Data Ingestion Tasks
- **LoadMarketDataTask**: Fetch OHLCV data using YahooFinanceTool
- **ScrapeNewsTask**: Collect news articles using FirecrawlScraper
- **PreprocessDataTask**: Clean and normalize datasets
- **StoreDataTask**: Save processed data to SQLite and ChromaDB

### Prediction Tasks
- **TrainModelTask**: Train NeuralProphet model on historical data
- **GenerateForecastTask**: Create price predictions with confidence intervals
- **EvaluateAccuracyTask**: Calculate RMSE, MAE, and other metrics
- **ValidatePredictionsTask**: Perform model validation checks

### Health Analysis Tasks
- **CalculateIndicatorsTask**: Compute RSI, MACD, Bollinger bands
- **AssessHealthTask**: Generate composite health score (0-100)
- **AnalyzeSentimentTask**: Track sentiment trends over time
- **IdentifyAnomaliesTask**: Detect unusual patterns or outliers

### Comparative Analysis Tasks
- **CompareReturnsTask**: Create return comparison visualizations
- **AnalyzeVolatilityTask**: Compare volatility across stocks
- **CalculateCorrelationsTask**: Generate correlation matrices
- **BenchmarkPerformanceTask**: Compare against sector indices

### Report Generation Tasks
- **ComposeReportTask**: Generate natural language insights
- **CreateChartsTask**: Produce all interactive visualizations
- **FormulateAdviceTask**: Generate buy/hold/sell recommendations
- **FormatOutputTask**: Structure final response for frontend

---

## 🔄 Crew Execution Process

1. **Sequential Execution**: Crews execute in order (Data → Prediction → Health → Comparative → Report)
2. **Shared Memory**: Each crew stores results in session-wide shared memory
3. **Error Handling**: Failed crews trigger graceful degradation
4. **Progress Tracking**: Real-time updates via WebSocket
5. **Result Aggregation**: Final crew combines all insights into dashboard format

---

## 📦 Dependencies (requirements.txt)

```
fastapi==0.104.1
uvicorn==0.24.0
sqlalchemy==2.0.23
chromadb==0.4.18
crewai==0.28.8
neuralprophet==0.6.2
transformers==4.35.2
torch==2.1.1
yfinance==0.2.28
firecrawl-py==0.0.8
plotly==5.17.0
pandas==2.1.3
numpy==1.25.2
scikit-learn==1.3.2
python-multipart==0.0.6
websockets==11.0.3
python-dotenv==1.0.0
pydantic==2.5.0
asyncio==3.4.3
aiofiles==23.2.1
```

This architecture ensures a robust, scalable McKinsey-style stock analysis system with proper separation of concerns, real-time updates, and comprehensive error handling.