"""
Data Ingestion Crew for McKinsey Stock Performance Monitor
Orchestrates data loading, news scraping, and preprocessing tasks
"""

from crewai import Crew, Task
from agents.data_agents import MarketDataLoaderAgent, NewsSentimentAgent, DataPreprocessingAgent
from tools.market_data_tools import YahooFinanceTool
from tools.sentiment_tools import FirecrawlScraper, FinBERTWrapper, ChromaDBTool
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataIngestionCrew:
    """Crew responsible for data ingestion and preprocessing"""
    
    def __init__(self):
        self.market_data_agent = MarketDataLoaderAgent()
        self.news_sentiment_agent = NewsSentimentAgent()
        self.preprocessing_agent = DataPreprocessingAgent()
        
        # Initialize tools
        self.yahoo_tool = YahooFinanceTool()
        self.firecrawl_scraper = FirecrawlScraper()
        self.finbert_wrapper = FinBERTWrapper()
        self.chroma_tool = ChromaDBTool()
    
    def create_crew(self) -> Crew:
        """Create and configure the data ingestion crew"""
        
        # Define tasks
        load_market_data_task = Task(
            description="Load historical market data for specified stock symbols",
            agent=self.market_data_agent,
            expected_output="Historical OHLCV data for all requested symbols with technical indicators"
        )
        
        scrape_news_task = Task(
            description="Scrape recent news articles and perform sentiment analysis",
            agent=self.news_sentiment_agent,
            expected_output="News articles with sentiment scores and embeddings stored in ChromaDB"
        )
        
        preprocess_data_task = Task(
            description="Clean and normalize all collected data for analysis",
            agent=self.preprocessing_agent,
            expected_output="Cleaned datasets ready for prediction and health analysis",
            dependencies=[load_market_data_task, scrape_news_task]
        )
        
        # Create crew
        crew = Crew(
            agents=[
                self.market_data_agent,
                self.news_sentiment_agent,
                self.preprocessing_agent
            ],
            tasks=[
                load_market_data_task,
                scrape_news_task,
                preprocess_data_task
            ],
            verbose=True,
            process="sequential"
        )
        
        return crew
    
    def execute_ingestion(self, symbols: List[str], period: str = "1y", news_days: int = 30) -> Dict[str, Any]:
        """Execute the complete data ingestion process"""
        try:
            logger.info(f"Starting data ingestion for symbols: {symbols}")
            
            # Prepare shared context
            shared_context = {
                'symbols': symbols,
                'period': period,
                'news_days': news_days,
                'market_data': {},
                'news_data': {},
                'technical_indicators': {},
                'sentiment_scores': {}
            }
            
            # Load market data
            logger.info("Loading market data...")
            market_task = {
                'symbols': symbols,
                'period': period,
                'context': shared_context
            }
            market_results = self.market_data_agent.execute_task(market_task)
            shared_context['market_data'] = market_results.get('market_data', {})
            shared_context['technical_indicators'] = market_results.get('technical_indicators', {})
            
            # Scrape news and analyze sentiment
            logger.info("Scraping news and analyzing sentiment...")
            news_task = {
                'symbols': symbols,
                'days': news_days,
                'context': shared_context
            }
            news_results = self.news_sentiment_agent.execute_task(news_task)
            shared_context['news_data'] = news_results.get('news_data', {})
            shared_context['sentiment_scores'] = news_results.get('sentiment_scores', {})
            
            # Preprocess all data
            logger.info("Preprocessing data...")
            preprocess_task = {
                'context': shared_context
            }
            preprocessing_results = self.preprocessing_agent.execute_task(preprocess_task)
            
            # Combine all results
            final_results = {
                'status': 'completed',
                'market_data': shared_context['market_data'],
                'technical_indicators': shared_context['technical_indicators'],
                'news_data': shared_context['news_data'],
                'sentiment_scores': shared_context['sentiment_scores'],
                'preprocessing_summary': preprocessing_results.get('summary', {}),
                'data_quality': preprocessing_results.get('quality_metrics', {}),
                'ingestion_metadata': {
                    'symbols_processed': len([s for s in symbols if s in shared_context['market_data']]),
                    'total_news_articles': sum(len(articles) for articles in shared_context['news_data'].values()),
                    'period': period,
                    'news_days': news_days
                }
            }
            
            logger.info("Data ingestion completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in data ingestion crew: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'partial_results': shared_context if 'shared_context' in locals() else {}
            }


class LoadMarketDataTask:
    """Task for loading market data using Yahoo Finance"""
    
    @staticmethod
    def execute(symbols: List[str], period: str, yahoo_tool: YahooFinanceTool) -> Dict[str, Any]:
        """Execute market data loading task"""
        results = {}
        
        for symbol in symbols:
            try:
                # Fetch historical data
                stock_data = yahoo_tool.fetch_stock_data(symbol, period)
                
                if stock_data is not None and not stock_data.empty:
                    # Calculate technical indicators
                    technical_indicators = yahoo_tool.get_technical_indicators(symbol, stock_data)
                    
                    results[symbol] = {
                        'data': stock_data,
                        'indicators': technical_indicators,
                        'status': 'success'
                    }
                    
                    logger.info(f"Successfully loaded data for {symbol}")
                else:
                    results[symbol] = {
                        'status': 'failed',
                        'error': 'No data available'
                    }
                    logger.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                results[symbol] = {
                    'status': 'failed',
                    'error': str(e)
                }
                logger.error(f"Error loading data for {symbol}: {str(e)}")
        
        return results


class ScrapeNewsTask:
    """Task for scraping news and analyzing sentiment"""
    
    @staticmethod
    def execute(symbols: List[str], days: int, firecrawl: FirecrawlScraper, 
                finbert: FinBERTWrapper, chroma: ChromaDBTool) -> Dict[str, Any]:
        """Execute news scraping and sentiment analysis task"""
        results = {}
        
        for symbol in symbols:
            try:
                # Scrape news articles
                news_articles = firecrawl.scrape_stock_news(symbol, days)
                
                if news_articles:
                    # Analyze sentiment for each article
                    sentiment_scores = []
                    processed_articles = []
                    
                    for article in news_articles:
                        try:
                            sentiment_score = finbert.analyze_sentiment(article.get('content', ''))
                            
                            article_with_sentiment = {
                                **article,
                                'sentiment_score': sentiment_score,
                                'symbol': symbol
                            }
                            
                            sentiment_scores.append(sentiment_score)
                            processed_articles.append(article_with_sentiment)
                            
                        except Exception as e:
                            logger.error(f"Error analyzing sentiment for article: {str(e)}")
                            continue
                    
                    # Store embeddings in ChromaDB
                    try:
                        chroma.store_embeddings(processed_articles, symbol)
                    except Exception as e:
                        logger.error(f"Error storing embeddings for {symbol}: {str(e)}")
                    
                    results[symbol] = {
                        'articles': processed_articles,
                        'avg_sentiment': sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
                        'sentiment_distribution': {
                            'positive': len([s for s in sentiment_scores if s > 0.1]),
                            'neutral': len([s for s in sentiment_scores if -0.1 <= s <= 0.1]),
                            'negative': len([s for s in sentiment_scores if s < -0.1])
                        },
                        'total_articles': len(processed_articles),
                        'status': 'success'
                    }
                    
                    logger.info(f"Successfully processed {len(processed_articles)} articles for {symbol}")
                    
                else:
                    results[symbol] = {
                        'articles': [],
                        'avg_sentiment': 0,
                        'total_articles': 0,
                        'status': 'no_news',
                        'message': 'No news articles found'
                    }
                    
            except Exception as e:
                results[symbol] = {
                    'status': 'failed',
                    'error': str(e)
                }
                logger.error(f"Error processing news for {symbol}: {str(e)}")
        
        return results


class PreprocessDataTask:
    """Task for cleaning and normalizing collected data"""
    
    @staticmethod
    def execute(context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data preprocessing task"""
        try:
            quality_metrics = {}
            preprocessing_summary = {}
            
            # Check market data quality
            market_data = context.get('market_data', {})
            valid_symbols = 0
            total_data_points = 0
            
            for symbol, data in market_data.items():
                if data.get('status') == 'success' and 'data' in data:
                    valid_symbols += 1
                    total_data_points += len(data['data'])
            
            quality_metrics['market_data'] = {
                'valid_symbols': valid_symbols,
                'total_symbols': len(market_data),
                'success_rate': (valid_symbols / len(market_data)) * 100 if market_data else 0,
                'total_data_points': total_data_points
            }
            
            # Check news data quality
            news_data = context.get('news_data', {})
            total_articles = 0
            symbols_with_news = 0
            
            for symbol, data in news_data.items():
                if data.get('status') == 'success':
                    symbols_with_news += 1
                    total_articles += data.get('total_articles', 0)
            
            quality_metrics['news_data'] = {
                'symbols_with_news': symbols_with_news,
                'total_symbols': len(news_data),
                'coverage_rate': (symbols_with_news / len(news_data)) * 100 if news_data else 0,
                'total_articles': total_articles
            }
            
            # Generate preprocessing summary
            preprocessing_summary = {
                'data_completeness': 'Good' if quality_metrics['market_data']['success_rate'] > 80 else 'Poor',
                'news_coverage': 'Good' if quality_metrics['news_data']['coverage_rate'] > 60 else 'Limited',
                'ready_for_analysis': quality_metrics['market_data']['success_rate'] > 50,
                'recommendations': []
            }
            
            # Add recommendations based on data quality
            if quality_metrics['market_data']['success_rate'] < 80:
                preprocessing_summary['recommendations'].append('Consider checking symbol validity or market hours')
            
            if quality_metrics['news_data']['coverage_rate'] < 60:
                preprocessing_summary['recommendations'].append('Limited news coverage may affect sentiment analysis')
            
            logger.info("Data preprocessing completed successfully")
            
            return {
                'status': 'completed',
                'quality_metrics': quality_metrics,
                'summary': preprocessing_summary
            }
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }