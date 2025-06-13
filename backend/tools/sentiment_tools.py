# tools/sentiment_tools.py
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import re

logger = logging.getLogger(__name__)

class FirecrawlScraper:
    """Tool for scraping financial news using Firecrawl"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.firecrawl.dev/v0"
        
    def scrape_stock_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Scrape financial news for a stock symbol"""
        try:
            # News sources to search
            sources = [
                f"https://finance.yahoo.com/quote/{symbol}/news",
                f"https://www.marketwatch.com/investing/stock/{symbol}",
                f"https://seekingalpha.com/symbol/{symbol}/news"
            ]
            
            articles = []
            
            for source in sources:
                try:
                    articles.extend(self._scrape_source(source, symbol, days))
                except Exception as e:
                    logger.warning(f"Failed to scrape {source}: {str(e)}")
                    continue
            
            # Remove duplicates based on title
            seen_titles = set()
            unique_articles = []
            
            for article in articles:
                title_hash = hash(article['title'].lower())
                if title_hash not in seen_titles:
                    seen_titles.add(title_hash)
                    unique_articles.append(article)
            
            logger.info(f"Scraped {len(unique_articles)} unique articles for {symbol}")
            return unique_articles
            
        except Exception as e:
            logger.error(f"Error scraping news for {symbol}: {str(e)}")
            return []
    
    def _scrape_source(self, url: str, symbol: str, days: int) -> List[Dict]:
        """Scrape a specific news source"""
        if not self.api_key:
            # Fallback to mock data if no API key
            return self._generate_mock_news(symbol, days)
        
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'url': url,
                'formats': ['markdown'],
                'onlyMainContent': True
            }
            
            response = requests.post(f"{self.base_url}/scrape", 
                                   json=payload, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_scraped_content(data, symbol)
            else:
                logger.warning(f"Scraping failed with status {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return []
    
    def _parse_scraped_content(self, data: Dict, symbol: str) -> List[Dict]:
        """Parse scraped content into structured articles"""
        articles = []
        content = data.get('data', {}).get('content', '')
        
        # Simple regex to extract news articles
        # This is a simplified parser - in production, you'd want more sophisticated parsing
        article_pattern = r'(?i)(.{0,100}' + symbol + r'.{0,300})'
        matches = re.findall(article_pattern, content)
        
        for i, match in enumerate(matches[:5]):  # Limit to 5 articles per source
            articles.append({
                'title': f"News about {symbol} #{i+1}",
                'content': match.strip(),
                'url': f"scraped_article_{i}",
                'published_date': datetime.now() - timedelta(days=i),
                'source': 'firecrawl'
            })
        
        return articles
    
    def _generate_mock_news(self, symbol: str, days: int) -> List[Dict]:
        """Generate mock news data for testing"""
        mock_articles = [
            {
                'title': f"{symbol} Reports Strong Q4 Earnings",
                'content': f"{symbol} announced better-than-expected quarterly results with revenue growth of 15% year-over-year. The company's strategic initiatives are showing positive results.",
                'url': f"mock_article_1_{symbol}",
                'published_date': datetime.now() - timedelta(days=1),
                'source': 'mock'
            },
            {
                'title': f"Analysts Upgrade {symbol} Rating",
                'content': f"Major investment banks have upgraded their rating on {symbol} citing improved market conditions and strong fundamentals.",
                'url': f"mock_article_2_{symbol}",
                'published_date': datetime.now() - timedelta(days=2),
                'source': 'mock'
            },
            {
                'title': f"{symbol} Announces New Partnership",
                'content': f"{symbol} has entered into a strategic partnership that is expected to drive growth in the coming quarters.",
                'url': f"mock_article_3_{symbol}",
                'published_date': datetime.now() - timedelta(days=3),
                'source': 'mock'
            }
        ]
        
        return mock_articles[:min(days, len(mock_articles))]

class FinBERTWrapper:
    """Wrapper for FinBERT sentiment analysis"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load FinBERT model"""
        try:
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.classifier = pipeline("sentiment-analysis", 
                                      model=self.model, 
                                      tokenizer=self.tokenizer)
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {str(e)}. Using fallback sentiment analysis.")
            self.classifier = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        try:
            if not self.classifier:
                return self._fallback_sentiment(text)
            
            # Truncate text to model's max length
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            result = self.classifier(text)[0]
            
            # Map FinBERT labels to scores
            label_mapping = {
                'positive': 1.0,
                'negative': -1.0,
                'neutral': 0.0
            }
            
            score = label_mapping.get(result['label'].lower(), 0.0)
            confidence = result['score']
            
            return {
                'score': score * confidence,
                'label': result['label'].lower(),
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return self._fallback_sentiment(text)
    
    def _fallback_sentiment(self, text: str) -> Dict[str, float]:
        """Simple fallback sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'strong', 'positive', 'up', 'gain', 'profit', 'growth']
        negative_words = ['bad', 'poor', 'weak', 'negative', 'down', 'loss', 'decline', 'drop']
        
        text_lower = text.lower()
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return {'score': 0.7, 'label': 'positive', 'confidence': 0.7}
        elif neg_count > pos_count:
            return {'score': -0.7, 'label': 'negative', 'confidence': 0.7}
        else:
            return {'score': 0.0, 'label': 'neutral', 'confidence': 0.5}
    
    def batch_analyze_sentiment(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment for multiple texts"""
        results = []
        for text in texts:
            results.append(self.analyze_sentiment(text))
        return results

class ChromaDBTool:
    """Tool for storing and retrieving news embeddings using ChromaDB"""
    
    def __init__(self, collection_name: str = "stock_news"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Initialize persistent client
            self.client = chromadb.PersistentClient(
                path="./data/chroma_db",
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB collection '{self.collection_name}' initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            self.client = None
            self.collection = None
    
    def store_embeddings(self, articles: List[Dict], embeddings: List[List[float]]) -> bool:
        """Store article embeddings in ChromaDB"""
        try:
            if not self.collection:
                logger.warning("ChromaDB not initialized, skipping embedding storage")
                return False
            
            # Prepare data for storage
            documents = []
            metadatas = []
            ids = []
            
            for i, article in enumerate(articles):
                documents.append(article['content'])
                metadatas.append({
                    'title': article['title'],
                    'url': article['url'],
                    'published_date': article['published_date'].isoformat(),
                    'source': article['source'],
                    'symbol': article.get('symbol', 'unknown')
                })
                ids.append(f"{article.get('symbol', 'unknown')}_{i}_{hash(article['title'])}")
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Stored {len(articles)} article embeddings in ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            return False
    
    def search_similar_articles(self, query_embedding: List[float], 
                              symbol: str = None, n_results: int = 5) -> List[Dict]:
        """Search for similar articles based on embedding"""
        try:
            if not self.collection:
                logger.warning("ChromaDB not initialized, returning empty results")
                return []
            
            # Build where clause for filtering
            where_clause = {}
            if symbol:
                where_clause["symbol"] = symbol
            
            # Search for similar embeddings
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            # Format results
            similar_articles = []
            if results['documents'] and len(results['documents']) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    similar_articles.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else None
                    })
            
            return similar_articles
            
        except Exception as e:
            logger.error(f"Error searching similar articles: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics about the collection"""
        try:
            if not self.collection:
                return {'total_articles': 0}
            
            stats = {
                'total_articles': self.collection.count()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {'total_articles': 0}

class SentimentAnalyzer:
    """Main sentiment analysis orchestrator"""
    
    def __init__(self, firecrawl_api_key: Optional[str] = None):
        self.scraper = FirecrawlScraper(firecrawl_api_key)
        self.sentiment_analyzer = FinBERTWrapper()
        self.vector_store = ChromaDBTool()
        self.embedding_model = None
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load sentence transformer for embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {str(e)}")
            self.embedding_model = None
    
    def analyze_stock_sentiment(self, symbol: str, days: int = 7) -> Dict:
        """Complete sentiment analysis pipeline for a stock"""
        try:
            # Step 1: Scrape news articles
            logger.info(f"Scraping news for {symbol}")
            articles = self.scraper.scrape_stock_news(symbol, days)
            
            if not articles:
                logger.warning(f"No articles found for {symbol}")
                return self._empty_sentiment_result(symbol)
            
            # Step 2: Analyze sentiment for each article
            logger.info(f"Analyzing sentiment for {len(articles)} articles")
            sentiment_results = []
            
            for article in articles:
                sentiment = self.sentiment_analyzer.analyze_sentiment(article['content'])
                sentiment_results.append({
                    **article,
                    'sentiment_score': sentiment['score'],
                    'sentiment_label': sentiment['label'],
                    'sentiment_confidence': sentiment['confidence']
                })
            
            # Step 3: Generate embeddings if model is available
            if self.embedding_model:
                logger.info("Generating embeddings for articles")
                contents = [article['content'] for article in articles]
                embeddings = self.embedding_model.encode(contents).tolist()
                
                # Store in vector database
                articles_with_symbol = [{**article, 'symbol': symbol} for article in articles]
                self.vector_store.store_embeddings(articles_with_symbol, embeddings)
            
            # Step 4: Calculate aggregate sentiment metrics
            sentiment_scores = [result['sentiment_score'] for result in sentiment_results]
            
            aggregate_metrics = {
                'overall_sentiment': sum(sentiment_scores) / len(sentiment_scores),
                'sentiment_volatility': pd.Series(sentiment_scores).std(),
                'positive_articles': len([s for s in sentiment_scores if s > 0.1]),
                'negative_articles': len([s for s in sentiment_scores if s < -0.1]),
                'neutral_articles': len([s for s in sentiment_scores if -0.1 <= s <= 0.1]),
                'total_articles': len(sentiment_results)
            }
            
            # Step 5: Create sentiment timeline
            sentiment_timeline = self._create_sentiment_timeline(sentiment_results)
            
            return {
                'symbol': symbol,
                'analysis_date': datetime.now().isoformat(),
                'articles': sentiment_results,
                'aggregate_metrics': aggregate_metrics,
                'sentiment_timeline': sentiment_timeline,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e),
                'status': 'error'
            }
    
    def _create_sentiment_timeline(self, sentiment_results: List[Dict]) -> List[Dict]:
        """Create timeline of sentiment changes"""
        timeline = []
        
        # Sort by published date
        sorted_results = sorted(sentiment_results, 
                              key=lambda x: x['published_date'])
        
        for result in sorted_results:
            timeline.append({
                'date': result['published_date'].isoformat(),
                'sentiment_score': result['sentiment_score'],
                'title': result['title'][:100] + '...' if len(result['title']) > 100 else result['title']
            })
        
        return timeline
    
    def _empty_sentiment_result(self, symbol: str) -> Dict:
        """Return empty result structure when no articles found"""
        return {
            'symbol': symbol,
            'analysis_date': datetime.now().isoformat(),
            'articles': [],
            'aggregate_metrics': {
                'overall_sentiment': 0.0,
                'sentiment_volatility': 0.0,
                'positive_articles': 0,
                'negative_articles': 0,
                'neutral_articles': 0,
                'total_articles': 0
            },
            'sentiment_timeline': [],
            'status': 'no_data'
        }
    
    def get_similar_sentiment_articles(self, query_text: str, symbol: str = None, 
                                     n_results: int = 5) -> List[Dict]:
        """Find articles with similar sentiment patterns"""
        try:
            if not self.embedding_model:
                logger.warning("Embedding model not available for similarity search")
                return []
            
            # Generate embedding for query
            query_embedding = self.embedding_model.encode(query_text).tolist()
            
            # Search for similar articles
            similar_articles = self.vector_store.search_similar_articles(
                query_embedding, symbol, n_results
            )
            
            return similar_articles
            
        except Exception as e:
            logger.error(f"Error finding similar articles: {str(e)}")
            return []
    
    def batch_analyze_sentiment(self, symbols: List[str], days: int = 7) -> Dict[str, Dict]:
        """Analyze sentiment for multiple stocks"""
        results = {}
        
        for symbol in symbols:
            logger.info(f"Processing sentiment analysis for {symbol}")
            results[symbol] = self.analyze_stock_sentiment(symbol, days)
        
        return results
    
    def get_sentiment_summary(self, symbols: List[str]) -> Dict:
        """Get high-level sentiment summary across multiple stocks"""
        try:
            # Analyze all symbols
            individual_results = self.batch_analyze_sentiment(symbols)
            
            # Calculate cross-stock metrics
            all_scores = []
            stock_sentiments = {}
            
            for symbol, result in individual_results.items():
                if result['status'] == 'success':
                    sentiment_score = result['aggregate_metrics']['overall_sentiment']
                    stock_sentiments[symbol] = sentiment_score
                    all_scores.append(sentiment_score)
            
            if not all_scores:
                return {'status': 'no_data', 'message': 'No sentiment data available'}
            
            summary = {
                'analysis_date': datetime.now().isoformat(),
                'total_stocks_analyzed': len(stock_sentiments),
                'market_sentiment': {
                    'overall_score': sum(all_scores) / len(all_scores),
                    'sentiment_range': [min(all_scores), max(all_scores)],
                    'bullish_stocks': len([s for s in all_scores if s > 0.2]),
                    'bearish_stocks': len([s for s in all_scores if s < -0.2]),
                    'neutral_stocks': len([s for s in all_scores if -0.2 <= s <= 0.2])
                },
                'individual_stocks': stock_sentiments,
                'detailed_results': individual_results,
                'status': 'success'
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating sentiment summary: {str(e)}")
            return {'status': 'error', 'error': str(e)}

# Utility functions for sentiment analysis

def calculate_sentiment_momentum(sentiment_timeline: List[Dict], window: int = 3) -> float:
    """Calculate sentiment momentum over time"""
    if len(sentiment_timeline) < window:
        return 0.0
    
    # Sort by date
    timeline = sorted(sentiment_timeline, key=lambda x: x['date'])
    
    # Calculate recent vs older sentiment
    recent_scores = [item['sentiment_score'] for item in timeline[-window:]]
    older_scores = [item['sentiment_score'] for item in timeline[:window]]
    
    recent_avg = sum(recent_scores) / len(recent_scores)
    older_avg = sum(older_scores) / len(older_scores)
    
    return recent_avg - older_avg

def identify_sentiment_anomalies(sentiment_scores: List[float], threshold: float = 2.0) -> List[int]:
    """Identify anomalous sentiment scores using z-score"""
    if len(sentiment_scores) < 3:
        return []
    
    import numpy as np
    
    scores_array = np.array(sentiment_scores)
    z_scores = np.abs((scores_array - np.mean(scores_array)) / np.std(scores_array))
    
    anomaly_indices = np.where(z_scores > threshold)[0].tolist()
    return anomaly_indices

def create_sentiment_features(sentiment_data: Dict) -> Dict[str, float]:
    """Create features from sentiment data for ML models"""
    if sentiment_data['status'] != 'success':
        return {}
    
    metrics = sentiment_data['aggregate_metrics']
    timeline = sentiment_data['sentiment_timeline']
    
    features = {
        'overall_sentiment': metrics['overall_sentiment'],
        'sentiment_volatility': metrics['sentiment_volatility'],
        'positive_ratio': metrics['positive_articles'] / max(metrics['total_articles'], 1),
        'negative_ratio': metrics['negative_articles'] / max(metrics['total_articles'], 1),
        'article_volume': metrics['total_articles']
    }
    
    # Add momentum feature if timeline available
    if timeline:
        features['sentiment_momentum'] = calculate_sentiment_momentum(timeline)
    else:
        features['sentiment_momentum'] = 0.0
    
    return features