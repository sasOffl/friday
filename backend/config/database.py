from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import chromadb
import os

# Import config and models
from config.settings import get_settings
from models.database_models import (
    StockData, NewsArticle, TechnicalIndicator, Prediction,
    AnalysisSession, SessionLog
)

settings = get_settings()

# SQLAlchemy setup
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ChromaDB setup
chroma_client = None
chroma_collection = None

async def init_database():
    """Initialize database and ChromaDB"""
    global chroma_client, chroma_collection

    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    Base.metadata.create_all(bind=engine)

    chroma_client = chromadb.PersistentClient(path=settings.CHROMADB_PATH)
    chroma_collection = chroma_client.get_or_create_collection(
        name="stock_news_embeddings",
        metadata={"description": "Stock news articles with embeddings"}
    )

def get_database_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_chroma_collection():
    return chroma_collection
