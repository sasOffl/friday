# ===== config/database.py =====
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import chromadb
import os
from pathlib import Path

from .settings import settings

# SQLAlchemy setup
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ChromaDB setup
chroma_client = None
chroma_collection = None

async def init_database():
    """Initialize database and ChromaDB"""
    global chroma_client, chroma_collection
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create SQLAlchemy tables
    from models.database_models import *  # Import all models
    Base.metadata.create_all(bind=engine)
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=settings.chromadb_path)
    chroma_collection = chroma_client.get_or_create_collection(
        name="stock_news_embeddings",
        metadata={"description": "Stock news articles with embeddings"}
    )

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_chroma_collection():
    """Get ChromaDB collection"""
    return chroma_collection
