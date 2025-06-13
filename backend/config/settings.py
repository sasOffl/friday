# ===== config/settings.py =====
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    database_url: str = "sqlite:///./data/stock_data.db"
    
    # ChromaDB
    chromadb_path: str = "./data/chroma_db"
    
    # API Keys
    alpha_vantage_api_key: Optional[str] = None
    firecrawl_api_key: Optional[str] = None
    
    # Model Configuration
    prediction_horizon_days: int = 30
    max_stocks_per_analysis: int = 10
    
    # Cache Settings
    cache_ttl_hours: int = 24
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
