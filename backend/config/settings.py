import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application settings
    ENVIRONMENT: str = "development"
    HOST: str = "localhost"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    
    # API Keys
    FIRECRAWL_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./data/stock_data.db"
    CHROMADB_PATH: str = "./data/chroma_db"
    
    # Analysis parameters
    DEFAULT_ANALYSIS_PERIOD: int = 90
    DEFAULT_PREDICTION_HORIZON: int = 30
    MAX_CONCURRENT_ANALYSES: int = 5
    
    # Data sources
    YAHOO_FINANCE_ENABLED: bool = True
    NEWS_SCRAPING_ENABLED: bool = False  # Disable by default until API keys are set
    
    # Model settings
    MODEL_CACHE_DIR: str = "./data/models"
    NEURALPROPHET_SEASONALITY: str = "auto"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create necessary directories
        self._create_directories()
        
        # Validate configuration
        self._validate_config()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            Path(self.DATABASE_URL.replace("sqlite:///", "")).parent,
            Path(self.CHROMADB_PATH),
            Path(self.MODEL_CACHE_DIR)
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _validate_config(self):
        """Validate configuration settings"""
        if self.NEWS_SCRAPING_ENABLED and not self.FIRECRAWL_API_KEY:
            print("⚠️  Warning: NEWS_SCRAPING_ENABLED is True but FIRECRAWL_API_KEY is not set")
            self.NEWS_SCRAPING_ENABLED = False
        
        if not self.OPENAI_API_KEY:
            print("⚠️  Warning: OPENAI_API_KEY is not set. Some AI features may not work.")
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()