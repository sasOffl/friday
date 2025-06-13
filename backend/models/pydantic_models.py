from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class AnalysisRequest(BaseModel):
    """Request model for stock analysis"""
    symbols: List[str] = Field(..., min_items=1, max_items=10)
    period: str = Field(default="1y", pattern="^(1d|5d|1mo|3mo|6mo|1y|2y|5y|10y|ytd|max)$")
    prediction_horizon: int = Field(default=30, ge=1, le=365)
    include_sentiment: bool = Field(default=True)
    include_technical: bool = Field(default=True)

    @field_validator('symbols')
    def validate_symbols(cls, v: List[str]) -> List[str]:
        """Validate stock symbols"""
        return [symbol.upper().strip() for symbol in v]


class StockInsight(BaseModel):
    """Individual stock analysis results"""
    symbol: str
    current_price: float
    health_score: int = Field(ge=0, le=100)
    trend: str = Field(pattern="^(bullish|bearish|neutral)$")
    prediction: Dict[str, Any]
    sentiment: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    risk_level: str = Field(pattern="^(low|medium|high)$")


class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    session_id: str
    status: str
    progress: int = Field(ge=0, le=100)
    stocks: List[StockInsight] = []
    comparative_analysis: Optional[Dict[str, Any]] = None
    recommendations: Optional[Dict[str, Any]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class SessionStatus(BaseModel):
    """Session status model"""
    session_id: str
    status: str
    progress: int
    current_task: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
