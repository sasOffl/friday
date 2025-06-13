"""
Router Package - McKinsey Stock Performance Monitor
Contains FastAPI routers for API endpoints and WebSocket connections
"""

from .analysis import router as analysis_router
from .websocket import router as websocket_router

__all__ = [
    'analysis_router',
    'websocket_router'
]