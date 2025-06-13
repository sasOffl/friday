# main.py - FastAPI application entry point
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import logging

from config.database import init_database
from routers import analysis, websocket
from utils.logger import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    try:
        logger.info("Starting McKinsey Stock Performance Monitor...")
        # Initialize database and ChromaDB
        await init_database()
        logger.info("Database initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    finally:
        logger.info("Shutting down McKinsey Stock Performance Monitor...")

# Create FastAPI app
app = FastAPI(
    title="McKinsey Stock Performance Monitor",
    description="AI-powered stock analysis system with real-time monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis.router, prefix="/api")
app.include_router(websocket.router, prefix="/ws")

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "McKinsey Stock Performance Monitor API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "stock-monitor"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )