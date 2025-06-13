import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from contextlib import asynccontextmanager

# Import local modules
from config.database import init_database, get_database_session
from backend.config.settings import get_settings
from routers import analysis, websocket

# Global settings
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting McKinsey Stock Performance Monitor...")
    
    # Initialize database
    try:
        await init_database()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down McKinsey Stock Performance Monitor...")

# Create FastAPI app
app = FastAPI(
    title="McKinsey Stock Performance Monitor",
    description="AI-powered stock analysis system with real-time monitoring and predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis.router, prefix="/api", tags=["analysis"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

# Mount static files (frontend)
frontend_dir = backend_dir.parent / "frontend"
if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - redirect to frontend"""
    return """
    <html>
        <head>
            <title>McKinsey Stock Performance Monitor</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
                .container { max-width: 600px; margin: 0 auto; }
                .button { 
                    display: inline-block; 
                    padding: 12px 24px; 
                    background-color: #007bff; 
                    color: white; 
                    text-decoration: none; 
                    border-radius: 4px; 
                    margin: 10px;
                }
                .button:hover { background-color: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏗️ McKinsey Stock Performance Monitor</h1>
                <p>AI-powered stock analysis system with real-time monitoring and predictions</p>
                <div>
                    <a href="/frontend/" class="button">📊 Dashboard</a>
                    <a href="/docs" class="button">📚 API Documentation</a>
                </div>
                <p><strong>Status:</strong> ✅ System Online</p>
            </div>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        session = next(get_database_session())
        session.close()
        return {
            "status": "healthy",
            "service": "McKinsey Stock Performance Monitor",
            "version": "1.0.0",
            "database": "connected"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
