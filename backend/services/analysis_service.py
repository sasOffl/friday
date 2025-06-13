"""
Analysis Service - Orchestrates all crews for comprehensive stock analysis
"""

import asyncio
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from backend.config.database import get_database_session
from backend.models.database_models import AnalysisSession
from backend.models.pydantic_models import AnalysisRequest, AnalysisResponse
from backend.crews.data_ingestion_crew import DataIngestionCrew
from backend.crews.model_prediction_crew import ModelPredictionCrew
from backend.crews.health_analytics_crew import HealthAnalyticsCrew
from backend.crews.comparative_analysis_crew import ComparativeAnalysisCrew
from backend.crews.report_generation_crew import ReportGenerationCrew
from backend.utils.logger import AnalysisLogger
from backend.utils.exceptions import AnalysisException
from backend.services.websocket_service import WebSocketService


class AnalysisService:
    """Service for orchestrating comprehensive stock analysis"""
    
    def __init__(self):
        self.logger = AnalysisLogger(session_id="temp-agent-session")
        self.websocket_service = WebSocketService()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def run_full_analysis(
        self, 
        symbols: List[str], 
        period: str = "1y", 
        horizon: int = 30
    ) -> str:
        """
        Orchestrate all crews for comprehensive analysis
        
        Args:
            symbols: List of stock symbols to analyze
            period: Time period for historical data
            horizon: Prediction horizon in days
            
        Returns:
            Session ID for tracking analysis progress
        """
        session_id = str(uuid.uuid4())
        
        try:
            # Initialize database session
            db_session = next(get_database_session())
            
            # Create analysis session record
            analysis_session = AnalysisSession(
                session_id=session_id,
                symbols=symbols,
                period=period,
                horizon=horizon,
                status="initializing",
                created_at=datetime.utcnow()
            )
            db_session.add(analysis_session)
            db_session.commit()
            
            # Initialize shared memory for crews
            shared_memory = {
                "session_id": session_id,
                "symbols": symbols,
                "period": period,
                "horizon": horizon,
                "results": {},
                "progress": 0
            }
            
            self.active_sessions[session_id] = shared_memory
            
            # Start analysis in background
            asyncio.create_task(self._execute_analysis_pipeline(session_id, shared_memory))
            
            return session_id
            
        except Exception as e:
            self.logger.log_crew_activity(
                "AnalysisService", 
                f"Failed to start analysis: {str(e)}", 
                "error"
            )
            raise AnalysisException(f"Failed to start analysis: {str(e)}")
    
    async def _execute_analysis_pipeline(self, session_id: str, shared_memory: Dict[str, Any]):
        """Execute the full analysis pipeline sequentially"""
        
        try:
            # Update status
            await self._update_session_status(session_id, "running")
            
            # 1. Data Ingestion Crew (20% progress)
            await self._broadcast_progress(session_id, 5, "Starting data ingestion...")
            data_crew = DataIngestionCrew()
            crew = data_crew.create_crew()
            data_results = await crew.kickoff(inputs=shared_memory)
            shared_memory["results"]["data_ingestion"] = data_results
            await self._broadcast_progress(session_id, 20, "Data ingestion completed")
            
            # 2. Model Prediction Crew (40% progress)
            await self._broadcast_progress(session_id, 25, "Training prediction models...")
            prediction_crew = ModelPredictionCrew()
            crew = prediction_crew.create_crew()
            prediction_results = await crew.kickoff(inputs=shared_memory)
            shared_memory["results"]["predictions"] = prediction_results
            await self._broadcast_progress(session_id, 40, "Predictions generated")
            
            # 3. Health Analytics Crew (60% progress)
            await self._broadcast_progress(session_id, 45, "Analyzing stock health...")
            health_crew = HealthAnalyticsCrew()
            crew = health_crew.create_crew()
            health_results = await crew.kickoff(inputs=shared_memory)
            shared_memory["results"]["health_analytics"] = health_results
            await self._broadcast_progress(session_id, 60, "Health analysis completed")
            
            # 4. Comparative Analysis Crew (80% progress)
            await self._broadcast_progress(session_id, 65, "Performing comparative analysis...")
            comparative_crew = ComparativeAnalysisCrew()
            crew = comparative_crew.create_crew()
            comparative_results = await crew.kickoff(inputs=shared_memory)
            shared_memory["results"]["comparative_analysis"] = comparative_results
            await self._broadcast_progress(session_id, 80, "Comparative analysis completed")
            
            # 5. Report Generation Crew (100% progress)
            await self._broadcast_progress(session_id, 85, "Generating reports...")
            report_crew = ReportGenerationCrew()
            crew = report_crew.create_crew()
            report_results = await crew.kickoff(inputs=shared_memory)
            shared_memory["results"]["final_report"] = report_results
            await self._broadcast_progress(session_id, 100, "Analysis completed successfully")
            
            # Update final status
            await self._update_session_status(session_id, "completed")
            
        except Exception as e:
            self.logger.log_crew_activity(
                "AnalysisService", 
                f"Analysis pipeline failed: {str(e)}", 
                "error"
            )
            await self._update_session_status(session_id, "failed")
            await self._broadcast_progress(session_id, -1, f"Analysis failed: {str(e)}")
    
    async def get_analysis_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get current analysis progress and status
        
        Args:
            session_id: Analysis session ID
            
        Returns:
            Dictionary containing progress and status information
        """
        try:
            db_session = next(get_database_session())
            analysis_session = db_session.query(AnalysisSession).filter(
                AnalysisSession.session_id == session_id
            ).first()
            
            if not analysis_session:
                raise AnalysisException(f"Session {session_id} not found")
            
            # Get from active sessions if running
            if session_id in self.active_sessions:
                shared_memory = self.active_sessions[session_id]
                return {
                    "session_id": session_id,
                    "status": analysis_session.status,
                    "progress": shared_memory.get("progress", 0),
                    "symbols": analysis_session.symbols,
                    "created_at": analysis_session.created_at,
                    "updated_at": analysis_session.updated_at
                }
            
            return {
                "session_id": session_id,
                "status": analysis_session.status,
                "progress": 100 if analysis_session.status == "completed" else 0,
                "symbols": analysis_session.symbols,
                "created_at": analysis_session.created_at,
                "updated_at": analysis_session.updated_at
            }
            
        except Exception as e:
            self.logger.log_crew_activity(
                "AnalysisService", 
                f"Failed to get status: {str(e)}", 
                "error"
            )
            raise AnalysisException(f"Failed to get analysis status: {str(e)}")
    
    async def get_analysis_results(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get completed analysis results
        
        Args:
            session_id: Analysis session ID
            
        Returns:
            Complete analysis results or None if not completed
        """
        try:
            if session_id not in self.active_sessions:
                return None
            
            shared_memory = self.active_sessions[session_id]
            
            # Check if analysis is completed
            db_session = next(get_database_session())
            analysis_session = db_session.query(AnalysisSession).filter(
                AnalysisSession.session_id == session_id
            ).first()
            
            if not analysis_session or analysis_session.status != "completed":
                return None
            
            return shared_memory.get("results", {})
            
        except Exception as e:
            self.logger.log_crew_activity(
                "AnalysisService", 
                f"Failed to get results: {str(e)}", 
                "error"
            )
            raise AnalysisException(f"Failed to get analysis results: {str(e)}")
    
    async def _update_session_status(self, session_id: str, status: str):
        """Update analysis session status in database"""
        try:
            db_session = next(get_database_session())
            analysis_session = db_session.query(AnalysisSession).filter(
                AnalysisSession.session_id == session_id
            ).first()
            
            if analysis_session:
                analysis_session.status = status
                analysis_session.updated_at = datetime.utcnow()
                db_session.commit()
            
        except Exception as e:
            self.logger.log_crew_activity(
                "AnalysisService", 
                f"Failed to update session status: {str(e)}", 
                "error"
            )
    
    async def _broadcast_progress(self, session_id: str, progress: int, message: str):
        """Broadcast progress updates via WebSocket"""
        try:
            # Update shared memory
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["progress"] = progress
            
            # Broadcast via WebSocket
            await self.websocket_service.broadcast_progress_update(
                session_id, 
                {"progress": progress, "message": message}
            )
            
            # Log progress
            self.logger.log_crew_activity(
                "AnalysisService", 
                f"Progress {progress}%: {message}", 
                "info"
            )
            
        except Exception as e:
            self.logger.log_crew_activity(
                "AnalysisService", 
                f"Failed to broadcast progress: {str(e)}", 
                "warning"
            )
    
    def cleanup_session(self, session_id: str):
        """Clean up completed analysis session from memory"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]