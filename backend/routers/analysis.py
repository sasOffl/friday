"""
Analysis Router - McKinsey Stock Performance Monitor
FastAPI router for stock analysis endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import asyncio
import uuid
from datetime import datetime, timedelta
import logging

from backend.models.pydantic_models import AnalysisRequest, AnalysisResponse, StockInsight
from backend.services.analysis_service import AnalysisService
from backend.services.data_service import DataService
from backend.utils.exceptions import AnalysisException, DataFetchException
from backend.utils.helpers import validate_stock_symbols, calculate_date_range
from backend.utils.logger import AnalysisLogger

# Initialize router
router = APIRouter(
    prefix="/api/analysis",
    tags=["analysis"],
    responses={404: {"description": "Not found"}}
)

# Initialize services
analysis_service = AnalysisService()
data_service = DataService()
logger = AnalysisLogger(session_id="temp-agent-session")

# In-memory storage for active sessions (in production, use Redis or database)
active_sessions: Dict[str, Dict[str, Any]] = {}


@router.post("/start", response_model=Dict[str, Any])
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Start new stock analysis session
    
    Args:
        request: Analysis request with symbols, period, and horizon
        background_tasks: FastAPI background tasks
    
    Returns:
        Session ID and initial status
    """
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Validate stock symbols
        validated_symbols = validate_stock_symbols(request.symbols)
        if not validated_symbols:
            raise HTTPException(
                status_code=400,
                detail="No valid stock symbols provided"
            )
        
        # Calculate date range
        start_date, end_date = calculate_date_range(request.period_days)
        
        # Create session record
        session_data = {
            'session_id': session_id,
            'symbols': validated_symbols,
            'period_days': request.period_days,
            'prediction_horizon': request.prediction_horizon,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'status': 'initializing',
            'created_at': datetime.now().isoformat(),
            'progress': 0,
            'current_crew': None,
            'error': None,
            'results': None
        }
        
        # Store session
        active_sessions[session_id] = session_data
        
        # Log session creation
        logger.log_crew_activity(
            "AnalysisRouter",
            f"Analysis session {session_id} created with symbols: {validated_symbols}",
            "INFO"
        )
        
        # Start analysis in background
        background_tasks.add_task(
            run_analysis_background,
            session_id,
            validated_symbols,
            request.period_days,
            request.prediction_horizon
        )
        
        return {
            'session_id': session_id,
            'status': 'started',
            'symbols': validated_symbols,
            'estimated_duration_minutes': len(validated_symbols) * 2,  # Rough estimate
            'message': 'Analysis started successfully'
        }
        
    except Exception as e:
        logger.log_crew_activity(
            "AnalysisRouter",
            f"Error starting analysis: {str(e)}",
            "ERROR"
        )
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")


@router.get("/{session_id}/status", response_model=Dict[str, Any])
async def get_analysis_status(session_id: str) -> Dict[str, Any]:
    """
    Get analysis progress and current status
    
    Args:
        session_id: Analysis session ID
    
    Returns:
        Analysis progress and current status
    """
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = active_sessions[session_id]
        
        # Get detailed status from analysis service
        detailed_status = await analysis_service.get_analysis_status(session_id)
        
        # Combine session data with detailed status
        status_response = {
            'session_id': session_id,
            'status': session_data['status'],
            'progress': session_data['progress'],
            'current_crew': session_data['current_crew'],
            'symbols': session_data['symbols'],
            'created_at': session_data['created_at'],
            'error': session_data.get('error'),
            'estimated_completion': _calculate_estimated_completion(session_data),
            'detailed_status': detailed_status
        }
        
        return status_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.log_crew_activity(
            "AnalysisRouter",
            f"Error getting status for session {session_id}: {str(e)}",
            "ERROR"
        )
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/{session_id}/results", response_model=AnalysisResponse)
async def get_analysis_results(session_id: str) -> AnalysisResponse:
    """
    Retrieve finished analysis results
    
    Args:
        session_id: Analysis session ID
    
    Returns:
        Complete analysis results
    """
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = active_sessions[session_id]
        
        if session_data['status'] != 'completed':
            raise HTTPException(
                status_code=400,
                detail=f"Analysis not completed. Current status: {session_data['status']}"
            )
        
        if not session_data['results']:
            raise HTTPException(status_code=404, detail="Results not found")
        
        # Get comprehensive results from analysis service
        results = await analysis_service.get_analysis_results(session_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Results not available")
        
        # Format response
        response = AnalysisResponse(
            session_id=session_id,
            status="completed",
            symbols=session_data['symbols'],
            analysis_period=session_data['period_days'],
            prediction_horizon=session_data['prediction_horizon'],
            completed_at=session_data.get('completed_at'),
            executive_summary=results.get('executive_summary', {}),
            stock_insights=[
                StockInsight(**insight) for insight in results.get('stock_insights', [])
            ],
            comparative_analysis=results.get('comparative_analysis', {}),
            visualizations=results.get('visualizations', {}),
            recommendations=results.get('recommendations', {}),
            risk_assessment=results.get('risk_assessment', {}),
            metadata=results.get('metadata', {})
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.log_crew_activity(
            "AnalysisRouter",
            f"Error getting results for session {session_id}: {str(e)}",
            "ERROR"
        )
        raise HTTPException(status_code=500, detail=f"Failed to get results: {str(e)}")


@router.get("/{session_id}/logs", response_model=List[Dict[str, Any]])
async def get_analysis_logs(
    session_id: str,
    limit: int = Query(100, ge=1, le=1000),
    level: Optional[str] = Query(None, regex="^(DEBUG|INFO|WARNING|ERROR)$")
) -> List[Dict[str, Any]]:
    """
    Get analysis logs for a session
    
    Args:
        session_id: Analysis session ID
        limit: Maximum number of logs to return
        level: Filter by log level
    
    Returns:
        List of log entries
    """
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        logs = logger.get_session_logs(session_id, limit=limit, level=level)
        
        return logs
        
    except HTTPException:
        raise
    except Exception as e:
        logger.log_crew_activity(
            "AnalysisRouter",
            f"Error getting logs for session {session_id}: {str(e)}",
            "ERROR"
        )
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")


@router.delete("/{session_id}", response_model=Dict[str, str])
async def cancel_analysis(session_id: str) -> Dict[str, str]:
    """
    Cancel ongoing analysis session
    
    Args:
        session_id: Analysis session ID
    
    Returns:
        Cancellation confirmation
    """
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = active_sessions[session_id]
        
        if session_data['status'] in ['completed', 'failed', 'cancelled']:
            return {
                'session_id': session_id,
                'message': f'Session already {session_data["status"]}'
            }
        
        # Cancel the analysis
        await analysis_service.cancel_analysis(session_id)
        
        # Update session status
        session_data['status'] = 'cancelled'
        session_data['cancelled_at'] = datetime.now().isoformat()
        
        logger.log_crew_activity(
            "AnalysisRouter",
            f"Analysis session {session_id} cancelled",
            "INFO"
        )
        
        return {
            'session_id': session_id,
            'message': 'Analysis cancelled successfully'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.log_crew_activity(
            "AnalysisRouter",
            f"Error cancelling session {session_id}: {str(e)}",
            "ERROR"
        )
        raise HTTPException(status_code=500, detail=f"Failed to cancel analysis: {str(e)}")


@router.get("/sessions", response_model=List[Dict[str, Any]])
async def list_active_sessions(
    limit: int = Query(50, ge=1, le=100),
    status: Optional[str] = Query(None)
) -> List[Dict[str, Any]]:
    """
    List active analysis sessions
    
    Args:
        limit: Maximum number of sessions to return
        status: Filter by session status
    
    Returns:
        List of active sessions
    """
    try:
        sessions = []
        
        for session_id, session_data in list(active_sessions.items())[:limit]:
            if status and session_data['status'] != status:
                continue
                
            sessions.append({
                'session_id': session_id,
                'symbols': session_data['symbols'],
                'status': session_data['status'],
                'progress': session_data['progress'],
                'created_at': session_data['created_at'],
                'current_crew': session_data['current_crew']
            })
        
        return sessions
        
    except Exception as e:
        logger.log_crew_activity(
            "AnalysisRouter",
            f"Error listing sessions: {str(e)}",
            "ERROR"
        )
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@router.post("/validate-symbols", response_model=Dict[str, Any])
async def validate_symbols(symbols: List[str]) -> Dict[str, Any]:
    """
    Validate stock symbols before analysis
    
    Args:
        symbols: List of stock symbols to validate
    
    Returns:
        Validation results
    """
    try:
        validated_symbols = validate_stock_symbols(symbols)
        invalid_symbols = list(set(symbols) - set(validated_symbols))
        
        return {
            'valid_symbols': validated_symbols,
            'invalid_symbols': invalid_symbols,
            'total_valid': len(validated_symbols),
            'total_invalid': len(invalid_symbols)
        }
        
    except Exception as e:
        logger.log_crew_activity(
            "AnalysisRouter",
            f"Error validating symbols: {str(e)}",
            "ERROR"
        )
        raise HTTPException(status_code=500, detail=f"Failed to validate symbols: {str(e)}")


async def run_analysis_background(
    session_id: str,
    symbols: List[str],
    period_days: int,
    prediction_horizon: int
) -> None:
    """
    Run analysis in background task
    
    Args:
        session_id: Analysis session ID
        symbols: List of stock symbols
        period_days: Analysis period in days
        prediction_horizon: Prediction horizon in days
    """
    try:
        # Update session status
        if session_id in active_sessions:
            active_sessions[session_id]['status'] = 'running'
            active_sessions[session_id]['started_at'] = datetime.now().isoformat()
        
        # Run comprehensive analysis
        results = await analysis_service.run_full_analysis(
            symbols=symbols,
            period_days=period_days,
            prediction_horizon=prediction_horizon,
            session_id=session_id
        )
        
        # Update session with results
        if session_id in active_sessions:
            active_sessions[session_id]['status'] = 'completed'
            active_sessions[session_id]['progress'] = 100
            active_sessions[session_id]['results'] = results
            active_sessions[session_id]['completed_at'] = datetime.now().isoformat()
        
        logger.log_crew_activity(
            "AnalysisRouter",
            f"Background analysis completed for session {session_id}",
            "INFO"
        )
        
    except Exception as e:
        # Update session with error
        if session_id in active_sessions:
            active_sessions[session_id]['status'] = 'failed'
            active_sessions[session_id]['error'] = str(e)
            active_sessions[session_id]['failed_at'] = datetime.now().isoformat()
        
        logger.log_crew_activity(
            "AnalysisRouter",
            f"Background analysis failed for session {session_id}: {str(e)}",
            "ERROR"
        )


def _calculate_estimated_completion(session_data: Dict[str, Any]) -> Optional[str]:
    """Calculate estimated completion time"""
    try:
        if session_data['status'] in ['completed', 'failed', 'cancelled']:
            return None
        
        if 'started_at' not in session_data or session_data['progress'] == 0:
            return None
        
        started_at = datetime.fromisoformat(session_data['started_at'])
        elapsed = datetime.now() - started_at
        
        if session_data['progress'] > 0:
            total_estimated = elapsed * (100 / session_data['progress'])
            completion_time = started_at + total_estimated
            return completion_time.isoformat()
        
        return None
        
    except Exception:
        return None


# Cleanup task to remove old sessions
@router.on_event("startup")
async def cleanup_old_sessions():
    """Cleanup old sessions on startup"""
    try:
        # Remove sessions older than 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        sessions_to_remove = []
        for session_id, session_data in active_sessions.items():
            created_at = datetime.fromisoformat(session_data['created_at'])
            if created_at < cutoff_time:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del active_sessions[session_id]
        
        if sessions_to_remove:
            logger.log_crew_activity(
                "AnalysisRouter",
                f"Cleaned up {len(sessions_to_remove)} old sessions",
                "INFO"
            )
            
    except Exception as e:
        logger.log_crew_activity(
            "AnalysisRouter",
            f"Error during session cleanup: {str(e)}",
            "ERROR"
        )