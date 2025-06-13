"""
WebSocket Service - Handles real-time communication with frontend
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import websockets
from websockets.exceptions import ConnectionClosed

from backend.utils.logger import AnalysisLogger
from backend.utils.exceptions import AnalysisException


class WebSocketService:
    """Service for real-time WebSocket communication"""
    
    def __init__(self):
        self.logger = AnalysisLogger(session_id="temp-agent-session")
        self.active_connections: Dict[str, List[websockets.WebSocketServerProtocol]] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def register_connection(
        self, 
        session_id: str, 
        websocket: websockets.WebSocketServerProtocol
    ):
        """
        Register a new WebSocket connection for a session
        
        Args:
            session_id: Analysis session ID
            websocket: WebSocket connection object
        """
        try:
            if session_id not in self.active_connections:
                self.active_connections[session_id] = []
            
            self.active_connections[session_id].append(websocket)
            
            # Store connection metadata
            self.connection_metadata[id(websocket)] = {
                "session_id": session_id,
                "connected_at": datetime.utcnow(),
                "remote_address": websocket.remote_address
            }
            
            self.logger.log_crew_activity(
                "WebSocketService", 
                f"New connection registered for session {session_id}", 
                "info"
            )
            
            # Send welcome message
            await self._send_to_connection(websocket, {
                "type": "connection_established",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "message": "WebSocket connection established successfully"
            })
            
        except Exception as e:
            self.logger.log_crew_activity(
                "WebSocketService", 
                f"Failed to register connection: {str(e)}", 
                "error"
            )
    
    async def unregister_connection(
        self, 
        session_id: str, 
        websocket: websockets.WebSocketServerProtocol
    ):
        """
        Unregister a WebSocket connection
        
        Args:
            session_id: Analysis session ID
            websocket: WebSocket connection object
        """
        try:
            if session_id in self.active_connections:
                if websocket in self.active_connections[session_id]:
                    self.active_connections[session_id].remove(websocket)
                
                # Clean up empty session lists
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
            
            # Remove connection metadata
            if id(websocket) in self.connection_metadata:
                del self.connection_metadata[id(websocket)]
            
            self.logger.log_crew_activity(
                "WebSocketService", 
                f"Connection unregistered for session {session_id}", 
                "info"
            )
            
        except Exception as e:
            self.logger.log_crew_activity(
                "WebSocketService", 
                f"Failed to unregister connection: {str(e)}", 
                "error"
            )
    
    async def broadcast_log_update(self, session_id: str, log_entry: Dict[str, Any]):
        """
        Send real-time log updates to frontend
        
        Args:
            session_id: Analysis session ID
            log_entry: Log entry dictionary with message, level, timestamp
        """
        try:
            if session_id not in self.active_connections:
                return
            
            message = {
                "type": "log_update",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "log_entry": {
                    "message": log_entry.get("message", ""),
                    "level": log_entry.get("level", "info"),
                    "crew_name": log_entry.get("crew_name", "System"),
                    "timestamp": log_entry.get("timestamp", datetime.utcnow().isoformat())
                }
            }
            
            await self._broadcast_to_session(session_id, message)
            
        except Exception as e:
            self.logger.log_crew_activity(
                "WebSocketService", 
                f"Failed to broadcast log update: {str(e)}", 
                "error"
            )
    
    async def broadcast_progress_update(self, session_id: str, progress_data: Dict[str, Any]):
        """
        Send analysis progress updates to frontend
        
        Args:
            session_id: Analysis session ID
            progress_data: Progress information with percentage and message
        """
        try:
            if session_id not in self.active_connections:
                return
            
            message = {
                "type": "progress_update",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "progress": {
                    "percentage": progress_data.get("progress", 0),
                    "message": progress_data.get("message", ""),
                    "current_step": progress_data.get("current_step", ""),
                    "estimated_completion": progress_data.get("estimated_completion")
                }
            }
            
            await self._broadcast_to_session(session_id, message)
            
        except Exception as e:
            self.logger.log_crew_activity(
                "WebSocketService", 
                f"Failed to broadcast progress update: {str(e)}", 
                "error"
            )
    
    async def broadcast_crew_status(
        self, 
        session_id: str, 
        crew_name: str, 
        status: str, 
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Send crew execution status updates
        
        Args:
            session_id: Analysis session ID
            crew_name: Name of the crew being executed
            status: Current status (starting, running, completed, failed)
            details: Additional status details
        """
        try:
            if session_id not in self.active_connections:
                return
            
            message = {
                "type": "crew_status",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "crew_status": {
                    "crew_name": crew_name,
                    "status": status,
                    "details": details or {},
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            await self._broadcast_to_session(session_id, message)
            
        except Exception as e:
            self.logger.log_crew_activity(
                "WebSocketService", 
                f"Failed to broadcast crew status: {str(e)}", 
                "error"
            )
    
    async def broadcast_error(self, session_id: str, error_message: str, error_type: str = "general"):
        """
        Send error notifications to frontend
        
        Args:
            session_id: Analysis session ID
            error_message: Error message to display
            error_type: Type of error (general, data_error, model_error, etc.)
        """
        try:
            if session_id not in self.active_connections:
                return
            
            message = {
                "type": "error",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "error": {
                    "message": error_message,
                    "type": error_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            await self._broadcast_to_session(session_id, message)
            
        except Exception as e:
            self.logger.log_crew_activity(
                "WebSocketService", 
                f"Failed to broadcast error: {str(e)}", 
                "error"
            )
    
    async def broadcast_analysis_complete(self, session_id: str, summary: Dict[str, Any]):
        """
        Send analysis completion notification
        
        Args:
            session_id: Analysis session ID
            summary: Analysis summary with key metrics and insights
        """
        try:
            if session_id not in self.active_connections:
                return
            
            message = {
                "type": "analysis_complete",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "summary": {
                    "total_symbols": summary.get("total_symbols", 0),
                    "success_count": summary.get("success_count", 0),
                    "duration_seconds": summary.get("duration_seconds", 0),
                    "key_insights": summary.get("key_insights", []),
                    "recommendations": summary.get("recommendations", [])
                }
            }
            
            await self._broadcast_to_session(session_id, message)
            
        except Exception as e:
            self.logger.log_crew_activity(
                "WebSocketService", 
                f"Failed to broadcast analysis completion: {str(e)}", 
                "error"
            )
    
    async def _broadcast_to_session(self, session_id: str, message: Dict[str, Any]):
        """
        Broadcast message to all connections in a session
        
        Args:
            session_id: Target session ID
            message: Message to broadcast
        """
        if session_id not in self.active_connections:
            return
        
        # Get all connections for this session
        connections = self.active_connections[session_id].copy()
        disconnected_connections = []
        
        # Send to all active connections
        for connection in connections:
            try:
                await self._send_to_connection(connection, message)
            except ConnectionClosed:
                disconnected_connections.append(connection)
            except Exception as e:
                self.logger.log_crew_activity(
                    "WebSocketService", 
                    f"Failed to send to connection: {str(e)}", 
                    "warning"
                )
                disconnected_connections.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected_connections:
            await self.unregister_connection(session_id, connection)
    
    async def _send_to_connection(
        self, 
        connection: websockets.WebSocketServerProtocol, 
        message: Dict[str, Any]
    ):
        """
        Send message to a specific connection
        
        Args:
            connection: WebSocket connection
            message: Message to send
        """
        try:
            message_json = json.dumps(message, default=str)
            await connection.send(message_json)
        except ConnectionClosed:
            raise
        except Exception as e:
            self.logger.log_crew_activity(
                "WebSocketService", 
                f"Failed to send message to connection: {str(e)}", 
                "error"
            )
            raise
    
    async def handle_client_message(
        self, 
        session_id: str, 
        websocket: websockets.WebSocketServerProtocol, 
        message: str
    ):
        """
        Handle incoming messages from clients
        
        Args:
            session_id: Analysis session ID
            websocket: WebSocket connection
            message: Raw message string from client
        """
        try:
            # Parse incoming message
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await self._send_to_connection(websocket, {
                    "type": "error",
                    "message": "Invalid JSON format"
                })
                return
            
            message_type = data.get("type", "unknown")
            
            # Handle different message types
            if message_type == "ping":
                await self._send_to_connection(websocket, {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            elif message_type == "subscribe_logs":
                # Client wants to receive log updates
                await self._send_to_connection(websocket, {
                    "type": "log_subscription_confirmed",
                    "session_id": session_id
                })
            
            elif message_type == "get_status":
                # Client requesting current status
                await self._send_to_connection(websocket, {
                    "type": "status_response",
                    "session_id": session_id,
                    "active_connections": len(self.active_connections.get(session_id, [])),
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            else:
                self.logger.log_crew_activity(
                    "WebSocketService", 
                    f"Unknown message type: {message_type}", 
                    "warning"
                )
            
        except Exception as e:
            self.logger.log_crew_activity(
                "WebSocketService", 
                f"Failed to handle client message: {str(e)}", 
                "error"
            )
            
            await self._send_to_connection(websocket, {
                "type": "error",
                "message": "Failed to process message"
            })
    
    def get_active_sessions(self) -> List[str]:
        """
        Get list of sessions with active WebSocket connections
        
        Returns:
            List of active session IDs
        """
        return list(self.active_connections.keys())
    
    def get_connection_count(self, session_id: str) -> int:
        """
        Get number of active connections for a session
        
        Args:
            session_id: Analysis session ID
            
        Returns:
            Number of active connections
        """
        return len(self.active_connections.get(session_id, []))
    
    async def cleanup_session(self, session_id: str):
        """
        Clean up all connections for a session
        
        Args:
            session_id: Session ID to clean up
        """
        try:
            if session_id in self.active_connections:
                connections = self.active_connections[session_id].copy()
                
                # Close all connections
                for connection in connections:
                    try:
                        await connection.close()
                    except Exception as e:
                        self.logger.log_crew_activity(
                            "WebSocketService", 
                            f"Error closing connection: {str(e)}", 
                            "warning"
                        )
                
                # Remove from active connections
                del self.active_connections[session_id]
                
                self.logger.log_crew_activity(
                    "WebSocketService", 
                    f"Cleaned up {len(connections)} connections for session {session_id}", 
                    "info"
                )
            
        except Exception as e:
            self.logger.log_crew_activity(
                "WebSocketService", 
                f"Failed to cleanup session {session_id}: {str(e)}", 
                "error"
            )