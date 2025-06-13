"""
WebSocket Router - McKinsey Stock Performance Monitor
FastAPI router for real-time WebSocket connections
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.websockets import WebSocketState
from typing import Dict, List, Any, Optional
import asyncio
import json
import logging
from datetime import datetime
import uuid

from services.websocket_service import WebSocketService
from utils.logger import AnalysisLogger
from utils.exceptions import AnalysisException

# Initialize router
router = APIRouter()

# Initialize services
websocket_service = WebSocketService()
logger = AnalysisLogger()

# Connection manager for active WebSocket connections
class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str) -> str:
        """Accept WebSocket connection and add to session"""
        await websocket.accept()
        
        # Generate connection ID
        connection_id = str(uuid.uuid4())
        
        # Add to active connections
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        
        self.active_connections[session_id].append(websocket)
        
        # Store connection metadata
        self.connection_metadata[connection_id] = {
            'session_id': session_id,
            'websocket': websocket,
            'connected_at': datetime.now().isoformat(),
            'last_ping': datetime.now().isoformat()
        }
        
        logger.log_crew_activity(
            "WebSocketRouter",
            f"New WebSocket connection for session {session_id}",
            "INFO"
        )
        
        return connection_id
    
    def disconnect(self, session_id: str, websocket: WebSocket):
        """Remove WebSocket connection"""
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)
            
            # Clean up if no more connections
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        
        # Clean up metadata
        connection_id_to_remove = None
        for conn_id, metadata in self.connection_metadata.items():
            if metadata['websocket'] == websocket:
                connection_id_to_remove = conn_id
                break
        
        if connection_id_to_remove:
            del self.connection_metadata[connection_id_to_remove]
        
        logger.log_crew_activity(
            "WebSocketRouter",
            f"WebSocket disconnected for session {session_id}",
            "INFO"
        )
    
    async def send_personal_message(self, message: Dict[str, Any], session_id: str):
        """Send message to all connections for a specific session"""
        if session_id not in self.active_connections:
            return
        
        message_json = json.dumps(message)
        disconnected_connections = []
        
        for websocket in self.active_connections[session_id]:
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(message_json)
                else:
                    disconnected_connections.append(websocket)
            except Exception as e:
                logger.log_crew_activity(
                    "WebSocketRouter",
                    f"Error sending message to WebSocket: {str(e)}",
                    "ERROR"
                )
                disconnected_connections.append(websocket)
        
        # Clean up disconnected connections
        for websocket in disconnected_connections:
            self.disconnect(session_id, websocket)
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all active connections"""
        message_json = json.dumps(message)
        
        for session_id, connections in self.active_connections.items():
            disconnected_connections = []
            
            for websocket in connections:
                try:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text(message_json)
                    else:
                        disconnected_connections.append(websocket)
                except Exception as e:
                    logger.log_crew_activity(
                        "WebSocketRouter",
                        f"Error broadcasting to WebSocket: {str(e)}",
                        "ERROR"
                    )
                    disconnected_connections.append(websocket)
            
            # Clean up disconnected connections
            for websocket in disconnected_connections:
                self.disconnect(session_id, websocket)
    
    def get_connection_count(self, session_id: Optional[str] = None) -> int:
        """Get number of active connections"""
        if session_id:
            return len(self.active_connections.get(session_id, []))
        
        return sum(len(connections) for connections in self.active_connections.values())
    
    def get_active_sessions(self) -> List[str]:
        """Get list of sessions with active connections"""
        return list(self.active_connections.keys())


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time analysis updates
    
    Args:
        websocket: WebSocket connection
        session_id: Analysis session ID
    """
    connection_id = None
    
    try:
        # Connect WebSocket
        connection_id = await manager.connect(websocket, session_id)
        
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            'type': 'connection_established',
            'session_id': session_id,
            'connection_id': connection_id,
            'timestamp': datetime.now().isoformat(),
            'message': 'WebSocket connection established'
        }))
        
        # Subscribe to real-time updates for this session
        await websocket_service.subscribe_to_session(session_id, manager)
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for message with timeout for ping/pong
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle incoming message
                await handle_websocket_message(websocket, session_id, data)
                
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_text(json.dumps({
                    'type': 'ping',
                    'timestamp': datetime.now().isoformat()
                }))
                
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        logger.log_crew_activity(
            "WebSocketRouter",
            f"WebSocket disconnected for session {session_id}",
            "INFO"
        )
    except Exception as e:
        logger.log_crew_activity(
            "WebSocketRouter",
            f"WebSocket error for session {session_id}: {str(e)}",
            "ERROR"
        )
        
        # Send error message if connection is still active
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'session_id': session_id,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }))
        except:
            pass
    
    finally:
        # Clean up connection
        if connection_id:
            manager.disconnect(session_id, websocket)
        
        # Unsubscribe from updates
        await websocket_service.unsubscribe_from_session(session_id, manager)


async def handle_websocket_message(websocket: WebSocket, session_id: str, message: str):
    """
    Handle incoming WebSocket messages
    
    Args:
        websocket: WebSocket connection
        session_id: Analysis session ID
        message: Received message
    """
    try:
        data = json.loads(message)
        message_type = data.get('type')
        
        if message_type == 'pong':
            # Handle pong response
            await websocket.send_text(json.dumps({
                'type': 'pong_received',
                'timestamp': datetime.now().isoformat()
            }))
            
        elif message_type == 'subscribe_logs':
            # Subscribe to log updates
            log_level = data.get('log_level', 'INFO')
            await websocket_service.subscribe_to_logs(session_id, manager, log_level)
            
            await websocket.send_text(json.dumps({
                'type': 'subscription_confirmed',
                'subscription': 'logs',
                'log_level': log_level,
                'timestamp': datetime.now().isoformat()
            }))
            
        elif message_type == 'subscribe_progress':
            # Subscribe to progress updates
            await websocket_service.subscribe_to_progress(session_id, manager)
            
            await websocket.send_text(json.dumps({
                'type': 'subscription_confirmed',
                'subscription': 'progress',
                'timestamp': datetime.now().isoformat()
            }))
            
        elif message_type == 'get_status':
            # Send current status
            status = await websocket_service.get_session_status(session_id)
            
            await websocket.send_text(json.dumps({
                'type': 'status_update',
                'session_id': session_id,
                'status': status,
                'timestamp': datetime.now().isoformat()
            }))
            
        elif message_type == 'unsubscribe':
            # Unsubscribe from updates
            subscription_type = data.get('subscription_type', 'all')
            await websocket_service.unsubscribe(session_id, manager, subscription_type)
            
            await websocket.send_text(json.dumps({
                'type': 'unsubscribe_confirmed',
                'subscription_type': subscription_type,
                'timestamp': datetime.now().isoformat()
            }))
            
        else:
            # Unknown message type
            await websocket.send_text(json.dumps({
                'type': 'error',
                'error': f'Unknown message type: {message_type}',
                'timestamp': datetime.now().isoformat()
            }))
            
    except json.JSONDecodeError:
        await websocket.send_text(json.dumps({
            'type': 'error',
            'error': 'Invalid JSON message format',
            'timestamp': datetime.now().isoformat()
        }))
        
    except Exception as e:
        logger.log_crew_activity(
            "WebSocketRouter",
            f"Error handling WebSocket message: {str(e)}",
            "ERROR"
        )
        
        await websocket.send_text(json.dumps({
            'type': 'error',
            'error': f'Message handling error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }))


@router.websocket("/ws/broadcast")
async def broadcast_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for system-wide broadcasts
    Used for admin notifications and system status updates
    """
    try:
        await websocket.accept()
        
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            'type': 'broadcast_connection_established',
            'timestamp': datetime.now().isoformat(),
            'message': 'Connected to broadcast channel'
        }))
        
        # Keep connection alive
        while True:
            try:
                # Wait for admin messages or system updates
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                
                # Handle broadcast message (admin only)
                await handle_broadcast_message(websocket, data)
                
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_text(json.dumps({
                    'type': 'broadcast_ping',
                    'timestamp': datetime.now().isoformat()
                }))
                
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        logger.log_crew_activity(
            "WebSocketRouter",
            "Broadcast WebSocket disconnected",
            "INFO"
        )
    except Exception as e:
        logger.log_crew_activity(
            "WebSocketRouter",
            f"Broadcast WebSocket error: {str(e)}",
            "ERROR"
        )


async def handle_broadcast_message(websocket: WebSocket, message: str):
    """Handle broadcast messages (admin functionality)"""
    try:
        data = json.loads(message)
        message_type = data.get('type')
        
        if message_type == 'system_announcement':
            # Broadcast system announcement to all connections
            announcement = {
                'type': 'system_announcement',
                'message': data.get('message'),
                'priority': data.get('priority', 'info'),
                'timestamp': datetime.now().isoformat()
            }
            
            await manager.broadcast_message(announcement)
            
            await websocket.send_text(json.dumps({
                'type': 'broadcast_sent',
                'recipients': manager.get_connection_count(),
                'timestamp': datetime.now().isoformat()
            }))
            
        elif message_type == 'connection_stats':
            # Send connection statistics
            stats = {
                'type': 'connection_stats',
                'total_connections': manager.get_connection_count(),
                'active_sessions': len(manager.get_active_sessions()),
                'sessions': manager.get_active_sessions(),
                'timestamp': datetime.now().isoformat()
            }
            
            await websocket.send_text(json.dumps(stats))
            
        elif message_type == 'shutdown_notification':
            # Notify all connections about system shutdown
            shutdown_msg = {
                'type': 'system_shutdown',
                'message': data.get('message', 'System maintenance scheduled'),
                'shutdown_time': data.get('shutdown_time'),
                'timestamp': datetime.now().isoformat()
            }
            
            await manager.broadcast_message(shutdown_msg)
            
            await websocket.send_text(json.dumps({
                'type': 'shutdown_notification_sent',
                'recipients': manager.get_connection_count(),
                'timestamp': datetime.now().isoformat()
            }))
            
    except json.JSONDecodeError:
        await websocket.send_text(json.dumps({
            'type': 'error',
            'error': 'Invalid JSON message format',
            'timestamp': datetime.now().isoformat()
        }))
        
    except Exception as e:
        logger.log_crew_activity(
            "WebSocketRouter",
            f"Error handling broadcast message: {str(e)}",
            "ERROR"
        )
        
        await websocket.send_text(json.dumps({
            'type': 'error',
            'error': f'Broadcast message handling error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }))


@router.get("/ws/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    try:
        return {
            'total_connections': manager.get_connection_count(),
            'active_sessions': len(manager.get_active_sessions()),
            'sessions': manager.get_active_sessions(),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@router.post("/ws/broadcast")
async def send_broadcast(message: Dict[str, Any]):
    """Send broadcast message to all connected clients"""
    try:
        broadcast_msg = {
            'type': 'admin_broadcast',
            'content': message,
            'timestamp': datetime.now().isoformat()
        }
        
        await manager.broadcast_message(broadcast_msg)
        
        return {
            'status': 'success',
            'recipients': manager.get_connection_count(),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Broadcast failed: {str(e)}")