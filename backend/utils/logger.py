# ===== utils/logger.py =====
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging():
    """Setup application logging"""
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler('logs/app.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

class AnalysisLogger:
    """Custom logger for analysis sessions"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.logger = logging.getLogger(f"analysis.{session_id}")
    
    def log_crew_activity(self, crew_name: str, message: str, level: str = "INFO"):
        """Log crew activity"""
        log_message = f"[{crew_name}] {message}"
        
        if level == "ERROR":
            self.logger.error(log_message)
        elif level == "WARNING":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def log_task_progress(self, task_name: str, progress: int):
        """Log task progress"""
        self.logger.info(f"Task '{task_name}' - Progress: {progress}%")