# ===== utils/exceptions.py =====
class AnalysisException(Exception):
    """Base exception for analysis errors"""
    pass

class DataFetchException(AnalysisException):
    """Exception for data retrieval failures"""
    pass

class ModelTrainingException(AnalysisException):
    """Exception for model training failures"""
    pass

class ValidationException(AnalysisException):
    """Exception for data validation failures"""
    pass
