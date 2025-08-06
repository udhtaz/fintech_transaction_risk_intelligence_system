"""
Configuration settings for the Fintech Transaction Risk Intelligence API
"""

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """API configuration settings"""
    API_TITLE: str = "Fintech Transaction Risk Intelligence API"
    API_DESCRIPTION: str = "API for detecting fraudulent or high-risk financial transactions"
    API_VERSION: str = "1.0.0"
    
    # Model paths
    MODEL_PATH: str = os.path.join("models", "fraud_detection_model.pkl")
    METADATA_PATH: str = os.path.join("models", "model_metadata.json")

    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Risk thresholds
    LOW_RISK_THRESHOLD: float = 0.3
    HIGH_RISK_THRESHOLD: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()
