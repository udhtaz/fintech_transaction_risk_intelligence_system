"""
Main FastAPI application for Fintech Transaction Risk Intelligence System
"""

from fastapi import FastAPI
from .config import settings
from .routers import health, model, prediction
import logging

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("fraud_detection_api")

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

# Include routers
app.include_router(health.router)
app.include_router(model.router)
app.include_router(prediction.router)

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD
    )
