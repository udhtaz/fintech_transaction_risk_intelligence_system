"""
FastAPI Server for Fintech Transaction Risk Intelligence System

This API provides endpoints for:
1. Making predictions on new transactions
2. Getting explanations for predictions
3. Batch processing of transactions
"""

import uvicorn
from api.main import app
from api.config import settings

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD
    )
