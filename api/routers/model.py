"""
Model information endpoints for the API
"""

from fastapi import APIRouter, HTTPException
from ..schemas.models import ModelInfoResponse
from ..helpers.prediction import load_model

router = APIRouter(prefix="/model", tags=["Model"])

@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the model"""
    try:
        model, metadata = load_model()
        
        return {
            "status": "success",
            "model_info": {
                "model_type": metadata.get("model_type", "unknown"),
                "model_version": metadata.get("model_version", "unknown"),
                "training_date": metadata.get("training_date", "unknown")
            },
            "features": metadata.get("features", []),
            "metrics": metadata.get("model_metrics", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
